"""Bucle de phase2 / improve: rollouts -> reward -> GRPO -> checkpoint/evaluación, con timing por etapa.

Por step se registra en metrics.jsonl (lo que lee el panel):
* tiempos: t_sample, t_gen (generación con el sandbox solapado), t_score_wait (espera al sandbox tras generar), t_repair,
  t_ref / t_old / t_update (trainer), t_ckpt, t_eval, secs (total). ms_per_tok_step = ms por paso de decodificación
  del lote (independiente de la longitud de las respuestas): si se dispara, la GPU va lenta, no el modelo.
* memoria: vram_gb (pico asignado), vram_reserved_gb, rss_mb, ram_avail_min_mb, swap_used_max_mb.
* sistema (ventana del step en el monitor): util_mean, power_mean_w, cpu_mean_pct, pcie_*_max_mbs, zombies.
* anomalías: slow (secs > 3x la mediana de los últimos 20 steps), vram_paging (uso >= 90 % a < 60 % de la potencia
  típica, o tok/s < 35 % de la mediana con la VRAM llena): el diagnóstico que faltaba para los steps de 359 s.
"""
from __future__ import annotations

import atexit, json, math, os, random, shutil, signal, statistics, sys, time
from collections import deque
from dataclasses import asdict, fields

from common import ROOT, new_run_dir, opt
from problems.problems import select
from rollouts.collect import collect
from rollouts.generate import build_prompt
from sandbox import executor

_STOP = {"f": False, "n": 0}


# ---------------------------------------------------------------------------------------------------- utilidades
def train_pool(cfg, a):
    """Problemas de entrenamiento: SIEMPRE del split train (nunca val/test, aunque se pidan por id)."""
    sets = None if a.problems else tuple(opt(cfg, "dataset.sets", ["core", "catalog"]))
    return select(a.levels or cfg.dataset.levels, a.problems, sets=sets, splits=("train",))


def resume_best(ckdir, last_metrics, best_reward, best_val):
    """Mejores valores al reanudar: los de best/ (el máximo visto), no los del último checkpoint. Así una evaluación
    peor justo después de reanudar nunca sobrescribe best/."""
    from training import checkpoint as ck
    best_reward = last_metrics.get("eval_reward", best_reward)
    best_val = last_metrics.get("best_val", best_val)
    try:
        bm = ck.read_meta(os.path.join(ckdir, "best"))["metrics"]
    except Exception:
        return best_reward, best_val
    if isinstance(bm.get("eval_reward"), (int, float)):
        best_reward = max(best_reward, bm["eval_reward"])
    v = bm.get("best_val", bm.get("val_pass@1"))
    if isinstance(v, (int, float)):
        best_val = max(best_val, v)
    return best_reward, best_val


def step_metrics(groups):
    """Métricas aditivas de un step a partir de los grupos [(Sample, Result)] (no tocan el entrenamiento)."""
    from collections import Counter

    from rollouts.collect import code_key
    rs = [r for g in groups for _, r in g]
    ss = [s for g in groups for s, _ in g]
    if not rs:
        return {}
    n = len(rs)
    errs = Counter((r.error or "wrong_answer").split(":")[0] for r in rs if not r.solved)
    comps = [r.components for r in rs if r.components]
    out = dict(compile_rate=sum(r.compiled for r in rs) / n, trunc_rate=sum(r.truncated for r in rs) / n,
               frac_mean=sum(r.frac for r in rs) / n, near_miss=sum(r.pv == 1.0 and not r.solved for r in rs) / n,
               err_rates={k: round(v / n, 4) for k, v in errs.items()},
               zero_var_frac=sum(len(g) > 1 and statistics.pstdev([r.reward for _, r in g]) < 1e-4
                                 for g in groups) / len(groups),
               diversity=sum(len({code_key(s.code) for s, _ in g}) / len(g) for g in groups if g) / len(groups),
               ntok_mean=sum(s.ntok for s in ss) / n, sandbox_s=sum(r.wall_s for r in rs))
    if comps:
        out.update({f"comp_{k}": sum(c[k] for c in comps) / len(comps) for k in comps[0]})
    return out


def informative(rs, eps=1e-4):
    """Un grupo aporta señal a GRPO si el reward de sus muestras varía."""
    return len(rs) > 1 and statistics.pstdev([r.reward for _, r in rs]) >= eps


def control_rewards(keyed, cfg, step):
    """reward.control: random -> control de recompensa espuria (arXiv 2506.10947): las ventajas salen de un reward
    Bernoulli(0.5) sin información, con el mismo pipeline. Si un brazo así también "mejora", la mejora no viene de la
    señal de los tests. No toca los Result (la caché de score los comparte): el reward real se sigue registrando."""
    if opt(cfg, "reward.control", None) != "random":
        return keyed
    rr = random.Random((int(cfg.seed) * 1_000_003 + step) ^ 0x5EED)  # RNG propio: no altera el muestreo de problemas
    return [(pid, float(rr.random() < 0.5)) for pid, _ in keyed]


def dynamic_fill(gen, cfg, probs, batch, results, k, improver, step):
    """grpo.dynamic_sampling: mientras haya menos de k grupos con varianza, genera grupos de problemas NUEVOS del pool
    (nunca repite uno del step) hasta grpo.dynamic_max_factor * k problemas en total. -> (batch, results, extra)."""
    budget = int(math.ceil(float(opt(cfg, "grpo.dynamic_max_factor", 2.0)) * k)) - len(batch)
    extra = 0
    while budget > 0:
        need = k - sum(informative(rs) for rs in results)
        used = {p.id for p in batch}
        rest = [p for p in probs if p.id not in used]
        if need <= 0 or not rest:
            break
        m = min(need, budget, len(rest))
        more = [p for p in (improver.sample(rest, m) if improver else random.sample(rest, m)) if p.id not in used][:m]
        if not more:
            break
        batch, results = batch + more, results + collect(gen, more, cfg.num_rollouts, cfg, cfg.seed + step)
        budget -= len(more)
        extra += len(more)
    return batch, results, extra


_PREV_EVAL = {}  # split -> {problema: pass@1} de la evaluación anterior (para detectar olvido)


def _changes(sp, rep, thr=0.25):
    now = {r["id"]: r["pass_at"]["1"] for r in rep["rows"]}
    prev, _PREV_EVAL[sp] = _PREV_EVAL.get(sp), now
    if prev is None:
        return None
    common = [i for i in now if i in prev]
    up = [i for i in common if now[i] - prev[i] >= thr]
    down = [i for i in common if prev[i] - now[i] >= thr]
    return len(up), len(down), sorted(down)[:10]


def _trim_metrics(run, step0):
    """Al reanudar en step0, las filas con step >= step0 son de una ejecución que murió tras su último checkpoint:
    esos steps se van a repetir, así que se quitan (y una última línea a medio escribir) para no duplicarlos."""
    path = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(path):
        return
    keep = []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and not (isinstance(row.get("step"), int) and row["step"] >= step0):
                keep.append(json.dumps(row) + "\n")
    with open(path + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(path + ".tmp", path)


def _log(run, row):
    """metrics.jsonl del run: una línea por step/eval/evento. Es lo que lee el panel."""
    with open(os.path.join(run, "metrics.jsonl"), "a") as f:
        f.write(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}) + "\n")


def growth_plan(cfg, n_layers):
    """Eventos de crecimiento previstos por la config: [{"step", "after", "zero"}] (vacío si growth.enabled=false).
    growth.schedule: [{step: 0, new_layers: 2}, {step: 20, new_layers: 2}] (o growth.new_layers + growth.at_step)."""
    if not opt(cfg, "growth.enabled", False):
        return []
    if opt(cfg, "growth.method", "depth") != "depth":
        raise ValueError(f"growth.method no soportado: {opt(cfg, 'growth.method')}")
    sched = opt(cfg, "growth.schedule", None) or [dict(step=int(opt(cfg, "growth.at_step", 0)),
                                                       new_layers=int(opt(cfg, "growth.new_layers", 4)))]
    from model.growth import DEFAULT_ZERO, schedule_positions
    sizes = [int(e["new_layers"]) for e in sched]
    pos = [e.get("after") for e in sched]
    auto = schedule_positions(n_layers, sizes)
    zero = list(opt(cfg, "growth.zero_modules", None) or DEFAULT_ZERO)
    return [dict(step=int(e["step"]), after=list(p) if p else a, zero=zero) for e, p, a in zip(sched, pos, auto)]


def _eval_base_cache(cfg, sp, probs, n):
    """Ruta de la evaluación cacheada del modelo de partida (mismo modelo, problemas, semillas y generación)."""
    import hashlib
    key = json.dumps([cfg["model"], cfg["dtype"], sp, sorted((p.id, getattr(p, "version", "")) for p in probs), n,
                      opt(cfg, "eval.seed", 999983), opt(cfg, "eval.greedy", True), cfg["generation"],
                      cfg["max_new_tokens"], cfg["prompt"], cfg["reward_weights"], cfg["reward"]], sort_keys=True,
                     default=str)
    return os.path.join(ROOT, "runs", "eval_cache", hashlib.sha256(key.encode()).hexdigest()[:16] + f"_{sp}.json")


def split_eval(gen, cfg, train_probs, run, step, improver, legacy, base=None):
    """eval.during_train: evaluación por split (sin alterar el RNG). Escribe evals/stepNNNNNN_<split>.json y una fila
    event=eval_split en metrics.jsonl con la comparación PAREADA contra el modelo de partida (base) si la hay.
    Devuelve {split: informe}."""
    from evaluation.compare import compare
    from evaluation.evaluate import evaluate as eval_split
    out = {}
    n = opt(cfg, "eval.during_train_n", None)
    for sp in list(opt(cfg, "eval.during_train", []) or []):
        probs = train_probs if sp == "train" else select(sets=("core", "catalog"), splits=(sp,))
        if not probs:
            continue
        rep = eval_split(gen, probs, cfg, n=n, split=sp)
        out[sp] = rep
        os.makedirs(os.path.join(run, "evals"), exist_ok=True)
        with open(os.path.join(run, "evals", f"step{step:06d}_{sp}.json"), "w") as f:
            json.dump(rep, f)
        if improver:
            improver.observe_eval(sp, rep)
        print(f"      eval {sp}: pass@1={rep['pass@1']:.3f} IC95={rep['ci_pass@1']} greedy={rep.get('greedy')} "
              f"compile={rep['compile_rate']} trunc={rep['trunc_rate']}")
    if out:
        row = {"step": step, "t": time.time(), "event": "eval_split"}
        for sp, rep in out.items():
            row.update({f"{sp}_pass@{k}": rep[f"pass@{k}"] for k in rep["ks"]})
            ch = _changes(sp, rep)
            if ch:
                row.update({f"{sp}_improved": ch[0], f"{sp}_regressed": ch[1], f"{sp}_regressed_ids": ch[2]})
            row.update({f"{sp}_greedy": rep.get("greedy"), f"{sp}_ci_pass@1": rep["ci_pass@1"],
                        f"{sp}_compile": rep["compile_rate"], f"{sp}_trunc": rep["trunc_rate"],
                        f"{sp}_reward": rep["reward"], f"{sp}_errors": rep["error_rates"],
                        f"{sp}_diversity": rep["diversity"], f"{sp}_wall_s": rep.get("wall_s"),
                        f"{sp}_by_level": {lv: v.get("pass@1") for lv, v in rep["by_level"].items()}})
            if base and sp in base:  # ¿mejora de verdad? pareado por problema contra el modelo de partida
                c = compare(base[sp], rep)
                m = c["metrics"].get("pass@1")
                if m:
                    row.update({f"{sp}_delta_pass@1": m["delta"], f"{sp}_delta_ci": m["ci"], f"{sp}_p_sign": m["p_sign"],
                                f"{sp}_verdict": c["verdict"]})
                    print(f"      {sp} vs base: Δpass@1={m['delta']:+.3f} IC95={m['ci']} (+{m['wins']}/-{m['losses']}) "
                          f"-> {c['verdict']}")
        if "val" in out and legacy is not None:
            ref = out["train"]["pass@1"] if "train" in out else legacy["all"]["pass@1"]
            row.update(gap_train_val=ref - out["val"]["pass@1"], gap_ref="train" if "train" in out else "legacy")
        _log(run, row)
    return out


def base_evals(gen, cfg, run):
    """Evaluación del modelo de PARTIDA (antes del primer update: LoRA con B=0 y bloques crecidos = identidad) en los
    splits de eval.during_train, cacheada en runs/eval_cache (todos los runs/brazos con el mismo modelo la comparten)."""
    from evaluation.evaluate import evaluate as eval_split
    out = {}
    n = opt(cfg, "eval.during_train_n", None)
    for sp in [s for s in (opt(cfg, "eval.during_train", []) or []) if s != "train"]:
        probs = select(sets=("core", "catalog"), splits=(sp,))
        if not probs:
            continue
        path = _eval_base_cache(cfg, sp, probs, n or opt(cfg, "eval.n", 16))
        rep = None
        if os.path.exists(path):
            try:
                with open(path) as f:
                    rep = json.load(f)
            except (OSError, ValueError):
                rep = None
        if rep is None:
            print(f"      evaluando el modelo de partida en {sp} (una vez; se cachea en runs/eval_cache)")
            rep = eval_split(gen, probs, cfg, n=n, split=sp)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                json.dump(rep, f)
            os.replace(path + ".tmp", path)
        out[sp] = rep
        _PREV_EVAL[sp] = {r["id"]: r["pass_at"]["1"] for r in rep["rows"]}
        _log(run, {"step": -1, "t": time.time(), "event": "eval_base", f"{sp}_pass@1": rep["pass@1"],
                   f"{sp}_greedy": rep.get("greedy"), f"{sp}_ci_pass@1": rep["ci_pass@1"], "cache": path})
    return out


def evaluate_train(gen, probs, cfg, n_eval, seed):
    """Evaluación "legacy" sobre los problemas de entrenamiento (best/ por eval_reward y brecha train/val)."""
    from evaluation.metrics import summarize
    results = collect(gen, probs, n_eval, cfg, seed)
    rows = [dict(id=p.id, level=p.level, n=len(rs), solved=sum(r.solved for _, r in rs),
                 reward=[r.reward for _, r in rs], compiled=[r.compiled for _, r in rs], frac=[r.frac for _, r in rs],
                 ntok=[s.ntok for s, _ in rs], runtime=[r.runtime_s for _, r in rs], mem=[r.mem_mb for _, r in rs],
                 errors=[r.error for _, r in rs if r.error]) for p, rs in zip(probs, results)]
    return summarize(rows, sorted({1, n_eval}))


class Telemetry:
    """Contexto de sistema por step y detector de anomalías (lento / paginación de VRAM)."""

    def __init__(self, mon):
        self.mon, self.secs, self.tps, self.power = mon, deque(maxlen=20), deque(maxlen=20), deque(maxlen=20)
        self.flags = 0

    def window(self, t0, t1):
        from model.vram import pressure
        from rollouts import monitor as rm
        rows = self.mon.recent(t0, t1) if hasattr(self.mon, "recent") else []
        out = rm.summarize(rows) if rows else {}
        ref = statistics.median(self.power) if self.power else None
        pr = pressure(rows, ref_power=ref) if rows else {}
        zs = [r["zombies"] for r in rows if r.get("zombies") is not None]
        if zs:
            out["zombies"] = max(zs)
        return out, pr

    def judge(self, secs, tok_s, win, pr):
        med_s = statistics.median(self.secs) if len(self.secs) >= 5 else None
        med_t = statistics.median(self.tps) if len(self.tps) >= 5 else None
        slow = bool(med_s and secs > 3 * med_s)
        paging = bool(pr.get("busy_lowpower") or (med_t and tok_s and tok_s < 0.35 * med_t and pr.get("vram_full")))
        if not paging and not slow:  # sólo los steps sanos alimentan las referencias
            self.secs.append(secs)
            if tok_s:
                self.tps.append(tok_s)
            if win.get("power_mean_w"):
                self.power.append(win["power_mean_w"])
        return dict(slow=slow or None, vram_paging=paging or None,
                    secs_median=round(med_s, 2) if med_s else None, tok_s_median=round(med_t, 1) if med_t else None)


# ---------------------------------------------------------------------------------------------------- bucle
def phase2(cfg, a):
    import torch

    from model import growth as gr
    from model import vram
    from model.policy import checkpoint_events, load_policy
    from rollouts import collect as rc
    from rollouts.generate import Generator
    from rollouts.monitor import GpuMonitor
    from training import checkpoint as ck
    from training import optim as topt
    from training.grpo import GrpoTrainer, RollItem, StepStats, group_advantages, zero_var_keys
    from training.improve import Improver

    probs = train_pool(cfg, a)
    if len(probs) < 2:
        sys.exit("hacen falta >=2 problemas (GRPO necesita grupos con varianza de reward)")
    b = executor.backend(cfg.sandbox)
    h = ck.config_hash(cfg)
    upd = int(opt(cfg, "grpo.updates_per_batch", 1)), int(opt(cfg, "grpo.minibatches", 1))
    print(f"modo={a.cmd} sandbox={b} pool={len(probs)} G={cfg.num_rollouts} problems/step={cfg.grpo.problems_per_step} "
          f"updates/batch={upd[0]}x{upd[1]} lora={cfg.lora.enabled} growth={bool(opt(cfg, 'growth.enabled', False))} "
          f"config_hash={h}")
    if cfg.checkpoint.eval_every_steps % cfg.checkpoint.every_steps:
        print(f"      [aviso] la evaluación sólo ocurre en steps de checkpoint: con eval_every_steps="
              f"{cfg.checkpoint.eval_every_steps} y every_steps={cfg.checkpoint.every_steps} se evalúa cada "
              f"{math.lcm(cfg.checkpoint.eval_every_steps, cfg.checkpoint.every_steps)} steps")

    run = getattr(a, "run", None)
    if run:  # run explícito (experimentos): siempre ese directorio
        run = os.path.abspath(run)
        os.makedirs(run, exist_ok=True)
        if not os.path.exists(os.path.join(run, "config.yaml")):
            import yaml
            with open(os.path.join(run, "config.yaml"), "w") as f:
                yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False)
    else:
        run = None if a.fresh else ck.find_run_dir(os.path.join(ROOT, "runs", "phase2"), h)
        run = run or new_run_dir(cfg, "phase2")
    try:
        lock = ck.claim_run(run)  # otro proceso entrenando en este mismo run -> no pisar sus checkpoints
    except RuntimeError as e:
        sys.exit(str(e))
    atexit.register(ck.release_run, lock)
    ckdir = os.path.join(run, "checkpoints")
    print(f"run: {run}")
    mon = GpuMonitor(cfg.monitor.interval_s, os.path.join(run, "gpu.jsonl")).start()
    budget = vram.apply_budget(cfg) if torch.cuda.is_available() else None
    if budget:
        print(f"      [vram] total={budget['total_gb']} GB, otros procesos={budget['others_gb']} GB, "
              f"presupuesto de este proceso={budget['budget_gb']} GB")

    # ------------------------------------------------------------------ reanudación / arranque
    step0, best_reward, meta, resume_path = 0, -1e9, None, None
    best_val, select_best = -1.0, opt(cfg, "eval.select_best", "legacy")
    if not a.fresh:
        for path in ck.candidates(ckdir):
            try:
                m = ck.read_meta(path)
                if ck.meta_hash(path) != h:
                    print(f"      [checkpoint] {path}: modelo/dtype/lora/lr/weight_decay/growth distintos a la config "
                          f"actual, se ignora")
                    continue
                ck.verify(path)
                resume_path, meta = path, m
                break
            except Exception as e:
                print(f"      [checkpoint] {path} corrupto/ilegible ({e!r}), probando el anterior")
    init_from = getattr(a, "init_from", None)
    lineage = dict(run=run, parent=None)
    if not resume_path and init_from:
        ck.verify(init_from)
        lineage["parent"] = os.path.abspath(init_from)
        print(f"      [init] pesos iniciales de {init_from} (optimizador nuevo, step 0)")
    print("reanudando desde checkpoint" if resume_path else "arranque limpio (sin checkpoint compatible)")

    model = tok = ref_ctx = optimizer = None
    events = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU: sólo para tests de extremo a extremo con modelos mínimos
    try:
        if resume_path:
            model, tok, ref_ctx = load_policy(cfg, dev, resume_from=resume_path)
            events = checkpoint_events(resume_path)
            optimizer = topt.build(model, cfg, events)
            ck.load_optim_rng(resume_path, optimizer, model.device)
            step0 = meta["step"] + 1
            best_reward, best_val = resume_best(ckdir, meta["metrics"], best_reward, best_val)
            lineage = meta.get("lineage") or lineage
            imp_path = os.path.join(resume_path, "improve.json")
            if os.path.exists(imp_path):  # el currículo vuelve al estado de ESE checkpoint
                shutil.copyfile(imp_path, os.path.join(run, "improve.json"))
            print(f"      [checkpoint] reanudado desde {resume_path} (step={meta['step']})")
        else:
            model, tok, ref_ctx = load_policy(cfg, dev, resume_from=init_from or None)
            events = checkpoint_events(init_from) if init_from else []
            optimizer = topt.build(model, cfg, events)
    except Exception as e:
        if not resume_path:
            raise  # el modelo base ni siquiera carga: nada razonable que reintentar
        print(f"      [checkpoint] {resume_path} no cargó ({e!r}); cae a arranque limpio en este run")
        del model, optimizer
        torch.cuda.empty_cache()
        model, tok, ref_ctx = load_policy(cfg, dev, resume_from=None)
        events = []
        optimizer = topt.build(model, cfg, events)
        step0, best_reward, best_val = 0, -1e9, -1.0

    n_orig = len(gr.original_indices(model))
    plan = growth_plan(cfg, n_orig)[len(events):]  # los ya aplicados vienen del checkpoint (meta.json)
    total0, train0 = gr.count_params(model)
    print(f"      params: total={total0 / 1e6:.1f}M entrenables={train0 / 1e6:.2f}M capas={len(gr.decoder(model).layers)} "
          f"crecimientos pendientes={[(e['step'], len(e['after'])) for e in plan]}")

    gen = Generator(model, tok, cfg)
    trainer = GrpoTrainer(model, ref_ctx, tok, optimizer, cfg)
    if not cfg.lora.enabled:
        print("      [aviso] full-FT: el autotuner de abajo NO incluye el estado de AdamW (varios GB en fp32); "
              "vram.budget_gb y vram_margin son la red de seguridad aquí")

    # autotune SIEMPRE (no cachear entre runs): el optimizer ya inicializado cambia el presupuesto de VRAM real.
    longest = max(probs, key=lambda p: len(p.prompt()))
    lp = build_prompt(tok, cfg.prompt.system, longest.prompt(), cfg.prompt.prefill_fence)

    def tune_all(regrow=False):
        if not torch.cuda.is_available():  # los autotuners miden VRAM: sin GPU se usan los valores de la config
            return
        for r in gen.tune(lp, start=gen.chunk if regrow else 1):
            print("tune[gen]", r)
        for r in trainer.tune(tok(lp).input_ids, gen.max_new):
            print("tune[train]", r)
        print(f"gen.chunk={gen.chunk} train.micro_bs={trainer.micro_bs} (peor caso: {trainer.ref_len} tokens)")
    tune_all()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    n_eval = max(1, cfg.num_rollouts // 2)
    improver = Improver(run, probs, cfg) if a.cmd == "improve" else None
    _trim_metrics(run, step0)
    _log(run, dict(event="start", t=time.time(), mode=a.cmd, step0=step0, resumed_from=resume_path if step0 else None,
                   init_from=lineage.get("parent"), chunk=gen.chunk, micro_bs=trainer.micro_bs, vram=budget,
                   params_total=total0, params_trainable=train0, layers=len(gr.decoder(model).layers)))
    if "test" in (opt(cfg, "eval.during_train", []) or []):
        print("      [aviso] eval.during_train incluye test: mirarlo durante el entrenamiento contamina la evaluación "
              "final; úsalo sólo al final con train.py eval --split test")
    if select_best == "val" and "val" not in (opt(cfg, "eval.during_train", []) or []):
        print("      [aviso] eval.select_best=val sin 'val' en eval.during_train: best/ se elige como siempre (legacy)")
    base = base_evals(gen, cfg, run) if opt(cfg, "eval.paired_base", True) and step0 == 0 and not events else {}
    if not base and opt(cfg, "eval.paired_base", True):  # reanudación: la base cacheada si existe
        base = _cached_base(cfg)
    adv_norm, skip_zero = opt(cfg, "grpo.adv_norm", "std"), bool(opt(cfg, "grpo.skip_zero_var", False))
    dynamic = bool(opt(cfg, "grpo.dynamic_sampling", False))
    patience, min_delta = int(opt(cfg, "eval.patience", 0)), float(opt(cfg, "eval.min_delta", 0.0))
    stall_evals, best_seen_val = 0, best_val
    tele = Telemetry(mon)
    step, last_metrics = step0, {}

    def extra_meta():
        return dict(arch=gr.arch(model), growth=events, lineage=lineage, code=ck.code_hash(),
                    config=json.loads(json.dumps(cfg, default=str)), env=_env())

    def save_ckpt(s, metrics, is_best):
        files = {}
        if improver:
            improver.save()
            files["improve.json"] = lambda p: shutil.copyfile(improver.path, p)
        return ck.save(ckdir, s, model, optimizer, metrics, cfg.lora.enabled, h, is_best, extra=extra_meta(),
                       files=files)

    while step < cfg.grpo.max_steps and not _STOP["f"]:
        T = {}
        t_step = time.time()
        tok0, sec0, steps0 = gen.stats["tokens"], gen.stats["seconds"], gen.stats.get("steps", 0)
        tm0 = dict(rc.TIMING)
        cache0, skipped_all, zero = rc.CACHE.stats(), False, set()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            # ---------------- crecimiento programado para este step (su tiempo, re-ajuste incluido, va en t_growth)
            t_grow = time.time()
            while plan and plan[0]["step"] <= step:
                ev = dict(plan.pop(0), step=step)
                p_before = gr.count_params(model)
                gr.apply_event(model, ev)
                topt.add_growth_group(optimizer, model, cfg, ev, len(events))
                events.append({k: v for k, v in ev.items()})
                p_after = gr.count_params(model)
                print(f"      [growth] step {step}: +{len(ev['new'])} bloques identidad tras las capas originales "
                      f"{ev['after']} -> {ev['layers_after']} capas; params {p_before[0] / 1e6:.1f}M -> "
                      f"{p_after[0] / 1e6:.1f}M, entrenables {p_before[1] / 1e6:.2f}M -> {p_after[1] / 1e6:.2f}M")
                _log(run, dict({k: v for k, v in ev.items() if k != "zero"}, event="growth", t=time.time(),
                               params_total=p_after[0], params_trainable=p_after[1]))  # ev ya trae "step"
                tune_all(regrow=True)  # la memoria del step cambió
                T["t_growth"] = time.time() - t_grow
            lrs = topt.set_lr(optimizer, step)

            t0 = time.time()
            k = min(cfg.grpo.problems_per_step, len(probs))
            batch = improver.sample(probs, k) if improver else random.sample(probs, k)
            T["t_sample"] = time.time() - t0
            # Los rollouts de cada step dependen sólo de (semilla, step, pesos): antes el RNG de torch llegaba aquí
            # consumido por el autoajuste (cuántas pruebas hace depende de tiempos medidos) y por el crecimiento, así
            # que dos brazos con los mismos pesos generaban rollouts distintos y un run no se podía reproducir.
            torch.manual_seed((cfg.seed * 1_000_003 + step) % 2 ** 63)
            results = collect(gen, batch, cfg.num_rollouts, cfg, cfg.seed + step)
            dyn_extra = None
            if dynamic:  # reponer grupos sin varianza con problemas nuevos
                batch, results, dyn_extra = dynamic_fill(gen, cfg, probs, batch, results, k, improver, step)
            flat = [(p.id, s, r) for p, rs in zip(batch, results) for s, r in rs]
            t0 = time.time()
            fixes = improver.repair(gen, batch, results, cfg.seed + step) if improver else []
            T["t_repair"] = time.time() - t0
            T["t_gen"] = rc.TIMING["gen_wall"] - tm0["gen_wall"]
            T["t_score_wait"] = rc.TIMING["score_wait"] - tm0["score_wait"]
            keyed = control_rewards([(pid, r.reward) for pid, _, r in flat + fixes], cfg, step)
            advs = group_advantages(keyed, norm=adv_norm)
            zero = zero_var_keys(keyed) if skip_zero or dynamic else set()
            items = [RollItem(s.prompt_ids, s.resp_ids, adv) for (pid, s, _), adv in zip(flat + fixes, advs)
                     if pid not in zero]
            torch.cuda.empty_cache()  # devolver lo que reservó la generación antes del backward
            if items:
                stats = trainer.step(items)
            else:  # ningún grupo con señal: no hay actualización este step
                stats, skipped_all = StepStats(**{f.name: 0 for f in fields(StepStats)}), True
        except BaseException as e:
            # model/optimizer reflejan el último step que SÍ completó, salvo que el trainer ya aplicara
            # `updates_done` actualizaciones de ESTE step: entonces el modelo es el de este step y se guarda como tal.
            last = step if getattr(e, "updates_done", 0) else step - 1
            if last >= step0:
                path = save_ckpt(last, last_metrics, False)
                print(f"      [emergencia] {type(e).__name__}; checkpoint del último estado completo "
                      f"(step={last}) -> {path}")
            raise
        if improver:
            improver.update(batch, results)
        rmean = sum(r.reward for _, _, r in flat) / len(flat)
        solved = sum(r.solved for _, _, r in flat)
        fsolved = sum(r.solved for _, _, r in fixes)
        gsec = gen.stats["seconds"] - sec0
        gsteps = gen.stats.get("steps", 0) - steps0
        tok_s = (gen.stats["tokens"] - tok0) / gsec if gsec > 0 else None
        secs = time.time() - t_step
        win, pr = tele.window(t_step, time.time())
        # el crecimiento (insertar bloques + re-ajuste) es un coste conocido, no una anomalía: no cuenta como lento
        verdict = tele.judge(secs - (T.get("t_growth") or 0), tok_s, win, pr)
        extra = f" fix={fsolved}/{len(fixes)} niveles<=L{improver.levels[improver.open - 1]}" if improver else ""
        warn = ""
        if stats.skipped:
            warn += " [aviso] gradiente no finito: update NO aplicado"
        if verdict["vram_paging"]:
            warn += (" [aviso] VRAM paginando a RAM (GPU ocupada a baja potencia / tok/s hundido): cierra otras apps de "
                     "GPU o baja vram.budget_gb")
            torch.cuda.empty_cache()
        elif verdict["slow"]:
            warn += f" [aviso] step lento ({secs:.0f}s vs mediana {verdict['secs_median']}s)"
        print(f"step={step} loss={stats.loss:+.4f} kl={stats.kl:.4f} clip={stats.clip_frac:.3f} "
              f"grad_norm={stats.grad_norm:.3f} reward={rmean:.3f} solved={solved}/{len(flat)} "
              f"micro_bs={trainer.micro_bs} oom={stats.oom} ({secs:.1f}s: gen {T['t_gen']:.1f} "
              f"sbx+{T['t_score_wait']:.1f} train {stats.t_ref + stats.t_old + stats.t_update:.1f}){extra}{warn}")
        last_metrics = dict(**asdict(stats), reward_mean=rmean)
        cache1 = rc.CACHE.stats()
        row = dict(step=step, t=time.time(), mode=a.cmd, problems=[p.id for p in batch], **asdict(stats), reward=rmean,
                   solved=solved, n=len(flat),
                   fix_solved=fsolved, fix_n=len(fixes), secs=secs, gen_secs=gsec, tok_s=tok_s,
                   ms_per_tok_step=1000 * gsec / gsteps if gsteps else None, chunk=gen.chunk, micro_bs=trainer.micro_bs,
                   open_level=improver.levels[improver.open - 1] if improver else None,
                   **step_metrics(results), gen_tokens=gen.stats["tokens"] - tok0,
                   cache_hits=cache1["hits"] - cache0["hits"], cache_misses=cache1["misses"] - cache0["misses"],
                   vram_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
                   vram_reserved_gb=torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None,
                   skipped_zero_var=len(zero) if skip_zero or dynamic else None, no_update=skipped_all or None,
                   dyn_extra=dyn_extra, dyn_informative=sum(informative(rs) for rs in results) if dynamic else None,
                   lr=lrs, **T, **{f"sys_{k}": v for k, v in win.items() if k != "samples"},
                   **{k: v for k, v in pr.items() if k in ("vram_full", "busy_lowpower")}, **verdict,
                   **({f"imp_{k}": v for k, v in improver.metrics().items()} if improver else {}))
        _log(run, row)

        do_ckpt = (step + 1) % cfg.checkpoint.every_steps == 0 or step + 1 == cfg.grpo.max_steps or _STOP["f"]
        if do_ckpt:
            t0 = time.time()
            metrics = dict(**asdict(stats), reward_mean=rmean)
            is_best = False
            if not _STOP["f"] and (step + 1) % cfg.checkpoint.eval_every_steps == 0:
                ev_ = evaluate_train(gen, probs, cfg, n_eval, cfg.seed - 1)
                metrics["eval_reward"], metrics["eval_pass@1"] = ev_["all"]["reward"], ev_["all"]["pass@1"]
                is_best = metrics["eval_reward"] > best_reward
                best_reward = max(best_reward, metrics["eval_reward"])
                sp = split_eval(gen, cfg, probs, run, step, improver, ev_, base)
                if "val" in sp and patience > 0:  # early stopping por val (eval.patience; 0 = desactivado)
                    v = sp["val"]["pass@1"]
                    stall_evals, best_seen_val = (0, v) if v > best_seen_val + min_delta else (stall_evals + 1,
                                                                                                 best_seen_val)
                    if stall_evals >= patience:
                        print(f"      [early stop] val pass@1 sin mejorar en {patience} evaluaciones seguidas "
                              f"(mejor {best_seen_val:.3f}): se detiene tras este checkpoint")
                        _log(run, {"step": step, "t": time.time(), "event": "early_stop", "patience": patience,
                                   "best_val_pass@1": best_seen_val})
                        _STOP["f"] = True
                if "val" in sp:
                    metrics["val_pass@1"] = sp["val"]["pass@1"]
                    if select_best == "val":
                        is_best = sp["val"]["pass@1"] > best_val
                        best_val = max(best_val, sp["val"]["pass@1"])
                        metrics["best_val"] = best_val
                print(f"      eval: reward={ev_['all']['reward']:.3f} pass@1={ev_['all']['pass@1']:.3f} "
                      f"{'(nuevo best)' if is_best else ''}")
                _log(run, {"step": step, "t": time.time(), "eval_reward": ev_["all"]["reward"],
                           "eval_pass@1": ev_["all"]["pass@1"], f"eval_pass@{n_eval}": ev_["all"][f"pass@{n_eval}"],
                           "eval_compile": ev_["all"]["compile"], "best": is_best, "t_eval": time.time() - t0})
            t1 = time.time()
            path = save_ckpt(step, metrics, is_best)
            for old in sorted((d for d in os.listdir(ckdir) if d.startswith("ckpt_")))[:-cfg.checkpoint.keep_last]:
                if old != os.path.basename(path):
                    shutil.rmtree(os.path.join(ckdir, old), ignore_errors=True)
            _log(run, {"step": step, "t": time.time(), "event": "checkpoint", "t_ckpt": time.time() - t1,
                       "path": os.path.basename(path)})
            print(f"      checkpoint -> {path}" + ("  (emergencia: señal recibida)" if _STOP["f"] else ""))
        step += 1

    if _STOP["f"]:
        print(f"\n[phase2] señal recibida tras completar step={step - 1}; checkpoint de ese step ya guardado "
              f"arriba. Reanuda con el mismo comando.")
    hw = mon.stop()
    print("gpu:", hw)
    print(f"guardado en {run}" + (" (interrumpido)" if _STOP["f"] else " (max_steps alcanzado)"))
    return run


def _cached_base(cfg):
    out = {}
    for sp in [s for s in (opt(cfg, "eval.during_train", []) or []) if s != "train"]:
        probs = select(sets=("core", "catalog"), splits=(sp,))
        path = _eval_base_cache(cfg, sp, probs, opt(cfg, "eval.during_train_n", None) or opt(cfg, "eval.n", 16))
        try:
            with open(path) as f:
                out[sp] = json.load(f)
        except (OSError, ValueError):
            pass
    return out


def _env():
    try:
        from common import env_info
        return env_info()
    except Exception:
        return {}


def _on_signal(*_):
    """1.ª señal: terminar el step en curso, guardar checkpoint y salir. 2.ª: abortar ya (checkpoint de emergencia
    del último step completo)."""
    _STOP["n"] += 1
    _STOP["f"] = True
    if _STOP["n"] == 1:
        print("\n[señal] se guardará checkpoint al terminar este step (otra vez Ctrl+C para abortar ya)", flush=True)
    else:
        raise KeyboardInterrupt
