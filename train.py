"""Fase 1: modelo -> generación -> sandbox -> tests -> reward. Mide el baseline del modelo sin entrenar.
Fase 2: + GRPO (LoRA por defecto, crecimiento opcional) + checkpoint/recovery automático (training/loop.py).
fix: generate -> test -> fix con feedback del sandbox (evaluación; sus fallos alimentan a improve).
improve: el bucle de phase2 + currículo + grupos de reparación. Continúa el último run compatible de phase2.
eval: evaluación por split (greedy + pass@k + IC95 + desgloses), de checkpoints, curvas o el modelo base.

python train.py phase1 [--n 8] [--levels 0 1 2 3] [--problems id ...] [--tune] [--dump]
python train.py phase2 [--n 8] [--levels 0 1 2 3] [--problems id ...] [--fresh] [--run DIR] [--init-from CKPT]
python train.py improve [...las mismas...]
python train.py fix [--iters 3] [--chains 1] [--levels ...] [--problems id ...] [--checkpoint DIR]
python train.py eval --split val [--checkpoint DIR ...] [--run RUN] [--with-base] [--n 16] [--out DIR]
python train.py distill [--teacher MODELO] [--k 8] [--per-problem 4] [--epochs 2] [--batch 16] [--lr 1e-4] [--run DIR]
python train.py genbench [--checkpoint DIR] [--problems 6] [--reps 3]   # sólo mide la generación (CUDA graphs)
python train.py humaneval [--checkpoint DIR] [--n 8] [--out DIR]   # benchmark externo (164 problemas) en el sandbox
"""
import argparse, json, os, sys
from dataclasses import asdict

from common import INFERENCE_VERSION, ROOT, load_config, new_run_dir, opt, seed_all, workers
from evaluation.metrics import summarize, table
from problems.problems import select
from rollouts.collect import collect
from sandbox import executor


def phase1(cfg, a):
    probs = select(a.levels or cfg.dataset.levels, a.problems)
    if not probs:
        sys.exit("ningún problema seleccionado")
    n = int(cfg.num_rollouts)
    b = executor.backend(cfg.sandbox)  # falla pronto si no hay aislamiento
    print(f"sandbox={b} problemas={len(probs)} rollouts={n} workers={workers(cfg)}")

    import torch
    from model.loader import load
    from rollouts.generate import Generator, build_prompt
    from rollouts.monitor import GpuMonitor

    run = new_run_dir(cfg, "phase1")
    mon = GpuMonitor(cfg.monitor.interval_s, os.path.join(run, "gpu.jsonl")).start()
    model, tok = load(cfg)
    gen = Generator(model, tok, cfg)
    prompts = [build_prompt(tok, cfg.prompt.system, p.prompt(), cfg.prompt.prefill_fence) for p in probs]
    print(f"dtype={next(model.parameters()).dtype} params={sum(x.numel() for x in model.parameters()) / 1e6:.0f}M "
          f"prompt_tokens(max)={max(len(tok(x).input_ids) for x in prompts)} vram={torch.cuda.memory_allocated() >> 20}MB")

    tuned = os.path.join(ROOT, "runs", "tuned.json")
    cache = json.load(open(tuned)) if os.path.exists(tuned) else {}
    key = f"{torch.cuda.get_device_name(0)}|{cfg.model}|{cfg.dtype}|{cfg.max_new_tokens}"
    if a.tune:
        for r in gen.tune(max(prompts, key=len)):
            print("tune", r)
        cache[key] = gen.chunk
        json.dump(cache, open(tuned, "w"), indent=1)
    elif key in cache:
        gen.chunk = cache[key]
    print(f"chunk={gen.chunk} max_new_tokens={gen.max_new}")

    torch.cuda.reset_peak_memory_stats()
    results = collect(gen, probs, n, cfg, cfg.seed)  # la GPU sigue generando mientras el sandbox verifica lotes anteriores
    rows, dump = [], []
    for p, rs in zip(probs, results):
        rows.append(dict(id=p.id, level=p.level, n=len(rs), solved=sum(r.solved for _, r in rs),
                         reward=[r.reward for _, r in rs], compiled=[r.compiled for _, r in rs],
                         frac=[r.frac for _, r in rs], ntok=[s.ntok for s, _ in rs],
                         runtime=[r.runtime_s for _, r in rs], mem=[r.mem_mb for _, r in rs],
                         errors=[r.error for _, r in rs if r.error]))
        if a.dump:
            dump += [dict(problem=p.id, code=s.code, ntok=s.ntok, closed=s.closed, **asdict(r)) for s, r in rs]

    ks = sorted({1, n})
    summ, hw, st = summarize(rows, ks), mon.stop(), gen.stats
    perf = dict(tokens_s=round(st["tokens"] / max(st["seconds"], 1e-9), 1),
                samples_s=round(st["samples"] / max(st["seconds"], 1e-9), 2), gen_seconds=round(st["seconds"], 1),
                oom_events=st["oom"], chunk=gen.chunk, max_new_tokens=gen.max_new,
                vram_peak_alloc_mb=torch.cuda.max_memory_allocated() >> 20,
                vram_peak_reserved_mb=torch.cuda.max_memory_reserved() >> 20)
    brief = [dict(id=r["id"], level=r["level"], n=r["n"], solved=r["solved"],
                  reward=round(sum(r["reward"]) / max(r["n"], 1), 3)) for r in rows]
    json.dump(dict(summary=summ, perf=perf, hw=hw, problems=brief), open(os.path.join(run, "metrics.json"), "w"), indent=1)
    if a.dump:
        with open(os.path.join(run, "samples.jsonl"), "w") as f:
            f.writelines(json.dumps(d) + "\n" for d in dump)
    print(table(summ, ks))
    print("errores:", summ["errors"])
    print("gen:", perf)
    print("gpu:", hw)
    print("guardado en", run)


def phase2(cfg, a):
    from training.loop import phase2 as run_phase2
    return run_phase2(cfg, a)


def distill(cfg, a):
    from training.distill import run as run_distill
    return run_distill(cfg, a)


def genbench(cfg, a):
    from rollouts.genbench import run as run_genbench
    return run_genbench(cfg, a)


def humaneval(cfg, a):
    from evaluation.humaneval import run as run_humaneval
    return run_humaneval(cfg, a)


def fix(cfg, a):
    probs = select(a.levels or cfg.dataset.levels, a.problems)
    if not probs:
        sys.exit("ningún problema seleccionado")
    b = executor.backend(cfg.sandbox)
    from model.infer import Engine
    from rollouts import fix as gtf
    from rollouts.monitor import GpuMonitor

    rounds = cfg.fix.iters if a.iters is None else a.iters
    for k in ("engine", "feedback", "candidates"):
        if getattr(a, k, None) is not None:
            cfg["fix"][k] = getattr(a, k)
    gen = Engine(cfg, a.checkpoint).generator()  # antes del run: si el modelo no carga, no queda un run vacío
    run = new_run_dir(cfg, "fix")
    mon = GpuMonitor(cfg.monitor.interval_s, os.path.join(run, "gpu.jsonl")).start()
    print(f"sandbox={b} problemas={len(probs)} cadenas/problema={a.chains} rondas={rounds} "
          f"checkpoint={a.checkpoint or 'modelo base'}")
    with open(os.path.join(run, "attempts.jsonl"), "a") as f:
        def log(att, r):
            f.write(json.dumps(dict(asdict(att), detail=r.detail)) + "\n")
            f.flush()
            print(f"  ronda {att.round} {att.problem}#{att.chain}: reward={att.reward:.3f} "
                  f"{'resuelto' if att.solved else att.error}")
        if getattr(a, "compare", False):
            res = gtf.compare(gen, probs, cfg, rounds, a.chains, cfg.seed, log)
            cmp = {m: v["summary"] for m, v in res.items()}
            with open(os.path.join(run, "compare.json"), "w") as fc:
                json.dump(dict(compare=cmp, checkpoint=a.checkpoint, rounds=rounds, engine=opt(cfg, "fix.engine", "v1"),
                               problems=[p.id for p in probs], chains=a.chains, seed=cfg.seed), fc, indent=1)
            keys = ("fix_rate", "oracle_any", "reward_gain", "repetition", "same_error", "success_per_candidate")
            print("métrica".ljust(22) + "basic".rjust(10) + "counterexample".rjust(16))
            for k in keys:
                print(k.ljust(22) + str(cmp["basic"].get(k)).rjust(10) + str(cmp["counterexample"].get(k)).rjust(16))
            chains = res["counterexample"]["chains"]
        else:
            chains = gtf.run(gen, probs, cfg, rounds, a.chains, cfg.seed, log)
    summ = dict(gtf.summary(chains), checkpoint=a.checkpoint, rounds=rounds)
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(dict(summary=summ, gen=gen.stats, hw=mon.stop()), f, indent=1)
    print(json.dumps(summ, indent=1))
    print("guardado en", run)


def eval_cmd(cfg, a):
    """python train.py eval --split train|val|test: greedy + pass@k + IC95 + desgloses (evaluation/evaluate.py)."""
    from evaluation.evaluate import evaluate as eval_split, table as eval_table
    from model.infer import Engine
    from rollouts.monitor import GpuMonitor
    sets = tuple(a.sets) if a.sets else (tuple(opt(cfg, "dataset.sets", ["core", "catalog"])) if a.split == "train"
                                         else ("core", "catalog"))
    probs = select(a.levels, a.problems, sets=sets, splits=(a.split,))
    if not probs:
        sys.exit(f"ningún problema en split={a.split} sets={list(sets)}")
    b = executor.backend(cfg.sandbox)
    cks = [None if c in ("base", "") else c for c in (a.checkpoint or [])]
    if getattr(a, "run", None):  # curva de aprendizaje: todos los ckpt_* del run (por step) y best/
        from training import checkpoint as ck
        d = os.path.join(a.run, "checkpoints")
        found = sorted(ck.candidates(d), key=lambda p: ck.read_meta(p).get("step", -1)) if os.path.isdir(d) else []
        cks += [p for p in found if os.path.basename(p) != "best"]
        if os.path.isdir(os.path.join(d, "best")):
            cks.append(os.path.join(d, "best"))
    if getattr(a, "with_base", False) or not cks:
        cks = [None] + [c for c in cks if c is not None]
    if getattr(a, "out", None):  # experimentos: la evaluación va dentro del directorio del brazo
        run = os.path.abspath(a.out)
        os.makedirs(run, exist_ok=True)
    else:
        run = new_run_dir(cfg, "eval")
    mon = GpuMonitor(cfg.monitor.interval_s, os.path.join(run, "gpu.jsonl")).start()
    curve = []
    for i, c in enumerate(cks):
        label = "base" if c is None else os.path.basename(os.path.normpath(c))
        print(f"sandbox={b} split={a.split} problemas={len(probs)} checkpoint={c or 'modelo base'}")
        eng = Engine(cfg, c)
        rep = eval_split(eng.generator(), probs, cfg, n=a.n, greedy=False if a.no_greedy else None, seed=a.seed,
                         split=a.split)
        step = None
        if c is not None:
            try:
                from training import checkpoint as ck
                step = ck.read_meta(c).get("step")
            except Exception:
                pass
        rep.update(checkpoint=c, sets=list(sets), step=step, label=label, arch=getattr(eng, "arch", None),
                   inference=INFERENCE_VERSION)  # experiments/run.py repite las evaluaciones de otra versión
        name = (f"eval_{a.split}.json" if getattr(a, "out", None) else "eval.json") if len(cks) == 1 else \
            f"{i:02d}_{label}.json"
        with open(os.path.join(run, name), "w") as f:
            json.dump(rep, f, indent=1)
        print(eval_table(rep))
        curve.append({"label": label, "checkpoint": c, "step": step, "file": name, "greedy": rep.get("greedy"),
                      **{f"pass@{k}": rep[f"pass@{k}"] for k in rep["ks"]}, "ci_pass@1": rep["ci_pass@1"]})
        if getattr(eng, "close", None):
            eng.close()
        del eng
        if "torch" in sys.modules:  # liberar el modelo antes de cargar el siguiente checkpoint
            import gc
            gc.collect()
            sys.modules["torch"].cuda.empty_cache()
    hw = mon.stop()
    if len(cks) > 1:
        with open(os.path.join(run, "curve.json"), "w") as f:
            json.dump(dict(split=a.split, sets=list(sets), curve=curve, hw=hw), f, indent=1)
        print("\ncurva:", "  ".join(f"{c_['label']}(step {c_['step']})={c_['pass@1']:.3f}" for c_ in curve))
    print("guardado en", run)


def main():
    import warnings
    # el código del modelo se parsea muchas veces (score, caché, diversidad, memoria de Fix): un "\d" en un string
    # emitiría un SyntaxWarning en cada parseo y llenaría la consola. No afecta al código del proyecto (ya compilado).
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p1 = sp.add_parser("phase1")
    p1.add_argument("--config")
    p1.add_argument("--n", type=int, help="rollouts por problema (sobrescribe num_rollouts)")
    p1.add_argument("--levels", type=int, nargs="*")
    p1.add_argument("--problems", nargs="*")
    p1.add_argument("--tune", action="store_true", help="busca la concurrencia máxima estable y la guarda")
    p1.add_argument("--dump", action="store_true", help="guarda cada muestra con su reward (samples.jsonl)")
    for name, h in (("phase2", None), ("improve", "phase2 + currículo + grupos de reparación (continúa el run compatible)")):
        p2 = sp.add_parser(name, help=h)
        p2.add_argument("--config")
        p2.add_argument("--n", type=int, help="rollouts por problema (sobrescribe num_rollouts)")
        p2.add_argument("--levels", type=int, nargs="*")
        p2.add_argument("--problems", nargs="*")
        p2.add_argument("--fresh", action="store_true", help="ignora checkpoints existentes, empieza de cero")
        p2.add_argument("--run", help="directorio del run (experimentos); se reanuda si tiene checkpoints compatibles")
        p2.add_argument("--init-from", dest="init_from",
                        help="run NUEVO que parte de los pesos de este checkpoint (rama; optimizador nuevo, step 0)")
    pf = sp.add_parser("fix", help="generate -> test -> fix con feedback del sandbox")
    pf.add_argument("--config")
    pf.add_argument("--iters", type=int, help="rondas de corrección tras el primer intento (defecto: fix.iters)")
    pf.add_argument("--chains", type=int, default=1, help="cadenas independientes por problema")
    pf.add_argument("--levels", type=int, nargs="*")
    pf.add_argument("--problems", nargs="*")
    pf.add_argument("--checkpoint", help="directorio de un checkpoint (ckpt_*/best); sin él, modelo base")
    pf.add_argument("--engine", choices=["v1", "v2"], help="v1 = bucle actual (defecto: fix.engine)")
    pf.add_argument("--feedback", choices=["basic", "counterexample"], help="defecto: fix.feedback")
    pf.add_argument("--candidates", type=int, help="candidatos por ronda (v2; defecto: fix.candidates)")
    pf.add_argument("--compare", action="store_true",
                    help="A/B basic vs counterexample sobre los mismos primeros intentos y semillas (compare.json)")
    pd = sp.add_parser("distill", help="destilación por rechazo: SFT del alumno con las soluciones del profesor que "
                                       "pasan los tests (training/distill.py)")
    pd.add_argument("--config")
    pd.add_argument("--teacher", help="modelo profesor (defecto: el propio config.model = RFT)")
    pd.add_argument("--k", type=int, default=8, help="muestras del profesor por problema de train")
    pd.add_argument("--per-problem", dest="per_problem", type=int, default=4, help="máx. soluciones distintas por problema")
    pd.add_argument("--epochs", type=int, default=2)
    pd.add_argument("--batch", type=int, default=16, help="soluciones por optimizer.step")
    pd.add_argument("--lr", type=float, default=1e-4)
    pd.add_argument("--levels", type=int, nargs="*")
    pd.add_argument("--problems", nargs="*")
    pd.add_argument("--run", help="directorio del run (se reutilizan las soluciones del profesor si ya están)")
    pg = sp.add_parser("genbench", help="mide la generación: eager vs caché estática + torch.compile (CUDA graphs)")
    pg.add_argument("--config")
    pg.add_argument("--checkpoint", help="checkpoint con LoRA entrenado (defecto: LoRA recién creado)")
    pg.add_argument("--problems", type=int, default=6, help="problemas de train por lote (x num_rollouts secuencias)")
    pg.add_argument("--reps", type=int, default=3)
    ph = sp.add_parser("humaneval", help="benchmark externo HumanEval (164) en el sandbox; filas pareables con "
                                         "evaluation.compare")
    ph.add_argument("--config")
    ph.add_argument("--checkpoint", help="checkpoint (ckpt_*/best); sin él, el modelo base")
    ph.add_argument("--n", type=int, default=8, help="muestras por problema (además de la greedy)")
    ph.add_argument("--seed", type=int, help="semilla de muestreo (defecto: eval.seed)")
    ph.add_argument("--timeout", type=float, default=10.0, help="segundos por programa")
    ph.add_argument("--chunk", type=int, default=128, help="secuencias por lote de generación (igual para todos)")
    ph.add_argument("--data", help="HumanEval.jsonl(.gz) local (defecto: runs/data/, se descarga si falta)")
    ph.add_argument("--limit", type=int, help="sólo los N primeros problemas (pruebas)")
    ph.add_argument("--out", help="directorio de salida (defecto: runs/humaneval/<checkpoint|base>)")
    pe = sp.add_parser("eval", help="evaluación por split: greedy + pass@k + IC95 + desgloses")
    pe.add_argument("--config")
    pe.add_argument("--split", choices=["train", "val", "test"], default="val")
    pe.add_argument("--sets", nargs="*", help="core | catalog (defecto: train -> dataset.sets; val/test -> ambos)")
    pe.add_argument("--levels", type=int, nargs="*")
    pe.add_argument("--problems", nargs="*")
    pe.add_argument("--checkpoint", nargs="*", help="uno o varios checkpoints (ckpt_*/best); 'base' = modelo base")
    pe.add_argument("--run", help="evalúa todos los ckpt_* y best/ de un run de phase2 (curva de aprendizaje)")
    pe.add_argument("--with-base", action="store_true", help="incluye también el modelo base (referencia)")
    pe.add_argument("--n", type=int, help="muestras por problema (defecto: eval.n)")
    pe.add_argument("--seed", type=int, help="semilla de evaluación (defecto: eval.seed)")
    pe.add_argument("--no-greedy", action="store_true")
    pe.add_argument("--out", help="directorio de salida (por defecto runs/eval/<fecha>)")
    a = ap.parse_args()
    cfg = load_config(a.config, num_rollouts=getattr(a, "n", None) if a.cmd not in ("eval", "humaneval") else None)
    seed_all(cfg.seed)
    {"phase1": phase1, "phase2": phase2, "improve": phase2, "fix": fix, "eval": eval_cmd, "distill": distill,
     "genbench": genbench, "humaneval": humaneval}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
