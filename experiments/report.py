"""Informe de un experimento (runs/exp/<plan>): coste, estabilidad y calidad de cada brazo, y comparación PAREADA por
problema contra el primer brazo (el baseline) y contra el modelo de partida.

python -m experiments.report runs/exp/<plan>   ->   report.json + report.md en ese directorio

Criterio: una diferencia de pass@1 es creíble si el IC 95 % bootstrap pareado (por problema) no incluye 0 Y el test de
signos da p < 0.05. El reward de entrenamiento NO decide nada (puede subir por crédito parcial o por memorizar los
problemas de train); se informa pareado por step (mismos problemas en el mismo step) como medida de velocidad de
aprendizaje sobre train.
"""
from __future__ import annotations

import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.compare import compare  # noqa: E402


def _rows(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    out.append(r)
    except OSError:
        pass
    return out


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _q(xs, q):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _r(x, d=3):
    return None if x is None else round(x, d)


def _growth_only(r):
    """Step marcado como lento sólo por el crecimiento (runs anteriores a que el detector descontara t_growth)."""
    med = r.get("secs_median")
    return bool(r.get("t_growth") and med and r.get("secs") is not None and r["secs"] - r["t_growth"] <= 3 * med)


def arm_summary(d):
    rows = _rows(os.path.join(d, "metrics.jsonl"))
    steps = [r for r in rows if "loss" in r and isinstance(r.get("step"), int) and r["step"] >= 0]
    evs = [r for r in rows if r.get("event") == "eval_split"]
    growth = [r for r in rows if r.get("event") == "growth"]
    start = [r for r in rows if r.get("event") == "start"]
    ck = os.path.join(d, "checkpoints")
    last = sorted(x for x in os.listdir(ck) if x.startswith("ckpt_")) if os.path.isdir(ck) else []
    meta = _json(os.path.join(ck, last[-1], "meta.json")) if last else None
    arch = (meta or {}).get("arch") or {}
    # s/step sin el crecimiento (inserción de bloques + re-ajuste de VRAM): ese coste va aparte, en growth_s
    secs = [r["secs"] - (r.get("t_growth") or 0) if r.get("secs") is not None else None for r in steps]
    k = max(1, len(steps) // 4)
    out = dict(
        steps=len(steps), wall_train_s=_r(sum(r.get("secs") or 0 for r in steps), 1),
        growth_s=_r(sum(r.get("t_growth") or 0 for r in steps), 1),
        step_s_p50=_r(_q(secs, 0.5), 1), step_s_p95=_r(_q(secs, 0.95), 1), step_s_max=_r(max(secs, default=None), 1),
        slow_steps=sum(bool(r.get("slow")) and not _growth_only(r) for r in steps),
        paging_steps=sum(bool(r.get("vram_paging")) for r in steps),
        tok_s_p50=_r(_q([r.get("tok_s") for r in steps], 0.5), 1),
        ms_per_tok_step_p50=_r(_q([r.get("ms_per_tok_step") for r in steps], 0.5), 2),
        t_gen_mean=_r(_mean([r.get("t_gen") for r in steps]), 2),
        t_score_wait_mean=_r(_mean([r.get("t_score_wait") for r in steps]), 2),
        t_train_mean=_r(_mean([(r.get("t_ref") or 0) + (r.get("t_old") or 0) + (r.get("t_update") or 0)
                               for r in steps]), 2),
        vram_peak_gb=_r(max((r.get("vram_gb") or 0 for r in steps), default=None), 2),
        vram_reserved_peak_gb=_r(max((r.get("vram_reserved_gb") or 0 for r in steps), default=None), 2),
        rss_peak_mb=max((r.get("sys_rss_max_mb") or 0 for r in steps), default=None),
        swap_peak_mb=max((r.get("sys_swap_used_max_mb") or 0 for r in steps), default=None),
        reward_first=_r(_mean([r.get("reward") for r in steps[:k]])),
        reward_last=_r(_mean([r.get("reward") for r in steps[-k:]])),
        solved_frac_last=_r(_mean([r["solved"] / r["n"] for r in steps[-k:] if r.get("n")])),
        kl_last=_r(_mean([r.get("kl") for r in steps[-k:]]), 4),
        approx_kl_mean=_r(_mean([r.get("approx_kl") for r in steps]), 5),
        clip_frac_mean=_r(_mean([r.get("clip_frac") for r in steps]), 4),
        ratio_max=_r(max((r.get("ratio_max") or 1 for r in steps), default=None), 3),
        entropy_first=_r(_mean([r.get("entropy") for r in steps[:k]])),
        entropy_last=_r(_mean([r.get("entropy") for r in steps[-k:]])),
        grad_norm_mean=_r(_mean([r.get("grad_norm") for r in steps])),
        skipped_updates=sum(r.get("skipped") or 0 for r in steps), oom=sum(r.get("oom") or 0 for r in steps),
        trunc_rate=_r(_mean([r.get("trunc_rate") for r in steps])),
        val_curve=[(r["step"], r.get("val_pass@1")) for r in evs if "val_pass@1" in r],
        val_delta_curve=[(r["step"], r.get("val_delta_pass@1"), r.get("val_delta_ci")) for r in evs
                         if "val_delta_pass@1" in r],
        growth=[dict(step=g.get("step"), after=g.get("after"), layers=g.get("layers_after"),
                     params_total=g.get("params_total"), params_trainable=g.get("params_trainable")) for g in growth],
        params_total=arch.get("params_total") or (start[0].get("params_total") if start else None),
        params_trainable=arch.get("params_trainable") or (start[0].get("params_trainable") if start else None),
        layers=arch.get("layers") or (start[0].get("layers") if start else None),
        micro_bs=steps[-1].get("micro_bs") if steps else None, chunk=steps[-1].get("chunk") if steps else None,
        reward_by_step={r["step"]: [r.get("reward"), r.get("problems")] for r in steps},
    )
    return out


def paired_train_reward(sa, sb, from_step=0):
    """Reward de train de b - a en los steps (>= from_step) en que ambos brazos vieron los MISMOS problemas: cada step
    es una pareja (mismos problemas, misma semilla de rollouts) y el IC es bootstrap sobre steps."""
    from evaluation.compare import paired_ci, sign_test
    ra, rb = sa.get("reward_by_step") or {}, sb.get("reward_by_step") or {}
    common = sorted(int(k) for k in ra if k in rb and int(k) >= from_step)
    same = [k for k in common if ra[k][1] is not None and ra[k][1] == rb[k][1] and None not in (ra[k][0], rb[k][0])]
    d = [rb[k][0] - ra[k][0] for k in same]
    if not d:
        return dict(steps=0, common=len(common))
    w, l_ = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return dict(steps=len(d), common=len(common), delta=round(sum(d) / len(d), 4), ci=paired_ci(d, 2000, 0),
                wins=w, losses=l_, p_sign=round(sign_test(w, l_), 4))


def _eval_metrics(rep):
    if not rep:
        return None
    ks = rep.get("ks", [1])
    return dict({f"pass@{k}": rep.get(f"pass@{k}") for k in ks}, greedy=rep.get("greedy"),
                ci_pass1=rep.get("ci_pass@1"), compile=rep.get("compile_rate"), trunc=rep.get("trunc_rate"),
                ntok=rep.get("ntok"), reward=rep.get("reward"), problems=rep.get("problems"), n=rep.get("n"),
                gen_s=rep.get("gen_s"), wall_s=rep.get("wall_s"))


def build(plan_dir):
    plan = {}
    try:
        import yaml
        with open(os.path.join(plan_dir, "plan.yaml")) as f:
            plan = yaml.safe_load(f) or {}
    except OSError:
        pass
    arms = [a for a in (plan.get("arms") or {}) if os.path.isdir(os.path.join(plan_dir, a))] or sorted(
        x for x in os.listdir(plan_dir) if os.path.isdir(os.path.join(plan_dir, x)) and not x.startswith("base_"))
    splits = (plan.get("final_eval") or {}).get("splits", ["val", "test"])
    out = dict(plan=plan.get("name"), dir=plan_dir, arms={}, comparisons={}, base={})
    for b in sorted(x for x in os.listdir(plan_dir) if x.startswith("base_")):
        out["base"][b] = {sp: _eval_metrics(_json(os.path.join(plan_dir, b, f"eval_{sp}.json"))) for sp in splits}
    evals = {}
    for a in arms:
        d = os.path.join(plan_dir, a)
        s = arm_summary(d)
        cfg = {}
        try:
            import yaml
            with open(os.path.join(d, "config.yaml")) as f:
                cfg = yaml.safe_load(f) or {}
        except OSError:
            pass
        s["model"] = cfg.get("model")
        s["overrides"] = (plan.get("arms") or {}).get(a)
        evals[a] = {sp: _json(os.path.join(d, f"eval_{sp}.json")) for sp in splits}
        s["final"] = {sp: _eval_metrics(evals[a][sp]) for sp in splits}
        out["arms"][a] = s
        for sp in splits:  # mismo checkpoint y semillas: LoRA fusionado en bf16 (carga antigua) frente a sin fusionar
            old = _json(os.path.join(d, f"eval_{sp}.inference1.json"))
            if old and evals[a][sp] and old.get("checkpoint") == evals[a][sp].get("checkpoint"):
                out["comparisons"][f"{a}_sinfusionar_vs_fusionado_{sp}"] = _cmp(compare(old, evals[a][sp]))
    ref = arms[0] if arms else None
    out["train_reward"] = {}
    for a in arms:
        if a != ref and ref:
            half = max((int(k) for k in out["arms"][ref].get("reward_by_step") or {}), default=0) // 2
            out["train_reward"][f"{a}_vs_{ref}"] = dict(
                all=paired_train_reward(out["arms"][ref], out["arms"][a]),
                second_half=paired_train_reward(out["arms"][ref], out["arms"][a], from_step=half + 1))
        for sp in splits:
            ra, rb = evals.get(ref, {}).get(sp), evals[a].get(sp)
            if a != ref and ra and rb:
                out["comparisons"][f"{a}_vs_{ref}_{sp}"] = _cmp(compare(ra, rb))
            base_dir = os.path.join(plan_dir, "base_" + _slug(out["arms"][a].get("model") or ""))
            rbase = _json(os.path.join(base_dir, f"eval_{sp}.json"))
            if rbase and rb:
                out["comparisons"][f"{a}_vs_base_{sp}"] = _cmp(compare(rbase, rb))
    out["markdown"] = markdown(out, splits)
    with open(os.path.join(plan_dir, "report.json"), "w") as f:
        json.dump({k: v for k, v in out.items() if k != "markdown"}, f, indent=1)
    with open(os.path.join(plan_dir, "report.md"), "w") as f:
        f.write(out["markdown"])
    return out


def _slug(model):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def _mde(m, z=2.8):
    """Efecto mínimo detectable (alfa 0.05 bilateral, potencia 80 %: 1.96 + 0.84 = 2.8 errores estándar), con el error
    estándar sacado del IC bootstrap pareado (semiancho / 1.96). Por debajo de esto, "sin diferencia" no significa
    "iguales": significa que esta evaluación no lo puede distinguir (Miller 2024, arXiv 2411.00640)."""
    ci = (m or {}).get("ci")
    if not ci or ci[0] is None or ci[1] is None:
        return None
    return round(z * (ci[1] - ci[0]) / (2 * 1.96), 4)


def _cmp(c):
    for m in c["metrics"].values():
        m["mde80"] = _mde(m)
    return dict(problems=c["problems"], verdict=c.get("verdict"), warnings=c.get("warnings"),
                metrics=c["metrics"], improved=c.get("improved"), regressed=c.get("regressed"))


def _fmt(x, d=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def markdown(rep, splits):
    arms = list(rep["arms"])
    L = [f"# Experimento `{rep['plan']}`", ""]
    if not arms:
        return "\n".join(L + ["(sin brazos todavía)"])
    L += ["## Coste y estabilidad", "",
          "| brazo | modelo | capas | params | entrenables | steps | total s | crecer s | s/step p50 | p95 | máx | lentos | "
          "paginación | tok/s | VRAM pico GB | KL final | clip | entropía ini→fin |",
          "|" + "---|" * 18]
    for a in arms:
        s = rep["arms"][a]
        L.append(f"| {a} | {(s.get('model') or '').split('/')[-1]} | {_fmt(s['layers'])} | "
                 f"{_fmt(s['params_total'] and s['params_total'] / 1e6, 1)}M | "
                 f"{_fmt(s['params_trainable'] and s['params_trainable'] / 1e6, 2)}M | {s['steps']} | "
                 f"{_fmt(s['wall_train_s'], 0)} | {_fmt(s.get('growth_s'), 0)} | {_fmt(s['step_s_p50'], 1)} | {_fmt(s['step_s_p95'], 1)} | {_fmt(s['step_s_max'], 1)} | "
                 f"{s['slow_steps']} | {s['paging_steps']} | {_fmt(s['tok_s_p50'], 0)} | {_fmt(s['vram_peak_gb'], 2)} | "
                 f"{_fmt(s['kl_last'], 4)} | {_fmt(s['clip_frac_mean'], 3)} | "
                 f"{_fmt(s['entropy_first'], 2)}→{_fmt(s['entropy_last'], 2)} |")
    L += ["", "## Calidad final (mismas semillas para todos)", ""]
    for sp in splits:
        L += [f"### {sp}", "", "| brazo | pass@1 | IC95 | pass@5 | pass@10 | pass@16 | greedy | compile | trunc | tokens |",
              "|" + "---|" * 10]
        for name, f in [(b, rep["base"][b].get(sp)) for b in rep["base"]] + [(a, rep["arms"][a]["final"].get(sp))
                                                                            for a in arms]:
            if not f:
                L.append(f"| {name} | (sin evaluar) |||||||||")
                continue
            L.append(f"| {name} | {_fmt(f.get('pass@1'))} | {f.get('ci_pass1')} | {_fmt(f.get('pass@5'))} | "
                     f"{_fmt(f.get('pass@10'))} | {_fmt(f.get('pass@16'))} | {_fmt(f.get('greedy'))} | "
                     f"{_fmt(f.get('compile'))} | {_fmt(f.get('trunc'))} | {_fmt(f.get('ntok'), 0)} |")
        L.append("")
    L += ["## Comparaciones pareadas (B - A por problema; IC 95 % bootstrap; test de signos)", "",
          "MDE80 = efecto mínimo detectable con potencia 80 %: una diferencia menor no la distingue esta evaluación.", "",
          "| comparación | problemas | Δpass@1 | IC95 | MDE80 | +/- | p | Δpass@16 | Δgreedy | veredicto |",
          "|" + "---|" * 10]
    for name, c in rep["comparisons"].items():
        m, g = c["metrics"].get("pass@1") or {}, c["metrics"].get("greedy") or {}
        k16 = c["metrics"].get("pass@16") or {}
        L.append(f"| {name} | {c['problems']} | {_fmt(m.get('delta'))} | {m.get('ci')} | {_fmt(m.get('mde80'))} | "
                 f"{m.get('wins')}/{m.get('losses')} | {_fmt(m.get('p_sign'))} | {_fmt(k16.get('delta'))} | "
                 f"{_fmt(g.get('delta'))} | {c.get('verdict')} |")
    if rep.get("train_reward"):
        L += ["", "## Reward de train pareado por step (mismos problemas en el mismo step; velocidad de aprendizaje, "
              "no calidad)", "", "| comparación | tramo | steps pareados | Δreward | IC95 | +/- | p |", "|" + "---|" * 7]
        for name, c in rep["train_reward"].items():
            for part, m in c.items():
                L.append(f"| {name} | {part} | {m.get('steps')}/{m.get('common')} | {_fmt(m.get('delta'))} | "
                         f"{m.get('ci')} | {m.get('wins')}/{m.get('losses')} | {_fmt(m.get('p_sign'))} |")
    L += ["", "## Curva de val durante el entrenamiento (pass@1; Δ pareado contra el modelo de partida)", ""]
    for a in arms:
        s = rep["arms"][a]
        cur = ", ".join(f"s{st}: {_fmt(v)}" for st, v in s["val_curve"])
        dl = ", ".join(f"s{st}: {_fmt(v)} {ci}" for st, v, ci in s["val_delta_curve"])
        L.append(f"- **{a}**: {cur or '–'}" + (f"  \n  Δ vs base: {dl}" if dl else ""))
        if s["growth"]:
            L.append(f"  crecimiento: " + "; ".join(f"step {g['step']}: tras {g['after']} -> {g['layers']} capas, "
                                                     f"{_fmt(g['params_total'] and g['params_total'] / 1e6, 1)}M"
                                                     for g in s["growth"]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = build(sys.argv[1])
    print(r["markdown"])
