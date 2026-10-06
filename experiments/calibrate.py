"""Calibración de dificultad: el modelo (base o un checkpoint) contra TODOS los problemas, con n muestras.

python -m experiments.calibrate [--n 8] [--checkpoint DIR] [--sets core catalog] [--out runs/exp/calibration]

Para qué: un split de evaluación sólo mide mejoras en problemas que el modelo ni resuelve siempre ni nunca. En el val
original 10 de 16 problemas estaban a 0 en el modelo base y en todos los checkpoints: el IC de pass@1 era [0.03, 0.26].
Esto da, por problema, la tasa de acierto del modelo de partida para (1) ver la distribución de dificultad por split y
(2) repartir problemas nuevos entre splits con una dificultad parecida, ANTES de cualquier experimento.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def summarize(rows, n):
    """Resumen por split: nº de problemas, pass@1 medio, muertos (0/n), saturados (n/n) e informativos."""
    out = defaultdict(lambda: dict(problems=0, pass1=0.0, dead=0, saturated=0, informative=0))
    for r in rows:
        s = out[r["split"]]
        s["problems"] += 1
        s["pass1"] += r["c"] / r["n"]
        s["dead"] += r["c"] == 0
        s["saturated"] += r["c"] == r["n"]
        s["informative"] += 0 < r["c"] < r["n"]
    for s in out.values():
        s["pass1"] = round(s["pass1"] / max(1, s["problems"]), 4)
    return dict(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dificultad de cada problema para un modelo (ver docstring)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--checkpoint")
    ap.add_argument("--config")
    ap.add_argument("--sets", nargs="*", default=["core", "catalog"])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "exp", "calibration"))
    a = ap.parse_args(argv)
    from common import load_config, seed_all
    from evaluation.evaluate import evaluate
    from model.infer import Engine
    from problems.problems import select
    from sandbox import executor
    cfg = load_config(a.config)
    seed_all(cfg.seed)
    executor.backend(cfg.sandbox)
    probs = select(sets=tuple(a.sets))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    eng = Engine(cfg, a.checkpoint)
    rep = evaluate(eng.generator(), probs, cfg, n=a.n, ks=[1, a.n], split="all")
    rows = [dict(id=r["id"], split=r["split"], level=r["level"], category=r["category"], n=r["n"], c=r["c"],
                 greedy=r["greedy"], errors=r["errors"], ntok=r["ntok"], trunc=r["trunc"]) for r in rep["rows"]]
    summ = summarize(rows, a.n)
    res = dict(model=cfg.model, checkpoint=a.checkpoint, n=a.n, seed=rep["seed"], wall_s=round(time.time() - t0, 1),
               summary=summ, rows=rows, arch=eng.arch)
    name = "calibration.json" if not a.checkpoint else f"calibration_{os.path.basename(os.path.normpath(a.checkpoint))}.json"
    with open(os.path.join(a.out, name), "w") as f:
        json.dump(res, f, indent=1)
    for sp, s in sorted(summ.items()):
        print(f"{sp:6s} problemas={s['problems']:3d} pass@1={s['pass1']:.3f} muertos={s['dead']} "
              f"saturados={s['saturated']} informativos={s['informative']}")
    print("guardado en", os.path.join(a.out, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
