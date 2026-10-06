"""¿Mejoró el modelo? Comparación PAREADA de dos evaluaciones del mismo split.

Uso: python -m evaluation.compare A.json B.json [--json salida.json]
A y B son eval.json de `train.py eval` (o los de una curva, o evals/stepNNNNNN_<split>.json de un run). Se comparan
sólo los problemas presentes en ambos. Para cada métrica (pass@k comunes y greedy):
  * delta = B - A (media por problema) con IC 95 % por bootstrap pareado sobre problemas (determinista);
  * test de signos exacto bilateral sobre los problemas que suben o bajan (los empates no cuentan);
  * problemas que mejoran / empeoran >= 0.25 en pass@1.
Una mejora es creíble si el IC no incluye 0 (y el test de signos da p < 0.05). Con 16 problemas los IC son anchos:
para efectos pequeños hacen falta más muestras (--n) o más problemas.
"""
from __future__ import annotations

import argparse, json, math, random, sys


def load(path):
    with open(path) as f:
        rep = json.load(f)
    if "rows" not in rep:
        raise ValueError(f"{path}: no es un informe de evaluación (sin 'rows')")
    return rep


def sign_test(wins, losses):
    """p-valor bilateral exacto del test de signos (empates excluidos)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_ci(deltas, B=2000, seed=0, alpha=0.05):
    if not deltas:
        return None, None
    r, n = random.Random(seed), len(deltas)
    means = sorted(sum(deltas[r.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(means[int(alpha / 2 * B)], 4), round(means[min(B - 1, int((1 - alpha / 2) * B))], 4)


def _metric(da, db, ids, B, seed, eps=1e-9):
    d = [db[i] - da[i] for i in ids]
    wins, losses = sum(x > eps for x in d), sum(x < -eps for x in d)
    return dict(a=round(sum(da[i] for i in ids) / len(ids), 4), b=round(sum(db[i] for i in ids) / len(ids), 4),
                delta=round(sum(d) / len(d), 4), ci=paired_ci(d, B, seed), wins=wins, losses=losses,
                ties=len(d) - wins - losses, p_sign=round(sign_test(wins, losses), 4))


def compare(a, b, B=2000, seed=0, thr=0.25):
    ra, rb = {r["id"]: r for r in a["rows"]}, {r["id"]: r for r in b["rows"]}
    ids = sorted(set(ra) & set(rb))
    out = dict(problems=len(ids), only_a=sorted(set(ra) - set(rb)), only_b=sorted(set(rb) - set(ra)),
               split=(a.get("split"), b.get("split")), n=(a.get("n"), b.get("n")),
               labels=(a.get("label") or a.get("checkpoint") or "A", b.get("label") or b.get("checkpoint") or "B"),
               metrics={}, warnings=[])
    if not ids:
        out["warnings"].append("no hay problemas en común")
        return out
    if a.get("split") != b.get("split"):
        out["warnings"].append(f"splits distintos: {a.get('split')} vs {b.get('split')}")
    if a.get("n") != b.get("n"):
        out["warnings"].append(f"muestras por problema distintas ({a.get('n')} vs {b.get('n')}): pass@k comparable, "
                               "pero el ruido de cada lado es distinto")
    for k in [k for k in a.get("ks", [1]) if k in b.get("ks", [1])]:
        out["metrics"][f"pass@{k}"] = _metric({i: ra[i]["pass_at"][str(k)] for i in ids},
                                              {i: rb[i]["pass_at"][str(k)] for i in ids}, ids, B, seed)
    if all(ra[i].get("greedy") is not None and rb[i].get("greedy") is not None for i in ids):
        out["metrics"]["greedy"] = _metric({i: float(ra[i]["greedy"]) for i in ids},
                                           {i: float(rb[i]["greedy"]) for i in ids}, ids, B, seed)
    d1 = {i: rb[i]["pass_at"]["1"] - ra[i]["pass_at"]["1"] for i in ids}
    out["improved"] = sorted(([i, round(ra[i]["pass_at"]["1"], 3), round(rb[i]["pass_at"]["1"], 3)]
                              for i in ids if d1[i] >= thr), key=lambda x: x[1] - x[2])
    out["regressed"] = sorted(([i, round(ra[i]["pass_at"]["1"], 3), round(rb[i]["pass_at"]["1"], 3)]
                               for i in ids if d1[i] <= -thr), key=lambda x: x[2] - x[1])
    # Veredicto: IC bootstrap que excluye 0 Y test de signos exacto p < 0.05. Sólo con el bootstrap, pocos problemas
    # discordantes y todos del mismo lado (p. ej. +5/-0 con n=2 muestras) dan un IC que no toca el 0 aunque el test de
    # signos diga p = 0.06: el bootstrap de percentiles no es fiable con tan pocos casos distintos de 0.
    m = out["metrics"].get("pass@1")
    sig = bool(m) and m["p_sign"] < 0.05
    out["verdict"] = ("mejora" if sig and m["ci"][0] > 0 else "empeora" if sig and m["ci"][1] < 0
                      else "sin diferencia significativa")
    return out


def table(c):
    la, lb = c["labels"]
    lines = [f"A={la}  B={lb}  problemas en común={c['problems']}  split={c['split'][0]}  n={c['n']}"]
    lines += [f"[aviso] {w}" for w in c["warnings"]]
    lines.append(f"{'métrica':10s} {'A':>7s} {'B':>7s} {'B-A':>7s}  {'IC95':17s} {'+':>3s} {'-':>3s} {'p(signos)':>9s}")
    for k, v in c["metrics"].items():
        lines.append(f"{k:10s} {v['a']:7.3f} {v['b']:7.3f} {v['delta']:+7.3f}  [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] "
                     f"{v['wins']:3d} {v['losses']:3d} {v['p_sign']:9.4f}")
    if c.get("improved"):
        lines.append("mejoran:  " + ", ".join(f"{i} {x:.2f}->{y:.2f}" for i, x, y in c["improved"]))
    if c.get("regressed"):
        lines.append("empeoran: " + ", ".join(f"{i} {x:.2f}->{y:.2f}" for i, x, y in c["regressed"]))
    lines.append(f"veredicto (pass@1): {c.get('verdict', '-')}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Comparación pareada de dos evaluaciones (B frente a A)")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="guardar la comparación en este fichero")
    x = ap.parse_args(argv)
    c = compare(load(x.a), load(x.b), x.bootstrap, x.seed)
    print(table(c))
    if x.json:
        with open(x.json, "w") as f:
            json.dump(c, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
