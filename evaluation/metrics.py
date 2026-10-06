import math
from collections import Counter


def pass_at_k(n, c, k):
    """Estimador insesgado de Chen et al. 2021 (Codex): n muestras, c correctas."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows, ks):
    """rows: un dict por problema con listas por muestra. pass@k es macro-promedio sobre problemas."""
    def agg(rs):
        cat = lambda key: [x for r in rs for x in r[key]]
        d = {"problems": len(rs), "samples": sum(r["n"] for r in rs)}
        for k in ks:
            d[f"pass@{k}"] = _mean(pass_at_k(r["n"], r["solved"], k) for r in rs if r["n"] >= k)
        d.update(reward=_mean(cat("reward")), compile=_mean(cat("compiled")), tests=_mean(cat("frac")),
                 tokens=_mean(cat("ntok")), runtime_ms=1000 * _mean(cat("runtime")), mem_mb=_mean(cat("mem")))
        return d
    out = {"all": agg(rows)}
    for lv in sorted({r["level"] for r in rows}):
        out[f"L{lv}"] = agg([r for r in rows if r["level"] == lv])
    out["errors"] = dict(Counter(e for r in rows for e in r["errors"]))
    return out


def table(s, ks):
    cols = [f"pass@{k}" for k in ks] + ["reward", "compile", "tests", "tokens", "runtime_ms", "mem_mb"]
    head = f"{'':5}{'probs':>6}" + "".join(f"{c:>11}" for c in cols)
    lines = [head]
    for name in [k for k in s if k != "errors"]:
        d = s[name]
        lines.append(f"{name:5}{d['problems']:>6}" + "".join(f"{d[c]:>11.3f}" if c not in ("tokens", "runtime_ms", "mem_mb")
                                                             else f"{d[c]:>11.1f}" for c in cols))
    return "\n".join(lines)
