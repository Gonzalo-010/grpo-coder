"""Evaluación por split (train | val | test) sin alterar el entrenamiento.

* Guarda y restaura TODO el RNG (python, numpy, torch, cuda) y siembra con eval.seed: evaluar a mitad de un run no
  cambia los rollouts que vienen después.
* Semillas independientes: muestreo y tests generados de eval usan eval.seed (distinta del rango de train
  seed..seed+max_steps y del pool de feedback, que lleva sal propia).
* greedy + pass@k (estimador insesgado) con n muestras; IC 95 % bootstrap determinista por problema y por split.
* Desglose por nivel y categoría, tasa por tipo de error, compile/truncation, tokens, tiempo de generación, tiempo de
  sandbox, VRAM, reward medio y componentes, diversidad por grupo y calidad (sólo resueltos, informativa).
"""
from __future__ import annotations

import contextlib, os, random, sys, time
from collections import Counter, defaultdict

from common import opt, seed_all
from evaluation.metrics import pass_at_k
from evaluation.quality import mean_quality
from rollouts.collect import CACHE, collect
from rollouts.solve import diversity


def _mods():
    mods = {}
    for name in ("numpy", "torch"):
        try:
            mods[name] = __import__(name)
        except ImportError:
            pass
    return mods


def rng_state():
    m = _mods()
    s = {"py": random.getstate(), "hashseed": os.environ.get("PYTHONHASHSEED")}
    if "numpy" in m:
        s["np"] = m["numpy"].random.get_state()
    if "torch" in m:
        s["torch"] = m["torch"].get_rng_state()
        if m["torch"].cuda.is_available():
            s["cuda"] = m["torch"].cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    m = _mods()
    random.setstate(s["py"])
    if s["hashseed"] is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = s["hashseed"]
    if "np" in s:
        m["numpy"].random.set_state(s["np"])
    if "torch" in s:
        m["torch"].set_rng_state(s["torch"])
        if "cuda" in s and m["torch"].cuda.is_available():
            m["torch"].cuda.set_rng_state_all(s["cuda"])


@contextlib.contextmanager
def rng_guard(seed):
    """Siembra con `seed` y al salir deja el RNG exactamente como estaba."""
    st = rng_state()
    seed_all(seed)
    try:
        yield
    finally:
        set_rng_state(st)


def bootstrap_ci(values, B=1000, seed=0, alpha=0.05):
    """IC percentil de la media (determinista para la misma semilla). (lo, hi) o (None, None)."""
    vals = list(values)
    if not vals:
        return None, None
    if len(vals) == 1 or B <= 0:
        return round(vals[0], 4), round(vals[0], 4)
    r, n = random.Random(seed), len(vals)
    means = sorted(sum(vals[r.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(means[int(alpha / 2 * B)], 4), round(means[min(B - 1, int((1 - alpha / 2) * B))], 4)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _err(r):
    return "ok" if r.solved else (r.error or "wrong_answer").split(":")[0]


def _vram():
    t = sys.modules.get("torch")
    try:
        return round(t.cuda.max_memory_allocated() / 1e9, 3) if t is not None and t.cuda.is_available() else None
    except Exception:
        return None


def _row(p, items, ks, greedy_item, B, seed):
    rs, ss = [r for _, r in items], [s for s, _ in items]
    n, c = len(rs), sum(r.solved for r in rs)
    comps = [r.components for r in rs if r.components]
    return dict(
        id=p.id, level=p.level, category=getattr(p, "category", None), split=getattr(p, "split", None), n=n, c=c,
        pass_at={str(k): round(pass_at_k(n, c, k), 4) for k in ks},
        ci_pass1=bootstrap_ci([float(r.solved) for r in rs], B, seed),
        greedy=None if greedy_item is None else bool(greedy_item[1].solved),
        reward=_mean(r.reward for r in rs), frac=_mean(r.frac for r in rs),
        components={k: _mean(cp[k] for cp in comps) for k in comps[0]} if comps else None,
        compile=_mean(float(r.compiled) for r in rs), trunc=_mean(float(r.truncated) for r in rs),
        errors=dict(Counter(_err(r) for r in rs)), ntok=_mean(s.ntok for s in ss),
        diversity=diversity([s.code for s in ss]), sandbox_s=round(sum(r.wall_s for r in rs), 3),
        quality=mean_quality([s.code for s, r in items if r.solved]),
    )


def _group(rows, key, ks):
    out = defaultdict(list)
    for r in rows:
        out[str(r[key])].append(r)
    return {g: dict(problems=len(rs), **{f"pass@{k}": _mean(r["pass_at"][str(k)] for r in rs) for k in ks},
                    greedy=_mean(float(r["greedy"]) for r in rs if r["greedy"] is not None))
            for g, rs in sorted(out.items())}


def evaluate(gen, probs, cfg, n=None, ks=None, greedy=None, seed=None, B=None, split=None):
    """Evalúa `probs` y devuelve un dict serializable (ver docstring del módulo)."""
    n = int(n or opt(cfg, "eval.n", 16))
    ks = [k for k in (ks or opt(cfg, "eval.ks", [1, 5, 10, 16])) if k <= n] or [1]
    greedy = bool(opt(cfg, "eval.greedy", True)) if greedy is None else greedy
    seed = int(opt(cfg, "eval.seed", 999983) if seed is None else seed)
    B = int(opt(cfg, "eval.bootstrap", 1000) if B is None else B)
    st0, c0, t0 = dict(gen.stats), CACHE.stats(), time.time()
    with rng_guard(seed):
        sampled = collect(gen, probs, n, cfg, seed)
        greedy_res = collect(gen, probs, 1, cfg, seed, temperature=0.0) if greedy else None
    rows = [_row(p, items, ks, greedy_res[i][0] if greedy else None, B, seed + i)
            for i, (p, items) in enumerate(zip(probs, sampled))]
    allr = [r for items in sampled for _, r in items]
    c1 = CACHE.stats()
    rep = dict(split=split, problems=len(rows), n=n, ks=ks, greedy_enabled=greedy, seed=seed, rows=rows)
    for k in ks:
        vals = [r["pass_at"][str(k)] for r in rows]
        rep[f"pass@{k}"] = _mean(vals)
        rep[f"ci_pass@{k}"] = bootstrap_ci(vals, B, seed)
    if greedy:
        g = [float(r["greedy"]) for r in rows]
        rep.update(greedy=_mean(g), ci_greedy=bootstrap_ci(g, B, seed))
    comps = [r.components for r in allr if r.components]
    rep.update(
        by_level=_group(rows, "level", ks), by_category=_group(rows, "category", ks),
        error_rates={e: round(v / len(allr), 4) for e, v in Counter(_err(r) for r in allr).items()} if allr else {},
        compile_rate=_mean(float(r.compiled) for r in allr), trunc_rate=_mean(float(r.truncated) for r in allr),
        reward=_mean(r.reward for r in allr), frac=_mean(r.frac for r in allr),
        components={k: _mean(c[k] for c in comps) for k in comps[0]} if comps else None,
        diversity=_mean(r["diversity"] for r in rows), ntok=_mean(r["ntok"] for r in rows),
        gen_tokens=gen.stats.get("tokens", 0) - st0.get("tokens", 0),
        gen_s=round(gen.stats.get("seconds", 0.0) - st0.get("seconds", 0.0), 3),
        sandbox_s=round(sum(r.wall_s for r in allr), 3), vram_gb=_vram(), wall_s=round(time.time() - t0, 2),
        cache_hits=c1["hits"] - c0["hits"], cache_misses=c1["misses"] - c0["misses"],
        quality=mean_quality([s.code for items in sampled for s, r in items if r.solved]),
    )
    return rep


def table(rep):
    ks = rep["ks"]
    head = f"{'problema':26s} L {'categoría':14s} " + " ".join(f"p@{k:<4}" for k in ks) + "  greedy reward"
    lines = [f"split={rep['split']} n={rep['n']} seed={rep['seed']} problemas={rep['problems']}", head]
    for r in rep["rows"]:
        lines.append(f"{r['id']:26s} {r['level']} {str(r['category']):14s} "
                     + " ".join(f"{r['pass_at'][str(k)]:.3f} " for k in ks)
                     + f"  {'-' if r['greedy'] is None else int(r['greedy']):>6} {r['reward']}")
    lines.append("MEDIA" + " " * 39 + " ".join(f"{rep[f'pass@{k}']:.3f} " for k in ks)
                 + f"  {rep.get('greedy', '-')} {rep['reward']}")
    lines.append(f"IC95 pass@1={rep['ci_pass@1']} greedy={rep.get('ci_greedy')} compile={rep['compile_rate']} "
                 f"trunc={rep['trunc_rate']} errores={rep['error_rates']}")
    return "\n".join(lines)
