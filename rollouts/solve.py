"""Resolver un problema: greedy | sample | best-of-n | repair. Lo reutilizan Fix, el panel y el agente.

Regla: cualquier SELECCIÓN entre candidatos usa sólo información permitida (tests visibles + pool de contraejemplos,
que nunca contiene entradas ocultas). El Result completo (con ocultos) es sólo para MEDIR, nunca para elegir.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor

from common import opt, workers
from rollouts.collect import CACHE, cached_score, code_key, evaluator_sig
from rollouts.generate import build_prompt

MODES = ("greedy", "sample", "best-of-n", "repair")


def sample(gen, prompts, n=1, temperature=None):
    """[[Sample] * n por prompt]. temperature: override sólo para esta llamada (<= 0: greedy)."""
    out = [[] for _ in prompts]
    stream = gen.stream(prompts, n) if temperature is None else gen.stream(prompts, n, temperature=temperature)
    for i, s in stream:
        out[i].append(s)
    return out


def score_many(items, cfg, seed=0):
    """[(problema, código)] -> [Result] en paralelo y con caché (el mismo código no se ejecuta dos veces)."""
    if not items:
        return []
    with ThreadPoolExecutor(workers(cfg)) as ex:
        return list(ex.map(lambda pc: cached_score(pc[0], pc[1], cfg, seed), items))


def allowed_cases(p, seed, pool_n=64):
    """Tests que se pueden usar para decidir: visibles + pool de feedback (sin ocultos por construcción)."""
    return p.visible() + (p.feedback_cases(seed, pool_n) if hasattr(p, "feedback_cases") else [])


def _allowed(p, code, cfg, seed, pool_n):
    from problems.problems import judge
    from rewards.reward import unsafe
    from sandbox import executor
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return 0.0
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == p.entry for n in tree.body):
        return 0.0
    if unsafe(tree, set(cfg.sandbox.allowed_imports)):
        return 0.0
    cases = allowed_cases(p, seed, pool_n)
    if not cases:
        return 0.0
    run = executor.run_cases(code, p.entry, [c.args for c in cases], cfg.sandbox_limits, cfg.sandbox.allowed_imports,
                             executor.backend(cfg.sandbox))
    return round(sum(judge(c, r) for c, r in zip(cases, run.cases)) / len(cases), 4)


def allowed_score(p, code, cfg, seed=0, pool_n=64):
    """Fracción de tests permitidos que pasa `code` (0..1). Cacheada como el score."""
    version = getattr(p, "version", None)
    if version is None or not opt(cfg, "rollouts.score_cache", True):
        return _allowed(p, code, cfg, seed, pool_n)
    key = ("allowed", p.id, version, seed, pool_n, code, evaluator_sig(cfg))
    return CACHE.get(key, lambda: _allowed(p, code, cfg, seed, pool_n))


def select(p, codes, cfg, seed=0, pool_n=64):
    """(índice del mejor, puntuaciones permitidas). Empate: el código más corto; luego el primero."""
    with ThreadPoolExecutor(workers(cfg)) as ex:
        sc = list(ex.map(lambda c: allowed_score(p, c, cfg, seed, pool_n), codes))
    best = max(range(len(codes)), key=lambda i: (sc[i], -len(codes[i]), -i))
    return best, sc


def diversity(codes):
    """Fracción de soluciones distintas (AST normalizado, sin formato ni comentarios) en un grupo."""
    return round(len({code_key(c) for c in codes}) / len(codes), 4) if codes else 0.0


def solve(gen, p, cfg, mode="greedy", n=4, seed=0, rounds=None, temperature=None):
    """Resuelve p y devuelve dict(code, result, mode, candidates, allowed). result incluye ocultos: sólo para medir."""
    if mode not in MODES:
        raise ValueError(f"modo desconocido: {mode} (usa {', '.join(MODES)})")
    if mode == "repair":
        from rollouts.fix import final_attempt, run_v2
        chain = run_v2(gen, [p], cfg, rounds if rounds is not None else int(opt(cfg, "fix.iters", 3)), 1, seed)[0]
        a = final_attempt(chain)
        r = score_many([(p, a.code)], cfg, seed)[0]
        return dict(code=a.code, result=r, mode=mode, candidates=len(chain), allowed=a.allowed)
    plain = build_prompt(gen.tok, cfg.prompt.system, p.prompt(), cfg.prompt.prefill_fence)
    if mode == "greedy":
        ss = sample(gen, [plain], 1, 0.0)[0]
    else:
        ss = sample(gen, [plain], n if mode == "best-of-n" else 1, temperature)[0]
    idx, sc = (select(p, [s.code for s in ss], cfg, seed, int(opt(cfg, "fix.pool", 64))) if len(ss) > 1 else (0, [None]))
    r = score_many([(p, ss[idx].code)], cfg, seed)[0]
    r.truncated = not getattr(ss[idx], "closed", True)
    return dict(code=ss[idx].code, result=r, mode=mode, candidates=len(ss), allowed=sc[idx])
