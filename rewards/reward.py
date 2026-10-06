"""Reward verificable. Correctitud >> estilo.

reward = w.compile*compiled + w.visible*pv + w.hidden*s(ph) + w.generated*s(pg) + w.solved*solved

compile  : parsea y define la función pedida. Señal densa mínima al principio; sola vale 0.05.
visible  : fracción cruda (pocos casos, peso bajo). Memorizar los ejemplos del prompt no da más de w.visible.
hidden / generated: acierto CORREGIDO POR AZAR: s(p) = (p - b) / (1 - b), con b = acierto del mejor predictor
           constante en ese conjunto. 'return True' en un test balanceado -> p = b -> s = 0.
solved   : TODOS los tests (incl. tamaños grandes -> obliga a complejidad adecuada y robustez). Es el RLVR binario.
unsafe   : imports fuera de la lista, eval/exec/open/dunders... -> NO se ejecuta y recibe penalización (moldea la política;
           la contención real la da el sandbox). Tiempo/memoria se miden pero no puntúan: los tests grandes ya los exigen.
Los esperados nunca entran en el sandbox: el candidato sólo recibe entradas.
"""
from __future__ import annotations

import ast, copy, json, threading, time
from dataclasses import dataclass

from problems.problems import baseline, judge
from sandbox import executor

BAD_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "globals", "locals", "vars",
             "memoryview", "exit", "quit"}
BAD_NAMES = {"__builtins__", "__import__", "__loader__", "__spec__", "__globals__"}
OK_DUNDER = {"__init__", "__name__", "__len__", "__eq__", "__lt__", "__le__", "__gt__", "__ge__", "__hash__",
             "__iter__", "__next__", "__getitem__", "__setitem__", "__contains__", "__repr__", "__str__",
             "__add__", "__call__"}
DYN = {"getattr", "setattr", "delattr", "hasattr"}


@dataclass
class Result:
    reward: float
    compiled: bool = False
    solved: bool = False
    pv: float = 0.0            # fracción cruda visible / hidden / generated
    ph: float = 0.0
    pg: float = 0.0
    frac: float = 0.0          # fracción global de tests pasados
    n_pass: int = 0
    n_cases: int = 0
    error: str | None = None   # syntax | no_entry | unsafe:* | timeout | load:* | crash:* | runtime:<Exc> | wrong_answer
    runtime_s: float = 0.0
    mem_mb: float = 0.0
    wall_s: float = 0.0
    detail: dict | None = None  # primer caso VISIBLE que falla (call/want/got); de los ocultos nunca hay detalle
    # --- campos aditivos (no intervienen en el reward) ---
    components: dict | None = None  # aportación de cada término al reward: compile/visible/hidden/generated/solved
    fail_case: dict | None = None   # primer caso que falla: {"kind": visible|hidden|generated, "index", "category"}
    line: int | None = None         # línea del candidato de la primera excepción/timeout o del error de carga
    tb: str | None = None           # traceback resumido: "IndexError: list index out of range (line 3)"
    truncated: bool = False         # la generación se cortó por max_new_tokens (lo rellena quien conoce el Sample)
    timeout: bool = False
    timeout_size: int | None = None  # tamaño (elementos) de la entrada que agotó el tiempo: sólo métrica interna
    ref_time: float | None = None   # segundos de la referencia en esa misma entrada
    cat_pass: dict | None = None    # {categoría: [pasados, total]}: visible, edge, random, large, error, generated


def _show(entry, c, rec, run):
    call = f"{entry}({', '.join(map(repr, c.args))})"
    want = f"raise {c.exc}" if c.exc else f"return {json.loads(c.exp)!r}"
    if rec is None:
        got = "did not run: " + (run.fatal or run.status)
    elif rec.get("exc") == "Timeout":
        got = "did not finish in time"
    elif "exc" in rec:
        got = f"raised {rec['exc']}" + (f": {rec['msg']}" if rec.get("msg") else "")
    elif "h" in rec:
        got = "returned a different (large) value"
    else:
        try:
            got = f"returned {json.loads(rec['out'])!r}"  # mismo formato que want (True, no true)
        except (KeyError, ValueError):
            got = "returned an unreadable value"
    return dict(call=call[:300], want=want[:300], got=got[:300])


def unsafe(tree, allowed):
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in allowed:
                    return "import:" + a.name
        elif isinstance(n, ast.ImportFrom):
            if n.level or (n.module or "").split(".")[0] not in allowed:
                return "import:" + (n.module or ".")
        elif isinstance(n, ast.Name) and n.id in BAD_NAMES:
            return "name:" + n.id
        elif isinstance(n, ast.Attribute) and n.attr.startswith("__") and n.attr.endswith("__") and n.attr not in OK_DUNDER:
            return "attr:" + n.attr
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in BAD_CALLS:
                return "call:" + n.func.id
            if n.func.id in DYN:
                a = n.args[1] if len(n.args) > 1 else None
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str) and not a.value.startswith("_")):
                    return "dynamic-attr"
    return None


def _chance(p, b):
    if b >= 0.999:  # test degenerado (todas las salidas iguales): sólo cuenta el acierto total
        return 1.0 if p >= 1.0 else 0.0
    return max(0.0, (p - b) / (1.0 - b))


def _frac(flags):
    return sum(flags) / len(flags) if flags else 0.0


def _size(x, cap=10 ** 7):
    """Tamaño de una entrada: hojas escalares, contando los caracteres de cada string."""
    if isinstance(x, str):
        return len(x)
    if isinstance(x, dict):
        x = list(x.values())
    if isinstance(x, (list, tuple)):
        n = 0
        for v in x:
            n += _size(v, cap)
            if n >= cap:
                break
        return n
    return 1


_REF_T, _REF_LOCK = {}, threading.Lock()


def _ref_time(p, kind, idx, case, seed=0):
    """(tamaño, segundos de la referencia) para un caso, medido una vez por (problema, versión, caso) en este proceso.
    Los casos generados dependen de la semilla: entra en la clave."""
    key = (p.id, getattr(p, "version", ""), kind, idx, seed if kind == "generated" else None)
    with _REF_LOCK:
        if key in _REF_T:
            return _REF_T[key]
    size, dt = _size(case.args), None
    ref = getattr(p, "ref", None)
    if ref is not None:
        t0 = time.perf_counter()
        try:
            ref(*copy.deepcopy(case.args))
            dt = round(time.perf_counter() - t0, 4)
        except Exception:
            dt = None
    with _REF_LOCK:
        if len(_REF_T) > 4096:
            _REF_T.clear()
        _REF_T[key] = (size, dt)
    return size, dt


def _diagnose(p, cases, kinds, ok, run, seed=0):
    """Campos estructurados del primer fallo (sin cambiar nada del reward)."""
    cats = {}
    for c, k, o in zip(cases, kinds, ok):
        cat = "error" if c.exc and k not in ("visible", "generated") else k
        v = cats.setdefault(cat, [0, 0])
        v[0] += bool(o)
        v[1] += 1
    out = dict(cat_pass=cats)
    first = next((i for i, o in enumerate(ok) if not o), None)
    if first is None:
        return out
    kind = "visible" if kinds[first] == "visible" else "generated" if kinds[first] == "generated" else "hidden"
    nv = kinds.count("visible")
    nh = len(kinds) - nv - kinds.count("generated")
    idx = first if kind == "visible" else first - nv if kind == "hidden" else first - nv - nh
    out["fail_case"] = {"kind": kind, "index": idx, "category": kinds[first]}
    recs = run.cases
    exc_i = next((i for i, (o, r) in enumerate(zip(ok, recs)) if not o and r and "exc" in r), None)
    if exc_i is not None:
        r = recs[exc_i]
        out["line"] = r.get("line")
        out["tb"] = f"{r['exc']}" + (f": {r['msg'][:120]}" if r.get("msg") else "") + (
            f" (line {r['line']})" if r.get("line") else "")
    elif run.status == "load_error":
        out["line"] = run.fatal_line
        out["tb"] = run.fatal[:160] + (f" (line {run.fatal_line})" if run.fatal_line else "")
    to_i = next((i for i, (o, r) in enumerate(zip(ok, recs)) if not o and (r is None and run.status == "timeout"
                                                                           or r and r.get("exc") == "Timeout")), None)
    if to_i is not None:
        out["timeout_size"], out["ref_time"] = _ref_time(p, kinds[to_i], to_i, cases[to_i], seed)
    return out


def score(p, code, cfg, seed=0) -> Result:
    w = cfg.reward_weights
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return Result(0.0, error="syntax")
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == p.entry for n in tree.body):
        return Result(0.0, error="no_entry")
    why = unsafe(tree, set(cfg.sandbox.allowed_imports))
    if why:
        return Result(cfg.reward.unsafe_penalty, error="unsafe:" + why)

    vis, hid, gen = p.visible(), p.hidden(), p.generated(seed, cfg.reward.n_generated)
    cases = vis + hid + gen
    run = executor.run_cases(code, p.entry, [c.args for c in cases], cfg.sandbox_limits,
                             cfg.sandbox.allowed_imports, executor.backend(cfg.sandbox))
    ok = [judge(c, r) for c, r in zip(cases, run.cases)]
    fv, fh, fg = ok[:len(vis)], ok[len(vis):len(vis) + len(hid)], ok[len(vis) + len(hid):]
    pv, ph, pg = _frac(fv), _frac(fh), _frac(fg)
    solved = all(ok)
    reward = (w.compile + w.visible * pv + w.hidden * _chance(ph, baseline(hid))
              + w.generated * _chance(pg, baseline(gen)) + w.solved * solved)

    err, detail = None, None
    if not solved:
        detail = next((_show(p.entry, c, r, run) for c, r, o in zip(vis, run.cases, fv) if not o), None)
        bad = [r for r, o in zip(run.cases, ok) if not o and r]
        if run.status == "timeout" or any(r.get("exc") == "Timeout" for r in bad):
            err = "timeout"
        elif run.status == "load_error":
            err = "load:" + run.fatal[:60]
        elif run.status.startswith("crash"):
            err = run.status
        elif any("exc" in r for r in bad):
            err = "runtime:" + next(r["exc"] for r in bad if "exc" in r)
        else:
            err = "wrong_answer"
    rt = sum(r.get("t", 0.0) for r in run.cases if r)
    comps = {"compile": w.compile, "visible": w.visible * pv, "hidden": w.hidden * _chance(ph, baseline(hid)),
             "generated": w.generated * _chance(pg, baseline(gen)), "solved": w.solved * solved}
    hk = p.hidden_kinds() if hasattr(p, "hidden_kinds") else ["random"] * len(hid)
    extra = _diagnose(p, cases, ["visible"] * len(vis) + list(hk) + ["generated"] * len(gen), ok, run, seed)
    return Result(round(reward, 6), True, solved, pv, ph, pg, _frac(ok), sum(ok), len(ok), err,
                  rt, run.rss_mb, run.wall_s, detail, {k: round(v, 6) for k, v in comps.items()},
                  timeout=err == "timeout", **extra)
