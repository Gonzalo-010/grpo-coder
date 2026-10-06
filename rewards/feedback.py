"""Feedback para el modelo a partir de un Result de score().

Sólo se detallan casos VISIBLES (ya están en el prompt). De los ocultos/generados se dice cuántos fallan y de qué
tipo, nunca sus entradas: si no, el bucle de corrección enseñaría a hardcodear tests.
"""
import ast


def _parse_error(code):
    try:
        ast.parse(code)
        return "invalid code"
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:  # null bytes, recursión del parser...
        return type(e).__name__


def feedback(p, code, r, allowed=()):
    """Texto corto que explica por qué falló `code`, o None si está resuelto."""
    if r.solved:
        return None
    e, d = r.error or "", r.detail
    if e == "syntax":
        return f"The code does not parse: {_parse_error(code)}."
    if e == "no_entry":
        return f"The code must define a top-level function named `{p.entry}`."
    if e.startswith("unsafe:"):
        return (f"Not allowed: {e[7:]}. Only these modules can be imported: {', '.join(allowed)}. "
                f"Do not use eval/exec/open or dunder attributes.")
    if e.startswith("load:"):
        return f"The code fails as soon as it is loaded: {e[5:]}."
    if e.startswith("crash"):
        return "The process crashed (too much memory or recursion?). Use an iterative, memory-bounded approach."
    n_fail = r.n_cases - r.n_pass
    if d:
        return f"`{d['call']}` should {d['want']} but {d['got']}. {n_fail} of {r.n_cases} tests fail."
    if e == "timeout":
        return (f"Passes the examples but is too slow on {n_fail} of {r.n_cases} tests: large inputs need an "
                f"efficient algorithm (avoid quadratic loops, deep recursion).")
    if e.startswith("runtime:"):
        return (f"Passes the examples but raises {e[8:]} on {n_fail} of {r.n_cases} hidden tests: "
                f"check edge cases (empty input, zero, negatives, duplicates).")
    return (f"Passes the examples but returns wrong results on {n_fail} of {r.n_cases} hidden tests: "
            f"check edge cases (empty input, zero, negatives, duplicates, large inputs).")


# ---------------------------------------------------------------------------------------------------------------
# Feedback v2 (fix.feedback: counterexample). Reglas de no-filtración:
#   * los contraejemplos salen SÓLO de Problem.feedback_cases(seed): semilla con sal propia y sin ninguna entrada
#     oculta ni visible (y la reducción vuelve a excluirlas);
#   * los timeouts se explican con entradas grandes NUEVAS (Problem.timeout_cases), nunca con las ocultas;
#   * la línea del error sólo se enseña si viene de un caso visible o del pool, nunca de un oculto.
# ---------------------------------------------------------------------------------------------------------------
import json, time

TRUNCATED = ("Your answer was cut off at the token limit before the code was complete. Write a shorter, complete "
             "solution: no comments, no docstrings, no examples.")


def _short(v, n=160):
    s = repr(v)
    return s if len(s) <= n else s[:n - 1] + "…"


def _call(entry, args, n=240):
    s = f"{entry}({', '.join(map(repr, args))})"
    return s if len(s) <= n else s[:n - 1] + "…"


def _code_line(code, n):
    lines = code.splitlines()
    return lines[n - 1].strip()[:120] if isinstance(n, int) and 0 < n <= len(lines) else None


def _tname(v):
    return ("None" if v is None else "bool" if isinstance(v, bool) else "number" if isinstance(v, (int, float))
            else "string" if isinstance(v, str) else "list" if isinstance(v, (list, tuple)) else "dict"
            if isinstance(v, dict) else type(v).__name__)


def _first_diff(want, got):
    """Frase con lo primero que difiere: el tipo, las claves de un dict o el primer índice de una lista/string."""
    from problems.problems import same
    tw, tg = _tname(want), _tname(got)
    if tw != tg:
        return f" Expected a {tw}, got a {tg}."
    if isinstance(want, dict):
        miss, extra = sorted(map(str, set(want) - set(got))), sorted(map(str, set(got) - set(want)))
        parts = ([f"missing keys {_short(miss, 80)}"] if miss else []) + ([f"unexpected keys {_short(extra, 80)}"]
                                                                         if extra else [])
        if not parts:
            k = next((k for k in want if not same(want[k], got[k])), None)
            if k is not None:
                parts.append(f"for key {k!r} expected {_short(want[k], 60)}, got {_short(got[k], 60)}")
        return (" " + "; ".join(parts)[0].upper() + "; ".join(parts)[1:] + ".") if parts else ""
    if not isinstance(want, (list, str)):
        return ""
    for i, (a, b) in enumerate(zip(want, got)):
        if not same(a, b):
            return f" First difference at index {i}: expected {_short(a, 60)}, got {_short(b, 60)}."
    if len(want) != len(got):
        return f" Expected length {len(want)}, got {len(got)}."
    return ""


def _syntax(code):
    try:
        ast.parse(code)
        return "The code does not parse."
    except SyntaxError as e:
        line = _code_line(code, e.lineno)
        caret = ""
        if line is not None and e.offset:
            raw = code.splitlines()[e.lineno - 1]
            caret = "\n    " + raw.strip()[:120] + "\n    " + " " * max(0, e.offset - 1 - (len(raw) - len(raw.lstrip()))) + "^"
        return (f"The code does not parse: SyntaxError at line {e.lineno}: {e.msg}.{caret}\n"
                f"Check brackets, colons and indentation around that line.")
    except Exception as e:
        return f"The code does not parse: {type(e).__name__}."


def _run(p, code, cases, cfg):
    from problems.problems import judge
    from sandbox import executor
    run = executor.run_cases(code, p.entry, [c.args for c in cases], cfg.sandbox_limits, cfg.sandbox.allowed_imports,
                             executor.backend(cfg.sandbox))
    return [(c, rec, judge(c, rec)) for c, rec in zip(cases, run.cases)], run


def _banned(p):
    from problems.problems import _key
    return {_key(c.args) for c in p.hidden() + p.visible()}


def counterexample(p, code, cfg, seed, pool_n=64, rounds=3):
    """(caso, registro, reducido) del contraejemplo más pequeño encontrado en el pool, o None. Nunca un oculto."""
    from problems.problems import _key, shrink_args
    pool = p.feedback_cases(seed, pool_n)
    if not pool:
        return None
    res, _ = _run(p, code, pool, cfg)
    fails = [(c, rec) for c, rec, ok in res if not ok]
    if not fails:
        return None
    best, rec, shrunk = fails[0][0], fails[0][1], False  # el pool está ordenado de menor a mayor
    banned = _banned(p)
    for _ in range(rounds):
        cands = [a for a in shrink_args(best.args) if _key(a) not in banned]
        cases = [c for c in (p._case(a, strict=False) for a in cands) if c]
        if not cases:
            break
        # sólo fallos EJECUTADOS: tras varios timeouts el harness aborta y los casos restantes quedan sin registro
        f2 = [(c, r2) for c, r2, ok in _run(p, code, cases, cfg)[0] if not ok and r2 is not None]
        if not f2:
            break
        c2, r2 = min(f2, key=lambda t: len(_key(t[0].args)))
        if len(_key(c2.args)) >= len(_key(best.args)):
            break
        best, rec, shrunk = c2, r2, True
    return best, rec, shrunk


def _describe(p, code, c, rec, cfg, where="Counterexample"):
    want = f"raise {c.exc}" if c.exc else f"return {_short(json.loads(c.exp))}"
    call = _call(p.entry, c.args)
    limit = cfg.sandbox_limits["case_timeout_s"]
    if rec is None or rec.get("exc") == "Timeout":
        from rewards.reward import _size
        t0 = time.perf_counter()
        try:
            p.ref(*json.loads(json.dumps(c.args)))
        except Exception:
            pass
        ref_t = time.perf_counter() - t0
        return (f"{where}: `{call}` (input size {_size(c.args)}) did not finish within {limit}s; the reference "
                f"solution takes {ref_t:.3f}s. Look for an infinite loop or a recursion that never stops.")
    if "exc" in rec:
        line = rec.get("line")
        at = f" at line {line}: `{_code_line(code, line)}`" if _code_line(code, line) else ""
        msg = f": {rec['msg'][:120]}" if rec.get("msg") else ""
        return f"{where}: `{call}` raised {rec['exc']}{msg}{at}. It should {want}."
    try:
        got = json.loads(rec["out"])
    except (KeyError, ValueError):
        return f"{where}: `{call}` should {want} but returned a different value."
    if c.exc:
        return f"{where}: `{call}` should raise {c.exc} but returned {_short(got)}."
    return f"{where}: `{call}` should {want} but returned {_short(got)}." + _first_diff(json.loads(c.exp), got)


def _too_slow(p, code, cfg, seed):
    """Mensaje de lentitud con una entrada grande NUEVA, o None si no se reproduce."""
    from rewards.reward import _size
    big = p.timeout_cases(seed)
    if not big:
        return None
    res, run = _run(p, code, big, cfg)
    for c, rec, ok in res:
        if not ok and (rec is None or rec.get("exc") == "Timeout"):
            t0 = time.perf_counter()
            p.ref(*json.loads(json.dumps(c.args)))
            ref_t = time.perf_counter() - t0
            return (f"Too slow: on a large input of size {_size(c.args)} (a new input, not one of the tests) your "
                    f"function did not finish within {cfg.sandbox_limits['case_timeout_s']}s, while the reference "
                    f"solution takes {ref_t:.2f}s. Use a more efficient algorithm (avoid nested loops over the input).")
        if not ok:
            return _describe(p, code, c, rec, cfg, "On a large new input")
    return None


def feedback_v2(p, code, r, cfg, seed=0, truncated=False, pool_n=64):
    """(texto, info) accionable. info: {mode, source: static|visible|pool|large|none, args?, shrunk?}."""
    info = {"mode": "counterexample", "source": "static"}
    if r.solved:
        return None, info
    e = r.error or ""
    if e == "syntax":
        return (TRUNCATED + "\n" + _syntax(code) if truncated else _syntax(code)), info
    if truncated:
        return TRUNCATED, info
    if e == "no_entry":
        names = [n.name for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return (f"The code must define `{p.sig}` at the top level (it defines: "
                f"{', '.join(names) if names else 'no functions'})."), info
    if e.startswith(("unsafe:", "crash")):
        return feedback(p, code, r, cfg.sandbox.allowed_imports), info
    if e.startswith("load:"):
        at = f" at line {r.line}: `{_code_line(code, r.line)}`" if _code_line(code, r.line) else ""
        return f"The code fails as soon as it is loaded: {e[5:]}{at}.", info
    n_fail = r.n_cases - r.n_pass
    tail = f" {n_fail} of {r.n_cases} tests fail."
    if r.detail:  # caso visible: ya está en el prompt
        d, fc = r.detail, r.fail_case or {}
        visible_exc = fc.get("kind") == "visible" and "raised" in d["got"]
        at = f" at line {r.line}: `{_code_line(code, r.line)}`" if visible_exc and _code_line(code, r.line) else ""
        diff = ""
        if fc.get("kind") == "visible" and d["got"].startswith("returned ") and isinstance(fc.get("index"), int):
            try:
                c = p.visible()[fc["index"]]
                if not c.exc:
                    diff = _first_diff(json.loads(c.exp), ast.literal_eval(d["got"][len("returned "):]))
            except (ValueError, SyntaxError, IndexError, TypeError, MemoryError, RecursionError):
                diff = ""
        return f"`{d['call']}` should {d['want']} but {d['got']}{at}.{diff}{tail}", dict(info, source="visible")
    cx = counterexample(p, code, cfg, seed, pool_n)
    if cx is None and e != "timeout":  # segundo intento: pool 4x mayor con otra semilla (igual de libre de ocultos)
        cx = counterexample(p, code, cfg, seed + 1_000_003, pool_n * 4)
    if cx:
        c, rec, shrunk = cx
        if (rec is None or rec.get("exc") == "Timeout") and e == "timeout":
            slow = _too_slow(p, code, cfg, seed)
            if slow:
                return slow + tail, dict(info, source="large")
        return _describe(p, code, c, rec, cfg) + tail, dict(info, source="pool", args=c.args, shrunk=shrunk)
    if e == "timeout":
        slow = _too_slow(p, code, cfg, seed)
        if slow:
            return slow + tail, dict(info, source="large")
    return feedback(p, code, r, cfg.sandbox.allowed_imports), dict(info, source="none")


def build(p, code, r, cfg, mode="basic", seed=0, truncated=False):
    """Feedback según fix.feedback. 'basic' = feedback() de siempre, sin cambios. 'counterexample' = v2 (si el
    problema no tiene generador, como una tarea personalizada del panel, cae a basic)."""
    if mode != "counterexample" or not hasattr(p, "feedback_cases"):
        return feedback(p, code, r, cfg.sandbox.allowed_imports), {"mode": "basic"}
    from common import opt
    return feedback_v2(p, code, r, cfg, seed, truncated, int(opt(cfg, "fix.pool", 64)))
