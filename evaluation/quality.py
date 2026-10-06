"""Calidad de código: métricas SÓLO informativas sobre soluciones RESUELTAS. No entran en el reward."""
import ast
from collections import Counter

_BRANCH = (ast.If, ast.For, ast.While, ast.Try, ast.IfExp, ast.BoolOp, ast.comprehension, ast.With, ast.Match)
_BLOCK = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Match)


def _depth(node, d=0):
    best = d
    for ch in ast.iter_child_nodes(node):
        best = max(best, _depth(ch, d + isinstance(ch, _BLOCK)))
    return best


def _unused_vars(fn):
    stores, loads = Counter(), set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            (loads.add(n.id) if isinstance(n.ctx, ast.Load) else stores.update([n.id]))
    return sum(1 for v in stores if v not in loads and not v.startswith("_"))


def quality(code):
    """dict de métricas o None si no parsea."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    nodes = list(ast.walk(tree))
    imported = {}
    for n in nodes:
        if isinstance(n, ast.Import):
            imported.update({(a.asname or a.name).split(".")[0]: 1 for a in n.names})
        elif isinstance(n, ast.ImportFrom):
            imported.update({a.asname or a.name: 1 for a in n.names})
    used = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    blocks = [b for n in nodes for b in (getattr(n, "body", None), getattr(n, "orelse", None)) if isinstance(b, list)]
    stmts = Counter(ast.dump(s) for b in blocks for s in b if isinstance(s, ast.stmt) and sum(1 for _ in ast.walk(s)) >= 4)
    module_state = sum(1 for s in tree.body if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and not all(
        isinstance(t, ast.Name) and t.id.isupper() for t in getattr(s, "targets", [getattr(s, "target", None)])))
    return dict(
        nodes=len(nodes),
        branches=sum(isinstance(n, _BRANCH) for n in nodes),
        depth=_depth(tree),
        duplication=sum(c - 1 for c in stmts.values() if c > 1),
        unused_imports=sum(1 for name in imported if name not in used),
        unused_vars=sum(_unused_vars(n) for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        bare_except=sum(isinstance(n, ast.ExceptHandler) and n.type is None for n in nodes),
        global_state=module_state + sum(isinstance(n, (ast.Global, ast.Nonlocal)) for n in nodes),
        prints=sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print" for n in nodes),
    )


def mean_quality(codes):
    qs = [q for q in map(quality, codes) if q]
    if not qs:
        return None
    return {k: round(sum(q[k] for q in qs) / len(qs), 3) for k in qs[0]}
