"""Reparación sintética: mutantes de las referencias convertidos en tareas de debugging (SÓLO split train).

Un mutante entra si compila, falla al menos un test (así no es equivalente a la referencia) y su problema base es de
train. run_batch ejecuta CÓDIGO PROPIO (referencias, sus mutantes, versiones lentas) en un proceso hijo con límites:
es para validar el dataset, no para código del modelo (ese va siempre al sandbox).
"""
from __future__ import annotations

import ast, copy, functools, hashlib, json, os, random, subprocess, sys, zlib

from common import ROOT

SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
        ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv,
        ast.And: ast.Or, ast.Or: ast.And}


def _sites(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            out += [(n, "ops", i) for i, o in enumerate(n.ops) if type(o) in SWAP]
        elif isinstance(n, (ast.BinOp, ast.AugAssign, ast.BoolOp)) and type(n.op) in SWAP:
            out.append((n, "op", None))
        elif isinstance(n, ast.Constant) and type(n.value) is int and abs(n.value) <= 10:
            out += [(n, "+1", None), (n, "-1", None)]
    return out


def mutants(src):
    """[(descripción, código)] con UN cambio de operador o constante cada uno (orden determinista)."""
    tree = ast.parse(src)
    out = []
    for k in range(len(_sites(tree))):
        t = copy.deepcopy(tree)
        n, kind, i = _sites(t)[k]
        if kind == "ops":
            old = type(n.ops[i]).__name__
            n.ops[i] = SWAP[type(n.ops[i])]()
            desc = f"línea {n.lineno}: {old} -> {type(n.ops[i]).__name__}"
        elif kind == "op":
            old = type(n.op).__name__
            n.op = SWAP[type(n.op)]()
            desc = f"línea {n.lineno}: {old} -> {type(n.op).__name__}"
        else:
            n.value += 1 if kind == "+1" else -1
            desc = f"línea {n.lineno}: constante {n.value - 1 if kind == '+1' else n.value + 1} -> {n.value}"
        code = ast.unparse(t)
        compile(code, "<mutant>", "exec")
        out.append((desc, code))
    return out


RUNNER = r"""
import copy, json, resource, signal, sys
job = json.loads(sys.stdin.read())
sys.path.insert(0, job["root"])
try:
    resource.setrlimit(resource.RLIMIT_AS, (job["mem"] << 20, job["mem"] << 20))
except (ValueError, OSError):
    pass
from problems.problems import Case, judge
from sandbox.harness import enc

class Timeout(BaseException):
    pass

def alarm(*_):
    raise Timeout()

signal.signal(signal.SIGALRM, alarm)
sys.setrecursionlimit(4000)
cases = [Case(a, e, x) for a, e, x in job["cases"]]
res = []
for code, entry in job["codes"]:
    ns = {"__name__": "trusted"}
    try:
        exec(compile(code, "<trusted>", "exec"), ns)
        fn = ns[entry]
    except BaseException as e:
        res.append({"ok": False, "first": -1, "why": "load:" + type(e).__name__})
        continue
    ok, first, why = True, None, None
    for i, c in enumerate(cases):
        try:
            signal.setitimer(signal.ITIMER_REAL, job["t"][i])
            out = fn(*copy.deepcopy(c.args))
            signal.setitimer(signal.ITIMER_REAL, 0)
            rec = {"out": enc(out)}
        except Timeout:
            rec = {"exc": "Timeout"}
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            rec = {"exc": type(e).__name__, "mro": [k.__name__ for k in type(e).__mro__]}
        if not judge(c, rec):
            ok, first, why = False, i, rec.get("exc") or "wrong"
            if job["stop"]:
                break
    res.append({"ok": ok, "first": first, "why": why})
print(json.dumps(res))
"""


def run_batch(codes, cases, case_timeout=2.0, stop=True, mem_mb=2048, timeout=None):
    """Ejecuta [(código, entry)] contra los mismos casos en UN proceso hijo (código propio de confianza).
    case_timeout: un número o una lista con el límite de cada caso.
    Devuelve por código {"ok": pasa todos, "first": índice del primer fallo, "why": excepción o 'wrong'}."""
    if not codes:
        return []
    ts = [float(t) for t in case_timeout] if isinstance(case_timeout, (list, tuple)) else [float(case_timeout)] * len(cases)
    job = dict(root=ROOT, codes=[list(c) for c in codes], cases=[[c.args, c.exp, c.exc] for c in cases],
               t=ts, stop=bool(stop), mem=int(mem_mb))
    budget = timeout or 60 + len(codes) * (sum(ts) + 0.05 * len(cases))
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER], input=json.dumps(job), capture_output=True, text=True,
                           timeout=budget, cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": "0"})
    except subprocess.TimeoutExpired:  # p. ej. un mutante que se traga la excepción del timeout en un bucle
        return [{"ok": False, "first": -1, "why": "batch_timeout"} for _ in codes]
    if p.returncode != 0:
        raise RuntimeError("run_batch falló: " + p.stderr[-500:])
    return json.loads(p.stdout.strip().splitlines()[-1])


SMALL_TIMEOUT = 0.25  # la referencia tarda microsegundos en los casos pequeños: un mutante en bucle cae enseguida


def ordered_cases(problem, case_timeout=2.0):
    """Todos los casos con los pequeños primero (límite corto) y los grandes al final (límite completo)."""
    hid, kinds = problem.hidden(), problem.hidden_kinds()
    small = problem.visible() + [c for c, k in zip(hid, kinds) if k != "large"] + problem.generated(0, 24)
    big = [c for c, k in zip(hid, kinds) if k == "large"]
    return small + big, [SMALL_TIMEOUT] * len(small) + [case_timeout] * len(big)


def killed(problem, mutant_codes, case_timeout=2.0):
    """Para cada mutante: True si algún test (visible, oculto o generado) lo detecta."""
    cases, ts = ordered_cases(problem, case_timeout)
    return [not r["ok"] for r in run_batch([(c, problem.entry) for c in mutant_codes], cases, ts)]


class RepairTask:
    """Tarea de debugging: el enunciado del problema base + una implementación con un bug real (mutante que falla
    tests). Tests, referencia y oráculo son los del problema base. Siempre train."""

    def __init__(self, base, desc, code, k):
        self.base, self.desc, self.buggy = base, desc, code
        self.id, self.entry, self.level = f"repair:{base.id}:{k}", base.entry, base.level
        self.split, self.category, self.family, self.source = "train", "debugging", "repair:" + base.family, "repair"
        self.ref, self.errs = base.ref, base.errs
        self.version = hashlib.sha1((base.version + code).encode()).hexdigest()[:12]

    def prompt(self):
        return (f"{self.base.prompt()}\n\n# This implementation is buggy:\n{self.buggy.rstrip()}\n"
                f"# Write a corrected version of {self.entry}.")

    def ref_source(self):
        return self.base.ref_source()

    def __getattr__(self, name):  # visible/hidden/generated/feedback_cases/timeout_cases/hidden_kinds del base
        if name in ("visible", "hidden", "generated", "feedback_cases", "timeout_cases", "hidden_kinds", "sig", "doc",
                    "_case", "large", "gen"):
            return getattr(self.base, name)
        raise AttributeError(name)


@functools.lru_cache(maxsize=8)
def _tasks_cached(ids, per_problem, seed):
    from problems.problems import ALL
    out = []
    for pid in ids:
        p = ALL[pid]
        muts = mutants(p.ref_source())
        if not muts:
            continue
        valid = [m for m, k in zip(muts, killed(p, [c for _, c in muts])) if k]
        rnd = random.Random(seed ^ zlib.crc32(p.id.encode()))
        for k, (desc, code) in enumerate(rnd.sample(valid, min(per_problem, len(valid)))):
            out.append(RepairTask(p, desc, code, k))
    return tuple(out)


def tasks(probs, per_problem=3, seed=0):
    """Tareas de reparación de los problemas TRAIN de `probs` (los de val/test se ignoran: nunca hay mutantes
    suyos en entrenamiento). Deterministas por (ids, per_problem, seed) y cacheadas en el proceso."""
    ids = tuple(sorted(p.id for p in probs if getattr(p, "split", "train") == "train" and p.source != "repair"))
    return list(_tasks_cached(ids, per_problem, seed))

