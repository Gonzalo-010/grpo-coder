"""python tests/run_tests.py [-k texto]
Código de salida 1 si algo falla. SKIP = NO verificado (falta GPU / torch / bwrap): no cuenta como éxito.
ALLOW_UNSAFE_SANDBOX=1 permite correr sin bubblewrap (backend rlimit; los tests de aislamiento se saltan).
"""
import json, hashlib, math, os, random, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import opt, load_config
from evaluation.metrics import pass_at_k
from problems.problems import PROBLEMS, baseline
from rewards.reward import score
from sandbox import executor

CFG = load_config()
if os.environ.get("ALLOW_UNSAFE_SANDBOX") == "1":
    CFG["sandbox"]["allow_unsafe"] = True


class Skip(Exception):
    pass


TESTS = []


def test(name):
    def deco(f):
        TESTS.append((name, f))
        return f
    return deco


def backend():
    return executor.backend(CFG.sandbox)


def run(code, cases, entry="f", **lim):
    L = dict(CFG.sandbox_limits)
    L.update(lim)
    return executor.run_cases(code, entry, cases, L, CFG.sandbox.allowed_imports, backend())


def harness_procs():
    n = 0
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            n += b"harness.py" in open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            pass
    return n


# ------------------------------------------------------------------ evaluación
@test("evaluación: pass@k")
def t_passk():
    assert pass_at_k(10, 0, 1) == 0 and pass_at_k(10, 10, 1) == 1
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-12
    assert abs(pass_at_k(4, 1, 2) - 0.5) < 1e-12 and pass_at_k(4, 3, 2) == 1.0


# ------------------------------------------------------------------ sandbox
@test("sandbox: ejecución básica y tipos de salida")
def t_sbx_basic():
    r = run("def f(x):\n    return x * 2\n", [[1], [21], [-3]])
    assert r.status == "ok" and [c["out"] for c in r.cases] == ["2", "42", "-6"], r
    r = run("def f(x):\n    return {'t': (1, 2), 's': {3, 1, 2}}[x]\n", [["t"], ["s"], ["zz"]])
    assert [c.get("out") for c in r.cases[:2]] == ["[1,2]", "[1,2,3]"] and r.cases[2]["exc"] == "KeyError"
    r = run("def f(x):\n    return object()\n", [[1]])
    assert r.cases[0]["exc"] == "TypeError"
    big = run("def f(x):\n    return 'a' * 10**6\n", [[1]]).cases[0]
    assert big["h"] == hashlib.sha256(('"' + "a" * 10**6 + '"').encode()).hexdigest() and "out" not in big


@test("sandbox: timeout por caso, continúa con los siguientes")
def t_sbx_case_timeout():
    r = run("def f(x):\n    if x == 1:\n        while True: pass\n    return x\n", [[0], [1], [2]], case_timeout_s=0.3)
    assert r.status == "ok" and r.cases[1]["exc"] == "Timeout" and r.cases[2]["out"] == "2", r


@test("sandbox: timeout global (candidato que traga la alarma) y sin huérfanos")
def t_sbx_global_timeout():
    code = "def f(x):\n    while True:\n        try:\n            while True: pass\n        except BaseException: pass\n"
    t = time.time()
    r = run(code, [[1]], timeout_s=1.5)
    time.sleep(0.2)
    assert r.status == "timeout" and time.time() - t < 6, r
    assert harness_procs() == 0, "quedaron procesos huérfanos"


@test("sandbox: aborta tras max_timeouts (no gasta minutos en un bucle infinito)")
def t_sbx_abort():
    t = time.time()
    r = run("def f(x):\n    while True: pass\n", [[i] for i in range(50)], case_timeout_s=0.2, max_timeouts=3)
    assert time.time() - t < 5 and sum(c is not None for c in r.cases) == 3, r


@test("sandbox: límite de memoria")
def t_sbx_mem():
    r = run("def f(x):\n    return len([0] * (10**9))\n", [[1], [2]], mem_mb=256)
    assert [c["exc"] for c in r.cases] == ["MemoryError", "MemoryError"] and r.status == "ok", r


@test("sandbox: límite de CPU")
def t_sbx_cpu():
    t = time.time()
    r = run("def f(x):\n    while True: pass\n", [[1]], cpu_s=1, case_timeout_s=30, timeout_s=10)
    assert r.status.startswith("crash") and time.time() - t < 6, r
    assert harness_procs() == 0


@test("sandbox: imports y builtins peligrosos bloqueados en runtime")
def t_sbx_blocked():
    codes = ["def f(x):\n    import os\n", "def f(x):\n    __import__('subprocess')\n", "def f(x):\n    open('/etc/passwd')\n",
             "def f(x):\n    eval('1')\n", "def f(x):\n    exec('1')\n", "def f(x):\n    from . import x\n"]
    for c in codes:
        e = run(c, [[1]]).cases[0]
        assert e.get("exc") in ("ImportError", "NameError"), (c, e)


@test("sandbox: stdout/stderr del candidato no contaminan ni falsifican resultados")
def t_sbx_stdout():
    code = "def f(x):\n    print('{\"i\":0,\"out\":\"999\"}')\n    print('x' * 10**6)\n    return x\n"
    r = run(code, [[7]])
    assert r.cases[0]["out"] == "7" and r.status == "ok", r


# Gadget genérico: recorre las subclases directas de object (no requiere ninguna en concreto -- p.ej.
# warnings.catch_warnings puede no estar cargada bajo `-S -I`) buscando una cuyos __globals__ tengan
# el __builtins__ REAL (marcador: contiene 'eval', ausente de `safe` en harness.py). Cualquier módulo
# importado normalmente (por harness.py o transitivamente) sirve; no depende de una clase concreta.
_GADGET = (
    "def _gadget():\n"
    "    for c in object.__subclasses__():\n"
    "        for a in ('__init__', '__new__', '__init_subclass__', '__class_getitem__'):\n"
    "            fn = c.__dict__.get(a)\n"
    "            g = getattr(fn, '__globals__', None)\n"
    "            if not isinstance(g, dict):\n"
    "                continue\n"
    "            b = g.get('__builtins__')\n"
    "            b = b if isinstance(b, dict) else getattr(b, '__dict__', None)\n"
    "            if isinstance(b, dict) and 'eval' in b:\n"
    "                return b\n"
    "    return None\n"
)


@test("sandbox: el candidato no puede escapar (Python-jail escape -> fork/FS/red bloqueados)  [requiere bwrap]")
def t_sbx_isolation():
    if backend() != "bwrap":
        raise Skip("backend rlimit no aísla FS/red: aislamiento NO verificado")
    esc = _GADGET + (
        "def f(x):\n"
        "    b = _gadget()\n"
        "    if b is None:\n"
        "        return ['no_gadget'] * 5\n"
        "    imp, op = b['__import__'], b.get('open')\n"
        "    try: osm = imp('os')\n"
        "    except BaseException: osm = None\n"
        "    try: skm = imp('socket')\n"
        "    except BaseException: skm = None\n"
        "    out = []\n"
        "    for probe in (lambda: osm.fork(), lambda: osm.listdir('/home'), lambda: osm.listdir(%r),\n"
        "                  lambda: op('/etc/passwd').read(),\n"
        "                  lambda: skm.create_connection(('1.1.1.1', 53), timeout=2)):\n"
        "        try:\n"
        "            probe(); out.append('OPEN')\n"
        "        except BaseException as e:\n"
        "            out.append('blocked:' + type(e).__name__)\n"
        "    return out\n") % ROOT
    import json
    labels = ["fork()", "listdir(/home)", "listdir(project ROOT)", "open(/etc/passwd)", "socket a IP externa"]
    out = run(esc, [[0]]).cases[0]
    assert "out" in out, out  # fatal/timeout/crash antes de devolver la lista
    got = json.loads(out["out"])
    if got == ["no_gadget"] * 5:
        raise Skip("ninguna clase cargada expone __builtins__ real: técnica no verificada en este proceso")
    bad = [(lbl, v) for lbl, v in zip(labels, got) if not v.startswith("blocked")]
    assert not bad, f"NO bloqueado: {bad}"
    print(f"      {dict(zip(labels, got))}")


@test("reward: el mismo escape ya es rechazado por el filtro AST antes de llegar al sandbox (capa 1)")
def t_reward_blocks_escape():
    code = _GADGET + "def abs_sum(a, b):\n    return _gadget() is not None\n"
    r = score(PROBLEMS["abs_sum"], code, CFG)
    assert r.error and r.error.startswith("unsafe:"), r
    assert r.reward == CFG.reward.unsafe_penalty and r.wall_s == 0, "unsafe debe rechazar SIN ejecutar"


@test("sandbox: el escape por subclases es real -> por eso existen la capa AST y el aislamiento del SO")
def t_sbx_escape_exists():
    if backend() != "rlimit":
        raise Skip("sólo informativo con backend rlimit")
    code = _GADGET + "def f(x):\n    b = _gadget()\n    return None if b is None else b['__import__']('os').getpid() > 0\n"
    out = run(code, [[0]]).cases[0]
    if out.get("out") == "null":
        raise Skip("ninguna clase cargada expone __builtins__ real: técnica no verificada en este proceso")
    assert out.get("out") == "true", out


# ------------------------------------------------------------------ problemas / reward
@test("problemas: tests bien formados (3 niveles, sin baseline trivial)")
def t_problems_wellformed():
    for p in PROBLEMS.values():
        assert len(p.visible()) >= 1 and len(p.hidden()) >= 14, p.id
        assert baseline(p.hidden()) <= 0.75, (p.id, "hidden con baseline constante alto")
        assert len(p.prompt()) < 1200, (p.id, len(p.prompt()))
        for seed in range(25):
            g = p.generated(seed, 24)
            assert len(g) >= 12 and baseline(g) <= 0.85, (p.id, seed, len(g), baseline(g))
        assert {c.exp for c in p.hidden() if c.exp} != {c.exp for c in p.visible() if c.exp} or len(p.hidden()) > len(p.visible())


@test("problemas: cada referencia resuelve todos sus tests a través del sandbox (reward 1.0)")
def t_problems_refs():
    bad = []
    with ThreadPoolExecutor(4) as ex:
        for p, r in zip(PROBLEMS.values(), ex.map(lambda p: score(p, p.ref_source(), CFG, 5), PROBLEMS.values())):
            if not r.solved or r.reward < 0.999:
                bad.append((p.id, r.error, r.n_pass, r.n_cases))
    assert not bad, bad


def _oracles():
    from collections import deque
    from functools import lru_cache
    import calendar, itertools, re

    def ex(f, *a):
        try:
            return f(*a)
        except ValueError:
            return "ValueError"

    def two_sum(n, t):
        return next(([i, j] for j in range(len(n)) for i in range(j) if n[i] + n[j] == t), [])

    def lis(n):
        b = [1] * len(n)
        for i in range(len(n)):
            for j in range(i):
                if n[j] < n[i]:
                    b[i] = max(b[i], b[j] + 1)
        return max(b, default=0)

    def edit(a, b):
        @lru_cache(None)
        def d(i, j):
            if i == len(a): return len(b) - j
            if j == len(b): return len(a) - i
            return min(d(i + 1, j) + 1, d(i, j + 1) + 1, d(i + 1, j + 1) + (a[i] != b[j]))
        return d(0, 0)

    def coin(cs, amt):
        if amt == 0: return 0
        seen, q = {0}, deque([(0, 0)])
        while q:
            s, k = q.popleft()
            for c in cs:
                if s + c == amt: return k + 1
                if s + c < amt and s + c not in seen:
                    seen.add(s + c); q.append((s + c, k + 1))
        return -1

    def path(n, es, s, t):
        d = [float("inf")] * n; d[s] = 0
        for _ in range(n):
            for u, v, w in es:
                d[v] = min(d[v], d[u] + w)
        return -1 if d[t] == float("inf") else d[t]

    def brackets(s):
        p = None
        while p != s:
            p, s = s, s.replace("()", "").replace("[]", "").replace("{}", "")
        return s == ""

    def topk(n, k):
        if k < 1: raise ValueError
        return sorted(set(n), key=lambda v: (-n.count(v), v))[:k]

    def second(n):
        v = sorted(set(n), reverse=True)
        if len(v) < 2: raise ValueError
        return v[1]

    def rot(n, k):
        n = list(n)
        for _ in range(k if n else 0): n.append(n.pop(0))
        return n

    def anagrams(ws):
        gs = []
        for w in ws:
            for g in gs:
                if sorted(g[0]) == sorted(w): g.append(w); break
            else: gs.append([w])
        return gs

    def lru(cap, ops):
        keys, val, out = [], {}, []  # keys: menos reciente -> más reciente
        for o in ops:
            if o[0] == "put":
                if o[1] in keys: keys.remove(o[1])
                keys.append(o[1]); val[o[1]] = o[2]
                if len(keys) > cap: del val[keys.pop(0)]
            elif o[1] in keys:
                keys.remove(o[1]); keys.append(o[1]); out.append(val[o[1]])
            else: out.append(-1)
        return out

    def clamp(x, lo, hi):
        if lo > hi: raise ValueError
        return sorted([lo, x, hi])[1]

    return {
        "abs_sum": lambda a, b: (a * a) ** .5 + (b * b) ** .5 if abs(a) < 1e7 and abs(b) < 1e7 else abs(a) + abs(b),
        "clamp": clamp,
        "fizzbuzz": lambda n: [("Fizz" * (i % 3 == 0) + "Buzz" * (i % 5 == 0)) or str(i) for i in range(1, n + 1)],
        "is_leap_year": lambda y: calendar.isleap(y),
        "reverse_words": lambda s: " ".join(s.split()[::-1]),
        "is_palindrome": lambda s: (lambda t: t == t[::-1])(re.sub(r"[\W_]+", "", s.lower())),
        "second_largest": second,
        "dedupe": lambda n: list(dict.fromkeys(n)),
        "rotate_left": rot,
        "rle_encode": lambda s: "".join(f"{k}{len(list(g))}" for k, g in itertools.groupby(s)),
        "two_sum": two_sum, "valid_brackets": brackets, "top_k_frequent": topk, "group_anagrams": anagrams, "lru": lru,
        "lis_length": lis, "edit_distance": edit, "coin_change": coin, "shortest_path": path,
    }, ex


@test("problemas: referencias == oráculos independientes (fuerza bruta) en entradas pequeñas")
def t_problems_oracles():
    orc, ex = _oracles()
    assert set(orc) == set(PROBLEMS), set(PROBLEMS) ^ set(orc)
    for pid, f in orc.items():
        p, r = PROBLEMS[pid], random.Random(99)
        cases = [c.args for c in p.visible() + p.hidden() if len(str(c.args)) < 400]
        cases += [p.gen(r) for _ in range(300)]
        for a in cases:
            e1, e2 = ex(p.ref, *a), ex(f, *a)
            ok = (abs(e1 - e2) < 1e-6) if pid == "abs_sum" else e1 == e2
            assert ok, (pid, a, e1, e2)


CANDS = {
    "always_true": ("is_palindrome", "def is_palindrome(s):\n    return True\n"),
    "always_false": ("is_palindrome", "def is_palindrome(s):\n    return False\n"),
    "hardcode_visible": ("is_palindrome", "def is_palindrome(s):\n    return {'A man, a plan, a canal: Panama': True, "
                                          "'race a car': False, '': True}.get(s, True)\n"),
    "constant_int": ("abs_sum", "def abs_sum(a, b):\n    return 0\n"),
}


@test("reward: 'siempre True', constantes y hardcodeo de visibles NO obtienen reward alto")
def t_reward_hacks():
    for name, (pid, code) in CANDS.items():
        r = score(PROBLEMS[pid], code, CFG, 3)
        assert not r.solved and r.reward < 0.25, (name, r)


@test("reward: correcto=1.0, parcial intermedio, sintaxis/sin entry=0, unsafe<0 sin ejecutar, bucle infinito=timeout")
def t_reward_ladder():
    P, ok = PROBLEMS, PROBLEMS["is_palindrome"]
    good = score(ok, ok.ref_source(), CFG, 3)
    naive = score(ok, "def is_palindrome(s):\n    return s == s[::-1]\n", CFG, 3)
    assert good.reward >= 0.999 and good.solved and 0.1 < naive.reward < 0.6 and not naive.solved, (good, naive)
    assert score(ok, "def is_palindrome(s:\n", CFG).error == "syntax" and score(ok, "def otra(s): return 1", CFG).error == "no_entry"
    for bad in ("import os\ndef is_palindrome(s):\n    return True\n", "def is_palindrome(s):\n    return eval('1')\n",
                "def is_palindrome(s):\n    return s.__class__ is None\n"):
        r = score(ok, bad, CFG)
        assert r.reward == CFG.reward.unsafe_penalty and r.error.startswith("unsafe") and r.wall_s == 0, r
    L = CFG.sandbox_limits
    old = L["case_timeout_s"]
    L["case_timeout_s"] = 0.3
    try:
        r = score(P["abs_sum"], "def abs_sum(a, b):\n    while True: pass\n", CFG)
    finally:
        L["case_timeout_s"] = old
    assert r.error == "timeout" and r.reward < 0.1, r


@test("reward: solución O(n^2) pasa lo pequeño pero NO 'solved' (los tests grandes exigen complejidad)")
def t_reward_complexity():
    quad = ("def two_sum(nums, target):\n    for j in range(len(nums)):\n        for i in range(j):\n"
            "            if nums[i] + nums[j] == target: return [i, j]\n    return []\n")
    L = CFG.sandbox_limits
    old = L["case_timeout_s"]
    L["case_timeout_s"] = 0.5
    try:
        r = score(PROBLEMS["two_sum"], quad, CFG, 3)
    finally:
        L["case_timeout_s"] = old
    assert r.error == "timeout" and not r.solved and 0.4 < r.reward < 0.9, r


@test("reward: los esperados no llegan al sandbox (sólo entradas)")
def t_reward_no_leak():
    p = PROBLEMS["abs_sum"]
    import json
    cases = [c.args for c in p.visible() + p.hidden()]
    blob = json.dumps({"code": p.ref_source(), "cases": cases})
    assert "exp" not in blob and all(isinstance(a, list) for a in cases)


# ------------------------------------------------------------------ rollout (lógica de backoff, sin GPU)
@test("rollout: backoff de OOM (concurrencia -> longitud), sin GPU")
def t_backoff():
    from rollouts.generate import Generator, Sample

    class Fake(Generator):
        def __init__(self, cap):
            self.chunk, self.max_new, self.floor, self.cap = 32, 384, 96, cap
            self.stats = dict(tokens=0, samples=0, seconds=0.0, oom=0)
            self.m = type("M", (), {"eval": lambda self: None})()  # stream() llama a self.m.eval(); sin GPU real aquí

        def _gen(self, texts, full=False):
            if len(texts) * self.max_new > self.cap:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return [Sample("", "", 1, True) for _ in texts], 0.01

    g = Fake(800)
    out = list(g.stream(["p"] * 5, 8))
    assert len(out) == 40 and g.chunk == 2 and g.max_new == 384 and g.stats["oom"] == 4, (g.chunk, g.max_new, g.stats)
    g = Fake(200)
    out = list(g.stream(["p"], 4))
    assert len(out) == 4 and g.chunk == 1 and g.max_new == 162, (g.chunk, g.max_new)
    try:
        list(Fake(50).stream(["p"], 2))
        raise AssertionError("debió fallar en la configuración mínima")
    except RuntimeError as e:
        assert "mínimo" in str(e)


# ------------------------------------------------------------------ GPU
_M = {}


def _model():
    if not _M:
        try:
            import torch
        except ImportError:
            raise Skip("torch no instalado")
        if not torch.cuda.is_available():
            raise Skip("CUDA no disponible")
        from model.loader import load
        from rollouts.generate import Generator
        m, tok = load(CFG)
        _M.update(m=m, tok=tok, gen=Generator(m, tok, CFG))
    return _M["m"], _M["tok"], _M["gen"]


def _prompt(tok, pid):
    from rollouts.generate import build_prompt
    return build_prompt(tok, CFG.prompt.system, PROBLEMS[pid].prompt(), CFG.prompt.prefill_fence)


@test("modelo: carga bf16/fp16 sin cuantización; PyTorch con kernels para esta GPU")
def t_model():
    m, tok, _ = _model()
    import torch
    if torch.cuda.get_device_capability(0)[0] >= 12:
        assert "sm_120" in torch.cuda.get_arch_list(), "PyTorch sin sm_120: instala el build cu128+"
    assert next(m.parameters()).dtype in (torch.bfloat16, torch.float16)
    assert not getattr(m.config, "quantization_config", None)
    ids = tok("def f(x):", return_tensors="pt").to(m.device)
    with torch.inference_mode():
        assert torch.isfinite(m(**ids).logits).all()
    print(f"      dtype={next(m.parameters()).dtype} vram={torch.cuda.memory_allocated() >> 20}MB")


@test("generación: sólo código, parada en el fence, respeta max_new_tokens")
def t_generation():
    m, tok, gen = _model()
    out = [s for _, s in gen.stream([_prompt(tok, "abs_sum")], 6)]
    assert len(out) == 6 and all(s.code.strip() for s in out)
    assert all(s.ntok <= gen.max_new and "```" not in s.code for s in out)
    assert any(s.ntok < gen.max_new and s.closed for s in out), "ninguna muestra paró sola (¿stop_strings?)"
    print(f"      tokens/muestra={sum(s.ntok for s in out) / 6:.0f}  tok/s={gen.stats['tokens'] / gen.stats['seconds']:.0f}")


@test("rollout: generar -> ejecutar -> verificar -> reward (extremo a extremo)")
def t_rollout():
    m, tok, gen = _model()
    ids = ["abs_sum", "clamp", "is_leap_year"]
    prompts = [_prompt(tok, i) for i in ids]
    pool, pend = ThreadPoolExecutor(2), [[] for _ in ids]
    for i, s in gen.stream(prompts, 4):
        pend[i].append(pool.submit(score, PROBLEMS[ids[i]], s.code, CFG, 1))
    res = [[f.result() for f in fs] for fs in pend]
    pool.shutdown()
    assert all(len(r) == 4 for r in res) and all(-0.5 <= x.reward <= 1.0 for r in res for x in r)
    print("      " + " ".join(f"{i}:{sum(x.solved for x in r)}/4" for i, r in zip(ids, res)))


@test("rollout: autotuner de concurrencia (rápido, max_new=64)")
def t_tune():
    m, tok, gen = _model()
    old_new, old_chunk = gen.max_new, gen.chunk
    gen.max_new = 64
    try:
        rows = gen.tune(_prompt(tok, "abs_sum"), max_bs=8)
    finally:
        gen.max_new = old_new
    assert rows and 1 <= gen.chunk <= 8, rows
    print("      " + str(rows))
    gen.chunk = old_chunk


@test("monitor: VRAM/GPU")
def t_monitor():
    from rollouts.monitor import snapshot
    s = snapshot()
    if "used_mb" not in s:
        raise Skip("sin NVML ni CUDA")
    assert s["total_mb"] > 1000
    print(f"      {s}")


# ==================================================================== FASE 2: GRPO + checkpoint
@test("grpo: group_advantages (varianza -> media 0; std=0 -> todo 0, sin NaN; grupos independientes)")
def t_group_advantages():
    from training.grpo import group_advantages
    advs = group_advantages([("p1", 1.0), ("p1", 0.0), ("p1", 0.5), ("p1", 0.5)])
    assert abs(sum(advs)) < 1e-9 and advs[2] == advs[3] == 0.0, advs
    assert group_advantages([("p2", 0.7), ("p2", 0.7), ("p2", 0.7)]) == [0.0, 0.0, 0.0]  # std=0 -> sin NaN/Inf
    inter = group_advantages([("a", 1.0), ("b", 0.0), ("a", 0.0), ("b", 1.0)])
    assert inter[0] == -inter[2] and inter[1] == -inter[3], inter  # cada grupo normalizado por separado


@test("grpo: KL k3 (Schulman) siempre >= 0, y = 0 cuando ref==policy")
def t_kl_k3_math():
    import math
    k3 = lambda d: math.exp(d) - d - 1  # misma fórmula que training.grpo.kl_k3, en math puro
    for d in (-3.0, -0.5, 0.0, 0.3, 2.0):
        assert k3(d) >= -1e-9, (d, k3(d))
    assert abs(k3(0.0)) < 1e-12


@test("checkpoint: candidates/read_meta (LATEST primero) y fallback al anterior si el más reciente está corrupto")
def t_checkpoint_logic():
    import shutil as sh

    from training import checkpoint as ck
    root = "/tmp/ck_logic_test"
    sh.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    import json as j
    for step, h in [(1, "AAA"), (2, "BBB"), (3, "AAA")]:
        d = os.path.join(root, f"ckpt_{step:07d}")
        os.makedirs(d)
        j.dump(dict(step=step, config_hash=h), open(os.path.join(d, "meta.json"), "w"))
    open(os.path.join(root, "LATEST"), "w").write("ckpt_0000003")
    cands = ck.candidates(root)
    assert os.path.basename(cands[0]) == "ckpt_0000003", cands

    root_runs = "/tmp/ck_logic_runs"
    sh.rmtree(root_runs, ignore_errors=True)
    sh.copytree(root, os.path.join(root_runs, "20260101-000000", "checkpoints"))
    assert ck.find_run_dir(root_runs, "AAA") is not None
    assert ck.find_run_dir(root_runs, "ZZZ") is None

    os.remove(os.path.join(cands[0], "meta.json"))  # simula el más reciente corrupto
    ok = next((p for p in ck.candidates(root) if os.path.exists(os.path.join(p, "meta.json"))), None)
    assert ok and os.path.basename(ok) == "ckpt_0000002", "no cayó al candidato anterior"
    sh.rmtree(root, ignore_errors=True)
    sh.rmtree(root_runs, ignore_errors=True)


@test("checkpoint: config_hash ignora cambios operativos, distingue los de estado y acepta checkpoints con el hash antiguo")
def t_config_hash():
    import copy
    import hashlib
    import json as j
    import shutil as sh

    import yaml

    from training import checkpoint as ck
    base = j.loads(j.dumps(load_config()))
    h = ck.config_hash(base)

    def with_(key, v, cfg=base):
        c = copy.deepcopy(cfg)
        *parents, last = key.split(".")
        d = c
        for k in parents:
            d = d[k]
        d[last] = v
        return c

    for key, v in [("grpo.max_steps", 8), ("checkpoint.every_steps", 2), ("checkpoint.eval_every_steps", 2),
                   ("num_rollouts", 4), ("dataset.levels", [0]), ("generation.temperature", 1.0), ("seed", 7)]:
        assert ck.config_hash(with_(key, v)) == h, f"cambiar {key} no debería impedir reanudar"
    for key, v in [("model", "otro/modelo"), ("dtype", "fp16"), ("lora.enabled", False), ("lora.r", 16),
                   ("lora.target_modules", ["q_proj"]), ("grpo.lr", 2e-5), ("grpo.weight_decay", 0.01)]:
        assert ck.config_hash(with_(key, v)) != h, f"cambiar {key} debería impedir reanudar"

    # runs de antes de acotar el hash: meta con el sha de la config completa, claves ya eliminadas incluidas
    root_runs = "/tmp/ck_hash_runs"
    sh.rmtree(root_runs, ignore_errors=True)

    def old_run(name, cfg, saved_cfg=None):
        run = os.path.join(root_runs, name)
        d = os.path.join(run, "checkpoints", "ckpt_0000007")
        os.makedirs(d)
        with open(os.path.join(run, "config.yaml"), "w") as f:  # como common.new_run_dir
            yaml.safe_dump(saved_cfg or cfg, f, sort_keys=False)
        old = hashlib.sha256(j.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
        j.dump(dict(step=7, config_hash=old), open(os.path.join(d, "meta.json"), "w"))
        return run, d

    legacy = with_("grpo", dict(base["grpo"], inner_epochs=1, warmup_steps=20, max_steps=8))
    ok_run, ok = old_run("20260101-000000", legacy)
    _, other = old_run("20260102-000000", with_("lora.r", 16, legacy))
    # config.yaml editada a mano: ya no reproduce el hash guardado, así que no se confía en ella
    _, edited = old_run("20260103-000000", with_("lora.r", 16, legacy), saved_cfg=legacy)
    assert ck.meta_hash(ok) == h, "un checkpoint antiguo compatible debería poder reanudarse"
    assert ck.meta_hash(other) != h and ck.meta_hash(edited) != h, "aceptó un checkpoint con otro adapter"
    assert ck.find_run_dir(root_runs, h) == ok_run, "debe saltarse los runs incompatibles más recientes"
    sh.rmtree(root_runs, ignore_errors=True)


_P = {}


def _policy():
    if not _P:
        try:
            import torch
        except ImportError:
            raise Skip("torch no instalado")
        if not torch.cuda.is_available():
            raise Skip("CUDA no disponible")
        try:
            import peft  # noqa: F401
        except ImportError:
            raise Skip("peft no instalado (pip install peft)")
        from model.policy import load_policy, trainable_params
        from training.grpo import GrpoTrainer
        model, tok, ref_ctx = load_policy(CFG, "cuda")
        opt = torch.optim.AdamW(trainable_params(model), lr=CFG.grpo.lr)
        _P.update(model=model, tok=tok, ref_ctx=ref_ctx, opt=opt, trainer=GrpoTrainer(model, ref_ctx, tok, opt, CFG))
    return _P["model"], _P["tok"], _P["ref_ctx"], _P["opt"], _P["trainer"]


@test("policy: LoRA recién inicializado == referencia (B=0 al empezar) y % de params entrenables es bajo")
def t_policy_lora_init():
    if not CFG.lora.enabled:
        raise Skip("lora.enabled=false en config")
    model, tok, ref_ctx, opt, trainer = _policy()
    import torch
    from training.logprob import full_token_logps as token_logps
    model.eval()  # explícito (ver nota en t_checkpoint_roundtrip): no depender del orden de ejecución de tests
    ids = tok("def f(x):\n    return x", return_tensors="pt").to(model.device)
    with torch.no_grad():
        lp_policy = token_logps(model, ids["input_ids"], ids["attention_mask"])
        with ref_ctx() as ref_model:
            lp_ref = token_logps(ref_model, ids["input_ids"], ids["attention_mask"])
    gap = (lp_policy - lp_ref).abs().max().item()
    assert gap < 1e-2, f"policy y ref difieren al iniciar (gap={gap}); ¿lora_B no se inicializó en 0?"
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    assert 0 < n_tr / n_tot < 0.1, f"% de params entrenables sospechoso: {n_tr}/{n_tot}"
    print(f"      trainable={n_tr / 1e6:.2f}M/{n_tot / 1e6:.1f}M ({100 * n_tr / n_tot:.2f}%) |Δlogp|max={gap:.2e}")


@test("grpo: training step end-to-end (loss finita, kl>=0; con varianza de reward: gradiente y pesos cambian)")
def t_grpo_step():
    if not CFG.lora.enabled:
        raise Skip("lora.enabled=false en config")
    model, tok, ref_ctx, opt, trainer = _policy()
    import torch
    from rollouts.collect import collect
    from rollouts.generate import Generator
    from training.grpo import RollItem, group_advantages

    gen = Generator(model, tok, CFG)
    gen.max_new = min(gen.max_new, 160)  # smoke test: no necesitamos generaciones largas
    probs = [PROBLEMS["abs_sum"], PROBLEMS["lru"], PROBLEMS["edit_distance"]]  # fácil+difícil: más chance de varianza
    results = collect(gen, probs, 6, CFG, 1)
    flat = [(p.id, s, r) for p, rs in zip(probs, results) for s, r in rs]
    advs = group_advantages([(pid, r.reward) for pid, _, r in flat])
    items = [RollItem(s.prompt_ids, s.resp_ids, a) for (_, s, _), a in zip(flat, advs)]
    has_signal = any(a != 0.0 for a in advs)

    lora_param = next(p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)
    before = lora_param.detach().clone()
    stats = trainer.step(items)
    after = lora_param.detach()

    assert stats.loss == stats.loss and abs(stats.loss) < 1e6, stats  # descarta NaN/Inf
    assert stats.kl >= -1e-6, stats
    if has_signal:
        assert stats.grad_norm > 0 and not torch.equal(before, after), "hubo varianza de reward pero sin gradiente/cambio"
    else:
        print("      (sin varianza de reward en los 3 grupos esta vez; nada más que verificar en este step)")
    print(f"      loss={stats.loss:+.4f} kl={stats.kl:.4f} clip={stats.clip_frac:.3f} grad_norm={stats.grad_norm:.3f} "
          f"solved={sum(r.solved for _, _, r in flat)}/{len(flat)} micro_bs={trainer.micro_bs} oom={stats.oom}")


@test("checkpoint: guardar+recargar LoRA (adapter y optimizer) reproduce los mismos logits (roundtrip exacto)")
def t_checkpoint_roundtrip():
    if not CFG.lora.enabled:
        raise Skip("lora.enabled=false en config")
    model, tok, ref_ctx, opt, trainer = _policy()  # tras t_grpo_step, opt ya trae estado real de AdamW
    import shutil as sh

    import torch
    from model.policy import load_policy, trainable_params
    from training import checkpoint as ck
    from training.logprob import full_token_logps as token_logps

    model.eval()  # explícito: no asumir que un test previo dejó el modo correcto (dropout de LoRA en
    # train() haría que `before` fuera estocástico -- no comparable con `after`, cargado en eval()).
    ids = tok("def f(x):\n    return x + 1", return_tensors="pt").to(model.device)
    with torch.no_grad():
        before = token_logps(model, ids["input_ids"], ids["attention_mask"]).clone()
        before2 = token_logps(model, ids["input_ids"], ids["attention_mask"])  # repetible en sí mismo?
    assert torch.equal(before, before2), "el propio modelo no es determinista en eval(): revisa dropout/modo"

    # snapshot DIRECTO de pesos (no vía forward) antes de guardar: adapter completo + una muestra de base.
    lora_before = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
    assert lora_before, "no se encontró ningún parámetro lora_* en el state_dict -- ¿target_modules vacío?"
    base_keys = [next(k for k in model.state_dict() if k.endswith("embed_tokens.weight")),
                 next(k for k in model.state_dict() if k.endswith("q_proj.base_layer.weight"))]
    base_before = {k: model.state_dict()[k].detach().clone() for k in base_keys}

    tmp = "/tmp/ck_roundtrip_test"
    sh.rmtree(tmp, ignore_errors=True)
    path = ck.save(tmp, 0, model, opt, {"reward_mean": 0.5}, True, "testhash")
    assert all(os.path.isfile(os.path.join(path, f)) for f in
              ("meta.json", "optim.pt", "rng.pt", "adapter_model.safetensors", "adapter_config.json"))

    # dtype EN DISCO, sin pasar por peft/torch de nuevo (aísla si la pérdida de precisión ocurre al guardar)
    from safetensors import safe_open
    with safe_open(os.path.join(path, "adapter_model.safetensors"), framework="pt") as f:
        disk_dtypes = {str(f.get_tensor(k).dtype) for k in f.keys()}
    print(f"      dtype en disco: {disk_dtypes}  |  dtype en memoria (antes): "
          f"{ {str(v.dtype) for v in lora_before.values()} }")

    model2, tok2, ref_ctx2 = load_policy(CFG, model.device, resume_from=path)
    model2.eval()

    # 1) ¿EL ADAPTER RESTAURADO ES EXACTAMENTE EL MISMO? comparación directa, sin forward de por medio.
    lora_after = {k: v.detach().clone() for k, v in model2.state_dict().items() if "lora_" in k}
    only_before, only_after = lora_before.keys() - lora_after.keys(), lora_after.keys() - lora_before.keys()
    assert not only_before and not only_after, f"claves LoRA distintas -- sólo antes: {only_before}; sólo después: {only_after}"
    mismatches = []
    for k, a in lora_before.items():
        b = lora_after[k]
        if a.dtype != b.dtype or a.device.type != b.device.type:
            mismatches.append((k, "dtype/device", str(a.dtype), str(b.dtype), a.device.type, b.device.type))
        elif not torch.equal(a, b.to(a.dtype)):
            mismatches.append((k, "valor", (a - b.to(a.dtype)).abs().max().item(), a.abs().max().item(), b.abs().max().item()))
    print(f"      lora params comparados: {len(lora_before)}  mismatches: {len(mismatches)}")
    for m in mismatches[:6]:
        print(f"        {m}")

    # 2) ¿LOS PESOS BASE SON IDÉNTICOS? (descarta no-determinismo de AutoModelForCausalLM.from_pretrained)
    base_after = {k: model2.state_dict()[k].detach().clone() for k in base_keys}
    base_mismatches = [k for k in base_keys if not torch.equal(base_before[k], base_after[k])]
    print(f"      pesos base comparados: {base_keys}  -- distintos: {base_mismatches or 'ninguno'}")

    # 5) ¿EL ADAPTER ACTIVO ES EL CORRECTO? -- diagnóstico oficial de PEFT para exactamente este tipo de bug
    # (ver "Check layer and model status" en la guía de troubleshooting de PEFT: adapters activos/mergeados).
    try:
        st_before, st_after = model.get_layer_status()[0], model2.get_layer_status()[0]
        print(f"      layer_status[0] antes: active={st_before.active_adapters} merged={st_before.merged_adapters}")
        print(f"      layer_status[0] después: active={st_after.active_adapters} merged={st_after.merged_adapters}")
        ms_before, ms_after = model.get_model_status(), model2.get_model_status()
        print(f"      model_status antes: enabled={ms_before.enabled} active={ms_before.active_adapters} "
              f"trainable={ms_before.trainable_params}")
        print(f"      model_status después: enabled={ms_after.enabled} active={ms_after.active_adapters} "
              f"trainable={ms_after.trainable_params}")
    except Exception as e:
        print(f"      get_layer_status()/get_model_status() no disponible en esta versión de peft: {e!r}")

    assert not mismatches, f"pesos LoRA distintos tras guardar+recargar (ver detalle arriba): {mismatches[:3]}"
    assert not base_mismatches, f"pesos BASE distintos entre las dos cargas de load_base(): {base_mismatches}"

    # sólo si 1) y 2) confirman pesos idénticos tiene sentido seguir comparando logits: si esto AÚN falla
    # con pesos bit-idénticos, la causa está en cómo se aplica el adapter en el forward, no en el guardado.
    with torch.no_grad():
        after = token_logps(model2, ids["input_ids"], ids["attention_mask"])
    gap = (before - after).abs().max().item()
    assert gap < 1e-3, (f"logits distintos (gap={gap}) pese a que los pesos LoRA y base son bit-idénticos "
                        "-- revisar active_adapters/merged_adapters arriba, o ref_ctx/disable_adapter")

    # 3) ¿optimizer restaurado?
    opt2 = torch.optim.AdamW(trainable_params(model2), lr=CFG.grpo.lr)
    ck.load_optim_rng(path, opt2, model.device)
    k1, k2 = set(opt.state_dict()["state"]), set(opt2.state_dict()["state"])
    assert k1 == k2, "el estado del optimizer (m, v de AdamW) no recargó las mismas claves"
    print(f"      |Δlogp| tras roundtrip: {gap:.2e}  optimizer state keys recargadas: {len(k2)}")
    sh.rmtree(tmp, ignore_errors=True)
    del model2, ref_ctx2, opt2




# ---------- Generate -> Test -> Fix, auto-mejora, GRPO avanzado, escalado ----------
class _FakeTok:
    """Tokenizer mínimo para tests: plantilla de chat legible y un token por palabra."""
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}\n" for m in msgs) + "<assistant>"

    def __call__(self, text):
        return type("E", (), {"input_ids": text.split()})()


class _ScriptGen:
    """Generador falso: devuelve `fixed` si el prompt trae feedback ("should"/"Passes") y `first` si no."""
    def __init__(self, first, fixed, ctx=4096):
        self.tok, self.max_new, self.ctx, self.first, self.fixed, self.prompts = _FakeTok(), 64, ctx, first, fixed, []
        self.stats = dict(tokens=0, samples=0, seconds=0.0, oom=0)

    def stream(self, prompts, n):
        from rollouts.generate import Sample
        self.prompts.append(list(prompts))
        for i, p in enumerate(prompts):
            for _ in range(n):
                code = self.fixed if ("should" in p or "Passes" in p) else self.first
                yield i, Sample(code, code, 10, True, [1, 2], [3, 4])


@test("fix: feedback por tipo de error (sintaxis, sin función, import prohibido; resuelto -> None)")
def t_feedback_kinds():
    from rewards.feedback import feedback
    from rewards.reward import Result
    p = PROBLEMS["clamp"]
    assert "SyntaxError at line 1" in feedback(p, "def clamp(:\n", score(p, "def clamp(:\n", CFG))
    assert "`clamp`" in feedback(p, "def other():\n    pass\n", score(p, "def other():\n    pass\n", CFG))
    code = "import os\ndef clamp(x, lo, hi):\n    return x\n"
    fb = feedback(p, code, score(p, code, CFG), CFG.sandbox.allowed_imports)
    assert "import:os" in fb and "math" in fb, fb
    assert feedback(p, "", Result(1.0, True, True)) is None


@test("fix: el detalle sale sólo del primer caso VISIBLE que falla; un fallo sólo oculto no revela entradas")
def t_feedback_no_leak():
    import copy
    from rewards.feedback import feedback
    from sandbox.harness import enc
    p = PROBLEMS["clamp"]
    nvis, hid = len(p.visible()), p.hidden()

    def fake_run(bad):
        def run_cases(code, entry, args_list, limits, allowed, b):
            recs = []
            for i, args in enumerate(args_list):
                try:
                    rec = {"i": i, "out": enc(p.ref(*copy.deepcopy(args)))}
                except Exception as e:
                    rec = {"i": i, "exc": type(e).__name__, "mro": [c.__name__ for c in type(e).__mro__]}
                recs.append({"i": i, "out": "123456789"} if i == bad else rec)
            return executor.Run("ok", recs)
        return run_cases

    real = executor.run_cases, executor.backend
    executor.backend = lambda sb: "rlimit"
    try:
        executor.run_cases = fake_run(0)
        r = score(p, "def clamp(x, lo, hi):\n    return x\n", CFG)
        assert not r.solved and r.detail and r.detail["got"] == "returned 123456789", r.detail
        assert r.detail["call"] in feedback(p, "", r)
        executor.run_cases = fake_run(nvis)
        r = score(p, "def clamp(x, lo, hi):\n    return x\n", CFG)
        fb = feedback(p, "", r)
        call = f"clamp({', '.join(map(repr, hid[0].args))})"
        assert not r.solved and r.detail is None and "hidden" in fb and call not in fb, fb
    finally:
        executor.run_cases, executor.backend = real


@test("fix: bucle sin sandbox: sólo se regeneran las cadenas pendientes; prompt de corrección que no cabe -> original")
def t_fix_loop_logic():
    import rollouts.collect as rc
    from rewards.reward import Result
    from rollouts import fix as gtf
    probs = [PROBLEMS["clamp"], PROBLEMS["abs_sum"]]
    bad = Result(0.1, True, False, error="wrong_answer", n_pass=3, n_cases=5,
                 detail=dict(call="clamp(9, 0, 5)", want="return 5", got="returned 9"))
    real = rc.score
    rc.score = lambda p, code, cfg, seed=0: Result(1.0, True, True) if "FIXED" in code or p.id == "abs_sum" else bad
    try:
        gen = _ScriptGen("def clamp(x, lo, hi): return x", "def clamp(x, lo, hi): return x  # FIXED")
        chains = gtf.run(gen, probs, CFG, rounds=3)
        assert [len(c) for c in chains] == [2, 1], [len(c) for c in chains]
        assert len(gen.prompts) == 2 and len(gen.prompts[1]) == 1 and "clamp(9, 0, 5)" in gen.prompts[1][0]
        s = gtf.summary(chains)
        assert s["solved@0"] == 0.5 and s["solved@1"] == 1.0 and s["fix_rate"] == 1.0, s
        gen = _ScriptGen("def clamp(x, lo, hi): return x", "def clamp(x, lo, hi): return x  # FIXED", ctx=90)
        chains = gtf.run(gen, probs[:1], CFG, rounds=2)
        assert len(chains[0]) == 3 and gen.prompts[1] == gen.prompts[0], "no volvió al prompt original"
    finally:
        rc.score = real


@test("fix: generate -> test -> fix con el sandbox real (feedback con el caso visible, la corrección resuelve)")
def t_fix_loop_sandbox():
    from rollouts import fix as gtf
    p = PROBLEMS["clamp"]
    gen = _ScriptGen("def clamp(x, lo, hi):\n    return x\n", p.ref_source())
    chains = gtf.run(gen, [p], CFG, rounds=2, n=2)
    assert [len(c) for c in chains] == [2, 2], chains
    first, second = chains[0]
    assert not first.solved and "should return" in first.feedback and "clamp(" in first.feedback, first
    assert second.solved and second.feedback is None
    s = gtf.summary(chains)
    assert s["solved@0"] == 0 and s["solved@1"] == 1 and s["fix_rate"] == 1, s


@test("tareas libres: asserts literales -> casos visibles; se rechaza lo que no es literal o no llama a la función")
def t_custom_task():
    from problems.custom import Task
    t = Task("def slug(s: str) -> str", "Lowercase words joined by '-'.",
             "assert slug('A b') == 'a-b'\nassert slug('') == ''\n")
    assert t.entry == "slug" and len(t.visible()) == 2 and not t.hidden() and not t.generated(0, 5)
    assert "# slug('A b') -> 'a-b'" in t.prompt() and t.prompt().startswith("def slug(s: str) -> str:")
    for bad in ("assert slug(x) == 1", "assert other(1) == 1", "print(1)", "assert slug(1) != 2", "",
                "assert slug(s='a') == 1"):
        try:
            Task("def slug(s)", "", bad)
        except (ValueError, SyntaxError):
            continue
        raise AssertionError(f"aceptó: {bad!r}")


@test("tareas libres: se puntúan con el sandbox y el reward de siempre")
def t_custom_task_sandbox():
    from problems.custom import Task
    t = Task("def slug(s: str) -> str", "Lowercase words joined by '-'.",
             "assert slug('A b') == 'a-b'\nassert slug('') == ''\n")
    assert score(t, "def slug(s):\n    return '-'.join(s.lower().split())\n", CFG).solved
    r = score(t, "def slug(s):\n    return s\n", CFG)
    assert not r.solved and r.detail["call"] == "slug('A b')", r.detail


@test("improve: niveles se abren sólo con promedios estables; buffer de fallos deduplicado por AST y persistente")
def t_improve_curriculum():
    import tempfile
    from rewards.reward import Result
    from rollouts.generate import Sample
    from training import improve as im

    class P:
        def __init__(self, pid, level):
            self.id = self.entry = pid
            self.level = level

    probs = [P("a", 0), P("b", 0), P("c", 1), P("d", 2)]
    d = tempfile.mkdtemp()
    imp = im.Improver(d, probs, CFG)
    random.seed(1234)
    assert {p.level for p in imp.sample(probs, 10)} == {0}
    ok, ko = (None, Result(1.0, True, True)), (None, Result(0.0, True, False))
    imp.update(probs[:2], [[ok] * 8, [ok] * 8])
    assert imp.open == 1, "un único step bueno no debe abrir nivel"
    imp.update(probs[:2], [[ok] * 8, [ko] * 8])
    assert imp.open == 2 and imp.weight("a") == im.FLOOR and imp.weight("c") == 0.25
    imp._remember([dict(pid="a", code="def f():\n    return 1\n", feedback="x"),
                   dict(pid="a", code="def f():  # otro formato\n  return 1", feedback="y"),
                   dict(pid="b", code="def f():\n    return 1\n", feedback="z")])
    assert len(imp.buffer) == 2, imp.buffer

    real = im.collect, im.prompt_for
    im.prompt_for = lambda gen, cfg, p, code=None, fb=None: f"FIX {p.id}: {fb}"
    im.collect = lambda gen, ps, n, cfg, seed=0, prompts=None: [[(Sample("", "", 1, True, [1], [2]), Result(0.5))] * n
                                                             for _ in ps]
    try:
        bad = (Sample("", "def a(x):\n    return 0\n", 1, True, [1], [2]),
               Result(0.2, True, False, error="wrong_answer", n_pass=1, n_cases=4))
        groups = imp.repair(None, probs[:2], [[bad] * 8, [ok] * 8], seed=0)
        keys = [k for k, _, _ in groups]
        k = CFG.improve.repair_per_step
        assert len(set(keys)) <= k and len(groups) == len(set(keys)) * CFG.num_rollouts, keys
        assert sum(1 for x in set(keys) if x.startswith("a#")) >= 1  # el fallo fresco siempre entra
    finally:
        im.collect, im.prompt_for = real
    imp.save()
    imp2 = im.Improver(d, probs, CFG)
    assert (imp2.rate, imp2.obs, imp2.open, len(imp2.buffer)) == (imp.rate, imp.obs, imp.open, len(imp.buffer))


@test("escalado: 0.5B/1.5B LoRA caben, 3B pide gradient checkpointing, 7B o full-FT de 1.5B no caben (12 GB)")
def t_fit_decide():
    from model import fit
    G = int(12.227 * 2**30)
    d = {n: fit.decide(fit.need_bytes(n, True, True), G) for n in (494e6, 1.54e9, 3.09e9, 7.62e9)}
    assert list(d.values()) == ["ok", "ok", "checkpointing", "too_big"], d
    assert fit.decide(fit.need_bytes(1.54e9, True, False), G) == "too_big"
    assert fit.decide(fit.need_bytes(7.62e9, False, True), G) == "too_big"  # ni para inferencia sin cuantizar


# ---------- agente de código ----------
def _mini_repo(bug=True):
    import tempfile
    d = tempfile.mkdtemp(prefix="repo_")
    with open(os.path.join(d, "calc.py"), "w") as f:
        f.write("def add(a, b):\n    return a - b\n" if bug else "def add(a, b):\n    return a + b\n")
    with open(os.path.join(d, "test_calc.py"), "w") as f:
        f.write("import unittest\nfrom calc import add\n\n\nclass T(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n")
    os.makedirs(os.path.join(d, ".git"))
    os.makedirs(os.path.join(d, "__pycache__"))
    with open(os.path.join(d, "__pycache__", "x.py"), "w") as f:
        f.write("junk")
    return d


class _FakeEngine:
    """Motor falso del agente: arregla calc.py e intenta además escribir fuera del repo."""
    def __init__(self):
        self.prompts = []

    def chat(self, msgs):
        return "\n".join(m["content"] for m in msgs)

    def n_tokens(self, text):
        return len(text) // 4

    def complete(self, prompt, max_new, temperature=None):
        self.prompts.append(prompt)
        return ("I will fix it.\nFILE: calc.py\n```python\ndef add(a, b):\n    return a + b\n```\n"
                "FILE: ../evil.py\n```python\nprint('x')\n```\n")


@test("agente: formato FILE + bloque, rutas fuera del repo/.git/symlink rechazadas, diff unificado")
def t_agent_edits():
    from agent.edits import diff, parse, safe
    e = parse("bla\nFILE: a/b.py\n```python\nx = 1\n```\ntexto\nFILE: `c.txt`\n```\nhola\n```\n")
    assert e == {"a/b.py": "x = 1\n", "c.txt": "hola\n"}, e
    d = _mini_repo()
    os.symlink("/etc", os.path.join(d, "link"))
    for bad in ("../x.py", "/etc/passwd", ".git/config", "link/passwd", "", "."):
        try:
            safe(d, bad)
        except ValueError:
            continue
        raise AssertionError(f"aceptó {bad!r}")
    assert safe(d, "pkg/new.py") == os.path.join(os.path.realpath(d), "pkg", "new.py")
    w = _mini_repo(bug=False)
    out = diff(d, w, ["calc.py"])
    assert "-    return a - b" in out and "+    return a + b" in out and "a/calc.py" in out, out


@test("agente: lectura del repo (sin .git/__pycache__) y ranking: la ruta del traceback sale primero")
def t_agent_repo():
    from agent.repo import context, rank, scan
    d = _mini_repo()
    files = scan(d)
    assert sorted(files) == ["calc.py", "test_calc.py"], sorted(files)
    assert rank(files, 'File "calc.py", line 2, in add')[0] == "calc.py"
    ctx = context(files, "fix add in calc.py", budget=120)
    assert "Repository files:" in ctx and "FILE: calc.py" in ctx and "FILE: test_calc.py" not in ctx, ctx


@test("agente: sesión con motor y tests falsos (itera, rechaza ruta mala, acepta, detecta conflicto, rechaza)")
def t_agent_session():
    import tempfile
    import agent.loop as al
    from sandbox.command import CmdResult
    cfg = load_config()

    def fake_run(cmd, workdir, sb, timeout_s, mem_mb):
        with open(os.path.join(workdir, "calc.py")) as f:
            ok = "a + b" in f.read()
        return CmdResult(0, "Ran 1 test\nOK", 0.1) if ok else CmdResult(1, 'File "calc.py", line 2\nAssertionError: -1 != 3', 0.1)

    real = al.command.run
    al.command.run = fake_run
    try:
        repo, runs = _mini_repo(), tempfile.mkdtemp()
        eng = _FakeEngine()
        s = al.Session(repo, "make add() add", cfg, "python3 -m unittest -q", root=runs).run(eng, iters=3)
        assert not s.baseline["passed"] and s.status == "passed" and len(s.iters) == 1, (s.baseline, s.iters)
        it = s.iters[0]
        assert it["files"] == ["calc.py"] and it["rejected"] and "AssertionError" in eng.prompts[0], it
        assert "+    return a + b" in s.last_diff and not os.path.exists(os.path.join(os.path.dirname(repo), "evil.py"))
        s.accept()
        with open(os.path.join(repo, "calc.py")) as f:
            assert "a + b" in f.read()

        repo = _mini_repo()
        s = al.Session(repo, "make add() add", cfg, "python3 -m unittest -q", root=runs).run(_FakeEngine(), iters=1)
        with open(os.path.join(repo, "calc.py"), "a") as f:
            f.write("# editado a mano durante la sesión\n")
        try:
            s.accept()
            raise AssertionError("aceptó pese al conflicto")
        except RuntimeError:
            pass
        with open(os.path.join(repo, "calc.py")) as f:
            assert "a - b" in f.read()
        s.reject()
        import json
        with open(os.path.join(s.dir, "session.json")) as f:
            j = json.load(f)
        assert not os.path.exists(s.work) and j["status"] == "rejected" and "a + b" in j["diff"]
    finally:
        al.command.run = real


@test("agente: los tests del repo corren en el sandbox (pasa, falla, timeout)")
def t_agent_command():
    from sandbox.command import run
    d = _mini_repo(bug=False)
    r = run(["python3", "-m", "unittest", "-q"], d, CFG.sandbox, 60, 2048)
    assert r.ok and "OK" in r.out, r
    d = _mini_repo(bug=True)
    r = run(["python3", "-m", "unittest", "-q"], d, CFG.sandbox, 60, 2048)
    assert not r.ok and "AssertionError" in r.out, r
    r = run(["python3", "-c", "while True: pass"], d, CFG.sandbox, 2, 512)
    assert r.code is None and "timeout" in r.out, r


@test("agente: aislamiento de los tests del repo (no ve el proyecto ni el home, sin red, sólo escribe en /work)")
def t_agent_command_isolation():
    from sandbox.command import run
    if executor.backend(CFG.sandbox) != "bwrap":
        raise Skip("requiere bwrap")
    d = _mini_repo()
    probe = ("import os, socket\n"
             f"print('proyecto', os.path.exists({ROOT!r}))\n"
             f"print('home', os.path.exists({os.path.expanduser('~')!r}))\n"
             "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2); print('red True')\n"
             "except OSError:\n    print('red False')\n"
             "try:\n    open('/usr/x', 'w'); print('usr True')\nexcept OSError:\n    print('usr False')\n"
             "open('/work/ok.txt', 'w').write('1'); print('work True')\n")
    r = run(["python3", "-c", probe], d, CFG.sandbox, 30, 1024)
    out = r.out.split()
    assert r.ok and out == ["proyecto", "False", "home", "False", "red", "False", "usr", "False", "work", "True"], r.out
    assert os.path.exists(os.path.join(d, "ok.txt"))



@test("agente: el sandbox de los tests no monta nada en /home ni en tu HOME (el venv va en /venv)")
def t_agent_sandbox_no_home():
    import tempfile
    from sandbox import command
    home = os.path.realpath(os.path.expanduser("~"))
    try:
        venv = tempfile.mkdtemp(dir=home, prefix=".venv-test-")  # como /home/<usuario>/.venvs/project
    except OSError:
        venv = tempfile.mkdtemp()
    real_prefix = sys.prefix
    sys.prefix = venv
    try:
        args = command._argv("bwrap", tempfile.mkdtemp(), ["python3", "-V"], 512, 5)
    finally:
        sys.prefix = real_prefix
        os.rmdir(venv)
    mounts = [(args[i + 1], args[i + 2]) for i, a in enumerate(args) if a in ("--ro-bind", "--bind", "--ro-bind-try")]
    inside = [dst for _, dst in mounts] + args[args.index("PATH") + 1].split(":")
    bad = [p for p in inside if p == "/home" or p.startswith("/home/") or p == home or p.startswith(home + os.sep)]
    assert not bad, f"rutas del home dentro del sandbox: {bad}"
    assert (venv, "/venv") in mounts and args[args.index("PATH") + 1].startswith("/venv/bin:"), (mounts, args[args.index("PATH") + 1])

# ---------- panel web ----------
@test("panel: API local (Host y JSON obligatorios, ajustes, fix de una tarea propia, agente + aplicar, entrenar/parar)")
def t_app_api():
    import http.client, json, shutil, tempfile, threading
    import agent.loop as al
    import app as appmod
    import rollouts.collect as rc
    from rewards.reward import Result
    from sandbox.command import CmdResult

    class Eng(_FakeEngine):
        def generator(self):
            return _ScriptGen("def slug(s):\n    return s\n", "def slug(s):\n    return '-'.join(s.lower().split())\n")

        def close(self):
            pass

    class TestApp(appmod.App):
        def external_trainings(self):  # aislado de entrenamientos reales que estén corriendo en la máquina
            return []

        def train_cmd(self, mode, fresh, cfg_path):  # un "entrenamiento" que sólo espera SIGINT
            return [sys.executable, "-c", "import signal, sys, time\nsignal.signal(signal.SIGINT, lambda *a: sys.exit(3))\n"
                                          "print('entrenando', flush=True)\ntime.sleep(60)"]

    ui = tempfile.mkdtemp()
    a = TestApp(engine_factory=lambda cfg, ck: Eng(), ui_dir=ui)
    srv = appmod.make_server(a, "127.0.0.1", 0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(method, path, body=None, host="localhost", ctype="application/json"):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers, data = {"Host": f"{host}:{port}"}, None
        if body is not None:
            data, headers["Content-Type"] = json.dumps(body).encode(), ctype
        c.request(method, path, body=data, headers=headers)
        r = c.getresponse()
        raw = r.read()
        return r.status, json.loads(raw) if r.getheader("Content-Type", "").startswith("application/json") else raw

    def wait(job_id):
        for _ in range(200):
            j = call("GET", f"/api/jobs/{job_id}")[1]
            if j["status"] in ("done", "error"):
                return j
            time.sleep(0.05)
        raise AssertionError("el trabajo no terminó")

    real = rc.score, al.command.run
    rc.score = lambda p, code, cfg, seed=0: (Result(1.0, True, True) if "join" in code else Result(
        0.1, True, False, error="wrong_answer", n_pass=0, n_cases=1,
        detail=dict(call="slug('A b')", want="return 'a-b'", got="returned 'A b'")))

    def fake_cmd(cmd, workdir, sb, timeout_s, mem_mb):
        with open(os.path.join(workdir, "calc.py")) as f:
            return CmdResult(0 if "a + b" in f.read() else 1, "salida de los tests", 0.1)
    al.command.run = fake_cmd
    made = []
    try:
        st, page = call("GET", "/")
        assert st == 200 and b"Coding AI local" in page
        assert call("GET", "/api/status", host="evil.example")[0] == 403
        assert call("POST", "/api/fix", {"problem": "clamp"}, ctype="text/plain")[0] == 415
        st, s = call("GET", "/api/status")
        assert st == 200 and not s["training"]["running"] and s["engine"]["loaded"] is False
        ps = call("GET", "/api/problems")[1]
        assert any(p["id"] == "clamp" for p in ps) and {"val", "test"} <= {p["split"] for p in ps} and len(ps) >= 71
        st, c = call("POST", "/api/config", {"grpo.max_steps": "7"})
        assert st == 200 and c["values"]["grpo.max_steps"] == 7 and a.cfg().grpo.max_steps == 7
        assert call("POST", "/api/config", {"model": "otro"})[0] == 400
        assert call("POST", "/api/fix", {"sig": "def slug(s)", "tests": "print(1)"})[0] == 400

        st, job = call("POST", "/api/fix", {"sig": "def slug(s: str) -> str", "doc": "Slug.",
                                            "tests": "assert slug('A b') == 'a-b'", "iters": 2})
        j = wait(job["id"])
        assert j["status"] == "done", j["error"]
        made.append(j["result"]["run"])
        assert [e["solved"] for e in j["events"]] == [False, True], j
        assert j["result"]["summary"]["fix_rate"] == 1.0 and os.path.exists(os.path.join(j["result"]["run"], "attempts.jsonl"))
        st, job = call("POST", "/api/fix", {"problem": "median", "iters": 1})  # problema del catálogo (split val)
        j = wait(job["id"])
        assert st == 200 and j["status"] == "done", (st, j.get("error"))
        made.append(j["result"]["run"])
        with open(os.path.join(j["result"]["run"], "attempts.jsonl")) as f:
            assert {json.loads(x)["problem"] for x in f} == {"median"}

        repo = _mini_repo()
        st, job = call("POST", "/api/agent", {"repo": repo, "task": "make add() add", "test_cmd": "python3 -m unittest", "iters": 2})
        j = wait(job["id"])
        assert j["status"] == "done", j["error"]
        made.append(j["result"]["dir"])
        assert j["result"]["status"] == "passed" and "+    return a + b" in j["result"]["diff"], j
        st, j = call("POST", f"/api/jobs/{job['id']}/accept", {})
        with open(os.path.join(repo, "calc.py")) as f:
            assert st == 200 and j["result"]["status"] == "accepted" and "a + b" in f.read()
        assert call("POST", f"/api/jobs/{job['id']}/reject", {})[0] == 409

        st, t = call("POST", "/api/train/start", {"mode": "phase2"})
        assert st == 200 and t["running"] and a.engine is None, "el modelo del panel debe liberar la GPU"
        assert call("POST", "/api/fix", {"problem": "clamp"})[0] == 409, "trabajo de GPU aceptado durante el entrenamiento"
        for _ in range(100):
            if "entrenando" in call("GET", "/api/status")[1]["training"]["log"]:
                break
            time.sleep(0.05)
        assert call("POST", "/api/train/stop", {})[0] == 200
        a.proc.wait(10)
        s = call("GET", "/api/status")[1]["training"]
        assert not s["running"] and s["exit_code"] == 3 and "entrenando" in s["log"], s
    finally:
        rc.score, al.command.run = real
        srv.shutdown()
        for d in made:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(ui, ignore_errors=True)
        for d in [os.path.dirname(x) for x in made] + [os.path.join(ROOT, "runs")]:  # sin carpetas vacías en el proyecto
            try:
                os.rmdir(d)
            except OSError:
                pass


# ---------- auditoría: regresiones de bugs encontrados ----------
def _dummy_train(seconds=30):
    """Proceso vivo cuyo argv termina en train.py, como un entrenamiento lanzado desde la terminal."""
    import subprocess, tempfile
    path = os.path.join(tempfile.mkdtemp(), "train.py")
    with open(path, "w") as f:
        f.write(f"import time\ntime.sleep({seconds})\n")
    p = subprocess.Popen([sys.executable, path, "phase2"])
    for _ in range(100):  # hasta que /proc muestre el argv real (justo tras el fork aún es el del padre)
        try:
            with open(f"/proc/{p.pid}/cmdline", "rb") as f:
                if any(a.endswith(b"train.py") for a in f.read().split(b"\0")):
                    break
        except OSError:
            pass
        time.sleep(0.02)
    return p


class _TextEngine(_FakeEngine):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def complete(self, prompt, max_new, temperature=None):
        self.prompts.append(prompt)
        return self.text


def _patch_cmd(code):
    import agent.loop as al
    from sandbox.command import CmdResult
    real = al.command.run
    al.command.run = lambda *a, **k: CmdResult(code, "salida", 0.1)
    return lambda: setattr(al.command, "run", real)


@test("auditoría: una tarea propia con el nombre de un problema del catálogo no se confunde con él")
def t_audit_task_id():
    from problems.custom import Task
    t = Task("def clamp(x, lo, hi)", "otra cosa", "assert clamp(1, 2, 3) == 9")
    assert t.entry == "clamp" and t.id not in PROBLEMS, t.id
    assert "# clamp(1, 2, 3) -> 9" in t.prompt(), t.prompt()


@test("auditoría: improve tolera improve.json corrupto y líneas inválidas, y reanuda niveles por valor")
def t_audit_improve_state():
    import tempfile
    from training import improve as im

    class P:
        def __init__(self, pid, level):
            self.id = self.entry = pid
            self.level = level

    probs, fix_root = [P("a", 0), P("b", 1), P("c", 2), P("d", 3)], tempfile.mkdtemp()
    os.makedirs(os.path.join(fix_root, "r1"))
    with open(os.path.join(fix_root, "r1", "attempts.jsonl"), "w") as f:
        f.write('1\n"x"\n{"problem": "a", "solved": false, "code": "x = 1", "feedback": "f"}\n{roto\n')
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "improve.json"), "w") as f:
        f.write("{")
    imp = im.Improver(d, probs, CFG, fix_root=fix_root)
    assert imp.open == 1 and [b["pid"] for b in imp.buffer] == ["a"], imp.buffer
    imp.open = 2  # abiertos L0 y L1
    imp.save()
    imp2 = im.Improver(d, [p for p in probs if p.level >= 2], CFG, fix_root=fix_root)
    assert imp2.open == 1, f"con el pool L2-L3 sólo debe quedar abierto L2 (open={imp2.open})"
    import json
    with open(os.path.join(d, "improve.json"), "w") as f:  # JSON válido pero con contenido inservible
        json.dump(dict(rate={"a": "alto"}, obs={"a": 1}, open_level=0, buffer=[{"pid": "a"}]), f)
    imp3 = im.Improver(d, probs, CFG, fix_root=fix_root)
    imp3.update(probs[:1], [[(None, type("R", (), {"solved": True})())] * 2])
    assert all("key" in b for b in imp3.buffer)


@test("auditoría: el agente edita un fichero binario sin reventar al calcular el diff")
def t_audit_agent_binary():
    import tempfile
    import agent.loop as al
    repo = _mini_repo()
    with open(os.path.join(repo, "data.bin"), "wb") as f:
        f.write(bytes(range(256)))
    undo = _patch_cmd(1)
    try:
        s = al.Session(repo, "t", load_config(), "true", root=tempfile.mkdtemp())
        s.run(_TextEngine("FILE: data.bin\n```\nhola\n```\n"), iters=1)
        assert s.iters[0]["files"] == ["data.bin"] and "+hola" in s.last_diff, s.iters
    finally:
        undo()


@test("auditoría: aceptar no deja el repo a medias si un destino deja de ser válido durante la sesión")
def t_audit_agent_accept_atomic():
    import shutil, tempfile
    import agent.loop as al
    repo, outside = _mini_repo(), tempfile.mkdtemp()
    os.makedirs(os.path.join(repo, "sub"))
    with open(os.path.join(repo, "sub", "keep.py"), "w") as f:
        f.write("x = 1\n")
    text = ("FILE: calc.py\n```python\ndef add(a, b):\n    return a + b\n```\n"
            "FILE: sub/new.py\n```python\ny = 2\n```\n")
    undo = _patch_cmd(0)
    try:
        s = al.Session(repo, "t", load_config(), "true", root=tempfile.mkdtemp()).run(_TextEngine(text), iters=1)
        shutil.rmtree(os.path.join(repo, "sub"))
        os.symlink(outside, os.path.join(repo, "sub"))  # el repo cambia: sub/ pasa a apuntar fuera
        try:
            s.accept()
            raise AssertionError("aceptó escribiendo fuera del repo")
        except ValueError:
            pass
        with open(os.path.join(repo, "calc.py")) as f:
            assert "a - b" in f.read(), "calc.py se aplicó aunque accept falló"
        assert not os.listdir(outside)
    finally:
        undo()


@test("auditoría: la copia de trabajo conserva carpetas legítimas (build/env/dist/runs) y no se copia a sí misma")
def t_audit_agent_copy():
    import tempfile
    import agent.loop as al
    repo = _mini_repo()
    for d in ("build", "env", "dist", "runs"):
        os.makedirs(os.path.join(repo, d))
        with open(os.path.join(repo, d, "m.py"), "w") as f:
            f.write("v = 1\n")
    s = al.Session(repo, "t", load_config(), "true", root=tempfile.mkdtemp())
    missing = [d for d in ("build", "env", "dist", "runs") if not os.path.exists(os.path.join(s.work, d, "m.py"))]
    assert not missing, f"no se copiaron: {missing}"
    s2 = al.Session(repo, "t", load_config(), "true", root=os.path.join(repo, "runs", "agent"))
    assert os.path.exists(os.path.join(s2.work, "calc.py")) and not os.path.exists(os.path.join(s2.work, "runs", "agent"))


@test("auditoría: si el modelo falla a mitad de sesión, queda session.json con estado error")
def t_audit_agent_error_state():
    import json, tempfile
    import agent.loop as al

    class Boom(_FakeEngine):
        def complete(self, prompt, max_new, temperature=None):
            raise RuntimeError("CUDA out of memory (simulado)")

    undo = _patch_cmd(1)
    try:
        s = al.Session(_mini_repo(), "t", load_config(), "true", root=tempfile.mkdtemp())
        try:
            s.run(Boom(), iters=1)
            raise AssertionError("se tragó el error")
        except RuntimeError:
            pass
        with open(os.path.join(s.dir, "session.json")) as f:
            assert json.load(f)["status"] == "error"
    finally:
        undo()


@test("auditoría: un fichero con bloques de código dentro (README) no se trunca al parsear la edición")
def t_audit_agent_fences():
    from agent.edits import parse
    text = ("Cambios:\nFILE: README.md\n```markdown\n# T\n```python\nx = 1\n```\nfin\n```\n"
            "FILE: a.py\n```python\ny = 2\n```\nlisto")
    e = parse(text)
    assert e == {"README.md": "# T\n```python\nx = 1\n```\nfin\n", "a.py": "y = 2\n"}, e


@test("auditoría: el panel rechaza JSON que no es objeto, tipos erróneos, NaN, rondas desmesuradas y rutas fuera de runs/")
def t_audit_app_validation():
    import http.client, json, shutil, tempfile, threading
    import app as appmod

    class A(appmod.App):
        def external_trainings(self):
            return []

    ui = tempfile.mkdtemp()
    a = A(engine_factory=lambda cfg, ck: None, ui_dir=ui)
    srv = appmod.make_server(a, "127.0.0.1", 0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(method, path, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request(method, path, body=raw, headers={"Host": f"localhost:{port}", "Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.read()

    nan_run = os.path.join(ROOT, "runs", "phase2", "zz-audit-nan")
    try:
        bad = [("/api/fix", b"[]"), ("/api/config", b'{"grpo.max_steps": [1]}'), ("/api/config", b'{"grpo.lr": "nan"}'),
               ("/api/fix", b'{"problem": "clamp", "iters": 1000000000}'),
               ("/api/fix", b'{"problem": "clamp", "checkpoint": "/etc"}'),
               ("/api/agent", json.dumps({"repo": ROOT, "task": 5}).encode())]
        for path, raw in bad:
            st, body = call("POST", path, raw)
            assert st == 400 and json.loads(body)["error"], (path, raw, st, body)
        assert a.jobs == {}, "se encoló un trabajo inválido"
        os.makedirs(os.path.join(nan_run, "checkpoints", "ckpt_00000001"))
        with open(os.path.join(nan_run, "checkpoints", "ckpt_00000001", "meta.json"), "w") as f:
            f.write('{"step": 1, "metrics": {"loss": NaN, "grad_norm": Infinity}}')
        st, body = call("GET", "/api/runs")

        def strict(c):
            raise ValueError(f"JSON no estándar para el navegador: {c}")
        assert st == 200 and json.loads(body, parse_constant=strict) is not None
    finally:
        srv.shutdown()
        shutil.rmtree(nan_run, ignore_errors=True)
        shutil.rmtree(ui, ignore_errors=True)
        for d in (os.path.dirname(nan_run), os.path.join(ROOT, "runs")):  # sin dejar carpetas vacías en el proyecto
            try:
                os.rmdir(d)
            except OSError:
                pass


@test("auditoría: métricas accesibles aunque runs/ sea un enlace simbólico; rutas fuera siguen rechazadas")
def t_audit_metrics_symlink():
    import tempfile
    import app as appmod
    real_root = appmod.ROOT
    base, target = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(os.path.join(target, "phase2", "r1"))
    with open(os.path.join(target, "phase2", "r1", "metrics.jsonl"), "w") as f:
        f.write('{"step": 0, "loss": 0.1}\n')
    os.symlink(target, os.path.join(base, "runs"))
    appmod.ROOT = base
    try:
        assert appmod.App.metrics(os.path.join(base, "runs", "phase2", "r1"))["metrics"] == [{"step": 0, "loss": 0.1}]
        try:
            appmod.App.metrics(os.path.join(base, "otro"))
            raise AssertionError("aceptó una ruta fuera de runs/")
        except ValueError:
            pass
    finally:
        appmod.ROOT = real_root


@test("auditoría: un entrenamiento lanzado desde la terminal bloquea trabajos de GPU y otro entrenamiento en el panel")
def t_audit_app_external_training():
    import tempfile
    import app as appmod
    p = _dummy_train()
    try:
        a = appmod.App(engine_factory=lambda cfg, ck: None, ui_dir=tempfile.mkdtemp())
        for call in (lambda: a.submit("fix", {"problem": "clamp"}), lambda: a.start_training("phase2")):
            try:
                call()
                raise AssertionError("aceptado con otro entrenamiento en marcha")
            except appmod.Busy:
                pass
        t = a.training()
        assert t["running"] and p.pid in t["pids"], t
    finally:
        p.kill()
        p.wait()


@test("auditoría: dos entrenamientos no pueden escribir en el mismo run (lock con PID comprobado en /proc)")
def t_audit_run_lock():
    import tempfile
    from training import checkpoint as ck
    run, p = tempfile.mkdtemp(), _dummy_train()
    try:
        with open(os.path.join(run, "train.pid"), "w") as f:
            f.write(str(p.pid))
        try:
            ck.claim_run(run)
            raise AssertionError("dos procesos en el mismo run")
        except RuntimeError:
            pass
    finally:
        p.kill()
        p.wait()
    path = ck.claim_run(run)  # el dueño anterior murió: el lock caducado se reclama
    with open(path) as f:
        assert f.read() == str(os.getpid())
    ck.release_run(path)
    assert not os.path.exists(path)


@test("auditoría: al reanudar se quitan de metrics.jsonl los steps perdidos tras el último checkpoint")
def t_audit_metrics_resume():
    import json, tempfile
    from training import loop as train
    run = tempfile.mkdtemp()
    with open(os.path.join(run, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"event": "start", "step0": 0}) + "\n")
        for s in range(26):
            f.write(json.dumps({"step": s, "loss": 0.1}) + "\n")
        f.write(json.dumps({"step": 19, "eval_reward": 0.5}) + "\n")
        f.write('{"step": 25, "lo')  # última línea a medio escribir (crash)
    train._trim_metrics(run, 20)
    with open(os.path.join(run, "metrics.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    assert rows[0]["event"] == "start" and max(r.get("step", -1) for r in rows) == 19 and len(rows) == 22, rows[-3:]


@test("auditoría: 0.5B en full-FT no activa gradient checkpointing; 3B con LoRA sí")
def t_audit_fit_small_full_ft():
    from model import fit
    G = int(12.227 * 2**30)
    assert fit.decide(fit.need_bytes(494e6, True, False), G) == "ok"
    assert fit.decide(fit.need_bytes(3.09e9, True, True), G) == "checkpointing"


@test("auditoría: editar train.py con un editor no cuenta como entrenamiento (sólo un intérprete Python que lo ejecuta)")
def t_audit_is_training_false_positive():
    import subprocess
    from common import is_training
    p = subprocess.Popen(["vim", "-c", "import time; time.sleep(30)", "train.py"], executable=sys.executable)
    try:
        for _ in range(100):  # hasta que /proc muestre el argv real
            with open(f"/proc/{p.pid}/cmdline", "rb") as f:
                if f.read().startswith(b"vim"):
                    break
            time.sleep(0.02)
        assert not is_training(p.pid), "un 'vim train.py' bloquearía la GPU en el panel"
    finally:
        p.kill()
        p.wait()
    q = _dummy_train()
    try:
        assert is_training(q.pid)
    finally:
        q.kill()
        q.wait()


@test("auditoría: con el venv bajo un enlace simbólico, el PATH del sandbox apunta a la ruta montada")
def t_audit_command_path_realpath():
    import tempfile
    from sandbox import command
    real_prefix = sys.prefix
    link = os.path.join(tempfile.mkdtemp(), "venv-link")
    target = tempfile.mkdtemp()
    os.symlink(target, link)
    sys.prefix = link
    try:
        args = command._argv("bwrap", tempfile.mkdtemp(), ["python", "-V"], 512, 5)
    finally:
        sys.prefix = real_prefix
    path = args[args.index("PATH") + 1]
    binds = [(args[i + 1], args[i + 2]) for i, a in enumerate(args) if a == "--ro-bind"]
    assert path.split(":")[0] == "/venv/bin" and (os.path.realpath(target), "/venv") in binds, (path, binds)


@test("auditoría: campos de texto nulos o en blanco toman su valor por defecto (doc: null, comando de tests vacío)")
def t_audit_null_fields():
    import tempfile
    import agent.loop as al
    import app as appmod

    class A(appmod.App):
        def external_trainings(self):
            return []

    a = A(engine_factory=lambda cfg, ck: None, ui_dir=tempfile.mkdtemp())
    a.q = type("Q", (), {"put": lambda self, job: None})()  # no ejecutar el trabajo: sólo validar y normalizar
    job = a.submit("fix", {"sig": "def f(x)", "doc": None, "tests": "assert f(1) == 1"})
    assert job["params"]["doc"] == "", job["params"]
    s = al.Session(_mini_repo(), "t", load_config(), "   ", root=tempfile.mkdtemp())
    assert s.test_cmd == load_config().agent.test_cmd, repr(s.test_cmd)


@test("auditoría: una edición sobre un directorio se rechaza sin tumbar la sesión del agente")
def t_audit_agent_edit_dir():
    import tempfile
    import agent.loop as al
    repo = _mini_repo()
    os.makedirs(os.path.join(repo, "pkg"))
    undo = _patch_cmd(1)
    try:
        s = al.Session(repo, "t", load_config(), "true", root=tempfile.mkdtemp())
        s.run(_TextEngine("FILE: pkg\n```\nx = 1\n```\n"), iters=1)
        assert s.iters[0]["files"] == [] and s.iters[0]["rejected"] and s.status == "failed", s.iters
    finally:
        undo()


@test("auditoría: una tarea propia con argumentos que no viajan como JSON (set, bytes) se rechaza al crearla")
def t_audit_task_json_args():
    from problems.custom import Task
    for bad in ("assert f({1, 2}) == 3", "assert f(b'x') == 1"):
        try:
            Task("def f(x)", "", bad)
        except ValueError:
            continue
        raise AssertionError(f"aceptó {bad!r}: fallaría luego dentro del sandbox")


@test("auditoría: un cambio en el repo antes de la primera edición del agente también cuenta como conflicto")
def t_audit_agent_early_conflict():
    import tempfile
    import agent.loop as al
    repo = _mini_repo()
    undo = _patch_cmd(0)
    try:
        s = al.Session(repo, "t", load_config(), "true", root=tempfile.mkdtemp())
        with open(os.path.join(repo, "calc.py"), "a") as f:
            f.write("# cambio del usuario tras copiar, antes de que el agente edite\n")
        s.run(_TextEngine("FILE: calc.py\n```python\ndef add(a, b):\n    return a + b\n```\n"), iters=1)
        try:
            s.accept()
            raise AssertionError("aceptar habría borrado el cambio del usuario")
        except RuntimeError:
            pass
        with open(os.path.join(repo, "calc.py")) as f:
            assert "cambio del usuario" in f.read()
    finally:
        undo()


@test("auditoría: si el modelo no carga, un trabajo de fix no deja un run vacío en runs/fix")
def t_audit_fix_job_no_empty_run():
    import glob, shutil, tempfile
    import app as appmod

    class A(appmod.App):
        def external_trainings(self):
            return []

    def boom(cfg, ck):
        raise RuntimeError("el modelo no carga (simulado)")

    a = A(engine_factory=boom, ui_dir=tempfile.mkdtemp())
    job = a.submit("fix", {"problem": "clamp"})
    for _ in range(500):
        if a.jobs[job["id"]]["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    left = glob.glob(os.path.join(ROOT, "runs", "fix", f"ui-*-{job['id']}"))
    try:
        assert a.jobs[job["id"]]["status"] == "error" and not left, (a.jobs[job["id"]]["status"], left)
    finally:
        for d in left:
            shutil.rmtree(d, ignore_errors=True)
        for d in (os.path.join(ROOT, "runs", "fix"), os.path.join(ROOT, "runs")):
            try:
                os.rmdir(d)
            except OSError:
                pass


@test("auditoría: si el modelo no carga, un trabajo del agente no deja una copia del repo huérfana")
def t_audit_agent_job_no_copy():
    import shutil, tempfile
    import app as appmod

    class A(appmod.App):
        def external_trainings(self):
            return []

    def boom(cfg, ck):
        raise RuntimeError("el modelo no carga (simulado)")

    root = os.path.join(ROOT, "runs", "agent")
    before = set(os.listdir(root)) if os.path.isdir(root) else set()
    a = A(engine_factory=boom, ui_dir=tempfile.mkdtemp())
    job = a.submit("agent", {"repo": _mini_repo(), "task": "arregla add"})
    for _ in range(500):
        if a.jobs[job["id"]]["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    new = (set(os.listdir(root)) if os.path.isdir(root) else set()) - before
    try:
        assert a.jobs[job["id"]]["status"] == "error" and not new, (a.jobs[job["id"]]["status"], new)
    finally:
        for d in new:
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        for d in (root, os.path.join(ROOT, "runs")):
            try:
                os.rmdir(d)
            except OSError:
                pass


def _app(**kw):
    import tempfile
    import app as appmod

    class A(appmod.App):
        def external_trainings(self):
            return kw.get("external", [])

    return A(engine_factory=lambda cfg, ck: None, ui_dir=kw.get("ui") or tempfile.mkdtemp())


@test("auditoría 2: los ajustes del panel rechazan valores que tumbarían el entrenamiento (0, top_p>1, 1 rollout, 2.5)")
def t_audit2_config_ranges():
    a = _app()
    for bad in ({"checkpoint.every_steps": 0}, {"generation.top_p": 1.5}, {"num_rollouts": 1}, {"num_rollouts": 2.5},
                {"num_rollouts": True}, {"generation.temperature": 0}):
        try:
            a.set_config(bad)
            raise AssertionError(f"aceptó {bad}")
        except ValueError:
            pass
    assert a.set_config({"num_rollouts": 6, "grpo.max_steps": "7"})["values"]["num_rollouts"] == 6


@test("auditoría 2: una clave desconocida o un valor inválido en overrides.json no inutiliza el panel")
def t_audit2_overrides_robust():
    import json, tempfile
    ui = tempfile.mkdtemp()
    with open(os.path.join(ui, "overrides.json"), "w") as f:
        json.dump({"nope.x": 1, "checkpoint.every_steps": 0, "grpo.lr": 0.001}, f)
    v = _app(ui=ui).config_view()["values"]
    assert v["grpo.lr"] == 0.001 and v["checkpoint.every_steps"] == load_config().checkpoint.every_steps, v


@test("auditoría 2: al submuestrear métricas largas se conservan todas las evaluaciones y la última fila")
def t_audit2_metrics_downsample():
    import json, shutil
    import app as appmod
    run = os.path.join(ROOT, "runs", "phase2", "zz-audit-long")
    os.makedirs(run, exist_ok=True)
    try:
        with open(os.path.join(run, "metrics.jsonl"), "w") as f:
            f.write("null\n")
            for s in range(4500):
                f.write(json.dumps({"step": s, "loss": 0.1}) + "\n")
                if s % 20 == 19:
                    f.write(json.dumps({"step": s, "eval_reward": 0.5}) + "\n")
        m = appmod.App.metrics(run)["metrics"]
        evals = [r for r in m if "eval_reward" in r]
        assert all(isinstance(r, dict) for r in m) and len(evals) == 225 and m[-1]["step"] == 4499, (len(evals), m[-1])
        assert len(m) < 3000
    finally:
        shutil.rmtree(run, ignore_errors=True)
        for d in (os.path.dirname(run), os.path.join(ROOT, "runs")):
            try:
                os.rmdir(d)
            except OSError:
                pass


@test("auditoría 2: detener no falla si el entrenamiento terminó entre la consulta y la señal")
def t_audit2_stop_race():
    import subprocess
    p = subprocess.Popen(["true"])
    p.wait()  # PID ya recogido: os.kill dará ProcessLookupError
    assert _app(external=[p.pid]).stop_training()["stopping"]


@test("auditoría 2: con un entrenamiento lanzado fuera del panel no se muestra el log viejo del panel")
def t_audit2_external_log():
    import json, subprocess, tempfile
    ui = tempfile.mkdtemp()
    log = os.path.join(ui, "train-viejo.log")
    with open(log, "w") as f:
        f.write("salida de un entrenamiento anterior del panel\n")
    p = subprocess.Popen(["true"])
    p.wait()
    with open(os.path.join(ui, "train.json"), "w") as f:
        json.dump(dict(pid=p.pid, mode="phase2", log=log), f)
    q = _dummy_train()
    try:
        t = _app(ui=ui, external=[q.pid]).training()
        assert t["running"] and t["pids"] == [q.pid] and t["log"] == "", t
    finally:
        q.kill()
        q.wait()


@test("auditoría 2: el agente conserva CRLF y BOM del fichero original (el diff sólo muestra cambios reales)")
def t_audit2_agent_crlf_bom():
    import tempfile
    import agent.loop as al
    repo = _mini_repo()
    path = os.path.join(repo, "calc.py")
    with open(path, encoding="utf-8") as f:
        orig = f.read()
    with open(path, "wb") as f:
        f.write(b"\xef\xbb\xbf" + orig.replace("\n", "\r\n").encode())
    new = orig.replace("a - b", "a + b")
    undo = _patch_cmd(0)
    try:
        s = al.Session(repo, "t", load_config(), "true", root=tempfile.mkdtemp())
        s.run(_TextEngine(f"FILE: calc.py\n```python\n{new}```\n"), iters=1)
        with open(os.path.join(s.work, "calc.py"), "rb") as f:
            raw = f.read()
        assert raw.startswith(b"\xef\xbb\xbf") and raw.count(b"\n") == raw.count(b"\r\n") > 0, raw[:60]
        changed = [l for l in s.last_diff.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        assert changed == ["-    return a - b", "+    return a + b"], changed
    finally:
        undo()


@test("auditoría 2: improve.json con observaciones sin tasa no tumba el entrenamiento a mitad")
def t_audit2_improve_obs_without_rate():
    import json, tempfile
    from training import improve as im

    class P:
        def __init__(self, pid, level):
            self.id = self.entry = pid
            self.level = level

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "improve.json"), "w") as f:
        json.dump(dict(rate={}, obs={"a": 3}, open_level=0, buffer=[]), f)
    imp = im.Improver(d, [P("a", 0), P("b", 1)], CFG, fix_root=tempfile.mkdtemp())
    imp.update([P("a", 0)], [[(None, type("R", (), {"solved": True})())] * 2])
    assert imp.rate["a"] == 1.0


@test("auditoría 2: `train.py fix` no deja un run vacío en runs/fix si el modelo no carga")
def t_audit2_fix_cli_no_empty_run():
    import argparse, glob, shutil, types
    import train
    root = os.path.join(ROOT, "runs", "fix")
    before = set(glob.glob(os.path.join(root, "*")))

    class Boom:
        def __init__(self, cfg, checkpoint=None):
            raise RuntimeError("el modelo no carga (simulado)")

    saved, backend = sys.modules.get("model.infer"), train.executor.backend
    sys.modules["model.infer"] = types.SimpleNamespace(Engine=Boom)
    train.executor.backend = lambda sb: "rlimit"  # sin bwrap aquí: que falle la carga del modelo, no el sandbox
    try:
        try:
            train.fix(load_config(), argparse.Namespace(levels=[0], problems=None, iters=1, chains=1, checkpoint=None))
            raise AssertionError("no propagó el fallo de carga")
        except RuntimeError as e:
            assert "no carga" in str(e), e
        new = set(glob.glob(os.path.join(root, "*"))) - before
        assert not new, f"run vacío: {new}"
    finally:
        train.executor.backend = backend
        if saved is None:
            sys.modules.pop("model.infer", None)
        else:
            sys.modules["model.infer"] = saved
        for d in set(glob.glob(os.path.join(root, "*"))) - before:
            shutil.rmtree(d, ignore_errors=True)
        for d in (root, os.path.join(ROOT, "runs")):
            try:
                os.rmdir(d)
            except OSError:
                pass


@test("auditoría 2: config con `extends` en ciclo o fichero vacío da un error claro (no RecursionError/AttributeError)")
def t_audit2_config_errors():
    import tempfile
    from common import read_config
    d = tempfile.mkdtemp()
    for name, text in (("a.yaml", "extends: b.yaml\nseed: 1\n"), ("b.yaml", "extends: a.yaml\nseed: 2\n"), ("vacio.yaml", "")):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)
    for name in ("a.yaml", "vacio.yaml"):
        try:
            read_config(os.path.join(d, name))
            raise AssertionError(f"{name}: sin error")
        except ValueError as e:
            assert name.split(".")[0][:1] in str(e) or "vac" in str(e) or "ciclo" in str(e), e


@test("auditoría 2: tests de una tarea propia con sintaxis inválida o anidamiento extremo -> ValueError (400, no 500)")
def t_audit2_task_syntax():
    from problems.custom import Task
    for bad in ("assert f(1) == ", "assert f(" + "[" * 5000 + "]" * 5000 + ") == 1"):
        try:
            Task("def f(x)", "", bad)
            raise AssertionError("aceptado")
        except ValueError:
            pass


# ======================================================================================================================
# Parche v2: dataset, evaluación, feedback, caché, Fix v2, improve v2, GRPO flags, agente, panel, compatibilidad
# ======================================================================================================================
GOOD_CLAMP = "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n"
W0, W1 = "def clamp(x, lo, hi):\n    return x\n", "def clamp(x, lo, hi):\n    return lo\n"


class _FnGen:
    """Generador falso con override de temperatura: fn(prompt, nº de veces visto ese prompt, temperatura) -> código."""
    def __init__(self, fn, ctx=4096):
        from collections import Counter
        self.tok, self.max_new, self.ctx, self.fn, self.calls, self.seen = _FakeTok(), 64, ctx, fn, [], Counter()
        self.stats = dict(tokens=0, samples=0, seconds=0.0, oom=0)

    def stream(self, prompts, n, temperature=None):
        from rollouts.generate import Sample
        self.calls.append((len(prompts), n, temperature))
        for i, p in enumerate(prompts):
            for _ in range(n):
                code = self.fn(p, self.seen[p], temperature)
                self.seen[p] += 1
                self.stats["tokens"] += 10
                self.stats["samples"] += 1
                yield i, Sample(code, code, 10, True, [1, 2], [3, 4])


def _fake_score(good, calls=None):
    from rewards.reward import Result
    good = {g.strip() for g in good}

    def f(p, code, cfg, seed=0):
        if calls is not None:
            calls.append(code)
        ok = code.strip() in good
        comps = dict(compile=0.05, visible=0.1, hidden=0.25 if ok else 0.1, generated=0.2 if ok else 0.05, solved=0.4 * ok)
        return Result(round(sum(comps.values()), 6), True, ok, 1.0, 1.0 if ok else 0.5, 1.0 if ok else 0.5,
                      1.0 if ok else 0.6, 10 if ok else 6, 10, None if ok else "wrong_answer", components=comps)
    return f


class _patched:
    """with _patched(modulo, nombre=valor, ...): sustituye atributos y los restaura; limpia la caché de score."""
    def __init__(self, mod, **kw):
        self.mod, self.kw, self.old = mod, kw, {}

    def __enter__(self):
        from rollouts.collect import CACHE
        CACHE.invalidate()
        for k, v in self.kw.items():
            self.old[k] = getattr(self.mod, k)
            setattr(self.mod, k, v)
        return self

    def __exit__(self, *a):
        from rollouts.collect import CACHE
        for k, v in self.old.items():
            setattr(self.mod, k, v)
        CACHE.invalidate()


def _cfg(**kw):
    import copy
    c = copy.deepcopy(CFG)
    for path, v in kw.items():
        d = c
        *head, last = path.split("__")
        for k in head:
            d = d[k]
        d[last] = v
    return c


@test("v2 dataset: los 19 originales intactos (misma huella de casos que antes del parche) y select() por defecto = 19")
def t_v2_core_intact():
    import hashlib
    h = hashlib.sha1()
    for pid in sorted(PROBLEMS):
        p = PROBLEMS[pid]
        for c in p.visible() + p.hidden() + p.generated(1234, 24) + p.generated(1233, 24):
            h.update(json.dumps([pid, c.args, c.exp, c.exc], sort_keys=True).encode())
    assert len(PROBLEMS) == 19 and h.hexdigest()[:16] == "3515217de1c07716", h.hexdigest()[:16]
    from problems.problems import select
    assert [p.id for p in select()] == [p.id for p in sorted(PROBLEMS.values(), key=lambda p: (p.level, p.id))]
    assert all(p.split == "train" and p.family == p.id and p.source == "core" and p.category for p in PROBLEMS.values())


@test("dataset: >=80 train nuevos, 50 val, 50 test, 14 categorías por split, familias sin compartir split, sin "
      "duplicados ni casi-duplicados entre splits")
def t_v2_splits_disjoint():
    from collections import Counter
    from problems.problems import ALL, CATALOG, CATEGORIES, select
    from problems.validate import check_global
    sp = Counter(p.split for p in CATALOG.values())
    assert len(CATALOG) >= 180 and sp["val"] == 50 and sp["test"] == 50 and sp["train"] >= 80, sp
    for s_ in ("train", "val", "test"):
        assert {p.category for p in CATALOG.values() if p.split == s_} == set(CATEGORIES), s_
    assert check_global(list(ALL.values())) == []
    from problems.validate import similar_pairs
    assert similar_pairs(list(ALL.values())) == [], similar_pairs(list(ALL.values()))[:5]
    fam = {}
    for p in ALL.values():
        fam.setdefault(p.family, set()).add(p.split)
    assert all(len(v) == 1 for v in fam.values())
    import argparse
    from training.loop import train_pool
    want = [p.id for p in select(sets=tuple(CFG.dataset.sets), splits=("train",))]
    assert [p.id for p in train_pool(CFG, argparse.Namespace(levels=None, problems=None))] == want
    leak = train_pool(CFG, argparse.Namespace(levels=None, problems=["move_zeros", "lcs_length", "dedupe_last"]))
    assert [p.id for p in leak] == ["dedupe_last"], [p.id for p in leak]  # val/test nunca entran en train


@test("v2 validate.py: todo el catálogo pasa (determinismo, referencia, diversidad, mutation score, O(n^2), fugas)")
def t_v2_validate_catalog():
    from problems.problems import CATALOG
    from problems.validate import validate
    res, glob_ = validate(list(CATALOG.values()), CFG, mutation=True, workers=4)
    bad = {pid: e for pid, (e, _) in res.items() if e}
    assert not bad and not glob_, (bad, glob_)
    opt_ = [i for pid, (_, i) in res.items() if CATALOG[pid].category == "optimizacion"]
    assert opt_ and all(i["slow_rejected"] for i in opt_)


@test("v2 validate.py: detecta salidas constantes, O(n^2) no detectado, oculto = visible y referencia rota")
def t_v2_validate_detects():
    from problems.problems import Problem
    from problems.validate import check_problem

    def const(xs):
        return 0
    p = Problem(1, "def const(xs: list) -> int", "Return 0.", const, [[[1]]], lambda r: [[r.randint(0, 9)]],
                [[[1]], [[]]], None, (), "train", "arrays", "fam_const", None, "catalog")
    errs, _ = check_problem(p, CFG, mutation=False)
    assert any("diversas" in e for e in errs) and any("visible" in e for e in errs), errs

    def slowish(xs):
        return sum(xs)

    def total(xs):
        return sum(xs)
    q = Problem(1, "def total(xs: list) -> int", "Sum.", total, [[[1, 2]]], lambda r: [[r.randint(-50, 50) for _ in range(5)]],
                [[[]], [[7]]], lambda r: [[[1] * 1000]], (), "train", "optimizacion", "fam_total", slowish, "catalog")
    errs, _ = check_problem(q, CFG, mutation=False)
    assert any("O(n^2)" in e for e in errs), errs
    assert any("optimización" in e for e in check_problem(Problem(1, "def total(xs)", "Sum.", total, [[[1]]],
        lambda r: [[r.randint(0, 99)]], [[[]]], None, (), "train", "optimizacion", "f2", None, "catalog"), CFG, False)[0])


@test("v2 feedback: el pool de contraejemplos y los grandes nuevos nunca contienen una entrada oculta ni visible")
def t_v2_pool_no_hidden():
    from problems.problems import ALL, _key
    for p in ALL.values():
        banned = {_key(c.args) for c in p.hidden() + p.visible()}
        for seed in (0, 1):
            pool = p.feedback_cases(seed) + p.timeout_cases(seed)
            assert not banned & {_key(c.args) for c in pool}, p.id
            sizes = [len(_key(c.args)) for c in p.feedback_cases(seed)]
            assert sizes == sorted(sizes), p.id  # de menor a mayor: el primer fallo ya es pequeño


@test("v2 sandbox: campo line en excepción, timeout y error de carga")
def t_v2_line():
    from sandbox import executor as ex
    b = backend()
    r = ex.run_cases("def f(x):\n    y = 1\n    return [0][x]\n", "f", [[0], [5]], CFG.sandbox_limits, CFG.sandbox.allowed_imports, b)
    assert r.cases[1]["exc"] == "IndexError" and r.cases[1]["line"] == 3, r.cases
    r = ex.run_cases("x = 1\nraise ValueError('a')\ndef f():\n    pass\n", "f", [[]], CFG.sandbox_limits, CFG.sandbox.allowed_imports, b)
    assert r.status == "load_error" and r.fatal_line == 2, (r.status, r.fatal_line)
    r = ex.run_cases("def f():\n    while True:\n        pass\n", "f", [[]], dict(CFG.sandbox_limits, case_timeout_s=0.3),
                     CFG.sandbox.allowed_imports, b)
    assert r.cases[0]["exc"] == "Timeout" and r.cases[0]["line"] in (2, 3), r.cases


@test("v2 Result: campos estructurados (componentes = reward, caso que falla, línea, timeout con tamaño y ref)")
def t_v2_result_fields():
    from problems.problems import ALL, baseline
    from rewards.reward import _chance
    w = CFG.reward_weights
    p = ALL["count_pairs_sum"]
    quad = ("def count_pairs_sum(nums, target):\n    c = 0\n    for i in range(len(nums)):\n"
            "        for j in range(i + 1, len(nums)):\n            c += nums[i] + nums[j] == target\n    return c\n")
    r = score(p, quad, CFG, 3)
    assert r.error == "timeout" and r.timeout and r.fail_case["category"] == "large" and r.timeout_size >= 30000
    assert r.ref_time is not None and r.ref_time < 1.0 and r.line in (4, 5), (r.fail_case, r.line, r.ref_time)
    assert abs(sum(r.components.values()) - r.reward) < 1e-5
    expect = (w.compile + w.visible * r.pv + w.hidden * _chance(r.ph, baseline(p.hidden()))
              + w.generated * _chance(r.pg, baseline(p.generated(3, CFG.reward.n_generated))) + w.solved * r.solved)
    assert abs(round(expect, 6) - r.reward) < 1e-9  # la fórmula no cambia
    assert sum(v[1] for v in r.cat_pass.values()) == r.n_cases and sum(v[0] for v in r.cat_pass.values()) == r.n_pass
    r = score(ALL["chunk"], "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n", CFG, 3)
    assert r.fail_case["kind"] == "hidden" and r.fail_case["category"] == "error" and r.detail is None, r.fail_case
    r = score(PROBLEMS["clamp"], "x = 1\ny = [][0]\ndef clamp(x, lo, hi):\n    return x\n", CFG, 3)
    assert r.error.startswith("load:") and r.line == 2 and "line 2" in r.tb, (r.error, r.line, r.tb)


@test("v2 feedback counterexample: mensaje accionable por tipo de error y basic idéntico al de siempre")
def t_v2_feedback_kinds():
    from problems.problems import ALL
    from rewards.feedback import build, feedback
    p = ALL["first_index"]

    def fb(prob, code, **kw):
        r = score(prob, code, CFG, 5)
        return build(prob, code, r, CFG, "counterexample", 11, **kw)[0], r
    t, _ = fb(p, "def first_index(xs, target)\n    return 0\n")
    assert "SyntaxError at line 1" in t and "^" in t, t
    t, _ = fb(p, "def first_idx(xs, target):\n    return 0\n")
    assert "def first_index(xs: list[int], target: int) -> int" in t and "first_idx" in t, t
    t, _ = fb(p, "y = [][0]\ndef first_index(xs, target):\n    return 0\n")
    assert "line 1" in t and "y = [][0]" in t, t
    t, _ = fb(ALL["rle_decode"], "def rle_decode(s):\n    out = ''\n    i = 0\n    while i < len(s):\n"
              "        out += s[i] * int(s[i + 1])\n        i += 2\n    return out\n")
    assert "line 5" in t and "IndexError" in t, t
    t, _ = fb(ALL["move_zeros"], "def move_zeros(xs):\n    return sorted(xs, key=lambda x: x == 0)[:len(xs) - 1] + [0] if xs else []\n")
    assert "should" in t, t
    t, _ = fb(ALL["dedupe_last"], "def dedupe_last(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n"
              "            out.append(x)\n    return out\n")
    assert "First difference at index" in t or "Expected length" in t, t
    t, _ = fb(p, "def first_index(xs, target):\n    for i in range(len(xs)):\n        if xs[i] == target:\n            return i\n",
              truncated=True)
    assert "cut off" in t, t
    code = "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]\n"
    r = score(ALL["median"], code, CFG, 5)
    assert build(ALL["median"], code, r, CFG, "basic")[0] == feedback(ALL["median"], code, r, CFG.sandbox.allowed_imports)


@test("v2 feedback counterexample: el contraejemplo nunca es una entrada oculta, se reduce y el timeout usa grandes nuevos")
def t_v2_counterexample_no_leak():
    from problems.problems import ALL, _key
    from rewards.feedback import build
    cases = [("chunk", "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n"),
             ("median", "def median(xs):\n    if not xs:\n        raise ValueError\n    s = sorted(xs)\n    return s[len(s) // 2]\n"),
             ("safe_join", "def safe_join(base, rel):\n    return base + '/' + rel\n"),
             ("count_pairs_sum", "def count_pairs_sum(nums, target):\n    return sum(nums[i] + nums[j] == target "
              "for i in range(len(nums)) for j in range(i + 1, len(nums)))\n")]
    for pid, code in cases:
        p = ALL[pid]
        r = score(p, code, CFG, 5)
        assert not r.solved, pid
        txt, info = build(p, code, r, CFG, "counterexample", 21)
        hidden = {_key(c.args) for c in p.hidden()}
        if info.get("args") is not None:
            assert _key(info["args"]) not in hidden, pid
        for c in p.hidden():
            s_ = repr(c.args)[1:-1]
            if len(s_) >= 12 and c.args not in [v.args for v in p.visible()]:
                assert s_ not in txt, (pid, s_)
        if pid == "count_pairs_sum":
            assert info["source"] == "large" and "new input" in txt, txt
        if pid == "chunk":
            assert info["source"] == "pool" and info["shrunk"], info


@test("v2 caché: código repetido entra una vez; AST igual con mismas posiciones comparte; semilla/versión/config no")
def t_v2_cache_logic():
    import rollouts.collect as rc
    calls = []
    p = PROBLEMS["clamp"]
    with _patched(rc, score=_fake_score([GOOD_CLAMP], calls)):
        a = rc.cached_score(p, GOOD_CLAMP, CFG, 1)
        b = rc.cached_score(p, GOOD_CLAMP, CFG, 1)
        assert len(calls) == 1 and a.solved and b.solved
        rc.cached_score(p, GOOD_CLAMP + "  # comentario\n", CFG, 1)
        assert len(calls) == 2  # clave = código exacto: otro texto, otra ejecución (nunca se mezclan)
        calls.pop()
        a.reward = -99  # las copias son independientes de la caché
        assert rc.cached_score(p, GOOD_CLAMP, CFG, 1).reward != -99 and len(calls) == 1
        rc.cached_score(p, GOOD_CLAMP, CFG, 2)
        assert len(calls) == 2  # otra semilla -> otros tests generados
        rc.cached_score(p, GOOD_CLAMP, _cfg(reward__n_generated=7), 1)
        assert len(calls) == 3  # otra config del evaluador
        rc.CACHE.invalidate("clamp")
        rc.cached_score(p, GOOD_CLAMP, CFG, 1)
        assert len(calls) == 4  # invalidación explícita
        old = p._version
        p._version = "otraversion0"
        try:
            rc.cached_score(p, GOOD_CLAMP, CFG, 1)
            assert len(calls) == 5  # otra versión del problema -> no se reutiliza
        finally:
            p._version = old
        rc.cached_score(p, GOOD_CLAMP, _cfg(rollouts__score_cache=False), 1)
        rc.cached_score(p, GOOD_CLAMP, _cfg(rollouts__score_cache=False), 1)
        assert len(calls) == 7  # desactivada: cada llamada ejecuta


@test("v2 caché: 8 hilos con el mismo código a la vez -> una sola ejecución (deduplicación en vuelo)")
def t_v2_cache_concurrency():
    import threading
    import rollouts.collect as rc
    calls, base = [], _fake_score([GOOD_CLAMP])

    def slow(p, code, cfg, seed=0):
        calls.append(code)
        time.sleep(0.3)
        return base(p, code, cfg, seed)
    with _patched(rc, score=slow):
        out = []
        ths = [threading.Thread(target=lambda: out.append(rc.cached_score(PROBLEMS["clamp"], GOOD_CLAMP, CFG, 9)))
               for _ in range(8)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        assert len(calls) == 1 and len(out) == 8 and all(r.solved for r in out), len(calls)
        assert len({id(r) for r in out}) == 8  # cada hilo recibe su propia copia


@test("v2 caché + sandbox: un grupo de 8 muestras idénticas sólo entra una vez al sandbox; truncado por muestra")
def t_v2_cache_sandbox_once():
    import rollouts.collect as rc
    from sandbox import executor as ex
    backend()
    real, n = ex.run_cases, []

    def counting(*a, **k):
        n.append(1)
        return real(*a, **k)
    good = PROBLEMS["clamp"].ref_source()
    gen = _FnGen(lambda p, i, t: good)
    with _patched(ex, run_cases=counting):
        res = rc.collect(gen, [PROBLEMS["clamp"]], 8, CFG, 4)
    assert len(n) == 1 and all(r.solved for _, r in res[0]), len(n)
    assert not any(r.truncated for _, r in res[0])


@test("v2 rollouts: override de temperatura por llamada y modos greedy / sample / best-of-n (selección permitida)")
def t_v2_solve_modes():
    import rollouts.collect as rc
    import rollouts.solve as rs
    gen = _FnGen(lambda p, i, t: GOOD_CLAMP if (t == 0.0 or i % 4 == 3) else W0)
    allowed = lambda p, code, cfg, seed, n: 1.0 if code == GOOD_CLAMP else 0.4
    with _patched(rc, score=_fake_score([GOOD_CLAMP])), _patched(rs, _allowed=allowed):
        g = rs.solve(gen, PROBLEMS["clamp"], CFG, "greedy")
        assert g["result"].solved and gen.calls[-1][2] == 0.0
        b = rs.solve(gen, PROBLEMS["clamp"], CFG, "best-of-n", n=4)
        assert b["result"].solved and b["allowed"] == 1.0 and b["candidates"] == 4
        assert rs.diversity([W0, W0 + "\n# x", GOOD_CLAMP]) == round(2 / 3, 4)
        try:
            rs.solve(gen, PROBLEMS["clamp"], CFG, "otro")
            raise AssertionError("modo inválido aceptado")
        except ValueError:
            pass


@test("v2 Fix v1 por defecto: mismo bucle, feedback basic idéntico, campos nuevos aditivos")
def t_v2_fix_v1_default():
    import rollouts.collect as rc
    from rewards.feedback import feedback
    from rollouts import fix as gtf
    gen = _ScriptGen(W0, GOOD_CLAMP)
    with _patched(rc, score=_fake_score([GOOD_CLAMP])):
        chains = gtf.run(gen, [PROBLEMS["clamp"]], CFG, 2, 1, 0)
    c = chains[0]
    assert [a.round for a in c] == [0, 1] and c[-1].solved and c[0].mode == "basic"
    assert c[0].feedback == feedback(PROBLEMS["clamp"], W0, _fake_score([GOOD_CLAMP])(None, W0, CFG), CFG.sandbox.allowed_imports)
    s_ = gtf.summary(chains)
    assert s_["engine"] == "v1" and s_["fix_rate"] == 1.0 and s_["solved@0"] == 0.0 and s_["solved@1"] == 1.0


@test("v2 Fix v2: k candidatos, repetidos sin ejecutar, selección permitida, parada al pasar lo permitido, métricas")
def t_v2_fix_v2_engine():
    import rollouts.collect as rc
    import rollouts.solve as rs
    from rollouts import fix as gtf

    def fn(prompt, i, t):
        if "Fix the function" not in prompt:
            return W0
        if "return lo" in prompt:
            return GOOD_CLAMP if i == 0 else W1
        return W0 if i == 0 else W1
    calls = []
    allowed = lambda p, code, cfg, seed, n: {GOOD_CLAMP: 1.0, W1: 0.6}.get(code, 0.5)
    cfg = _cfg(fix__engine="v2", fix__candidates=2)
    with _patched(rc, score=_fake_score([GOOD_CLAMP], calls)), _patched(rs, _allowed=allowed):
        chains = gtf.run(_FnGen(fn), [PROBLEMS["clamp"]], cfg, 4, 1, 0)
    c = chains[0]
    assert sorted(set(calls)) == sorted({W0, W1, GOOD_CLAMP}) and len(calls) == 3, calls  # repetidos no se ejecutan
    assert any(a.repeat for a in c) and gtf.final_attempt(c).code == GOOD_CLAMP
    s_ = gtf.summary(chains)
    assert s_["engine"] == "v2" and s_["fix_rate"] == 1.0 and s_["solved@1"] == 0.0 and s_["solved@2"] == 1.0, s_
    assert s_["repetition"] > 0 and s_["oracle_any"] == 1.0 and "fix_by_error" in s_
    assert max(a.round for a in c) == 2  # paró al pasar todo lo permitido


@test("v2 Fix v2: atasco -> reinicio desde cero; paciencia y presupuesto acotan la cadena")
def t_v2_fix_v2_restart_budget():
    import rollouts.collect as rc
    import rollouts.solve as rs
    from rollouts import fix as gtf
    calls = []
    allowed = lambda p, code, cfg, seed, n: 0.5
    with _patched(rc, score=_fake_score([GOOD_CLAMP], calls)), _patched(rs, _allowed=allowed):
        c = gtf.run(_FnGen(lambda p, i, t: W0), [PROBLEMS["clamp"]], _cfg(fix__engine="v2", fix__candidates=2), 8, 1, 0)[0]
        assert len(calls) == 1 and any(a.strategy == "restart" for a in c), [(a.round, a.strategy) for a in c]
        assert max(a.round for a in c) < 8  # paciencia agotada tras el reinicio: no gasta todas las rondas
        c = gtf.run(_FnGen(lambda p, i, t: W0 if i == 0 else f"def clamp(x, lo, hi):\n    return {i}\n"),
                    [PROBLEMS["clamp"]], _cfg(fix__engine="v2", fix__candidates=2, fix__budget=3), 8, 1, 0)[0]
        assert len(c) == 3, len(c)  # presupuesto: 3 candidatos en total


@test("v2 Fix A/B: basic vs counterexample sobre los mismos primeros intentos; sólo cambia el feedback")
def t_v2_fix_compare():
    import rollouts.collect as rc
    from rollouts import fix as gtf

    def fake_fb(p, code, r, cfg, mode="basic", seed=0, truncated=False):
        return (None if r.solved else f"[{mode}] should fix", {"source": "x"})
    gen = _FnGen(lambda p, i, t: GOOD_CLAMP if "[counterexample]" in p else W0)
    with _patched(rc, score=_fake_score([GOOD_CLAMP])), _patched(gtf, build_feedback=fake_fb):
        res = gtf.compare(gen, [PROBLEMS["clamp"]], CFG, 2, 2, 0)
    b, cx = res["basic"]["chains"], res["counterexample"]["chains"]
    assert [c[0].code for c in b] == [c[0].code for c in cx] and gen.calls[0][1] == 2  # primeros intentos compartidos
    assert res["basic"]["summary"]["fix_rate"] == 0.0 and res["counterexample"]["summary"]["fix_rate"] == 1.0
    assert all(a.mode == "counterexample" for c in cx for a in c)


@test("v2 eval: RNG restaurado, determinista, greedy con temperatura 0, pass@k, IC bootstrap y desgloses")
def t_v2_eval_rng():
    import random
    import numpy as np
    import rollouts.collect as rc
    from evaluation.evaluate import bootstrap_ci, evaluate
    good = {"clamp": GOOD_CLAMP, "abs_sum": "def abs_sum(a, b):\n    return abs(a) + abs(b)\n"}
    gen = _FnGen(lambda p, i, t: good["clamp"] if "clamp" in p and (t == 0.0 or i % 2) else W0)
    probs = [PROBLEMS["clamp"], PROBLEMS["abs_sum"]]
    random.seed(7)
    np.random.seed(7)
    st, nst = random.getstate(), np.random.get_state()
    with _patched(rc, score=_fake_score(good.values())):
        r1 = evaluate(gen, probs, CFG, n=4, ks=[1, 4], seed=5, B=200, split="train")
        r2 = evaluate(gen, probs, CFG, n=4, ks=[1, 4], seed=5, B=200, split="train")
    assert random.getstate() == st and all((a == b).all() if hasattr(a, "all") else a == b
                                           for a, b in zip(np.random.get_state(), nst))
    assert r1["pass@1"] == r2["pass@1"] and r1["ci_pass@1"] == r2["ci_pass@1"]
    assert any(c[2] == 0.0 for c in gen.calls) and r1["greedy"] == 0.5
    row = {r["id"]: r for r in r1["rows"]}
    assert row["clamp"]["pass_at"]["1"] == 0.5 and row["abs_sum"]["pass_at"]["4"] == 0.0
    assert set(r1["by_level"]) and set(r1["by_category"]) and "wrong_answer" in r1["error_rates"]
    assert r1["components"] and r1["compile_rate"] == 1.0 and r1["quality"]
    assert bootstrap_ci([1, 0, 1, 1], 500, 3) == bootstrap_ci([1, 0, 1, 1], 500, 3)
    assert 999983 not in range(CFG.seed, CFG.seed + CFG.grpo.max_steps + 1)


@test("v2 GRPO: adv_norm std por defecto idéntico, mean = r - media, skip_zero_var detecta grupos sin varianza")
def t_v2_grpo_flags():
    import statistics as st
    from training.grpo import group_advantages, zero_var_keys
    items = [("a", 1.0), ("a", 0.0), ("a", 0.5), ("b", 0.3), ("b", 0.3), ("c", 2.0), ("c", -1.0)]
    m = {k: (st.fmean([r for kk, r in items if kk == k]), st.pstdev([r for kk, r in items if kk == k])) for k in "abc"}
    ref = [0.0 if m[k][1] < 1e-4 else (r - m[k][0]) / m[k][1] for k, r in items]
    assert group_advantages(items) == ref == group_advantages(items, norm="std")
    assert group_advantages(items, norm="mean") == [0.0 if m[k][1] < 1e-4 else r - m[k][0] for k, r in items]
    sd = st.pstdev([r for _, r in items])  # batch (Lite PPO): media del grupo, std de todo el lote
    assert group_advantages(items, norm="batch") == [0.0 if m[k][1] < 1e-4 else (r - m[k][0]) / sd for k, r in items]
    assert zero_var_keys(items) == {"b"}
    try:
        group_advantages(items, norm="max")
        raise AssertionError("norm inválida aceptada")
    except ValueError:
        pass
    assert opt(CFG, "grpo.skip_zero_var") is False and opt(CFG, "grpo.adv_norm") == "std"


@test("v2 improve: hard mining (fórmula y tope 3x), cuotas de repaso/frontera/recuperación, dominio y apertura")
def t_v2_improve_curriculum():
    import tempfile
    from training.improve import Improver
    probs, V2 = sorted(PROBLEMS.values(), key=lambda p: (p.level, p.id)), _cfg(improve__curriculum="v2")
    assert Improver(tempfile.mkdtemp(), probs, CFG, fix_root=tempfile.mkdtemp()).mode == "v1"  # por defecto: el anterior
    imp = Improver(tempfile.mkdtemp(), probs, V2, fix_root=tempfile.mkdtemp())
    imp.rate, imp.near, imp.fix_fail, imp.rep = {"clamp": 0.5}, {"clamp": 1.0}, {"clamp": 1.0}, {"clamp": 1.0}
    assert abs(imp.weight("clamp") - 0.25 * 2.25) < 1e-12
    pool = [PROBLEMS[k] for k in ("clamp", "abs_sum", "fizzbuzz", "is_leap_year")]
    for k in ("abs_sum", "fizzbuzz", "is_leap_year"):
        imp.rate[k] = 0.0
    w = imp.weights(pool)
    assert abs(w[0] - 3 * (0.5625 + 3 * 0.05) / 4) < 1e-9 and w[1] == 0.05, w
    imp.open = 2
    imp.mastered = {imp.levels[0]}
    nf, nr, nm, *_ = imp.quotas(6)
    assert nr >= 1 and nr / 6 >= 0.15 and nf + nr + nm == 6, (nf, nr, nm)
    imp.recovering = {imp.levels[0]}
    assert imp.quotas(6)[1] / 6 >= 0.25
    imp.stalled = True
    assert imp.quotas(10)[0] <= 2
    batch = imp.sample(probs, 6)
    assert len(batch) == 6 and all(p.level in imp.levels[:2] for p in batch)
    imp2 = Improver(tempfile.mkdtemp(), probs, V2, fix_root=tempfile.mkdtemp())
    for p in probs:
        if p.level == imp2.levels[0]:
            imp2.rate[p.id], imp2.obs[p.id] = 0.9, 5
    imp2._refresh()
    assert imp2.levels[0] in imp2.mastered and imp2.open == 2
    imp2.observe_eval("val", {"by_level": {str(imp2.levels[0]): {"pass@1": 0.9}}})
    imp2.observe_eval("val", {"by_level": {str(imp2.levels[0]): {"pass@1": 0.5}}})
    assert imp2.levels[0] in imp2.recovering


@test("v2 improve: estado versionado (guardar/cargar), migración desde v1, reinicio limpio y modo v1")
def t_v2_improve_state():
    import json, tempfile
    from training.improve import Improver
    probs = sorted(PROBLEMS.values(), key=lambda p: (p.level, p.id))
    d, V2 = tempfile.mkdtemp(), _cfg(improve__curriculum="v2")
    imp = Improver(d, probs, V2, fix_root=tempfile.mkdtemp())
    imp.rate, imp.obs, imp.near, imp.mastered, imp.open = {"clamp": 0.4}, {"clamp": 3}, {"clamp": 0.2}, {0}, 2
    imp.save()
    data = json.load(open(os.path.join(d, "improve.json")))
    assert data["version"] == 2 and data["mastered"] == [0]
    imp2 = Improver(d, probs, V2, fix_root=tempfile.mkdtemp())
    assert imp2.rate == {"clamp": 0.4} and imp2.near == {"clamp": 0.2} and imp2.mastered == {0} and imp2.open == 2
    with open(os.path.join(d, "improve.json"), "w") as f:  # formato v1 (antes del parche)
        json.dump(dict(rate={"clamp": 0.7}, obs={"clamp": 4}, open_level=1, buffer=[]), f)
    imp3 = Improver(d, probs, V2, fix_root=tempfile.mkdtemp())
    assert imp3.rate == {"clamp": 0.7} and imp3.obs == {"clamp": 4} and imp3.near == {}
    imp3.save()
    assert json.load(open(os.path.join(d, "improve.json")))["version"] == 2
    imp4 = Improver(d, probs, _cfg(improve__reset=True, improve__curriculum="v2"), fix_root=tempfile.mkdtemp())
    assert imp4.rate == {} and imp4.open == 1
    v1 = Improver(tempfile.mkdtemp(), probs, _cfg(improve__curriculum="v1"), fix_root=tempfile.mkdtemp())
    v1.rate, v1.near = {"clamp": 0.5}, {"clamp": 1.0}
    assert v1.weight("clamp") == 0.25 and v1.weights([PROBLEMS["clamp"]]) == [0.25]


@test("v2 reparación sintética: mutantes compilan, fallan algún test, no son la referencia y sólo hay de train")
def t_v2_repair_tasks():
    from problems import repair
    from problems.problems import select
    ts = repair.tasks(select(), per_problem=1, seed=0)
    assert len(ts) >= 15, len(ts)
    for t in ts:
        compile(t.buggy, "<m>", "exec")
        assert t.split == "train" and t.category == "debugging" and t.family.startswith("repair:")
        assert t.buggy.strip() != t.ref_source().strip() and "buggy" in t.prompt()
    r = repair.run_batch([(ts[0].buggy, ts[0].entry)], ts[0].visible() + ts[0].hidden() + ts[0].generated(0, 24))
    assert not r[0]["ok"]
    assert repair.tasks(select(sets=("catalog",), splits=("val", "test")), 2) == []


@test("v2 agente: feedback estructurado de pytest/traceback, línea del error y firma")
def t_v2_agent_summary():
    from agent.loop import summarize_tests
    out = ("E       assert -1 == 3\n\ntests/test_calc.py:4: AssertionError\n=== short test summary info ===\n"
           "FAILED tests/test_calc.py::test_add - assert -1 == 3\n1 failed, 2 passed in 0.03s\n")
    r = summarize_tests(out)
    assert r["failed"] == ["tests/test_calc.py::test_add"] and r["counts"] == {"failed": 1, "passed": 2}
    d = _mini_repo()
    r = summarize_tests('Traceback (most recent call last):\n  File "/w/calc.py", line 2, in add\nTypeError: boom\n', d)
    assert r["where"] == ("/w/calc.py", 2) and r["code"] == "return a - b" and "TypeError" in r["sig"], r


@test("v2 agente: una edición repetida no se escribe ni se testea; memoria de fallos en el prompt")
def t_v2_agent_repeat():
    import tempfile
    import agent.loop as al
    from agent.loop import Session
    from sandbox.command import CmdResult
    n = []
    real = al.command.run
    al.command.run = lambda *a, **k: (n.append(1), CmdResult(1, "FAILED test_calc.py::test_add - AssertionError\n1 failed", 0.1))[1]
    try:
        eng = _TextEngine("FILE: calc.py\n```python\ndef add(a, b):\n    return a * b\n```\n")
        cfg = load_config()
        cfg["agent"]["patience"] = 0  # aquí sólo se prueba la memoria de repeticiones (la paciencia tiene su test)
        s = Session(_mini_repo(), "arregla add", cfg, root=tempfile.mkdtemp())
        s.run(eng, 3)
    finally:
        al.command.run = real
    assert [it.get("repeat") for it in s.iters] == [False, True, True], [it.get("repeat") for it in s.iters]
    assert len(n) == 2  # tests iniciales + la primera edición; las repetidas no
    assert "repeated an earlier edit" in eng.prompts[-1] and "Failing:" in eng.prompts[-1]


@test("v2 agente: k candidatos se prueban en copias y se aplica el que pasa los tests")
def t_v2_agent_candidates():
    import tempfile
    import agent.loop as al
    from agent.loop import Session
    from sandbox.command import CmdResult

    def fake_run(argv, work, *a, **k):
        ok = "a + b" in open(os.path.join(work, "calc.py")).read()
        return CmdResult(0 if ok else 1, "1 passed" if ok else "FAILED x - AssertionError\n1 failed", 0.1)
    texts = iter(["FILE: calc.py\n```python\ndef add(a, b):\n    return a * b\n```\n",
                  "FILE: calc.py\n```python\ndef add(a, b):\n    return a + b\n```\n"] * 3)

    class Eng(_FakeEngine):
        def complete(self, prompt, max_new, temperature=None):
            self.prompts.append(prompt)
            return next(texts)
    real = al.command.run
    al.command.run = fake_run
    try:
        s = Session(_mini_repo(), "arregla add", _cfg(agent__candidates=2), root=tempfile.mkdtemp())
        s.run(Eng(), 1)
    finally:
        al.command.run = real
    assert s.status == "passed" and s.iters[0]["cand_scores"] == [[0, 0], [1, 1]], s.iters[0].get("cand_scores")
    assert not [d for d in os.listdir(os.path.dirname(s.work)) if d.startswith("cand_")]  # copias borradas


@test("v2 benchmark del agente: mini-repo con bug real; sus tests fallan con el bug y pasan con la referencia")
def t_v2_agent_bench():
    import subprocess, tempfile
    import agent.loop as al
    from agent import bench
    from sandbox.command import CmdResult
    t = bench.bench_tasks(1, 0)[0]
    d = tempfile.mkdtemp()
    task = bench.make_repo(d, t)
    run_ = lambda w: subprocess.run([sys.executable, "test_solution.py"], cwd=w, capture_output=True, text=True, timeout=60)
    assert run_(d).returncode == 1 and t.entry in task
    ref = t.ref_source()

    class Eng(_FakeEngine):
        def complete(self, prompt, max_new, temperature=None):
            return f"FILE: solution.py\n```python\n{ref.rstrip()}\n```\n"

    def fake_run(argv, work, *a, **k):
        r = run_(work)
        return CmdResult(r.returncode, r.stdout + r.stderr, 0.1)
    real = al.command.run
    al.command.run = fake_run
    try:
        rep = bench.run_bench(Eng(), CFG, 2, 0, "python test_solution.py", 2, root=tempfile.mkdtemp())
        seen = []
        al.command.run = lambda argv, work, *a, **k: seen.append(argv) or fake_run(argv, work)
        rep2 = bench.run_bench(Eng(), CFG, 1, 0, None, 1, root=tempfile.mkdtemp())  # sin --test: el del benchmark
    finally:
        al.command.run = real
    assert rep["n"] == 2 and rep["tampered"] == 0, rep
    assert rep["repos"][0]["solved"], rep["repos"][0]
    assert rep2["test_cmd"] == "python test_solution.py" and seen and all(a[-1] == "test_solution.py" for a in seen)
    assert rep2["repos"][0]["solved"], rep2  # no depende de que pytest esté instalado


@test("agente: una respuesta con un único bloque de código y sin cabecera FILE: se aplica al único fichero que nombra la "
      "tarea; la plantilla del prompt copiada tal cual nunca se escribe; con varios bloques o sin fichero claro, nada")
def t_agent_lone_block():
    import subprocess, tempfile
    import agent.loop as al
    from agent import bench
    from agent.edits import parse
    from sandbox.command import CmdResult
    assert parse("```python\nx = 1\n```", default="a.py") == {"a.py": "x = 1\n"}
    assert parse("```python\nx = 1\n```\n```python\ny = 2\n```", default="a.py") == {}  # ambiguo: dos bloques
    assert parse("```python\nx = 1\n```") == {}  # sin fichero por defecto: como antes
    assert parse("FILE: b.py\n```python\nz = 3\n```", default="a.py") == {"b.py": "z = 3\n"}  # la cabecera manda
    t = bench.bench_tasks(1, 0)[0]
    ref = t.ref_source().rstrip()
    run_ = lambda w: subprocess.run([sys.executable, "test_solution.py"], cwd=w, capture_output=True, text=True, timeout=60)

    def fake_run(argv, work, *a, **k):
        r = run_(work)
        return CmdResult(r.returncode, r.stdout + r.stderr, 0.1)

    class Lone(_FakeEngine):  # el arreglo correcto, sin la cabecera FILE:
        def complete(self, prompt, max_new, temperature=None):
            return f"```python\n{ref}\n```"

    class Template(_FakeEngine):  # copia la plantilla del prompt de sistema
        def complete(self, prompt, max_new, temperature=None):
            return "FILE: solution.py\n```python\n<the complete new content of the file>\n```\n"
    real = al.command.run
    al.command.run = fake_run
    try:
        rep = bench.run_bench(Lone(), CFG, 1, 0, None, 2, root=tempfile.mkdtemp())
        assert rep["repos"][0]["solved"], rep
        d = tempfile.mkdtemp()
        task = bench.make_repo(d, t)
        before = open(os.path.join(d, "solution.py")).read()
        s = al.Session(d, task, CFG, "python test_solution.py", root=tempfile.mkdtemp())
        assert s.target() == "solution.py"
        s.run(Template(), iters=1)
        it = s.iters[-1]
        assert it["files"] == [] and it["rejected"] and "plantilla" in it["rejected"][0], it
        assert open(os.path.join(s.work, "solution.py")).read() == before  # la copia de trabajo no se tocó
        s2 = al.Session(d, "Fix the bug so the tests pass.", CFG, "python test_solution.py", root=tempfile.mkdtemp())
        assert s2.target() is None  # la tarea no nombra ningún fichero: un bloque sin cabecera no se aplica
    finally:
        al.command.run = real


@test("v2 calidad: métricas informativas (AST, ramas, anidamiento, imports/variables sin usar, except, global, print)")
def t_v2_quality():
    from evaluation.quality import quality
    q = quality("import os\ncache = []\ndef f(a):\n    global cache\n    t = 1\n    try:\n        print(a)\n    except:\n        pass\n"
                "    if a:\n        for i in a:\n            if i:\n                return i\n    return 0\n")
    assert q["unused_imports"] == 1 and q["unused_vars"] == 1 and q["bare_except"] == 1 and q["prints"] == 1
    assert q["global_state"] == 2 and q["depth"] >= 4 and q["branches"] >= 4 and quality("def (") is None
    q = quality("def g(xs):\n    k = lambda v: v if v else 0\n    a = sorted(xs, key=k)\n    a = sorted(xs, key=k)\n    return a\n")
    assert q["duplication"] == 1 and q["branches"] == 1, q  # lambda / IfExp: su body es una expresión


@test("v2 panel: opciones nuevas editables con validación de opciones y booleanos; choices en /api/config")
def t_v2_app_editable():
    import app
    assert app._check("fix.feedback", "counterexample") == "counterexample"
    assert app._check("grpo.skip_zero_var", "true") is True and app._check("rollouts.score_cache", False) is False
    for k, v in (("fix.feedback", "otro"), ("grpo.adv_norm", "max"), ("grpo.skip_zero_var", 1), ("fix.candidates", 0)):
        try:
            app._check(k, v)
            raise AssertionError((k, v))
        except ValueError:
            pass
    a = _app()
    view = a.config_view()
    assert view["choices"]["fix.engine"] == ["v1", "v2"] and view["values"]["fix.engine"] == "v1"
    assert view["values"]["grpo.skip_zero_var"] is False


@test("compatibilidad: la config de los runs existentes da el MISMO config_hash (se siguen reanudando); Result/Attempt "
      "posicionales; opt() con config antigua; claves nuevas con valor seguro")
def t_v2_compat():
    # tests/fixtures/config_v1.yaml = config.yaml con la que se creó runs/phase2/20261002-234419 (hash 46bcf41273df).
    # Antes este test comparaba con la config.yaml VIVA del usuario: cualquier ajuste suyo lo rompía (2 fallos falsos).
    from common import read_config
    from training.checkpoint import config_hash
    from rewards.reward import Result
    from rollouts.fix import Attempt
    old = read_config(os.path.join(ROOT, "tests", "fixtures", "config_v1.yaml"))
    assert config_hash(old) == "46bcf41273df", config_hash(old)
    assert config_hash(dict(old, growth={"enabled": False, "lr": 1.0})) == "46bcf41273df"  # growth apagado: mismo hash
    assert config_hash(dict(old, growth={"enabled": True, "schedule": [{"step": 0, "new_layers": 2}]})) != "46bcf41273df"
    r = Result(0.5, True, False, 0.1, 0.2, 0.3, 0.4, 1, 2, "wrong_answer", 0.0, 0.0, 0.0, None)
    assert r.components is None and r.truncated is False and r.line is None
    a = Attempt("p", 0, 0, "c", 0.1, False, True, "e", "fb", 3)
    assert a.selected and a.mode == "basic" and a.allowed is None
    assert opt({"fix": {"iters": 3}}, "fix.feedback", "basic") == "basic" and opt({}, "a.b.c", 7) == 7
    for key, dflt in (("grpo.minibatches", 1), ("growth.enabled", False), ("vram.budget_gb", "auto"),
                      ("eval.paired_base", True), ("agent.patience", 0)):  # config antigua sin las claves nuevas
        assert opt(old, key, dflt) == dflt, key



PHASE2_RUNNER = r"""
import argparse, os, pickle, sys, types
sys.path.insert(0, os.getcwd())
mode = sys.argv[1]


class _Any:  # cualquier otro atributo de torch: decorador/función/contexto inocuo
    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else _Any()

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def __getattr__(self, name):
        return _Any()


class _T:  # estado de RNG con la forma que espera checkpoint.py
    def __init__(self, v):
        self.v, self.device = v, types.SimpleNamespace(type="cpu")

    def cpu(self):
        return self


import importlib.abc, importlib.machinery


class _TorchFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):  # torch.nn, torch.nn.functional...: inertes
    def find_spec(self, name, path, target=None):
        if name.startswith("torch.") and name not in sys.modules:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__getattr__, m.__path__ = (lambda name: _Any()), []
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _TorchFinder())
torch = types.ModuleType("torch")
torch.__getattr__, torch.__path__ = (lambda name: _Any()), []
_rng = {"v": 0}
torch.get_rng_state = lambda: _T(_rng["v"])
torch.set_rng_state = lambda t: _rng.__setitem__("v", t.v)
torch.manual_seed = lambda s_: _rng.__setitem__("v", s_)


def _save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path, map_location=None, weights_only=True):
    with open(path, "rb") as f:
        return pickle.load(f)


torch.save, torch.load = _save, _load
cuda = types.ModuleType("torch.cuda")
for k_, v_ in dict(is_available=lambda: False, empty_cache=lambda: None, manual_seed_all=lambda s_: None,
                   reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
                   max_memory_reserved=lambda: 0).items():
    setattr(cuda, k_, v_)
optim = types.ModuleType("torch.optim")


class AdamW:
    def __init__(self, params, lr=0.0, weight_decay=0.0):
        self.state, self.param_groups = {"lr": lr, "updates": 0}, []

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, d):
        self.state = dict(d)


optim.AdamW = AdamW
torch.cuda, torch.optim = cuda, optim
sys.modules.update({"torch": torch, "torch.cuda": cuda, "torch.optim": optim})


class Tok:
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}\n" for m in msgs) + "<assistant>"

    def __call__(self, text):
        return types.SimpleNamespace(input_ids=text.split())


pol = types.ModuleType("model.policy")
_fake_model = types.SimpleNamespace(device="cpu", get_decoder=lambda: types.SimpleNamespace(layers=[]),
                                    parameters=lambda: iter(()), config=types.SimpleNamespace(model_type="doble"))
pol.load_policy = lambda cfg, device, resume_from=None, events=None: (_fake_model, Tok(), None)
pol.checkpoint_events = lambda path: []
pol.trainable_params = lambda m: []
pol.save_policy = lambda model, path, lora: open(os.path.join(path, "adapter.txt"), "w").write("fake")
sys.modules["model.policy"] = pol

from collections import Counter
import rollouts.generate as rg, rollouts.monitor as rmon, training.grpo as tg
from problems.problems import ALL


class Gen:
    # referencia en greedy, al reparar y en muestras alternas; el nivel 1 siempre falla al primer intento;
    # el "modelo base" (base=True) siempre falla
    def __init__(self, model, tok, cfg, base=False):
        self.tok, self.max_new, self.ctx, self.chunk, self.base = tok, 64, 4096, 4, base
        self.stats, self.seen = dict(tokens=0, samples=0, seconds=0.0, oom=0), Counter()

    def tune(self, prompt):
        yield {"chunk": 4}

    def stream(self, prompts, n, temperature=None):
        for i, p in enumerate(prompts):
            prob = next((q for q in ALL.values() if q.prompt() in p), None)
            for _ in range(n):
                k = self.seen[p]
                self.seen[p] += 1
                good = not self.base and prob is not None and (
                    temperature == 0.0 or "Fix the function" in p or "buggy" in p
                    or (prob.level != 1 and prob.id not in ("abs_sum", "clamp") and k % 2 == 0))  # 2 de L0: sin varianza
                code = prob.ref_source() if good else "def broken(:\n"
                self.stats["tokens"] += 10
                self.stats["samples"] += 1
                self.stats["seconds"] += 0.01
                yield i, rg.Sample(code, code, 10, True, [1, 2], [3, 4])


class Trainer:
    def __init__(self, model, ref_ctx, tok, optimizer, cfg):
        self.micro_bs, self.opt, self.ref_len = 2, optimizer, 8

    def tune(self, *a, **k):
        yield {"micro_bs": 2}

    def step(self, items):
        self.opt.state["updates"] += 1
        return tg.StepStats(loss=0.1, kl=0.01, clip_frac=0.0, grad_norm=1.0, tokens=10 * len(items), microbatches=1)


class Mon:
    def __init__(self, *a):
        pass

    def start(self):
        return self

    def stop(self):
        return {}


class Engine:
    def __init__(self, cfg, ck=None):
        self.cfg, self.ck = cfg, ck

    def generator(self):
        return Gen(None, Tok(), self.cfg, base=self.ck is None)


rg.Generator, tg.GrpoTrainer, rmon.GpuMonitor = Gen, Trainer, Mon
sys.modules["model.infer"] = types.SimpleNamespace(Engine=Engine)
import common
import train
from common import load_config
common.env_info = lambda: {"torch": "doble de test"}  # sin GPU no hay versiones reales que registrar
cfg = load_config()
cfg["sandbox"]["allow_unsafe"] = True
cfg["num_rollouts"] = 4
cfg["dataset"]["levels"] = [0, 1]
if mode == "early":  # early stopping: el val del doble no mejora entre evaluaciones -> para tras la 2.ª
    cfg["grpo"].update(problems_per_step=2, max_steps=6)
    cfg["checkpoint"].update(every_steps=1, eval_every_steps=1)
    cfg["eval"].update(n=2, ks=[1, 2], during_train=["val"], patience=1, bootstrap=50)
    train.phase2(cfg, argparse.Namespace(cmd="phase2", levels=None, problems=None, fresh=True, n=None, config=None))
    sys.exit(0)
if mode == "curve":  # curva: modelo base + todos los checkpoints y best/ de un run
    train.eval_cmd(cfg, argparse.Namespace(split="val", sets=None, levels=None, problems=["word_wrap", "safe_join", "rpn_eval"],
                                           checkpoint=None, run=sys.argv[2], with_base=True, n=2, seed=None, no_greedy=False))
    sys.exit(0)
if mode == "default":  # Phase 2 con la config por defecto (sólo tamaños pequeños)
    cfg["grpo"].update(problems_per_step=3, max_steps=2)
    cfg["checkpoint"].update(every_steps=2, eval_every_steps=2)
    train.phase2(cfg, argparse.Namespace(cmd="phase2", levels=None, problems=None, fresh=True, n=None, config=None))
    sys.exit(0)
# pool de train = los 19 originales: con levels [0, 1] la mitad de los L0 (abs_sum, clamp) no tienen varianza en el
# doble, así que cada step tiene grupos que dynamic sampling debe reponer (con el catálogo eso pasaba sólo a veces)
cfg["dataset"]["sets"] = ["core"]
cfg["grpo"].update(problems_per_step=3, max_steps=4 if mode == "first" else 6, skip_zero_var=True, dynamic_sampling=True)
cfg["checkpoint"].update(every_steps=2, eval_every_steps=2, keep_last=3)
cfg["eval"].update(n=2, ks=[1, 2], during_train=["val"], select_best="val", bootstrap=100)
cfg["improve"].update(curriculum="v2", feedback="basic", synthetic_per_step=1, repair_per_step=2)
if mode in ("first", "resume"):
    train.phase2(cfg, argparse.Namespace(cmd="improve", levels=None, problems=None, fresh=mode == "first", n=None,
                                         config=None))
elif mode == "eval":
    train.eval_cmd(cfg, argparse.Namespace(split="val", sets=None, levels=None, problems=["move_zeros", "median", "grade"],
                                           checkpoint=["ckpt_doble"], n=2, seed=None, no_greedy=False))
else:
    train.fix(cfg, argparse.Namespace(levels=None, problems=["dedupe", "rle_encode"], iters=2, chains=2, checkpoint="ckpt_doble",
                                      engine=None, feedback=None, candidates=None, compare=True))
"""


@test("v2 bucle improve/phase2 real sin GPU (torch y modelo dobles): métricas, eval val, best por val, currículo v2, "
      "checkpoint y reanudación reales, CLI eval --split y fix --compare")
def t_v2_phase2_loop():
    import glob, shutil, subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = os.path.join(root, "runs")
    before = {k: set(glob.glob(os.path.join(runs, k, "*"))) for k in ("phase2", "eval", "fix", "eval_cache")}

    def go(mode, *extra):
        r = subprocess.run([sys.executable, "-c", PHASE2_RUNNER, mode, *extra], cwd=root, capture_output=True,
                           text=True, timeout=900)
        assert r.returncode == 0, (mode, r.stdout[-2000:], r.stderr[-2000:])
        return r.stdout
    made = []
    try:
        go("first")
        new = sorted(set(glob.glob(os.path.join(runs, "phase2", "*"))) - before["phase2"])
        assert len(new) == 1, new
        run_ = new[0]
        made.append(run_)
        rows = [json.loads(x) for x in open(os.path.join(run_, "metrics.jsonl"))]
        steps = [r for r in rows if "loss" in r]
        assert [r["step"] for r in steps] == [0, 1, 2, 3], [r["step"] for r in steps]
        for k in ("compile_rate", "trunc_rate", "comp_solved", "zero_var_frac", "diversity", "cache_hits", "gen_tokens",
                  "near_miss", "err_rates", "imp_open_level", "imp_quota_front", "skipped_zero_var", "sandbox_s"):
            assert all(k in r for r in steps), k
        assert any(r["skipped_zero_var"] for r in steps), "ningún grupo sin varianza descartado"
        assert any(r["dyn_extra"] for r in steps) and all(r["dyn_informative"] is not None for r in steps), \
            [(r["dyn_extra"], r["dyn_informative"]) for r in steps]  # dynamic sampling repuso grupos sin varianza
        ev = [r for r in rows if r.get("event") == "eval_split"]
        assert [r["step"] for r in ev] == [1, 3] and all("val_pass@1" in r and "gap_train_val" in r for r in ev), ev
        assert "val_regressed" in ev[1] and "val_improved" in ev[1]  # olvido (la 1.ª se compara con el modelo base)
        assert all("val_delta_pass@1" in r and "val_delta_ci" in r for r in ev), ev  # pareado contra el de partida
        base_rows = [r for r in rows if r.get("event") == "eval_base"]
        assert base_rows and base_rows[0]["step"] == -1, base_rows
        assert os.path.exists(os.path.join(run_, "evals", "step000001_val.json"))
        assert json.load(open(os.path.join(run_, "improve.json")))["version"] == 2
        best = json.load(open(os.path.join(run_, "checkpoints", "best", "meta.json")))["metrics"]
        assert "best_val" in best and "val_pass@1" in best, best
        out = go("resume")
        assert "reanudado desde" in out, out[-1500:]
        rows = [json.loads(x) for x in open(os.path.join(run_, "metrics.jsonl"))]
        steps = [r for r in rows if "loss" in r]
        assert [r["step"] for r in steps] == list(range(6)), [r["step"] for r in steps]  # sin duplicados
        last = sorted(d for d in os.listdir(os.path.join(run_, "checkpoints")) if d.startswith("ckpt_"))[-1]
        with open(os.path.join(run_, "checkpoints", last, "optim.pt"), "rb") as f:
            import pickle
            upd = pickle.load(f)["updates"]
        assert last == "ckpt_0000005" and upd == sum(not r.get("no_update") for r in steps), (last, upd)
        go("default")
        dflt = sorted(set(glob.glob(os.path.join(runs, "phase2", "*"))) - before["phase2"] - {run_})
        made += dflt
        rows = [json.loads(x) for x in open(os.path.join(dflt[0], "metrics.jsonl"))]
        steps = [r for r in rows if "loss" in r]
        assert [r["step"] for r in steps] == [0, 1] and [r["step"] for r in rows if r.get("event") == "eval_split"] == [1]
        assert all(r["skipped_zero_var"] is None and r["open_level"] is None and "compile_rate" in r for r in steps)
        for k in ("t_gen", "t_score_wait", "t_update", "ms_per_tok_step", "vram_reserved_gb", "entropy", "updates"):
            assert all(k in r for r in steps), k  # timing por etapa y diagnóstico del trainer en cada step
        assert any("eval_reward" in r for r in rows) and not os.path.exists(os.path.join(dflt[0], "improve.json"))
        best = json.load(open(os.path.join(dflt[0], "checkpoints", "best", "meta.json")))
        assert "eval_reward" in best["metrics"] and "best_val" in best["metrics"], best  # config por defecto: best por val
        assert best["format"] == 2 and best["arch"]["layers"] == 0 and best["growth"] == [] and "code" in best, best
        assert os.path.exists(os.path.join(dflt[0], "checkpoints", "best", "MANIFEST.json"))
        go("early")
        early = sorted(set(glob.glob(os.path.join(runs, "phase2", "*"))) - before["phase2"] - {run_} - set(dflt))
        made += early
        rows = [json.loads(x) for x in open(os.path.join(early[0], "metrics.jsonl"))]
        assert [r["step"] for r in rows if "loss" in r] == [0, 1] and any(r.get("event") == "early_stop" for r in rows)
        go("curve", run_)
        cv = sorted(set(glob.glob(os.path.join(runs, "eval", "*"))) - before["eval"])
        made += cv
        curve = json.load(open(os.path.join(cv[0], "curve.json")))["curve"]
        assert curve[0]["label"] == "base" and curve[-1]["label"] == "best" and len(curve) >= 3, curve
        assert curve[0]["pass@1"] == 0.0 and curve[-1]["pass@1"] == 0.5 and curve[-1]["greedy"] == 1.0, curve
        from evaluation.compare import compare, load
        c = compare(load(os.path.join(cv[0], curve[0]["file"])), load(os.path.join(cv[0], curve[-1]["file"])))
        # 3 problemas mejoran y ninguno empeora: test de signos p = 0.25 -> no significativo (con el criterio
        # anterior, sólo bootstrap, esto salía "mejora"; ver evaluation/compare.py)
        assert c["metrics"]["pass@1"]["delta"] == 0.5 and len(c["improved"]) == 3, c
        assert c["metrics"]["pass@1"]["p_sign"] == 0.25 and c["verdict"] == "sin diferencia significativa", c
        before["eval"] |= set(cv)
        go("eval")
        ev_run = sorted(set(glob.glob(os.path.join(runs, "eval", "*"))) - before["eval"])
        made += ev_run
        rep = json.load(open(os.path.join(ev_run[0], "eval.json")))
        assert rep["split"] == "val" and rep["problems"] == 3 and rep["greedy"] == 1.0 and "ci_pass@1" in rep, rep
        go("fix")
        fx = sorted(set(glob.glob(os.path.join(runs, "fix", "*"))) - before["fix"])
        made += fx
        cmp = json.load(open(os.path.join(fx[0], "compare.json")))["compare"]
        assert cmp["basic"]["fix_rate"] == 1.0 and cmp["counterexample"]["fix_rate"] == 1.0, cmp
        att = [json.loads(x) for x in open(os.path.join(fx[0], "attempts.jsonl"))]
        assert {a["mode"] for a in att} == {"basic", "counterexample"}
    finally:
        for d in made:
            shutil.rmtree(d, ignore_errors=True)
        for f in set(glob.glob(os.path.join(runs, "eval_cache", "*"))) - before["eval_cache"]:
            os.remove(f)



@test("v2 bugs: Fix v2 con candidatos idénticos en la misma ronda no desalinea resultados (cada intento con el suyo)")
def t_v2_fix_v2_dups_same_round():
    import rollouts.collect as rc
    import rollouts.solve as rs
    from rollouts import fix as gtf

    def fn(prompt, i, t):
        return W0 if "Fix the function" not in prompt else (W1, W1, GOOD_CLAMP)[i % 3]
    calls = []
    allowed = lambda p, code, cfg, seed, n: {GOOD_CLAMP: 1.0, W1: 0.6}.get(code, 0.5)
    with _patched(rc, score=_fake_score([GOOD_CLAMP], calls)), _patched(rs, _allowed=allowed):
        c = gtf.run(_FnGen(fn), [PROBLEMS["clamp"]], _cfg(fix__engine="v2", fix__candidates=3), 2, 1, 0)[0]
    assert len(calls) == 3, calls  # W0, W1 y GOOD: el W1 repetido de la ronda no se ejecuta
    assert all(a.solved == (a.code == GOOD_CLAMP) for a in c), [(a.code[-10:], a.solved) for a in c]
    assert sum(a.repeat for a in c) == 1 and gtf.final_attempt(c).solved


@test("v2 bugs: al reanudar, best_reward/best_val salen de best/ (una eval peor no sobrescribe best/)")
def t_v2_resume_best():
    import tempfile
    from training.loop import resume_best
    d = tempfile.mkdtemp()
    assert resume_best(d, {"eval_reward": 0.6}, -1e9, -1.0) == (0.6, -1.0)
    os.makedirs(os.path.join(d, "best"))
    json.dump({"step": 9, "metrics": {"eval_reward": 0.8, "val_pass@1": 0.5}}, open(os.path.join(d, "best", "meta.json"), "w"))
    assert resume_best(d, {"eval_reward": 0.6}, -1e9, -1.0) == (0.8, 0.5)
    assert resume_best(d, {"eval_reward": 0.9, "best_val": 0.7}, -1e9, -1.0) == (0.9, 0.7)


@test("v2 bugs: un crash (posible fallo transitorio) no se cachea; greedy del override es greedy puro")
def t_v2_cache_crash_greedy():
    import types
    import rollouts.collect as rc
    from rewards.reward import Result
    from rollouts.generate import Generator
    n = []

    def crash(p, code, cfg, seed=0):
        n.append(1)
        return Result(0.0, True, False, error="crash:-9")
    with _patched(rc, score=crash):
        rc.cached_score(PROBLEMS["clamp"], W0, CFG, 1)
        rc.cached_score(PROBLEMS["clamp"], W0, CFG, 1)
    assert len(n) == 2
    fake = types.SimpleNamespace(sampling=dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0, repetition_penalty=1.0))
    assert Generator._sampling(fake, 0.0) == dict(do_sample=False, repetition_penalty=1.0)
    assert Generator._sampling(fake, None) is fake.sampling  # por defecto: exactamente el de siempre
    assert Generator._sampling(fake, 0.5)["temperature"] == 0.5 and Generator._sampling(fake, 0.5)["top_p"] == 0.95


@test("v2 bugs: improve v2 con 1-2 problemas por step no deja sin muestras a la frontera")
def t_v2_quotas_small_k():
    import tempfile
    from training.improve import Improver
    probs = sorted(PROBLEMS.values(), key=lambda p: (p.level, p.id))
    imp = Improver(tempfile.mkdtemp(), probs, _cfg(improve__curriculum="v2"), fix_root=tempfile.mkdtemp())
    imp.open, imp.mastered = 2, {imp.levels[0]}
    nf, nr, _, *_ = imp.quotas(2)
    assert nf == 1 and nr == 1, (nf, nr)
    assert all(sum(imp.quotas(1)[:2]) == 1 for _ in range(20))
    assert any(imp.quotas(1)[0] == 1 for _ in range(50))


@test("v2 bugs: run_batch con un código que se traga el timeout devuelve fallo en vez de colgarse o reventar")
def t_v2_run_batch_hang():
    from problems import repair
    from problems.problems import Case
    code = ("def f():\n    while True:\n        try:\n            while True:\n                pass\n"
            "        except BaseException:\n            pass\n")
    t0 = time.time()
    r = repair.run_batch([(code, "f")], [Case([], "1", None)], 0.2, timeout=3)
    assert r == [{"ok": False, "first": -1, "why": "batch_timeout"}] and time.time() - t0 < 10, r



@test("v2 bugs: benchmark con más de 19 repos y validate --ids desconocido devuelve error")
def t_v2_bench_ids():
    from agent.bench import bench_tasks
    from problems import validate
    ts = bench_tasks(25, 0)
    assert len(ts) == 25 and len({t.id for t in ts}) == 25, len(ts)
    assert validate.main(["--ids", "no_existe", "--no-mutation"]) == 2



@test("v2 mejoras: comparación pareada de evaluaciones (delta, IC bootstrap, test de signos, mejoran/empeoran)")
def t_v2_compare():
    from evaluation.compare import compare, sign_test, table

    def rep_(vals, split="val", n=16):
        return dict(split=split, n=n, ks=[1], rows=[dict(id=f"p{i}", pass_at={"1": v}, greedy=v > 0.5)
                                                    for i, v in enumerate(vals)])
    a, b = rep_([0.0] * 8 + [0.5] * 4), rep_([0.5] * 8 + [0.5] * 4)
    c = compare(a, b, 500, 1)
    m = c["metrics"]["pass@1"]
    assert m["delta"] == round(4 / 12, 4) and m["wins"] == 8 and m["losses"] == 0 and m["ties"] == 4
    assert m["p_sign"] == round(2 / 256, 4) and m["ci"][0] > 0 and c["verdict"] == "mejora" and len(c["improved"]) == 8
    assert compare(a, b, 500, 1) == c and "greedy" in c["metrics"]  # determinista
    assert compare(b, a, 500, 1)["verdict"] == "empeora" and compare(a, a, 500, 1)["verdict"].startswith("sin")
    assert sign_test(0, 0) == 1.0 and abs(sign_test(5, 5) - 1.0) < 1e-12
    w = compare(a, rep_([0.1] * 12, split="test", n=8), 100, 0)["warnings"]
    assert any("splits" in x for x in w) and any("muestras" in x for x in w) and "veredicto" in table(c)


@test("v2 mejoras: aviso de problemas parecidos entre splits (variante no declarada) y no entre el mismo split")
def t_v2_similar_pairs():
    from problems.problems import Problem
    from problems.validate import similar_pairs

    def total(xs):
        acc = 0
        for x in xs:
            if x > 0:
                acc += x
        return acc

    def total2(xs):
        acc = 0
        for x in xs:
            if x >= 0:
                acc += x
        return acc
    mk = lambda fn, split, doc: Problem(1, f"def {fn.__name__}(xs)", doc, fn, [[[1]]], lambda r: [[1]], [], None, (),
                                        split, "arrays", fn.__name__, None, "catalog")
    a, b = mk(total, "train", "Sum of the positive numbers."), mk(total2, "val", "Sum of the non-negative numbers.")
    assert [x[:2] for x in similar_pairs([a, b])] == [("total", "total2")]
    assert similar_pairs([a, mk(total2, "train", "Sum of the non-negative numbers.")]) == []


@test("v2 mejoras: feedback con pistas de tipo y diferencias de claves en dicts")
def t_v2_feedback_types():
    from rewards.feedback import _first_diff
    assert _first_diff([1, 2], {"a": 1}) == " Expected a list, got a dict."
    assert _first_diff({"a": 1, "b": 2}, {"a": 1, "c": 2}) == " Missing keys ['b']; unexpected keys ['c']."
    assert _first_diff({"a": 1}, {"a": 2}) == " For key 'a' expected 1, got 2."
    assert _first_diff(True, 1) == " Expected a bool, got a number." and _first_diff(1.0, 1) == ""
    assert _first_diff([1, 2, 3], [1, 5, 3]) == " First difference at index 1: expected 2, got 5."


@test("v2 mejoras: el panel lista las evaluaciones (una y curva) con split, pass@1, IC y greedy")
def t_v2_app_evals():
    import shutil
    from common import ROOT
    d1, d2 = os.path.join(ROOT, "runs", "eval", "zz-uno"), os.path.join(ROOT, "runs", "eval", "zz-curva")
    os.makedirs(d1, exist_ok=True)
    os.makedirs(d2, exist_ok=True)
    try:
        json.dump({"split": "val", "pass@1": 0.25, "ci_pass@1": [0.1, 0.4], "greedy": 0.5, "checkpoint": None, "rows": []},
                  open(os.path.join(d1, "eval.json"), "w"))
        json.dump({"split": "val", "curve": [{"label": "base", "step": None, "pass@1": 0.1, "ci_pass@1": [0, 0.2],
                                              "greedy": 0.0}, {"label": "best", "step": 40, "pass@1": 0.3,
                                                               "ci_pass@1": [0.2, 0.4], "greedy": 0.5}]},
                  open(os.path.join(d2, "curve.json"), "w"))
        ev = {r["name"]: r["evals"] for r in _app().runs() if r["kind"] == "eval"}
        assert ev["zz-uno"] == [dict(label="base", step=None, split="val", pass1=0.25, ci=[0.1, 0.4], greedy=0.5)]
        assert [e["label"] for e in ev["zz-curva"]] == ["base", "best"] and ev["zz-curva"][1]["step"] == 40
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)



@test("v2 bugs: dos runs creados en el mismo segundo no comparten directorio")
def t_v2_run_dir_unique():
    import shutil
    from common import new_run_dir
    ds = [new_run_dir(CFG, "zz_test_unico") for _ in range(3)]
    try:
        assert len(set(ds)) == 3 and all(os.path.exists(os.path.join(d, "config.yaml")) for d in ds), ds
    finally:
        shutil.rmtree(os.path.dirname(ds[0]), ignore_errors=True)



@test("v2 dataset: longest_unique_substring detecta la solución O(n^2) (trabajo cuadrático real, timeout y validate)")
def t_v2_unique_substring_quadratic():
    import inspect, textwrap
    from problems import repair
    from problems.problems import CATALOG, _key
    from problems.validate import check_problem
    p = CATALOG["longest_unique_substring"]
    big = [c for c, k in zip(p.hidden(), p.hidden_kinds()) if k == "large"]
    assert len(big) == 2 and not {_key(c.args) for c in big} & {_key(c.args) for c in p.visible()}

    def slow_steps(s_):  # pasos exactos de la versión ingenua (cada inicio hasta el primer repetido), en O(n)
        seen, r, total = set(), 0, 0
        for i in range(len(s_)):
            while r < len(s_) and s_[r] not in seen:
                seen.add(s_[r])
                r += 1
            total += r - i + (r < len(s_))
            seen.discard(s_[i])
        return total
    # independiente de la CPU: >= 5e8 iteraciones de Python no caben en el timeout de 2 s en ninguna máquina
    assert max(slow_steps(c.args[0]) for c in big) >= 5e8, [slow_steps(c.args[0]) for c in big]
    assert all(len(json.dumps(c.args)) < 2_000_000 for c in big)  # razonable para el sandbox
    t0 = time.perf_counter()
    assert [p.ref(*c.args) for c in big] == [json.loads(c.exp) for c in big] and time.perf_counter() - t0 < 1.0
    limit = CFG.sandbox_limits["case_timeout_s"]
    slow_src = textwrap.dedent(inspect.getsource(p.slow))
    r = repair.run_batch([(slow_src, p.slow.__name__), (p.ref_source(), p.entry)], big, limit)
    assert r[0]["ok"] is False and r[0]["why"] == "Timeout" and r[1]["ok"], r  # la O(n^2) agota el tiempo, la O(n) no
    errs, info = check_problem(p, CFG, mutation=False)
    assert errs == [] and info["slow_rejected"] is True, (errs, info)


@test("v2 sandbox: el scorer real marca timeout en un caso grande a la O(n^2) de longest_unique_substring")
def t_v2_unique_substring_quadratic_score():
    import inspect, textwrap
    from problems.problems import CATALOG
    p = CATALOG["longest_unique_substring"]
    backend()
    code = textwrap.dedent(inspect.getsource(p.slow)).replace(f"def {p.slow.__name__}(", f"def {p.entry}(")
    rs = score(p, code, CFG, 3)
    assert not rs.solved and rs.error == "timeout" and rs.fail_case["category"] == "large", (rs.error, rs.fail_case)
    assert score(p, p.ref_source(), CFG, 3).solved


def _extra_tests():
    """Tests de la v3 (tests/test_v3.py): crecimiento, logprobs por trozos, GRPO nuevo, checkpoints, experimentos."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_v3", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "test_v3.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TESTS, mod.Skip


def main():
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    res = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    extra, Skip3 = _extra_tests()
    for name, fn in TESTS + extra:
        if sel not in name:
            continue
        t = time.time()
        try:
            fn()
            st, msg = "PASS", ""
        except (Skip, Skip3, executor.SandboxUnavailable) as e:
            st, msg = "SKIP", str(e)
        except Exception:
            st, msg = "FAIL", traceback.format_exc(limit=3)
        res[st] += 1
        print(f"[{st}] {name} ({time.time() - t:.1f}s)" + (f"\n      {msg}" if msg else ""))
    print(f"\n{res['PASS']} pass, {res['FAIL']} fail, {res['SKIP']} skip (skip = NO verificado)")
    sys.exit(1 if res["FAIL"] else 0)


if __name__ == "__main__":
    main()
