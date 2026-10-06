"""Problemas con tres niveles de tests. Referencias y generadores son código PROPIO (originales, sin datasets externos).

visible  : ejemplos que van en el prompt.
hidden   : casos límite + errores esperados + aleatorios (semilla fija por problema) + tamaños grandes. Nunca en el prompt.
generated: fuzz contra la referencia con semilla variable (no memorizable); expected = ref(args).
Los esperados los calcula la referencia en el proceso confiable; el candidato sólo ve entradas.
"""
from __future__ import annotations

import ast, copy, hashlib, inspect, json, math, random, textwrap, threading, zlib
from collections import ChainMap, Counter
from dataclasses import dataclass

from sandbox.harness import enc

PROBLEMS: dict[str, "Problem"] = {}   # los 19 originales (baseline de Phase 1/2): no cambia
CATALOG: dict[str, "Problem"] = {}    # problems/catalog: nuevos, con split train/val/test por familia
ALL = ChainMap(PROBLEMS, CATALOG)     # búsqueda por id en ambos
SPLITS = ("train", "val", "test")
CATEGORIES = ("algoritmos", "arrays", "strings", "estructuras", "matematicas", "parsing", "seguridad", "edge_cases",
              "optimizacion", "refactoring", "debugging", "comprension", "especificacion", "multi_step")
N_RANDOM = 14  # casos aleatorios de semilla fija en hidden()


def _key(args):
    """Clave canónica de unos argumentos (como viajan al sandbox: JSON)."""
    return json.dumps(args, sort_keys=True, default=repr)


def _src(fn):
    if fn is None:
        return ""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return getattr(fn, "__qualname__", repr(fn))


def shrink_args(args, limit=48):
    """Variantes más pequeñas de unos argumentos (contraejemplo mínimo, bordes del feedback): vaciar, partir por la
    mitad o quitar un elemento de listas/strings y acercar enteros a 0. Nunca devuelve los mismos argumentos."""
    out, seen = [], {_key(list(args))}
    for i, a in enumerate(args):
        if isinstance(a, (list, str)) and a:
            subs = [a[:0], a[:len(a) // 2], a[len(a) // 2:], a[1:], a[:-1]]
            if len(a) <= 12:
                subs += [a[:j] + a[j + 1:] for j in range(len(a))]
        elif isinstance(a, int) and not isinstance(a, bool) and a:
            subs = [0, a // 2, a - 1 if a > 0 else -a]
        else:
            continue
        for v in subs:
            cand = list(args[:i]) + [v] + list(args[i + 1:])
            k = _key(cand)
            if k not in seen:
                seen.add(k)
                out.append(cand)
    return out[:limit]


@dataclass(frozen=True)
class Case:
    args: list
    exp: str | None    # JSON canónico esperado
    exc: str | None    # excepción esperada


class Problem:
    def __init__(self, level, sig, doc, ref, ex, gen, edge, large, errs, split="train", category=None, family=None,
                 slow=None, source="core"):
        self.id = self.entry = ref.__name__
        self.level, self.sig, self.doc, self.ref = level, sig, doc, ref
        self.ex, self.gen, self.edge, self.large, self.errs = ex, gen, edge, large, errs
        self.split, self.category, self.family = split, category, family or ref.__name__
        self.slow, self.source = slow, source  # slow: versión correcta pero cuadrática (sólo optimización)
        self._memo, self._lock, self._version = {}, threading.RLock(), None  # reentrante: feedback_cases usa hidden() dentro de _get

    @property
    def version(self):
        """Huella de todo lo que define el problema y sus tests: si cambia, la caché de score no se reutiliza."""
        if self._version is None:
            desc = json.dumps([self.id, self.level, self.sig, self.doc, _key(self.ex), _key(self.edge), list(self.errs),
                               _src(self.ref), _src(self.gen), _src(self.large), N_RANDOM])
            self._version = hashlib.sha1(desc.encode()).hexdigest()[:12]
        return self._version

    def _case(self, args, strict=True):
        try:
            out = self.ref(*copy.deepcopy(args))
        except Exception as e:
            n = type(e).__name__
            if n in self.errs:
                return Case(args, None, n)
            if strict:
                raise AssertionError(f"{self.id}: la referencia lanzó {n} con {str(args)[:80]}") from e
            return None
        return Case(args, enc(out), None)

    def _get(self, key, build):
        with self._lock:
            if key not in self._memo:
                if len(self._memo) > 64:  # generated() con semillas nuevas: no crecer sin límite
                    self._memo = {k: v for k, v in self._memo.items() if k in ("v", "h")}
                self._memo[key] = build()
            return self._memo[key]

    def visible(self):
        return self._get("v", lambda: [self._case(a) for a in self.ex])

    def hidden(self):
        def build():
            r = random.Random(zlib.crc32(self.id.encode()))
            vis = {_key(a) for a in self.ex} if self.source != "core" else set()  # core: idéntico al original

            def draw():  # catálogo: ningún caso oculto puede ser un ejemplo visible del prompt
                a = self.gen(r)
                for _ in range(50):
                    if _key(a) not in vis:
                        break
                    a = self.gen(r)
                return a
            cs = [self._case(a) for a in self.edge] + [self._case(draw()) for _ in range(N_RANDOM)]
            if self.large:
                cs += [self._case(a) for a in self.large(random.Random(r.random()))]
            return cs
        return self._get("h", build)

    def generated(self, seed, n):
        def build():
            r = random.Random((seed << 20) ^ zlib.crc32(self.id.encode()))
            return [c for c in (self._case(self.gen(r), strict=False) for _ in range(n)) if c]
        return self._get(("g", seed, n), build)

    def hidden_kinds(self):
        """Tipo de cada caso oculto, alineado con hidden(): edge | random | large, o error si espera excepción."""
        hs = self.hidden()
        kinds = ["edge"] * len(self.edge) + ["random"] * N_RANDOM
        kinds += ["large"] * (len(hs) - len(kinds))
        return ["error" if c.exc else k for c, k in zip(hs, kinds)]

    def _fresh(self, arg_lists):
        """Casos nuevos: sin ninguna entrada de los ocultos ni de los visibles, deduplicados, válidos para la
        referencia (lo que la referencia rechaza con una excepción no declarada se descarta), de menor a mayor."""
        banned = {_key(c.args) for c in self.hidden() + self.visible()}
        out, seen = [], set()
        for a in arg_lists:
            k = _key(a)
            if k in banned or k in seen:
                continue
            seen.add(k)
            c = self._case(a, strict=False)
            if c:
                out.append(c)
        return sorted(out, key=lambda c: len(_key(c.args)))

    def feedback_cases(self, seed, n=64):
        """Pool de contraejemplos para el feedback: semilla con sal propia (no coincide con la de train, eval ni
        ocultos) más variantes reducidas de los ejemplos visibles (vacío, un elemento...). Nunca contiene ocultos."""
        def build():
            r = random.Random(zlib.crc32(f"feedback|{seed}|{self.id}".encode()))
            return self._fresh([self.gen(r) for _ in range(n)] + [v for c in self.visible() for v in shrink_args(c.args)])
        return self._get(("f", seed, n), build)

    def timeout_cases(self, seed):
        """Entradas grandes NUEVAS (mismo generador que los grandes ocultos, otra semilla) para el feedback de timeouts."""
        if not self.large:
            return []
        return self._get(("t", seed), lambda: self._fresh(self.large(random.Random(zlib.crc32(f"feedback-large|{seed}|{self.id}".encode())))))

    def prompt(self):
        ex = []
        for c in self.visible():
            call = f"{self.id}({', '.join(map(repr, c.args))})"
            ex.append(f"# {call} raises {c.exc}" if c.exc else f"# {call} -> {json.loads(c.exp)!r}")
        return f'{self.sig}:\n    """{self.doc}"""\n' + "\n".join(ex)

    def ref_source(self):
        """Fuente de la referencia sin decoradores (para ejecutarla como si fuera un candidato)."""
        fn = ast.parse(textwrap.dedent(inspect.getsource(self.ref))).body[0]
        fn.decorator_list = []
        return ast.unparse(fn)


def P(level, sig, doc, ex, gen, edge=(), large=None, errs=(), *, split="train", category=None, family=None, slow=None,
      source="core"):
    def deco(fn):
        assert fn.__name__ not in ALL, f"id de problema repetido: {fn.__name__}"
        nonlocal edge
        if source != "core":  # un borde oculto nunca repite un ejemplo visible
            edge = [e for e in edge if _key(e) not in {_key(x) for x in ex}]
        (PROBLEMS if source == "core" else CATALOG)[fn.__name__] = Problem(
            level, sig, doc, fn, list(ex), gen, list(edge), large, tuple(errs), split, category, family, slow, source)
        return fn
    return deco


def select(levels=None, ids=None, sets=None, splits=None):
    """Por defecto sólo los 19 originales (comportamiento validado). sets: core | catalog; splits: train | val | test.
    Con ids explícitos se buscan en todos los conjuntos salvo que se indique `sets`."""
    sets = sets or (("core", "catalog") if ids else ("core",))
    ps = [p for p in ALL.values() if p.source in sets and (not levels or p.level in levels)
          and (not ids or p.id in ids) and (not splits or p.split in splits)]
    return sorted(ps, key=lambda p: (p.level, p.id))


# ---------- comparación y baseline ----------
def same(a, b):
    """a = salida del candidato (JSON decodificado), b = esperado. bool estricto; floats con tolerancia."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            return a == b or math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9) or (a != a and b != b)
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def judge(c: Case, rec) -> bool:
    if rec is None or "fatal" in rec:
        return False
    if c.exc:
        return c.exc in rec.get("mro", ())
    if "exc" in rec:
        return False
    if "h" in rec:
        import hashlib
        return rec["h"] == hashlib.sha256(c.exp.encode()).hexdigest()
    try:
        return same(json.loads(rec["out"]), json.loads(c.exp))
    except (KeyError, ValueError, TypeError):
        return False


def baseline(cases):
    """Acierto del mejor predictor constante en este conjunto (p. ej. 'siempre True' en un test balanceado = 0.5)."""
    if not cases:
        return 1.0
    return max(Counter(c.exc and "!" + c.exc or c.exp for c in cases).values()) / len(cases)


# ---------- generadores ----------
def _ints(r, n=12, lo=-20, hi=20, min_n=0):
    return [r.randint(lo, hi) for _ in range(r.randint(min_n, n))]


def _word(r, k=6, alpha="abc"):
    return "".join(r.choice(alpha) for _ in range(r.randint(0, k)))


def _bal(r, ref, make, tries=400):
    """Para problemas booleanos: ~50% True / ~50% False (un 'return True' no puede pasar por azar)."""
    want = r.random() < 0.5
    for _ in range(tries):
        a = make(r)
        if bool(ref(*a)) == want:
            return a
    return make(r)


def _g_clamp(r):
    lo = r.randint(-30, 30)
    return [r.randint(-50, 50), lo, lo + r.randint(-3, 30)]


def _g_words(r):
    ws = [_word(r, 5, "abcxyz") or "q" for _ in range(r.randint(0, 6))]
    body = "".join((r.choice([" ", "  ", "\t"]) if i else "") + w for i, w in enumerate(ws))
    return [r.choice(["", " ", "  "]) + body + r.choice(["", " ", "   "])]


def _g_pal(r):
    def make(r):
        h = _word(r, 6, "abAB1")
        c = h + r.choice(["", "x"]) + h[::-1]
        if c and r.random() < 0.5:  # casi-palíndromo
            i = r.randrange(len(c))
            c = c[:i] + r.choice("ab1") + c[i + 1:]
        elif r.random() < 0.3:
            c = _word(r, 12, "abAB1")
        return ["".join(ch + r.choice(["", "", " ", ",", "!"]) for ch in c)]
    return _bal(r, is_palindrome, make)


def _g_rot(r):
    xs = _ints(r, 10, -9, 9)
    return [xs, r.randint(0, 3 * len(xs) + 2)]


def _g_rle(r):
    return ["".join(r.choice("abc") * r.randint(1, 12) for _ in range(r.randint(0, 5)))]


def _g_two(r):
    xs = _ints(r, 12, -15, 15)
    t = r.choice(xs) + r.choice(xs) if xs and r.random() < 0.7 else r.randint(-30, 30)
    return [xs, t]


def _valid(r, d=0):
    s = ""
    for _ in range(r.randint(1, 3) if d == 0 else r.randint(0, 2) if d < 3 else 0):
        o, c = r.choice(["()", "[]", "{}"])
        s += o + _valid(r, d + 1) + c
    return s


def _g_brk(r):
    def make(r):
        s = _valid(r)
        if r.random() < 0.6:
            i = r.randrange(len(s))
            s = r.choice([s[:i] + s[i + 1:], s[:i] + r.choice("()[]{}") + s[i + 1:], s[:i] + r.choice("()[]{}") + s[i:]])
        return [s]
    return _bal(r, valid_brackets, make)


def _g_lru(r):
    ops = [["put", r.randint(0, 4), r.randint(0, 9)] if r.random() < 0.5 else ["get", r.randint(0, 4)]
           for _ in range(r.randint(0, 14))]
    return [r.randint(1, 3), ops]


def _g_graph(r):
    n = r.randint(1, 7)
    es = [[r.randrange(n), r.randrange(n), r.randint(0, 9)] for _ in range(r.randint(0, 12))]
    return [n, es, r.randrange(n), r.randrange(n)]


def _l_two(r):  # valores enormes: el par correcto es único y está al final (O(n^2) no termina)
    xs = [r.randint(-10**12, 10**12) for _ in range(200000)]
    return [[xs, xs[-3] + xs[-1]], [xs, 4 * 10**12]]


def _l_brk(r):
    v = "([{" * 60000 + "}])" * 60000
    return [[v], [v[:-1] + "]"]]


def _l_graph(r):
    n = 20000
    es = [[i, i + 1, 1000] for i in range(n - 1)] + [[r.randrange(n), r.randrange(n), r.randint(1, 100)] for _ in range(80000)]
    return [[n, es, 0, n - 1]]


# ---------- L0: sintaxis y funciones simples ----------
@P(0, "def abs_sum(a: int, b: int) -> int", "Return |a| + |b|.",
   ex=[[3, -4], [0, 0]], edge=[[-10**9, 10**9], [-1, 0]],
   gen=lambda r: [r.randint(-1000, 1000), r.randint(-1000, 1000)])
def abs_sum(a, b):
    return abs(a) + abs(b)


@P(0, "def clamp(x: int, lo: int, hi: int) -> int", "Limit x to the range [lo, hi]. Raise ValueError if lo > hi.",
   ex=[[5, 0, 3], [-2, 0, 3], [1, 5, 2]], edge=[[0, 0, 0], [10**9, -10**9, 10**9], [2, 0, 3]],
   gen=_g_clamp, errs=("ValueError",))
def clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo > hi")
    return max(lo, min(x, hi))


@P(0, "def fizzbuzz(n: int) -> list[str]",
   'For i in 1..n: "FizzBuzz" if i%15==0, "Fizz" if i%3==0, "Buzz" if i%5==0, else str(i).',
   ex=[[3], [5]], edge=[[0], [1], [15], [30]], gen=lambda r: [r.randint(0, 60)], large=lambda r: [[100000]])
def fizzbuzz(n):
    return ["FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i)
            for i in range(1, n + 1)]


@P(0, "def is_leap_year(year: int) -> bool", "Gregorian leap year.",
   ex=[[2024], [1900], [2000], [2023]], edge=[[4], [100], [400], [1], [2100], [1600]],
   gen=lambda r: _bal(r, is_leap_year, lambda r: [r.randint(1, 3000)]))
def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# ---------- L1: arrays / strings ----------
@P(1, "def reverse_words(s: str) -> str",
   "Reverse the order of whitespace-separated words. Single spaces in the result, no leading/trailing spaces.",
   ex=[["hello world"], ["  a   b c "]], edge=[[""], ["   "], ["x"], ["a\tb\nc"]], gen=_g_words)
def reverse_words(s):
    return " ".join(reversed(s.split()))


@P(1, "def is_palindrome(s: str) -> bool",
   "True if s reads the same forwards and backwards, ignoring case and non-alphanumeric characters.",
   ex=[["A man, a plan, a canal: Panama"], ["race a car"], [""]],
   edge=[[" "], ["a."], ["0P"], ["ab_a"], ["Aa"]], gen=_g_pal)
def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]


@P(1, "def second_largest(nums: list[int]) -> int",
   "Second largest DISTINCT value. Raise ValueError if there are fewer than 2 distinct values.",
   ex=[[[3, 1, 2]], [[5, 5, 4]], [[7, 7]]], edge=[[[1, 2]], [[-1, -1, -2]], [[2, 2, 3, 3]], [[]], [[1]]],
   gen=lambda r: [_ints(r, 12, -9, 9)], errs=("ValueError",),
   large=lambda r: [[[r.randint(-10**9, 10**9) for _ in range(200000)]]])
def second_largest(nums):
    s = sorted(set(nums))
    if len(s) < 2:
        raise ValueError("need 2 distinct values")
    return s[-2]


@P(1, "def dedupe(nums: list[int]) -> list[int]",
   "Remove duplicates keeping the first occurrence of each value, preserving order.",
   ex=[[[1, 2, 1, 3, 2]], [[]]], edge=[[[5]], [[2, 2, 2]], [[-1, 0, -1, 0]]],
   gen=lambda r: [_ints(r, 15, -6, 6)], large=lambda r: [[[r.randint(0, 100000) for _ in range(200000)]]])
def dedupe(nums):
    seen, out = set(), []
    for x in nums:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@P(1, "def rotate_left(nums: list[int], k: int) -> list[int]",
   "Rotate left by k >= 0 positions (k may exceed len(nums)). Empty list -> [].",
   ex=[[[1, 2, 3, 4, 5], 2], [[1, 2, 3], 0]], edge=[[[], 5], [[7], 100], [[1, 2, 3], 3], [[1, 2, 3], 4]],
   gen=_g_rot, large=lambda r: [[list(range(100000)), 10**12 + 7]])
def rotate_left(nums, k):
    if not nums:
        return []
    k %= len(nums)
    return nums[k:] + nums[:k]


@P(1, "def rle_encode(s: str) -> str",
   "Run-length encode: each run becomes char+count, e.g. 'aaabcc' -> 'a3b1c2'. '' -> ''.",
   ex=[["aaabcc"], ["z"]], edge=[[""], ["aaaaaaaaaaaa"], ["abab"]], gen=_g_rle,
   large=lambda r: [["".join(r.choice("abc") * r.randint(1, 30) for _ in range(6000))]])
def rle_encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str(j - i))
        i = j
    return "".join(out)


# ---------- L2: estructuras de datos ----------
@P(2, "def two_sum(nums: list[int], target: int) -> list[int]",
   "Return [i, j] with i < j and nums[i] + nums[j] == target, choosing the smallest j (then the smallest i). [] if none.",
   ex=[[[2, 7, 11, 15], 9], [[3, 3], 6], [[1, 2], 9]], edge=[[[], 0], [[5], 10], [[3, 2, 4], 6], [[0, 4, 3, 0], 0]],
   gen=_g_two, large=_l_two)
def two_sum(nums, target):
    seen = {}
    for j, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], j]
        seen.setdefault(x, j)
    return []


@P(2, "def valid_brackets(s: str) -> bool",
   "s contains only ()[]{}. True if every bracket is correctly matched and nested.",
   ex=[["()[]{}"], ["(]"], ["([)]"], ["{[]}"]], edge=[[""], ["("], [")"], ["(("], ["(()"], ["]["]],
   gen=_g_brk, large=_l_brk)
def valid_brackets(s):
    st, pair = [], {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in pair:
            if not st or st.pop() != pair[c]:
                return False
        else:
            st.append(c)
    return not st


@P(2, "def top_k_frequent(nums: list[int], k: int) -> list[int]",
   "The k most frequent values, most frequent first; ties broken by smaller value first. "
   "If there are fewer than k distinct values return all of them. Raise ValueError if k < 1.",
   ex=[[[1, 1, 1, 2, 2, 3], 2], [[4, 4, 5, 5], 1], [[1], 0]],
   edge=[[[], 3], [[7], 1], [[1, 2, 3], 5], [[1], -2]],
   gen=lambda r: [_ints(r, 20, -5, 5), r.randint(0, 5)], errs=("ValueError",),
   large=lambda r: [[[r.randint(0, 50000) for _ in range(200000)], 10]])
def top_k_frequent(nums, k):
    if k < 1:
        raise ValueError("k < 1")
    from collections import Counter
    return [v for v, _ in sorted(Counter(nums).items(), key=lambda t: (-t[1], t[0]))[:k]]


@P(2, "def group_anagrams(words: list[str]) -> list[list[str]]",
   "Group words that are anagrams of each other. Keep input order inside each group; order groups by first appearance.",
   ex=[[["eat", "tea", "tan", "ate", "nat", "bat"]]], edge=[[[]], [[""]], [["a", "a"]], [["ab", "ba", "abc"]]],
   gen=lambda r: [[_word(r, 4, "abc") for _ in range(r.randint(0, 8))]])
def group_anagrams(words):
    g = {}
    for w in words:
        g.setdefault("".join(sorted(w)), []).append(w)
    return list(g.values())


@P(2, "def lru(capacity: int, ops: list[list]) -> list[int]",
   'LRU cache. ops are ["put", key, value] or ["get", key]. Return the result of each get (the value, or -1 if missing). '
   "put and get make the key most recent; evict the least recent key when size exceeds capacity.",
   ex=[[2, [["put", 1, 1], ["put", 2, 2], ["get", 1], ["put", 3, 3], ["get", 2], ["get", 3]]]],
   edge=[[1, []], [1, [["put", 1, 1], ["put", 2, 2], ["get", 1], ["get", 2]]], [2, [["put", 1, 1], ["put", 1, 5], ["get", 1]]]],
   gen=_g_lru)
def lru(capacity, ops):
    from collections import OrderedDict
    c, out = OrderedDict(), []
    for op in ops:
        if op[0] == "put":
            c[op[1]] = op[2]
            c.move_to_end(op[1])
            if len(c) > capacity:
                c.popitem(last=False)
        elif op[1] in c:
            c.move_to_end(op[1])
            out.append(c[op[1]])
        else:
            out.append(-1)
    return out


# ---------- L3: algoritmos ----------
@P(3, "def lis_length(nums: list[int]) -> int", "Length of the longest STRICTLY increasing subsequence.",
   ex=[[[10, 9, 2, 5, 3, 7, 101, 18]], [[]]], edge=[[[5]], [[2, 2, 2]], [[1, 2, 3, 4]], [[4, 3, 2, 1]]],
   gen=lambda r: [_ints(r, 14, -9, 9)], large=lambda r: [[[r.randint(0, 10**6) for _ in range(100000)]]])
def lis_length(nums):
    from bisect import bisect_left
    t = []
    for x in nums:
        i = bisect_left(t, x)
        if i == len(t):
            t.append(x)
        else:
            t[i] = x
    return len(t)


@P(3, "def edit_distance(a: str, b: str) -> int", "Levenshtein distance (insert, delete, substitute; each costs 1).",
   ex=[["kitten", "sitting"], ["", "abc"]], edge=[["", ""], ["a", "a"], ["abc", "abc"], ["abc", "xyz"]],
   gen=lambda r: [_word(r, 12, "abc"), _word(r, 12, "abc")],
   large=lambda r: [["".join(r.choice("abcd") for _ in range(600)), "".join(r.choice("abcd") for _ in range(600))]])
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@P(3, "def coin_change(coins: list[int], amount: int) -> int",
   "Fewest coins (unlimited supply of each positive denomination) summing to amount; -1 if impossible; 0 if amount == 0.",
   ex=[[[1, 2, 5], 11], [[2], 3], [[1], 0]],
   edge=[[[], 0], [[], 5], [[3, 7], 5], [[186, 419, 83, 408], 6249]],
   gen=lambda r: [r.sample(range(1, 12), r.randint(1, 4)), r.randint(0, 40)],
   large=lambda r: [[r.sample(range(2, 300), 30), 20000]])
def coin_change(coins, amount):
    inf = amount + 1
    dp = [0] + [inf] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] < inf else -1


@P(3, "def shortest_path(n: int, edges: list[list[int]], src: int, dst: int) -> int",
   "Directed graph with nodes 0..n-1 and edges [u, v, w] (w >= 0). Minimum total weight from src to dst, or -1 if unreachable.",
   ex=[[3, [[0, 1, 4], [1, 2, 1], [0, 2, 7]], 0, 2], [2, [], 0, 1]],
   edge=[[1, [], 0, 0], [3, [[0, 1, 0], [1, 2, 0]], 0, 2], [4, [[0, 1, 1], [2, 3, 1]], 0, 3]],
   gen=_g_graph, large=_l_graph)
def shortest_path(n, edges, src, dst):
    import heapq
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [None] * n
    dist[src] = 0
    h = [(0, src)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        if u == dst:
            return d
        for v, w in adj[u]:
            nd = d + w
            if dist[v] is None or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return -1


# categoría de los 19 originales (sus definiciones no cambian: split train, familia = id)
CORE_CATEGORY = {"abs_sum": "matematicas", "clamp": "edge_cases", "fizzbuzz": "especificacion", "is_leap_year": "matematicas",
                 "dedupe": "arrays", "is_palindrome": "strings", "reverse_words": "strings", "rle_encode": "strings",
                 "rotate_left": "arrays", "second_largest": "edge_cases", "group_anagrams": "estructuras",
                 "lru": "estructuras", "top_k_frequent": "estructuras", "two_sum": "arrays", "valid_brackets": "parsing",
                 "coin_change": "algoritmos", "edit_distance": "algoritmos", "lis_length": "algoritmos",
                 "shortest_path": "algoritmos"}
for _pid, _cat in CORE_CATEGORY.items():
    PROBLEMS[_pid].category = _cat

from problems import catalog  # noqa: E402,F401  registra CATALOG (necesita P y los helpers de arriba)
