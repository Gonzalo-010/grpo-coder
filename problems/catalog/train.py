"""Catálogo, split train (20 problemas, las 14 categorías). Las referencias importan dentro de la función: así
también se pueden ejecutar como si fueran un candidato (validate.py, reparación sintética)."""
from problems.problems import P, _bal, _ints, _word


def T(level, sig, doc, ex, gen, edge=(), large=None, errs=(), *, category, family, slow=None):
    return P(level, sig, doc, ex, gen, edge, large, errs, split="train", category=category, family=family, slow=slow,
             source="catalog")


# ---------- especificación (variantes reales de problemas del baseline) ----------
@T(1, "def dedupe_last(xs: list) -> list",
   "Remove duplicates keeping the LAST occurrence of each value; the result keeps the order of those last occurrences.",
   [[[1, 2, 1, 3, 2]], [["a", "b", "a"]]], lambda r: [_ints(r, 12, -3, 3)],
   [[[]], [[5]], [[1, 1, 1]], [[1, 2, 3]], [[2, 1, 2, 1]]],
   lambda r: [[[r.randint(0, 999) for _ in range(200000)]]], category="especificacion", family="dedupe_last")
def dedupe_last(xs):
    last = {}
    for i, x in enumerate(xs):
        last[x] = i
    return [x for i, x in enumerate(xs) if last[x] == i]


@T(2, "def two_sum_all(nums: list[int], target: int) -> list[list[int]]",
   "All index pairs [i, j] with i < j and nums[i] + nums[j] == target, sorted by i and then by j.",
   [[[1, 2, 3, 2], 4], [[5], 5]], lambda r: [_ints(r, 12, -5, 5), r.randint(-6, 6)],
   [[[], 0], [[1], 2], [[1, 1, 1], 2], [[0, 0], 0], [[3, -3, 0], 0]], category="especificacion", family="two_sum_all")
def two_sum_all(nums, target):
    out, pos = [], {}
    for j, x in enumerate(nums):
        for i in pos.get(target - x, []):
            out.append([i, j])
        pos.setdefault(x, []).append(j)
    return sorted(out)


# ---------- algoritmos ----------
def _g_intervals(r):
    out = []
    for _ in range(r.randint(0, 8)):
        a = r.randint(0, 20)
        out.append([a, a + r.randint(0, 6)])
    if r.random() < 0.1 and out:
        out[0] = [out[0][1] + 1, out[0][0]]  # intervalo inválido
    return [out]


@T(2, "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
   "Merge overlapping or touching intervals [a, b] (touching: [1, 2] and [2, 3]) and return them sorted by start. "
   "Raise ValueError if some interval has a > b.",
   [[[[1, 3], [2, 6], [8, 10]]], [[[1, 4], [4, 5]]]], _g_intervals,
   [[[]], [[[1, 1]]], [[[1, 2], [3, 4]]], [[[5, 6], [1, 2], [2, 5]]], [[[3, 1]]]],
   lambda r: [[[[a, a + r.randint(0, 30)] for a in (r.randint(0, 10 ** 6) for _ in range(100000))]]],
   ("ValueError",), category="algoritmos", family="merge_intervals")
def merge_intervals(intervals):
    if any(a > b for a, b in intervals):
        raise ValueError("a > b")
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


@T(1, "def first_index(xs: list[int], target: int) -> int",
   "xs is sorted in ascending order and may contain duplicates. Return the index of the FIRST occurrence of target, "
   "or -1 if it is not present.",
   [[[1, 2, 2, 2, 5], 2], [[1, 3], 2]], lambda r: [sorted(_ints(r, 15, -5, 5)), r.randint(-6, 6)],
   [[[], 1], [[1], 1], [[1, 1, 1], 1], [[1, 2, 3], 4], [[1, 2, 3], 0]], category="algoritmos", family="first_index")
def first_index(xs, target):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(xs) and xs[lo] == target else -1


# ---------- arrays ----------
def _g_matrix(r):
    rows, cols = r.randint(1, 5), r.randint(1, 5)
    return [[[r.randint(-9, 9) for _ in range(cols)] for _ in range(rows)]]


@T(2, "def spiral_order(m: list[list[int]]) -> list[int]",
   "Elements of a rectangular matrix in clockwise spiral order, starting at the top-left corner.",
   [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2], [3, 4], [5, 6]]]], _g_matrix,
   [[[]], [[[]]], [[[7]]], [[[1, 2, 3]]], [[[1], [2], [3]]]], category="arrays", family="spiral_order")
def spiral_order(m):
    out = []
    top, bot, left, right = 0, len(m) - 1, 0, (len(m[0]) - 1 if m else -1)
    while top <= bot and left <= right:
        for j in range(left, right + 1):
            out.append(m[top][j])
        top += 1
        for i in range(top, bot + 1):
            out.append(m[i][right])
        right -= 1
        if top <= bot:
            for j in range(right, left - 1, -1):
                out.append(m[bot][j])
            bot -= 1
        if left <= right:
            for i in range(bot, top - 1, -1):
                out.append(m[i][left])
            left += 1
    return out


def _l_products(r):
    xs = [r.choice((-1, 1)) for _ in range(100000)]
    xs[r.randrange(len(xs))] = 0
    return [[xs], [[r.choice((-1, 1)) for _ in range(100000)]]]


@T(2, "def product_except_self(nums: list[int]) -> list[int]",
   "out[i] is the product of every element except nums[i] (exact integers; nums may contain zeros).",
   [[[1, 2, 3, 4]], [[0, 5]]], lambda r: [_ints(r, 8, -4, 4)],
   [[[]], [[5]], [[0, 4]], [[0, 0, 3]], [[2, 3, 4]]], _l_products, category="arrays", family="product_except_self")
def product_except_self(nums):
    n = len(nums)
    out, acc = [1] * n, 1
    for i in range(n):
        out[i] = acc
        acc *= nums[i]
    acc = 1
    for i in range(n - 1, -1, -1):
        out[i] *= acc
        acc *= nums[i]
    return out


# ---------- strings ----------
def _g_rle(r):
    s = "".join(r.choice("abc") + str(r.randint(1, 12)) for _ in range(r.randint(0, 5)))
    u = r.random()
    if u < 0.1:
        s = "1" + s
    elif u < 0.2:
        s = s + r.choice("abc")
    elif u < 0.25:
        s = s + "a0"
    return [s]


@T(1, "def rle_decode(s: str) -> str",
   "Decode run-length text: every letter is followed by its count, a positive integer without leading zeros that may "
   "have several digits ('a3b1c12' -> 'aaab' + 12 'c'). Raise ValueError if the text does not follow that format.",
   [["a3b1"], ["x2y10"]], _g_rle,
   [[""], ["a1"], ["a12"], ["a0"], ["1a"], ["ab2"], ["a01"]], errs=("ValueError",), category="strings",
   family="rle_decode")
def rle_decode(s):
    out, i = [], 0
    while i < len(s):
        ch, j = s[i], i + 1
        while j < len(s) and s[j] in "0123456789":
            j += 1
        num = s[i + 1:j]
        if not ch.isalpha() or not num or num[0] == "0":
            raise ValueError(s)
        out.append(ch * int(num))
        i = j
    return "".join(out)


def _g_lcp(r):
    base = _word(r, 4, "ab")
    return [[base + _word(r, 3, "ab") for _ in range(r.randint(0, 4))]]


@T(1, "def longest_common_prefix(words: list[str]) -> str",
   "Longest string that is a prefix of every word ('' if words is empty).",
   [[["flower", "flow", "flight"]], [["dog", "car"]]], _g_lcp,
   [[[]], [[""]], [["abc"]], [["abc", "abd"]], [["a", "b"]], [["ab", "abc", ""]]], category="strings",
   family="longest_common_prefix")
def longest_common_prefix(words):
    if not words:
        return ""
    p = words[0]
    for w in words[1:]:
        k = 0
        while k < len(p) and k < len(w) and p[k] == w[k]:
            k += 1
        p = p[:k]
    return p


# ---------- estructuras de datos ----------
def _ops_stack(r, n):
    ops, size = [], 0
    for _ in range(n):
        u = r.random()
        if u < 0.45 or size == 0 and u < 0.9:
            ops.append(["push", r.randint(-9, 9)])
            size += 1
        else:
            op = r.choice(("pop", "top", "min"))
            ops.append([op])
            size -= op == "pop"
    return ops


@T(2, "def min_stack(ops: list[list]) -> list[int]",
   "Run stack operations: ['push', x], ['pop'], ['top'], ['min']. Return, in order, the values produced by pop, top "
   "and min (min = smallest value currently in the stack). pop, top or min on an empty stack raise IndexError.",
   [[[["push", 3], ["push", 1], ["min"], ["pop"], ["min"]]], [[["push", 2], ["top"]]]],
   lambda r: [_ops_stack(r, r.randint(0, 15))],
   [[[]], [[["pop"]]], [[["push", 3], ["min"], ["push", 1], ["min"], ["pop"], ["min"]]], [[["push", 1], ["pop"], ["top"]]]],
   lambda r: [[_ops_stack(r, 200000)]], ("IndexError",), category="estructuras", family="min_stack")
def min_stack(ops):
    st, mins, out = [], [], []
    for op in ops:
        if op[0] == "push":
            st.append(op[1])
            mins.append(op[1] if not mins else min(op[1], mins[-1]))
        elif not st:
            raise IndexError(op[0])
        elif op[0] == "pop":
            mins.pop()
            out.append(st.pop())
        elif op[0] == "top":
            out.append(st[-1])
        else:
            out.append(mins[-1])
    return out


def _g_graph(r):
    n = r.randint(0, 10)
    return [n, [[r.randrange(n), r.randrange(n)] for _ in range(r.randint(0, n))] if n else []]


@T(2, "def count_components(n: int, edges: list[list[int]]) -> int",
   "Undirected graph with nodes 0..n-1 and edges [u, v]. Return the number of connected components.",
   [[5, [[0, 1], [1, 2], [3, 4]]], [3, []]], _g_graph,
   [[0, []], [1, []], [3, [[0, 1], [1, 2]]], [3, [[0, 0]]], [4, [[0, 1], [2, 3], [1, 0]]]],
   lambda r: [[100000, [[r.randrange(100000), r.randrange(100000)] for _ in range(90000)]]],
   category="estructuras", family="count_components")
def count_components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    comps = n
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            comps -= 1
    return comps


# ---------- matemáticas ----------
@T(1, "def primes_below(n: int) -> list[int]",
   "All prime numbers p with 2 <= p < n, in ascending order.",
   [[10], [2]], lambda r: [r.randint(0, 200)],
   [[0], [1], [2], [3], [10], [11]], lambda r: [[2000000]], category="matematicas", family="primes_below")
def primes_below(n):
    if n < 3:
        return []
    sieve = bytearray([1]) * n
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, n, i)))
    return [i for i in range(n) if sieve[i]]


def _g_base(r):
    u = r.random()
    b = r.choice((0, 1, 37, 40)) if u < 0.1 else r.randint(2, 36)
    n = -r.randint(1, 50) if u > 0.95 else r.choice((r.randint(0, 40), r.randint(0, 10 ** 6)))
    return [n, b]


@T(1, "def to_base(n: int, b: int) -> str",
   "Write the non-negative integer n in base b (2..36) with digits 0-9 then a-z (lowercase). Raise ValueError if b is "
   "outside 2..36 or n < 0.",
   [[255, 16], [5, 2]], _g_base,
   [[0, 2], [255, 16], [35, 36], [5, 1], [-1, 10], [10, 10]], errs=("ValueError",), category="matematicas",
   family="to_base")
def to_base(n, b):
    if not 2 <= b <= 36 or n < 0:
        raise ValueError((n, b))
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    out = []
    while n:
        n, d = divmod(n, b)
        out.append(digits[d])
    return "".join(reversed(out))


# ---------- parsing ----------
def _g_duration(r):
    parts = [(u, r.randint(0, 99)) for u in "hms" if r.random() < 0.6] or [("s", r.randint(0, 99))]
    u = r.random()
    if u < 0.1:
        parts = parts[::-1] if len(parts) > 1 else parts + parts
    elif u < 0.18:
        return [r.choice(("h", "5", "1x", "1h 2m", "-1s", ""))]
    return ["".join(f"{n}{unit}" for unit, n in parts)]


@T(1, "def parse_duration(s: str) -> int",
   "Convert a duration like '1h30m15s' to seconds. Units h, m and s may each appear at most once and in that order, "
   "each preceded by a non-negative integer; at least one unit must be present. Raise ValueError otherwise.",
   [["1h30m15s"], ["90m"]], _g_duration,
   [[""], ["0s"], ["1h"], ["1m1h"], ["h"], ["2h2h"], ["1h 1m"]], errs=("ValueError",), category="parsing",
   family="parse_duration")
def parse_duration(s):
    import re
    m = re.fullmatch(r"(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?", s)
    if not s or m is None:
        raise ValueError(s)
    h, mi, se = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + se


# ---------- seguridad ----------
def _ip(r):
    return ".".join(str(r.randint(0, 255)) for _ in range(4))


def _bad_ip(r):
    ip = _ip(r).split(".")
    k = r.randrange(4)
    ip[k] = r.choice(("256", "01", "", "1a", " 1", "+1", "-1", "١", "999", "00"))
    if r.random() < 0.2:
        ip = ip[:r.choice((3, 5))] if len(ip) == 4 else ip
    return ".".join(ip)


@T(1, "def is_valid_ipv4(s: str) -> bool",
   "True only for a dotted-quad IPv4 address: exactly 4 parts separated by '.', each made only of ASCII digits, value "
   "0..255 and no leading zeros (except '0' itself). No spaces, signs or any other character.",
   [["192.168.0.1"], ["256.1.1.1"]],
   lambda r: _bal(r, is_valid_ipv4, lambda r: [_ip(r) if r.random() < 0.5 else _bad_ip(r)]),
   [["0.0.0.0"], ["255.255.255.255"], ["01.1.1.1"], ["1.1.1"], ["1.1.1.1."], [" 1.1.1.1"], ["1.1.1.١"], ["+1.1.1.1"]],
   category="seguridad", family="is_valid_ipv4")
def is_valid_ipv4(s):
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p or not all(c in "0123456789" for c in p):
            return False
        if len(p) > 1 and p[0] == "0":
            return False
        if int(p) > 255:
            return False
    return True


# ---------- edge cases ----------
@T(0, "def chunk(xs: list, n: int) -> list[list]",
   "Split xs into consecutive chunks of size n; the last chunk may be shorter. Raise ValueError if n < 1.",
   [[[1, 2, 3, 4, 5], 2], [[1, 2], 5]], lambda r: [_ints(r, 12), r.choice((r.randint(1, 6), r.randint(-1, 6)))],
   [[[], 3], [[1, 2, 3], 1], [[1, 2, 3], 5], [[1], 0], [[1, 2, 3, 4], 2], [[1, 2], -1]], errs=("ValueError",),
   category="edge_cases", family="chunk")
def chunk(xs, n):
    if n < 1:
        raise ValueError(n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]


# ---------- optimización ----------
def _slow_count_pairs_sum(nums, target):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                count += 1
    return count


@T(2, "def count_pairs_sum(nums: list[int], target: int) -> int",
   "Number of index pairs i < j with nums[i] + nums[j] == target. It must be fast for lists of tens of thousands "
   "of numbers.",
   [[[1, 5, 7, -1, 5], 6], [[1, 1, 1], 2]], lambda r: [_ints(r, 15, -5, 5), r.randint(-6, 6)],
   [[[], 0], [[1, 1, 1], 2], [[0], 0], [[2, -2, 0, 0], 0], [[3, 3], 5]],
   lambda r: [[[r.randint(-1000, 1000) for _ in range(30000)], r.randint(-50, 50)],
              [[r.randint(0, 3) for _ in range(30000)], 3]],
   category="optimizacion", family="count_pairs_sum", slow=_slow_count_pairs_sum)
def count_pairs_sum(nums, target):
    seen, count = {}, 0
    for x in nums:
        count += seen.get(target - x, 0)
        seen[x] = seen.get(x, 0) + 1
    return count


# ---------- refactoring ----------
@T(1, "def summarize_stats(xs: list[int]) -> dict",
   "Refactor the function below into summarize_stats(xs), returning a dict {'min': ..., 'max': ..., 'mean': ...} with "
   "the same values and the same error:\n\n"
   "    def stats(xs):\n        if len(xs) == 0:\n            raise ValueError('empty')\n        lo = xs[0]\n"
   "        hi = xs[0]\n        total = 0\n        for x in xs:\n            if x < lo:\n                lo = x\n"
   "            if x > hi:\n                hi = x\n            total = total + x\n"
   "        return (lo, hi, total / len(xs))\n",
   [[[1, 2, 3]], [[5]]], lambda r: [_ints(r, 10, -50, 50, min_n=0 if r.random() < 0.1 else 1)],
   [[[]], [[5]], [[1, 2]], [[-3, -1]]], errs=("ValueError",), category="refactoring", family="summarize_stats")
def summarize_stats(xs):
    if not xs:
        raise ValueError("empty")
    return {"min": min(xs), "max": max(xs), "mean": sum(xs) / len(xs)}


# ---------- debugging ----------
@T(1, "def average_positive(xs: list[int]) -> float",
   "Mean of the strictly positive numbers in xs, or 0.0 if there are none. The implementation below has bugs; write "
   "a correct version:\n\n"
   "    def average_positive(xs):\n        total = 0\n        count = 0\n        for x in xs:\n"
   "            if x >= 0:\n                total += x\n        count += 1\n        return total / count\n",
   [[[1, 2, 3]], [[-1, 4]]], lambda r: [_ints(r, 10, -5, 5)],
   [[[]], [[0]], [[-1, -2]], [[0, 2, 4]], [[3]]], category="debugging", family="average_positive")
def average_positive(xs):
    pos = [x for x in xs if x > 0]
    return sum(pos) / len(pos) if pos else 0.0


# ---------- comprensión ----------
@T(1, "def doubling_totals(xs: list[int]) -> list[int]",
   "Given this function:\n\n    def total(xs):\n        t = 0\n        for x in xs:\n            t = t * 2 + x\n"
   "        return t\n\nReturn the list of values that t takes after each iteration (one value per element of xs).",
   [[[1, 1, 1]], [[3, -2]]], lambda r: [_ints(r, 10, -5, 5)],
   [[[]], [[1]], [[0, 0]], [[-1, 2]]], category="comprension", family="doubling_totals")
def doubling_totals(xs):
    t, out = 0, []
    for x in xs:
        t = t * 2 + x
        out.append(t)
    return out


# ---------- multi-step ----------
def _g_inventory(r):
    cmds, stock = [], {}
    for _ in range(r.randint(0, 12)):
        it, u = r.choice("abc"), r.random()
        if u < 0.4:
            q = r.randint(1, 9)
            cmds.append(["add", it, q])
            stock[it] = stock.get(it, 0) + q
        elif u < 0.6 and stock.get(it, 0):
            q = r.randint(1, stock[it]) if r.random() < 0.9 else stock[it] + 1
            cmds.append(["remove", it, q])
            stock[it] = max(0, stock[it] - q)
        elif u < 0.98:
            cmds.append(["count", it])
        else:
            cmds.append([r.choice(("sell", "add")), it, 0])
    if r.random() < 0.8:
        cmds.append(["count", max(stock, key=stock.get) if stock else "a"])
    return [cmds]


@T(2, "def inventory(commands: list[list]) -> list[int]",
   "Process commands on an inventory that starts empty: ['add', item, qty] adds qty; ['remove', item, qty] removes "
   "qty; ['count', item] appends the current quantity of item (0 if never added) to the result. Return the list of "
   "counts. Raise ValueError for an unknown command, a qty < 1, or a remove of more than the current stock.",
   [[[["add", "a", 2], ["count", "a"], ["remove", "a", 1], ["count", "a"]]], [[["count", "x"]]]], _g_inventory,
   [[[]], [[["count", "x"]]], [[["add", "a", 2], ["remove", "a", 2], ["count", "a"]]], [[["remove", "a", 1]]],
    [[["add", "a", 0]]], [[["sell", "a", 1]]]], errs=("ValueError",), category="multi_step", family="inventory")
def inventory(commands):
    stock, out = {}, []
    for c in commands:
        if c[0] == "count":
            out.append(stock.get(c[1], 0))
            continue
        if c[0] not in ("add", "remove") or c[2] < 1:
            raise ValueError(c)
        if c[0] == "add":
            stock[c[1]] = stock.get(c[1], 0) + c[2]
        elif stock.get(c[1], 0) < c[2]:
            raise ValueError(c)
        else:
            stock[c[1]] -= c[2]
    return out
