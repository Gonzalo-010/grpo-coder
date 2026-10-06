"""Catálogo, split val (16 problemas, las 14 categorías). Nunca se entrena con ellos: sirven para elegir best/ y
para el currículo. Familias propias, distintas de las de train y test."""
from problems.problems import P, _ints, _word


def V(level, sig, doc, ex, gen, edge=(), large=None, errs=(), *, category, family, slow=None):
    return P(level, sig, doc, ex, gen, edge, large, errs, split="val", category=category, family=family, slow=slow,
             source="catalog")


# ---------- algoritmos ----------
def _g_topo(r):
    n = r.randint(0, 8)
    order = list(range(n))
    r.shuffle(order)
    deps = []
    for _ in range(r.randint(0, n + 2) if n > 1 else 0):
        i, j = sorted(r.sample(range(n), 2))
        deps.append([order[i], order[j]])
    if deps and r.random() < 0.2:
        a, b = deps[0]
        deps.append([b, a])  # ciclo
    return [n, deps]


def _l_topo(r):
    n = 50000
    order = list(range(n))
    r.shuffle(order)
    deps = []
    for _ in range(100000):
        i, j = sorted(r.sample(range(n), 2))
        deps.append([order[i], order[j]])
    return [[n, deps]]


@V(3, "def topo_order(n: int, deps: list[list[int]]) -> list[int]",
   "Tasks 0..n-1 and dependencies [a, b] meaning that a must be done before b. Return an order of all the tasks that "
   "respects every dependency, choosing at each step the smallest task that is available. Raise ValueError if the "
   "dependencies contain a cycle.",
   [[4, [[1, 0], [2, 0]]], [2, [[0, 1], [1, 0]]]], _g_topo,
   [[0, []], [3, []], [3, [[2, 0]]], [2, [[0, 1], [1, 0]]], [1, [[0, 0]]]], _l_topo, ("ValueError",),
   category="algoritmos", family="topo_order")
def topo_order(n, deps):
    import heapq
    indeg, adj = [0] * n, [[] for _ in range(n)]
    for a, b in deps:
        adj[a].append(b)
        indeg[b] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        x = heapq.heappop(heap)
        out.append(x)
        for y in adj[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                heapq.heappush(heap, y)
    if len(out) != n:
        raise ValueError("cycle")
    return out


def _g_grid(r):
    rows, cols = r.randint(1, 6), r.randint(1, 6)
    g = [["#" if r.random() < 0.25 else "." for _ in range(cols)] for _ in range(rows)]
    if r.random() < 0.85:  # casi siempre inicio y fin libres: la respuesta es una distancia, no siempre -1
        g[0][0] = g[-1][-1] = "."
    return [["".join(row) for row in g]]


def _l_grid(r):
    return [[["".join("#" if r.random() < 0.2 else "." for _ in range(300)) for _ in range(300)]]]


@V(3, "def grid_shortest(grid: list[str]) -> int",
   "grid is a list of equal-length strings of '.' (free) and '#' (wall). Return the minimum number of moves "
   "(up, down, left, right) from the top-left cell to the bottom-right cell, or -1 if it is impossible, if either of "
   "those cells is a wall, or if the grid is empty.",
   [[["..", ".."]], [[".#", "#."]]], _g_grid,
   [[[]], [["."]], [["#"]], [[".#", ".."]], [["...", "##.", "..."]]], _l_grid, category="algoritmos",
   family="grid_shortest")
def grid_shortest(grid):
    from collections import deque
    if not grid or not grid[0]:
        return -1
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == "#" or grid[rows - 1][cols - 1] == "#":
        return -1
    dist = {(0, 0): 0}
    q = deque([(0, 0)])
    while q:
        r, c = q.popleft()
        if (r, c) == (rows - 1, cols - 1):
            return dist[(r, c)]
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "." and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return -1


# ---------- arrays ----------
@V(1, "def move_zeros(xs: list[int]) -> list[int]",
   "Return a new list with every 0 moved to the end, keeping the relative order of the other elements.",
   [[[0, 1, 0, 3, 12]], [[1, 2]]], lambda r: [_ints(r, 12, -3, 3)],
   [[[]], [[0]], [[0, 0, 1]], [[1, 2]], [[0, 1, 0, 2]]],
   lambda r: [[[r.choice((0, 0, r.randint(-9, 9))) for _ in range(200000)]]], category="arrays", family="move_zeros")
def move_zeros(xs):
    nz = [x for x in xs if x != 0]
    return nz + [0] * (len(xs) - len(nz))


# ---------- strings ----------
def _g_wrap(r):
    words = [_word(r, 7, "abcde") or "a" for _ in range(r.randint(0, 8))]
    text = "".join(w + r.choice((" ", " ", "  ")) for w in words).strip()
    return [text, r.randint(0, 12) if r.random() < 0.1 else r.randint(1, 12)]


@V(2, "def word_wrap(text: str, width: int) -> list[str]",
   "Greedily wrap the words of text (separated by one or more spaces) into lines of at most width characters, words "
   "joined by a single space. A word longer than width goes alone on its own line. Return the list of lines. Raise "
   "ValueError if width < 1.",
   [["the quick brown fox", 10], ["a bb ccc", 3]], _g_wrap,
   [["", 5], ["a b c", 1], ["abcdef", 3], ["aa bb cc", 5], ["x", 0]], errs=("ValueError",), category="strings",
   family="word_wrap")
def word_wrap(text, width):
    if width < 1:
        raise ValueError(width)
    lines, cur = [], ""
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ---------- estructuras de datos ----------
def _ops_queue(r, n):
    ops, size = [], 0
    for _ in range(n):
        u = r.random()
        if u < 0.45 or size == 0 and u < 0.9:
            ops.append(["enq", r.randint(-9, 9)])
            size += 1
        else:
            op = r.choice(("deq", "peek", "size"))
            ops.append([op])
            size -= op == "deq"
    return ops


@V(2, "def queue_ops(ops: list[list]) -> list[int]",
   "Simulate a FIFO queue with ['enq', x], ['deq'], ['peek'] and ['size']. Return, in order, the values produced by "
   "deq, peek and size. deq or peek on an empty queue raise IndexError.",
   [[[["enq", 1], ["enq", 2], ["deq"], ["peek"], ["size"]]], [[["size"]]]], lambda r: [_ops_queue(r, r.randint(0, 15))],
   [[[]], [[["deq"]]], [[["size"]]], [[["enq", 5], ["deq"], ["peek"]]]], lambda r: [[_ops_queue(r, 200000)]],
   ("IndexError",), category="estructuras", family="queue_ops")
def queue_ops(ops):
    from collections import deque
    q, out = deque(), []
    for op in ops:
        if op[0] == "enq":
            q.append(op[1])
        elif op[0] == "size":
            out.append(len(q))
        elif not q:
            raise IndexError(op[0])
        elif op[0] == "deq":
            out.append(q.popleft())
        else:
            out.append(q[0])
    return out


# ---------- matemáticas ----------
def _roman(n):
    out = ""
    for v, s in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                 (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out += s
            n -= v
    return out


@V(1, "def roman_to_int(s: str) -> int",
   "Convert a Roman numeral written with I, V, X, L, C, D, M (a smaller symbol before a larger one is subtracted, as "
   "in IV or CM) to an integer. Raise ValueError if s is empty or contains any other character.",
   [["MCMXCIV"], ["III"]],
   lambda r: [_roman(r.randint(1, 3999)) if r.random() < 0.85 else r.choice(("", "iv", "XA", "M M", "IIII"))],
   [[""], ["III"], ["IV"], ["MCMXCIV"], ["iv"], ["IIII"], ["MMMCMXCIX"]], errs=("ValueError",),
   category="matematicas", family="roman_to_int")
def roman_to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not s or any(c not in vals for c in s):
        raise ValueError(s)
    total = 0
    for i, c in enumerate(s):
        v = vals[c]
        total += -v if i + 1 < len(s) and vals[s[i + 1]] > v else v
    return total


@V(1, "def pascal_row(k: int) -> list[int]",
   "Row k (0-indexed) of Pascal's triangle. Raise ValueError if k < 0.",
   [[4], [0]], lambda r: [r.randint(-1, 30)], [[0], [1], [2], [-1], [10]], errs=("ValueError",),
   category="matematicas", family="pascal_row")
def pascal_row(k):
    if k < 0:
        raise ValueError(k)
    row = [1]
    for i in range(k):
        row = [1] + [row[j] + row[j + 1] for j in range(i)] + [1]
    return row


# ---------- parsing ----------
def _g_kv(r):
    segs = []
    for _ in range(r.randint(0, 4)):
        k = r.choice("abc")
        v = r.choice((str(r.randint(-20, 99)), _word(r, 3, "xyz"), "", "-", "1a"))
        segs.append(r.choice(("", " ")) + k + r.choice(("", " ")) + "=" + r.choice(("", " ")) + v)
    if r.random() < 0.15:
        segs.append(r.choice(("bad", "=1", " ")))
    return [";".join(segs) + r.choice(("", ";"))]


@V(1, "def parse_kv(s: str) -> dict",
   "Parse text like 'a=1;b=x' into a dict. Segments are separated by ';' and empty (or blank) segments are ignored. "
   "Keys and values are stripped of surrounding spaces. A value made only of digits, optionally with a leading '-', "
   "becomes an int; any other value stays a string. A later key overwrites an earlier one. Raise ValueError for a "
   "segment without '=' or with an empty key.",
   [["a=1;b=x"], ["k=-5;k=7"]], _g_kv,
   [[""], ["a=1"], ["a=1;a=2"], ["a"], ["=1"], [" a = -5 ; b = x "], ["a=-"], ["a=;;b=2;"]], errs=("ValueError",),
   category="parsing", family="parse_kv")
def parse_kv(s):
    out = {}
    for seg in s.split(";"):
        if not seg.strip():
            continue
        if "=" not in seg:
            raise ValueError(seg)
        k, v = seg.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(seg)
        body = v[1:] if v.startswith("-") else v
        out[k] = int(v) if body and all(c in "0123456789" for c in body) else v
    return out


# ---------- seguridad ----------
def _g_join(r):
    segs = [r.choice(("a", "b", "..", ".", "", "c", "docs")) for _ in range(r.randint(0, 6))]
    rel = "/".join(segs)
    if r.random() < 0.1:
        rel = "/" + rel
    return [r.choice(("/srv/app", "/data", "/home/u/www")), rel]


@V(2, "def safe_join(base: str, rel: str) -> str",
   "Join a relative path rel to the directory base (paths use '/'; base is absolute and normalized, without a "
   "trailing '/'). Ignore empty and '.' segments and resolve '..' segments. Raise ValueError if rel is absolute "
   "(starts with '/') or if the result would leave base. Return base + '/' + the normalized relative part, or base "
   "itself when that part is empty.",
   [["/srv/app", "img/../css/site.css"], ["/srv/app", "../etc/passwd"]], _g_join,
   [["/srv", ""], ["/srv", "a/./b"], ["/srv", "a/../b"], ["/srv", "../etc"], ["/srv", "/etc"],
    ["/srv", "a/b/../../.."], ["/srv", "a//b/"]], errs=("ValueError",), category="seguridad", family="safe_join")
def safe_join(base, rel):
    if rel.startswith("/"):
        raise ValueError(rel)
    parts = []
    for seg in rel.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                raise ValueError(rel)
            parts.pop()
        else:
            parts.append(seg)
    return base + "/" + "/".join(parts) if parts else base


# ---------- edge cases ----------
@V(1, "def median(xs: list[int]) -> float",
   "Median of xs: the middle element of the sorted values, or the mean of the two middle ones when the length is "
   "even. Raise ValueError if xs is empty.",
   [[[3, 1, 2]], [[4, 1, 3, 2]]], lambda r: [_ints(r, 9, -10, 10)],
   [[[]], [[3]], [[1, 2]], [[-1, -1, 5, 5]], [[7, 7, 7]]], errs=("ValueError",), category="edge_cases",
   family="median")
def median(xs):
    if not xs:
        raise ValueError("empty")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ---------- optimización ----------
def _slow_max_subarray(xs):
    if not xs:
        raise ValueError("empty")
    best = xs[0]
    for i in range(len(xs)):
        acc = 0
        for j in range(i, len(xs)):
            acc += xs[j]
            best = max(best, acc)
    return best


@V(2, "def max_subarray(xs: list[int]) -> int",
   "Largest sum of a non-empty contiguous run of xs. Raise ValueError if xs is empty. It must be fast for lists of "
   "tens of thousands of numbers.",
   [[[-2, 1, -3, 4, -1, 2, 1, -5, 4]], [[-3, -1]]], lambda r: [_ints(r, 12, -9, 9)],
   [[[]], [[-3]], [[-2, -1]], [[1, -2, 3]], [[2, -1, 2]]],
   lambda r: [[[r.randint(-100, 100) for _ in range(30000)]]], ("ValueError",), category="optimizacion",
   family="max_subarray", slow=_slow_max_subarray)
def max_subarray(xs):
    if not xs:
        raise ValueError("empty")
    best = cur = xs[0]
    for x in xs[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


# ---------- refactoring ----------
@V(1, "def grade(score: int) -> str",
   "Rewrite this function as grade(score) with simple, flat logic and exactly the same behaviour:\n\n"
   "    def g(s):\n        if s >= 90:\n            r = 'A'\n        else:\n            if s >= 80:\n                r = 'B'\n"
   "            else:\n                if s >= 65:\n                    r = 'C'\n                else:\n"
   "                    r = 'F'\n        if s > 100 or s < 0:\n            raise ValueError('out of range')\n"
   "        return r\n",
   [[95], [70]], lambda r: [r.randint(-10, 110)],
   [[90], [89], [80], [79], [65], [64], [0], [100], [101], [-1]], errs=("ValueError",), category="refactoring",
   family="grade")
def grade(score):
    if score > 100 or score < 0:
        raise ValueError("out of range")
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 65 else "F"


# ---------- debugging ----------
def _g_nested(r):
    out = []
    for _ in range(r.randint(0, 6)):
        u = r.random()
        out.append(r.randint(-5, 5) if u < 0.4 else [r.randint(-5, 5) for _ in range(r.randint(0, 3))] if u < 0.85
                   else [[r.randint(0, 3)], r.randint(0, 3)])
    return [out]


@V(1, "def flatten_once(xs: list) -> list",
   "Flatten one level: elements that are lists are replaced by their elements, other elements are kept as they are. "
   "The implementation below has a bug; write a correct version:\n\n"
   "    def flatten_once(xs):\n        out = []\n        for x in xs:\n            if isinstance(x, list):\n"
   "                out.append(x)\n            else:\n                out.extend(x)\n        return out\n",
   [[[1, [2, 3], 4]], [[[1], [2, [3]]]]], _g_nested,
   [[[]], [[[]]], [[1, [2, 3]]], [[[1], [2, [3]]]], [[5]]], category="debugging", family="flatten_once")
def flatten_once(xs):
    out = []
    for x in xs:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


# ---------- comprensión ----------
@V(1, "def collatz_peak(n: int) -> int",
   "Given this function:\n\n    def g(n):\n        steps = 0\n        while n != 1:\n"
   "            n = n // 2 if n % 2 == 0 else 3 * n + 1\n            steps += 1\n        return steps\n\n"
   "Return the largest value that n takes during g(n), including its starting value. Raise ValueError if n < 1.",
   [[3], [1]], lambda r: [r.randint(-2, 3000)], [[1], [2], [3], [0], [27], [-5]], errs=("ValueError",),
   category="comprension", family="collatz_peak")
def collatz_peak(n):
    if n < 1:
        raise ValueError(n)
    best = n
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        best = max(best, n)
    return best


# ---------- especificación ----------
@V(2, "def two_sum_closest(nums: list[int], target: int) -> list[int]",
   "Return [i, j] with i < j such that nums[i] + nums[j] is as close as possible to target. Ties: prefer the smaller "
   "sum, then the smaller i, then the smaller j. Raise ValueError if nums has fewer than 2 numbers.",
   [[[1, 3, 5], 4], [[-1, 1, 3], 1]], lambda r: [_ints(r, 10, -10, 10), r.randint(-15, 15)],
   [[[1], 0], [[1, 2], 0], [[1, 3, 5], 4], [[-1, 1, 3], 1], [[2, 2, 2], 4]], errs=("ValueError",),
   category="especificacion", family="two_sum_closest")
def two_sum_closest(nums, target):
    if len(nums) < 2:
        raise ValueError("need two numbers")
    best = None
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            key = (abs(s - target), s, i, j)
            if best is None or key < best:
                best = key
    return [best[2], best[3]]


# ---------- multi-step ----------
def _g_rpn(r):
    def expr(d):
        if d == 0 or r.random() < 0.35:
            return [str(r.randint(-9, 9))]
        return expr(d - 1) + expr(d - 1) + [r.choice("+-*/")]
    toks = expr(r.randint(0, 3))
    u = r.random()
    if u < 0.1:
        toks = toks + [str(r.randint(0, 9))]
    elif u < 0.18:
        toks = toks[:-1] if len(toks) > 1 else ["+"]
    elif u < 0.22:
        toks = toks + ["x"]
    return [toks]


@V(2, "def rpn_eval(tokens: list[str]) -> int",
   "Evaluate a Reverse Polish Notation expression given as tokens: integers written as strings (possibly negative) "
   "and the operators + - * /. Division truncates toward zero and dividing by zero raises ZeroDivisionError. Raise "
   "ValueError if the expression is malformed (an operator without two operands, operands left over, an unknown "
   "token, or no tokens at all).",
   [[["2", "3", "+", "4", "*"]], [["7", "-2", "/"]]], _g_rpn,
   [[[]], [["3"]], [["2", "3", "+"]], [["7", "-2", "/"]], [["1", "0", "/"]], [["1", "+"]], [["1", "2"]]],
   errs=("ValueError", "ZeroDivisionError"), category="multi_step", family="rpn_eval")
def rpn_eval(tokens):
    st = []
    for t in tokens:
        if t in ("+", "-", "*", "/"):
            if len(st) < 2:
                raise ValueError("missing operand")
            b, a = st.pop(), st.pop()
            if t == "+":
                st.append(a + b)
            elif t == "-":
                st.append(a - b)
            elif t == "*":
                st.append(a * b)
            else:
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                q = abs(a) // abs(b)
                st.append(q if (a >= 0) == (b >= 0) else -q)
        else:
            try:
                st.append(int(t))
            except ValueError:
                raise ValueError(t) from None
    if len(st) != 1:
        raise ValueError("leftover operands")
    return st[0]
