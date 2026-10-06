"""Catálogo v3, split val (34 problemas, las 14 categorías). Nunca se entrena con ellos: sirven para elegir best/ y
medir durante el entrenamiento. Familias propias: ninguna está en train ni en test, y se evitaron variantes cercanas
de problemas de train (mismo algoritmo con otro envoltorio). Sesgado a niveles 0-2: con el modelo de 0.5B el val
original tenía 10 de 16 problemas a 0 y no podía medir mejoras."""
from problems.problems import P, _bal, _ints, _word


def V(level, sig, doc, ex, gen, edge=(), large=None, errs=(), *, category, family=None, slow=None):
    def deco(fn):
        return P(level, sig, doc, ex, gen, edge, large, errs, split="val", category=category,
                 family=family or fn.__name__, slow=slow, source="catalog")(fn)
    return deco


# ============================================================================================== arrays
@V(1, "def leaders(xs: list[int]) -> list[int]",
   "Elements that are strictly greater than every element to their right, in their original order (the last element "
   "is always one).",
   [[[16, 17, 4, 3, 5, 2]], [[1, 2, 3]]], lambda r: [_ints(r, 10, 0, 9)],
   [[[]], [[5]], [[3, 3]], [[5, 4, 3]], [[2, 2, 1]], [[-1, -5, -1]]], category="arrays")
def leaders(xs):
    out, best = [], None
    for x in reversed(xs):
        if best is None or x > best:
            out.append(x)
            best = x
    return out[::-1]


@V(0, "def count_rises(xs: list[int]) -> int",
   "Number of positions i >= 1 where xs[i] is strictly greater than xs[i - 1].",
   [[[1, 2, 2, 5, 3]], [[]]], lambda r: [_ints(r, 10, 0, 5)],
   [[[7]], [[1, 1, 1]], [[3, 2, 1]], [[1, 2, 3, 4]], [[-1, 0]]], category="arrays")
def count_rises(xs):
    return sum(1 for a, b in zip(xs, xs[1:]) if b > a)


@V(1, "def longest_plateau(xs: list[int]) -> int",
   "Length of the longest run of consecutive equal values (0 for an empty list).",
   [[[1, 1, 2, 2, 2, 1]], [[4]]], lambda r: [_ints(r, 12, 0, 2)],
   [[[]], [[5, 5]], [[1, 2, 3]], [[2, 2, 1, 2, 2, 2]], [[0, 0, 0, 0]]], category="arrays")
def longest_plateau(xs):
    best = cur = 0
    for i, x in enumerate(xs):
        cur = cur + 1 if i and x == xs[i - 1] else 1
        best = max(best, cur)
    return best


# ============================================================================================== strings
@V(1, "def reverse_vowels(s: str) -> str",
   "Reverse the order of the vowels (a, e, i, o, u, in either case) in s, leaving every other character in place.",
   [["hello"], ["Aeb"]], lambda r: [_word(r, 10, "aeiouAEbcdxy ")],
   [[""], ["xyz"], ["a"], ["aA"], ["Uber cool"], ["AEIOU"]], category="strings")
def reverse_vowels(s):
    vs = [c for c in s if c in "aeiouAEIOU"]
    return "".join(vs.pop() if c in "aeiouAEIOU" else c for c in s)


def _g_rot(r):
    a = _word(r, 7, "ab")
    if a and r.random() < 0.5:
        k = r.randint(0, len(a))
        return [a, a[k:] + a[:k]]
    return [a, _word(r, 7, "ab")]


@V(1, "def is_rotation(a: str, b: str) -> bool",
   "True if b can be obtained by moving some prefix of a (possibly empty) to its end ('waterbottle' -> "
   "'erbottlewat').",
   [["waterbottle", "erbottlewat"], ["abc", "acb"]], lambda r: _bal(r, is_rotation, _g_rot),
   [["", ""], ["a", ""], ["", "a"], ["aa", "aa"], ["ab", "ba"], ["abc", "abcabc"], ["aab", "abb"]], category="strings")
def is_rotation(a, b):
    return len(a) == len(b) and b in a + a


@V(1, "def remove_adjacent_pairs(s: str) -> str",
   "Repeatedly delete two adjacent equal characters until no such pair remains ('abbaca' -> 'ca').",
   [["abbaca"], ["azxxzy"]], lambda r: [_word(r, 12, "abc")],
   [[""], ["aa"], ["aaa"], ["abba"], ["abc"], ["aabbcc"]], category="strings")
def remove_adjacent_pairs(s):
    st = []
    for c in s:
        if st and st[-1] == c:
            st.pop()
        else:
            st.append(c)
    return "".join(st)


# ============================================================================================== estructuras
@V(1, "def group_by_initial(words: list[str]) -> dict",
   "Group words by their first letter, lowercased: return a dict letter -> list of the distinct words that start with "
   "that letter in either case, sorted with Python's default string order. Empty strings are ignored.",
   [[["apple", "avocado", "banana", "Apricot"]], [[]]],
   lambda r: [[_word(r, 4, "abcAB") for _ in range(r.randint(0, 7))]],
   [[[""]], [["a", "a"]], [["B", "b"]], [["xyz"]], [["", "q", ""]]], category="estructuras")
def group_by_initial(words):
    out = {}
    for w in words:
        if w:
            out.setdefault(w[0].lower(), set()).add(w)
    return {k: sorted(v) for k, v in out.items()}


def _g_chain(r):
    names = "abcdef"
    return [{k: r.choice(names) for k in r.sample(names, r.randint(0, 5))}, r.choice(names)]


@V(2, "def lookup_chain(mapping: dict, start: str) -> list[str]",
   "Follow the chain start -> mapping[start] -> mapping[mapping[start]] -> ... and return the names visited, starting "
   "with start. The chain ends at a name that is not a key of mapping (that name is included) or just before a name "
   "that was already visited (a cycle).",
   [[{"a": "b", "b": "c"}, "a"], [{"x": "y", "y": "x"}, "x"]], _g_chain,
   [[{}, "a"], [{"a": "a"}, "a"], [{"a": "b"}, "b"], [{"a": "b", "b": "c", "c": "b"}, "a"]], category="estructuras")
def lookup_chain(mapping, start):
    path, seen, cur = [start], {start}, start
    while cur in mapping:
        nxt = mapping[cur]
        if nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path


# ============================================================================================== algoritmos
@V(1, "def max_profit(prices: list[int]) -> int",
   "Best profit from buying on one day and selling on a later day; 0 if no profit is possible.",
   [[[7, 1, 5, 3, 6, 4]], [[7, 6, 4, 3, 1]]], lambda r: [_ints(r, 10, 1, 20)],
   [[[]], [[5]], [[1, 2]], [[2, 1]], [[3, 3, 3]], [[1, 9, 0, 8]]], category="algoritmos")
def max_profit(prices):
    best, low = 0, None
    for p in prices:
        if low is not None and p - low > best:
            best = p - low
        if low is None or p < low:
            low = p
    return best


@V(2, "def josephus(n: int, k: int) -> int",
   "n people stand in a circle, numbered 1..n. Starting from person 1, count k people around the circle; the k-th is "
   "removed and the count restarts at the next person. Return the number of the last person remaining. Raise "
   "ValueError if n < 1 or k < 1.",
   [[5, 2], [1, 3]], lambda r: [r.randint(-1, 12), r.randint(-1, 6)],
   [[1, 1], [2, 1], [2, 2], [7, 3], [0, 2], [3, 0], [10, 10]], errs=("ValueError",), category="algoritmos")
def josephus(n, k):
    if n < 1 or k < 1:
        raise ValueError((n, k))
    pos = 0
    for m in range(2, n + 1):
        pos = (pos + k) % m
    return pos + 1


def _g_msearch(r):
    rows, cols = r.randint(0, 4), r.randint(1, 4)
    vals = sorted(r.sample(range(0, 60), rows * cols))
    m = [vals[i * cols:(i + 1) * cols] for i in range(rows)]
    return [m, r.choice(vals) if vals and r.random() < 0.6 else r.randint(-1, 61)]


@V(1, "def matrix_search(m: list[list[int]], target: int) -> list[int]",
   "Every row of m is non-empty and sorted ascending, and every row starts with a value greater than the last value of "
   "the previous row. Return [row, col] of target, or [-1, -1] if it is not in m.",
   [[[[1, 3, 5], [7, 9, 11]], 9], [[[1, 3]], 2]], _g_msearch,
   [[[], 1], [[[5]], 5], [[[5]], 4], [[[1, 2], [3, 4]], 4], [[[1, 2], [3, 4]], 0], [[[1, 2], [3, 4]], 3]],
   category="algoritmos")
def matrix_search(m, target):
    for i, row in enumerate(m):
        if row[0] <= target <= row[-1]:
            return [i, row.index(target)] if target in row else [-1, -1]
    return [-1, -1]


# ============================================================================================== matemáticas
_ARMSTRONG = (0, 1, 5, 9, 153, 370, 371, 407, 1634, 8208, 9474, 54748, 92727, 93084)


@V(0, "def is_armstrong(n: int) -> bool",
   "True if n >= 0 is equal to the sum of its digits, each raised to the number of digits (153 = 1**3 + 5**3 + 3**3). "
   "Negative numbers are never Armstrong numbers.",
   [[153], [154]], lambda r: _bal(r, is_armstrong, lambda r: [r.choice((r.randint(-50, 10 ** 5), r.choice(_ARMSTRONG)))]),
   [[0], [1], [10], [-153], [9474], [9475], [370], [100]], category="matematicas")
def is_armstrong(n):
    if n < 0:
        return False
    ds = str(n)
    return n == sum(int(d) ** len(ds) for d in ds)


@V(1, "def trailing_zeros(n: int) -> int",
   "Number of trailing zeros of n! (n factorial). Raise ValueError if n < 0. It must be fast for n up to 10**18.",
   [[5], [3]], lambda r: [r.choice((r.randint(-2, 200), r.randint(0, 10 ** 9)))],
   [[0], [4], [10], [24], [25], [-1], [125]], lambda r: [[10 ** 18], [10 ** 18 - 1]], ("ValueError",),
   category="matematicas")
def trailing_zeros(n):
    if n < 0:
        raise ValueError(n)
    z = 0
    while n:
        n //= 5
        z += n
    return z


@V(1, "def reduce_fraction(num: int, den: int) -> list[int]",
   "Reduce num/den to lowest terms and return [num, den] with den > 0 (the sign goes in num; zero is [0, 1]). Raise "
   "ZeroDivisionError if den == 0.",
   [[6, 8], [3, -9]], lambda r: [r.randint(-30, 30), r.randint(-12, 12)],
   [[0, 5], [0, -5], [5, 0], [-4, -6], [7, 1], [1, -1]], errs=("ZeroDivisionError",), category="matematicas")
def reduce_fraction(num, den):
    from math import gcd
    if den == 0:
        raise ZeroDivisionError("den")
    if den < 0:
        num, den = -num, -den
    g = gcd(num, den)
    return [num // g, den // g]


# ============================================================================================== parsing
def _g_bool(r):
    w = r.choice(("true", "yes", "on", "1", "false", "no", "off", "0", "maybe", "", "2", "y", "nope", "tru"))
    w = "".join(c.upper() if r.random() < 0.3 else c for c in w)
    return [r.choice(("", " ", "  ")) + w + r.choice(("", " ", "\t"))]


@V(0, "def parse_bool(s: str) -> bool",
   "Parse a boolean ignoring case and surrounding whitespace: 'true', 'yes', 'on' and '1' are True; 'false', 'no', "
   "'off' and '0' are False. Raise ValueError for anything else.",
   [["Yes"], [" off "]], _g_bool,
   [["TRUE"], ["0"], [""], ["   "], ["y"], ["1 "], ["n o"], ["False"]], errs=("ValueError",), category="parsing")
def parse_bool(s):
    t = s.strip().lower()
    if t in ("true", "yes", "on", "1"):
        return True
    if t in ("false", "no", "off", "0"):
        return False
    raise ValueError(s)


def _g_hex(r):
    if r.random() < 0.7:
        return ["#" + "".join(r.choice("0123456789abcdefABCDEF") for _ in range(r.choice((3, 6))))]
    bad = r.choice(("len", "hash", "char"))
    n = r.choice((2, 4, 5, 7)) if bad == "len" else r.choice((3, 6))
    body = "".join(r.choice("0123456789abcdef") for _ in range(n))
    if bad == "char":
        i = r.randrange(n)
        body = body[:i] + r.choice("gxz -") + body[i + 1:]
    return [("" if bad == "hash" else "#") + body]


@V(1, "def hex_color(s: str) -> list[int]",
   "Parse a CSS hex color '#rrggbb', or the short form '#rgb' where each digit is doubled ('#fa0' = '#ffaa00'), "
   "case-insensitive, into [r, g, b]. Raise ValueError for anything else.",
   [["#ff8000"], ["#Fa0"]], _g_hex,
   [["#000"], ["#FFFFFF"], ["#12345"], ["123456"], ["#gg0000"], [""], ["#"], ["#1234567"], ["# 12345"]],
   errs=("ValueError",), category="parsing")
def hex_color(s):
    h = s[1:]
    if not s.startswith("#") or len(h) not in (3, 6) or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(s)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


# ============================================================================================== seguridad
@V(0, "def strip_control(s: str) -> str",
   "Remove every ASCII control character (code points 0-31 and 127) except tab and newline.",
   [["a\x00b\tc"], ["ok\r\n"]],
   lambda r: ["".join(r.choice(("a", "b", " ", "\t", "\n", "\r", "\x00", "\x1b", "\x7f", "\u00e9", "\x85"))
                      for _ in range(r.randint(0, 10)))],
   [[""], ["\x7f"], ["\t\n"], ["\x1b[31mred\x1b[0m"], ["\x80\x9f"], ["plain text"]], category="seguridad")
def strip_control(s):
    return "".join(c for c in s if c in "\t\n" or not (ord(c) < 32 or ord(c) == 127))


def _g_upload(r):
    name = r.choice(("a", "photo", "", "x/y", "doc", "a\\b"))
    exts = [r.choice(("jpg", "PNG", "php", "", "Jpg", "gif")) for _ in range(r.choice((1, 1, 1, 2)))]
    return [name + "".join("." + e for e in exts), ["jpg", "png", "gif"][:r.randint(0, 3)]]


@V(1, "def allowed_upload(filename: str, allowed: list[str]) -> bool",
   "True if the file may be uploaded: the name contains exactly one dot, the part before it is non-empty and has no "
   "'/' or '\\\\', and the extension after it is non-empty and, lowercased, is in allowed (lowercase extensions without "
   "the dot).",
   [["photo.JPG", ["jpg", "png"]], ["shell.php.jpg", ["jpg"]]], lambda r: _bal(r, allowed_upload, _g_upload),
   [["a.jpg", []], [".jpg", ["jpg"]], ["a.", ["jpg"]], ["a", ["jpg"]], ["dir/a.png", ["png"]], ["A.PNG", ["png"]],
    ["a.jpg ", ["jpg"]], ["b.gif", ["gif", "png"]], ["x.png", ["png"]], ["IMG_1.Jpg", ["jpg"]],
    ["my file.gif", ["gif"]]], category="seguridad")
def allowed_upload(filename, allowed):
    if filename.count(".") != 1:
        return False
    name, ext = filename.split(".")
    if not name or "/" in name or "\\" in name or not ext:
        return False
    return ext.lower() in allowed


# ============================================================================================== edge cases
@V(0, "def safe_int(s: str, default: int) -> int",
   "int(s) if Python's int() accepts s (surrounding spaces, a sign, underscores between digits...), otherwise default.",
   [["42", 0], ["4x", -1]],
   lambda r: [r.choice(("12", " -3 ", "+7", "", "abc", "1.5", "0x1f", "007", "1_000", "--1", "9" * 30, "_1")),
              r.randint(-5, 5)],
   [[" ", 3], ["-0", 1], ["+", 2], ["1e3", 0], ["\u0663", 9], ["1__0", 4]], category="edge_cases")
def safe_int(s, default):
    try:
        return int(s)
    except ValueError:
        return default


@V(0, "def min_max(xs: list) -> list | None",
   "[smallest, largest] of the numbers in xs, ignoring None entries; None if there are no numbers.",
   [[[3, None, 1, 2]], [[]]], lambda r: [[r.choice((None, r.randint(-9, 9), r.randint(-9, 9))) for _ in range(r.randint(0, 7))]],
   [[[None]], [[5]], [[None, -1, None]], [[0, None, 0]], [[3, -3]]], category="edge_cases")
def min_max(xs):
    ys = [x for x in xs if x is not None]
    return [min(ys), max(ys)] if ys else None


@V(0, "def last_n(xs: list, n: int) -> list",
   "The last n elements of xs in their order: [] if n <= 0 and the whole list if n > len(xs).",
   [[[1, 2, 3, 4], 2], [[1, 2], 5]], lambda r: [_ints(r, 6, 0, 9), r.randint(-2, 8)],
   [[[], 0], [[], 3], [[1, 2, 3], 0], [[1, 2, 3], -1], [[1, 2, 3], 3]], category="edge_cases")
def last_n(xs, n):
    return xs[-n:] if n > 0 else []


# ============================================================================================== optimización
def _slow_next_greater(xs):
    out = []
    for i, x in enumerate(xs):
        nxt = -1
        for y in xs[i + 1:]:
            if y > x:
                nxt = y
                break
        out.append(nxt)
    return out


@V(2, "def next_greater(xs: list[int]) -> list[int]",
   "For each element, the first element to its right that is strictly greater, or -1 if there is none. It must be "
   "fast for lists of hundreds of thousands of numbers.",
   [[[2, 1, 3]], [[5, 4]]], lambda r: [_ints(r, 12, 0, 9)],
   [[[]], [[1]], [[1, 1]], [[1, 2, 3]], [[3, 2, 1]], [[2, 7, 2, 7]]],
   lambda r: [[list(range(200000, 0, -1)) + [200001]], [[r.randint(0, 10 ** 6) for _ in range(200000)]]],
   category="optimizacion", slow=_slow_next_greater)
def next_greater(xs):
    out, st = [-1] * len(xs), []
    for i, x in enumerate(xs):
        while st and xs[st[-1]] < x:
            out[st.pop()] = x
        st.append(i)
    return out


def _slow_range_sums(xs, queries):
    return [sum(xs[i:j + 1]) for i, j in queries]


def _g_rsum(r):
    xs = _ints(r, 8, -9, 9, min_n=1)
    qs = []
    for _ in range(r.randint(0, 4)):
        i = r.randrange(len(xs))
        qs.append([i, r.randrange(i, len(xs))])
    return [xs, qs]


def _l_rsum(r):
    n = 100000
    return [[[r.randint(-100, 100) for _ in range(n)], [[r.randint(0, 50), r.randint(n - 50, n - 1)] for _ in range(n)]]]


@V(1, "def range_sums(xs: list[int], queries: list[list[int]]) -> list[int]",
   "For each query [i, j] (0 <= i <= j < len(xs)) the sum of xs[i..j], both ends included. It must be fast for 10**5 "
   "numbers and 10**5 queries.",
   [[[1, 2, 3, 4], [[0, 1], [1, 3]]], [[5], [[0, 0]]]], _g_rsum,
   [[[1], []], [[0, 0], [[0, 1]]], [[-5, 5], [[0, 1], [1, 1], [0, 0]]]], _l_rsum,
   category="optimizacion", slow=_slow_range_sums)
def range_sums(xs, queries):
    pre = [0]
    for x in xs:
        pre.append(pre[-1] + x)
    return [pre[j + 1] - pre[i] for i, j in queries]


# ============================================================================================== refactoring
@V(1, "def ticket_price(age: int, student: bool, weekday: bool) -> int",
   "Rewrite this function as ticket_price(age, student, weekday) with flat, readable logic and exactly the same "
   "behaviour:\n\n"
   "    def p(a, s, w):\n        if a < 0:\n            raise ValueError('age')\n        if a < 3:\n            return 0\n"
   "        else:\n            if a < 12:\n                price = 6\n            else:\n                if a >= 65:\n"
   "                    price = 7\n                else:\n                    if s == True:\n"
   "                        price = 8\n                    else:\n                        price = 12\n"
   "        if w == True:\n            if price > 6:\n                price = price - 2\n        return price\n",
   [[30, False, True], [70, True, False]],
   lambda r: [r.choice((r.randint(-1, 90), r.choice((2, 3, 11, 12, 64, 65)))), r.random() < 0.5, r.random() < 0.5],
   [[0, False, False], [3, True, True], [12, True, True], [65, False, True], [-5, False, False], [64, False, False]],
   errs=("ValueError",), category="refactoring")
def ticket_price(age, student, weekday):
    if age < 0:
        raise ValueError("age")
    if age < 3:
        return 0
    if age < 12:
        price = 6
    elif age >= 65:
        price = 7
    else:
        price = 8 if student else 12
    return price - 2 if weekday and price > 6 else price


@V(1, "def discount_price(price: float, member: bool, coupon: str) -> float",
   "Rewrite this function as discount_price(price, member, coupon) with clear code and exactly the same results:\n\n"
   "    def d(p, m, c):\n        if p < 0:\n            raise ValueError('price')\n        r = p\n        if m:\n"
   "            r = r * 0.9\n        if c == 'HALF':\n            if p >= 20:\n                r = r * 0.5\n"
   "        else:\n            if c == 'FIVE':\n                if r > 5:\n                    r = r - 5\n"
   "                else:\n                    r = 0\n        if r < 0:\n            r = 0\n        return round(r, 2)\n",
   [[100.0, True, "HALF"], [4.0, False, "FIVE"]],
   lambda r: [r.choice((round(r.uniform(0, 60), 2), r.randint(0, 40), -1)), r.random() < 0.5,
              r.choice(("", "HALF", "FIVE", "half", "TEN"))],
   [[20, False, "HALF"], [19.99, True, "HALF"], [5, False, "FIVE"], [5.5, True, "FIVE"], [0, True, ""],
    [-0.01, False, ""], [21, True, "FIVE"]], errs=("ValueError",), category="refactoring")
def discount_price(price, member, coupon):
    if price < 0:
        raise ValueError("price")
    r = price * 0.9 if member else price
    if coupon == "HALF":
        if price >= 20:
            r *= 0.5
    elif coupon == "FIVE":
        r = r - 5 if r > 5 else 0
    return round(r, 2)


# ============================================================================================== debugging
@V(0, "def is_sorted(xs: list[int]) -> bool",
   "True if xs is in non-decreasing order. The implementation below has bugs; write a correct version:\n\n"
   "    def is_sorted(xs):\n        for i in range(len(xs) - 2):\n            if xs[i] >= xs[i + 1]:\n"
   "                return False\n        return True\n",
   [[[1, 2, 2, 3]], [[3, 1]]],
   lambda r: _bal(r, is_sorted, lambda r: [sorted(_ints(r, 6, 0, 5)) if r.random() < 0.5 else _ints(r, 6, 0, 5)]),
   [[[]], [[1]], [[2, 2]], [[1, 3, 2]], [[1, 2, 3, 0]], [[-1, -1, 0]], [[2, 1]], [[0, 0, -1]], [[5, 1, 2, 3]],
    [[0, 1, 1, 2]], [[-3, 5]], [[7, 7, 7]]], category="debugging")
def is_sorted(xs):
    return all(a <= b for a, b in zip(xs, xs[1:]))


@V(1, "def swap_pairs(xs: list) -> list",
   "Return a new list with the elements swapped in pairs: [1, 2, 3, 4, 5] -> [2, 1, 4, 3, 5] (a last unpaired element "
   "stays where it is). The implementation below has bugs; write a correct version:\n\n"
   "    def swap_pairs(xs):\n        for i in range(0, len(xs), 2):\n            xs[i], xs[i + 1] = xs[i + 1], xs[i]\n"
   "        return xs\n",
   [[[1, 2, 3, 4, 5]], [[]]], lambda r: [_ints(r, 9, 0, 9)],
   [[[1]], [[1, 2]], [["a", "b", "c"]], [[0, 0, 1]]], category="debugging")
def swap_pairs(xs):
    out = list(xs)
    for i in range(0, len(out) - 1, 2):
        out[i], out[i + 1] = out[i + 1], out[i]
    return out


# ============================================================================================== comprensión
@V(1, "def trace_digits(n: int) -> int",
   "Given this code:\n\n    r = 0\n    while n > 0:\n        d = n % 10\n        if d % 2 == 0:\n"
   "            r = r * 10 + d\n        n = n // 10\n\nReturn the final value of r for the given n.",
   [[1234], [0]], lambda r: [r.choice((r.randint(-50, 10 ** 6), r.randint(0, 10 ** 12)))],
   [[-24], [8], [13579], [2020], [10 ** 15]], category="comprension")
def trace_digits(n):
    r = 0
    while n > 0:
        d = n % 10
        if d % 2 == 0:
            r = r * 10 + d
        n //= 10
    return r


@V(2, "def subtract_steps(a: int, b: int) -> int",
   "Given this code:\n\n    steps = 0\n    while a != b:\n        if a > b:\n            a = a - b\n        else:\n"
   "            b = b - a\n        steps += 1\n\nReturn the final value of steps. Raise ValueError if a < 1 or b < 1. "
   "It must be fast even when a and b are as large as 10**18.",
   [[6, 4], [5, 5]], lambda r: [r.randint(-1, 60), r.randint(-1, 60)],
   [[1, 1], [1, 7], [7, 1], [0, 3], [12, 18], [10 ** 6, 3]],
   lambda r: [[10 ** 18, 1], [1, 10 ** 18 - 1], [10 ** 18 + 7, 3]], ("ValueError",), category="comprension")
def subtract_steps(a, b):
    if a < 1 or b < 1:
        raise ValueError((a, b))
    steps = 0
    while b:
        steps += a // b
        a, b = b, a % b
    return steps - 1


# ============================================================================================== especificación
@V(2, "def int_to_words(n: int) -> str",
   "English words for 0 <= n <= 999, lowercase: 0 -> 'zero', 21 -> 'twenty-one', 100 -> 'one hundred', 115 -> 'one "
   "hundred fifteen', 342 -> 'three hundred forty-two' (no 'and'). Raise ValueError outside that range.",
   [[42], [900]], lambda r: [r.choice((r.randint(-5, 1005), r.randint(0, 99)))],
   [[0], [10], [11], [19], [20], [99], [101], [110], [999], [1000], [-1]], errs=("ValueError",),
   category="especificacion")
def int_to_words(n):
    if not 0 <= n <= 999:
        raise ValueError(n)
    ones = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen").split()
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

    def two(k):
        return ones[k] if k < 20 else tens[k // 10] + ("-" + ones[k % 10] if k % 10 else "")
    if n < 100:
        return two(n)
    h, rest = divmod(n, 100)
    return ones[h] + " hundred" + (" " + two(rest) if rest else "")


def _g_phrase(r):
    out = ""
    for _ in range(r.randint(0, 6)):
        out += _word(r, 5, "abcXY\u00e9") + r.choice((" ", "-", "  ", " - ", ""))
    return [out]


@V(0, "def abbreviate(phrase: str) -> str",
   "Acronym of a phrase: the first letter of each word, uppercased. Words are separated by spaces and/or hyphens; "
   "empty words are ignored ('Portable network-graphics' -> 'PNG').",
   [["Portable network-graphics"], ["as soon  as possible"]], _g_phrase,
   [[""], ["-"], ["a"], ["  hello   world "], ["self-contained underwater-breathing apparatus"], ["\u00e9lan vital"]],
   category="especificacion")
def abbreviate(phrase):
    return "".join(w[0].upper() for w in phrase.replace("-", " ").split())


@V(1, "def time_ago(seconds: int) -> str",
   "Human-readable age: under 60 -> 'just now'; under 3600 -> '<m> minutes ago' with m = seconds // 60; under 86400 -> "
   "'<h> hours ago' with h = seconds // 3600; otherwise '<d> days ago' with d = seconds // 86400. Use the singular when "
   "the number is 1 ('1 minute ago'). Raise ValueError if seconds < 0.",
   [[30], [7200]],
   lambda r: [r.choice((r.randint(-5, 59), r.randint(60, 3599), r.randint(3600, 86399), r.randint(86400, 10 ** 7),
                        r.choice((60, 119, 120, 3600, 7199, 86400, 172799))))],
   [[0], [59], [60], [119], [3599], [3600], [86399], [86400], [172800], [-1]], errs=("ValueError",),
   category="especificacion")
def time_ago(seconds):
    if seconds < 0:
        raise ValueError(seconds)
    if seconds < 60:
        return "just now"
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            n = seconds // size
            return f"{n} {unit}{'' if n == 1 else 's'} ago"


# ============================================================================================== multi-paso
@V(2, "def elevator_trip(start: int, requests: list[int]) -> list[int]",
   "An elevator at floor start serves the requested floors in order, skipping a request for the floor where it "
   "already is. Return [total floors travelled, number of direction changes] (a change is a move up after a move down "
   "or vice versa).",
   [[0, [5, 2, 8]], [3, [3]]], lambda r: [r.randint(0, 9), [r.randint(0, 9) for _ in range(r.randint(0, 6))]],
   [[0, []], [2, [2, 2]], [0, [1, 2, 3]], [5, [0, 9, 0]], [4, [4, 6, 6, 1]], [1, [3, 3, 2, 2, 5]]],
   category="multi_step")
def elevator_trip(start, requests):
    pos, dist, changes, last = start, 0, 0, 0
    for f in requests:
        if f == pos:
            continue
        d = 1 if f > pos else -1
        if last and d != last:
            changes += 1
        dist += abs(f - pos)
        pos, last = f, d
    return [dist, changes]


def _g_todo(r):
    tasks = ("buy milk", "call bo", "gym", "pay rent")
    out = []
    for _ in range(r.randint(0, 8)):
        u = r.random()
        if u < 0.45:
            out.append("add " + r.choice(tasks))
        elif u < 0.75:
            out.append("done " + r.choice(tasks))
        elif u < 0.95:
            out.append("undo")
        else:
            out.append(r.choice(("list", "add", "redo")))
    return [out]


@V(2, "def todo_commands(cmds: list[str]) -> list[str]",
   "Run commands on a to-do list that starts empty and return the final list of pending tasks in insertion order. "
   "'add <task>' appends the task (everything after the first space; must be non-empty) unless it is already pending; "
   "'done <task>' removes it if it is pending; 'undo' restores the list as it was before the most recent add or done "
   "that changed it (repeated undos go further back; nothing happens if there is nothing to undo). Any other command "
   "is ignored.",
   [[["add buy milk", "add gym", "done buy milk"]], [["add a", "undo", "undo"]]], _g_todo,
   [[[]], [["undo"]], [["add a", "add a", "undo"]], [["add a", "done a", "undo"]],
    [["add a", "add b", "done a", "undo", "undo"]], [["done x", "undo"]], [["add a b", "add", "done a"]]],
   category="multi_step")
def todo_commands(cmds):
    tasks, history = [], []
    for c in cmds:
        op, _, arg = c.partition(" ")
        if op == "add" and arg and arg not in tasks:
            history.append(list(tasks))
            tasks.append(arg)
        elif op == "done" and arg in tasks:
            history.append(list(tasks))
            tasks.remove(arg)
        elif c == "undo" and history:
            tasks = history.pop()
    return tasks
