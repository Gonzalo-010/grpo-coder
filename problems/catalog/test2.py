"""Catálogo v3, split test (34 problemas, las 14 categorías). Ni se entrena ni se usa para elegir nada: sólo
`train.py eval --split test` al final. Familias propias: ninguna está en train ni en val, y se evitaron variantes
cercanas de problemas de train (mismo algoritmo con otro envoltorio). Sesgado a niveles 0-2, como val2."""
from problems.problems import P, _bal, _ints, _word


def S(level, sig, doc, ex, gen, edge=(), large=None, errs=(), *, category, family=None, slow=None):
    def deco(fn):
        return P(level, sig, doc, ex, gen, edge, large, errs, split="test", category=category,
                 family=family or fn.__name__, slow=slow, source="catalog")(fn)
    return deco


# ============================================================================================== arrays
@S(1, "def sign_changes(xs: list[int]) -> int",
   "Number of times the sign changes between consecutive non-zero elements (zeros are skipped: [1, 0, -1] has one "
   "change).",
   [[[1, -2, 0, -3, 4]], [[0, 0]]], lambda r: [_ints(r, 10, -3, 3)],
   [[[]], [[5]], [[-1, 1]], [[0, -1, 0, 1, 0]], [[2, 3, 4]], [[-1, -2, 3, -4]]], category="arrays")
def sign_changes(xs):
    n, prev = 0, 0
    for x in xs:
        if x:
            if prev and (x > 0) != (prev > 0):
                n += 1
            prev = x
    return n


@S(1, "def max_gap(xs: list[int]) -> int",
   "Largest difference between two consecutive values of sorted(xs); 0 if xs has fewer than 2 elements.",
   [[[3, 6, 9, 1]], [[10]]], lambda r: [_ints(r, 8, -20, 20)],
   [[[]], [[5, 5]], [[1, 100]], [[-5, 5, 0]], [[7, 7, 7, 8]]], category="arrays")
def max_gap(xs):
    ys = sorted(xs)
    return max((b - a for a, b in zip(ys, ys[1:])), default=0)


def _g_equil(r):
    if r.random() < 0.5:
        left, right = _ints(r, 4, -5, 5), _ints(r, 3, -5, 5)
        return [left + [r.randint(-5, 5)] + right + [sum(left) - sum(right)]]
    return [_ints(r, 7, -5, 5)]


@S(1, "def equilibrium_index(xs: list[int]) -> int",
   "Smallest index i such that the sum of the elements before i equals the sum of the elements after i; -1 if there "
   "is none.",
   [[[1, 7, 3, 6, 5, 6]], [[1, 2, 3]]], _g_equil,
   [[[]], [[0]], [[5]], [[1, -1, 0]], [[0, 0, 0]], [[2, 1, -1]]], category="arrays")
def equilibrium_index(xs):
    total, left = sum(xs), 0
    for i, x in enumerate(xs):
        if left == total - left - x:
            return i
        left += x
    return -1


# ============================================================================================== strings
@S(0, "def count_overlapping(s: str, sub: str) -> int",
   "Number of occurrences of sub in s, counting overlapping ones ('aaa' contains 'aa' twice). Raise ValueError if sub "
   "is empty.",
   [["aaaa", "aa"], ["abc", "d"]], lambda r: [_word(r, 10, "ab"), _word(r, 3, "ab")],
   [["", "a"], ["a", "aa"], ["abab", "ab"], ["aaa", "a"], ["x", ""], ["ababa", "aba"]], errs=("ValueError",),
   category="strings")
def count_overlapping(s, sub):
    if not sub:
        raise ValueError("empty")
    return sum(1 for i in range(len(s) - len(sub) + 1) if s[i:i + len(sub)] == sub)


@S(0, "def alternate_case(s: str) -> str",
   "Make the letters alternate between upper and lower case, starting with upper; other characters are kept and do "
   "not count ('hello world' -> 'HeLlO wOrLd').",
   [["hello world"], ["a1b2"]], lambda r: [_word(r, 10, "abcXY 1!")],
   [[""], ["123"], ["AAAA"], ["a b c d"], ["\u00c9a"]], category="strings")
def alternate_case(s):
    out, up = [], True
    for c in s:
        if c.isalpha():
            out.append(c.upper() if up else c.lower())
            up = not up
        else:
            out.append(c)
    return "".join(out)


@S(0, "def is_isogram(s: str) -> bool",
   "True if no letter appears more than once, ignoring case; spaces and hyphens may repeat ('six-year-old' is an "
   "isogram).",
   [["six-year-old"], ["Alpha"]], lambda r: _bal(r, is_isogram, lambda r: [_word(r, 7, "abcdefgA -")]),
   [[""], ["a"], ["aA"], ["--  --"], ["abc def"], ["ab-ba"], ["Dermatoglyphics"]], category="strings")
def is_isogram(s):
    letters = [c.lower() for c in s if c not in " -"]
    return len(letters) == len(set(letters))


# ============================================================================================== estructuras
def _g_nested(r):
    def make(depth):
        if depth == 0 or r.random() < 0.3:
            return r.randint(0, 9)
        return {k: make(depth - 1) for k in r.sample("abc", r.randint(1, 3))}
    d = make(3)
    if not isinstance(d, dict):
        d = {"a": d}
    path, cur = [], d
    while isinstance(cur, dict) and r.random() < 0.8:  # ~70 % caminos que existen (en parte o del todo)
        k = r.choice(sorted(cur))
        path.append(k)
        cur = cur[k]
    if not path or r.random() < 0.3:
        path.append(r.choice("abc"))
    return [d, ".".join(path), -1]


@S(1, "def nested_get(d: dict, path: str, default)",
   "Look up a dotted path such as 'a.b.c' in nested dicts and return the value found, or default if some key along "
   "the path is missing or a value before the end of the path is not a dict.",
   [[{"a": {"b": 1}}, "a.b", None], [{"a": 1}, "a.b", 0]], _g_nested,
   [[{}, "a", None], [{"a": {"b": {}}}, "a.b", 5], [{"a": {"b": None}}, "a.b", 5], [{"a": {"b": 2}}, "a", 0],
    [{"": 1}, "", 0]], category="estructuras")
def nested_get(d, path, default):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@S(1, "def dict_diff(old: dict, new: dict) -> list[list[str]]",
   "Compare two dicts and return [added, removed, changed]: the keys only in new, the keys only in old and the keys in "
   "both whose values differ, each list sorted.",
   [[{"a": 1, "b": 2}, {"b": 3, "c": 4}], [{}, {}]],
   lambda r: [{k: r.randint(0, 2) for k in r.sample("abcde", r.randint(0, 5))},
              {k: r.randint(0, 2) for k in r.sample("abcde", r.randint(0, 5))}],
   [[{"a": 1}, {"a": 1}], [{"a": 1}, {}], [{}, {"z": 0}], [{"a": [1]}, {"a": [1, 2]}], [{"b": 1, "a": 2}, {"a": 1}]],
   category="estructuras")
def dict_diff(old, new):
    return [sorted(k for k in new if k not in old), sorted(k for k in old if k not in new),
            sorted(k for k in old if k in new and old[k] != new[k])]


# ============================================================================================== algoritmos
@S(2, "def house_robber(xs: list[int]) -> int",
   "Largest sum of elements of xs (all >= 0) that can be picked without ever picking two adjacent elements.",
   [[[2, 7, 9, 3, 1]], [[]]], lambda r: [_ints(r, 10, 0, 9)],
   [[[5]], [[1, 2]], [[2, 1, 1, 2]], [[0, 0, 0]], [[4, 1, 1, 4, 1]]], category="algoritmos")
def house_robber(xs):
    take, skip = 0, 0
    for x in xs:
        take, skip = skip + x, max(take, skip)
    return max(take, skip)


def _g_kth(r):
    xs = _ints(r, 8, -9, 9)
    k = r.randint(1, len(xs)) if xs and r.random() < 0.75 else r.choice((0, -1, len(xs) + 1))
    return [xs, k]


@S(1, "def kth_smallest(xs: list[int], k: int) -> int",
   "The k-th smallest value of xs, counting from 1 and counting repeated values (k = 1 is the minimum). Raise "
   "IndexError if k is outside 1..len(xs).",
   [[[7, 2, 9, 2], 2], [[5], 1]], _g_kth,
   [[[], 1], [[1, 2], 0], [[3, 1, 2], 3], [[4, 4, 4], 2], [[1], 2], [[5, -5], -1]], errs=("IndexError",),
   category="algoritmos")
def kth_smallest(xs, k):
    if not 1 <= k <= len(xs):
        raise IndexError(k)
    return sorted(xs)[k - 1]


def _g_wb(r):
    words = [_word(r, 3, "ab") or "a" for _ in range(r.randint(1, 4))]
    if r.random() < 0.5:
        return ["".join(r.choice(words) for _ in range(r.randint(0, 4))), words]
    return [_word(r, 8, "abc"), words]


@S(2, "def word_break(s: str, words: list[str]) -> bool",
   "True if s can be split into a sequence of words from words (each word can be used any number of times); the "
   "empty string gives True. It must be fast for strings of a few thousand characters.",
   [["leetcode", ["leet", "code"]], ["catsandog", ["cats", "dog", "sand", "and", "cat"]]],
   lambda r: _bal(r, word_break, _g_wb),
   [["", []], ["a", []], ["aaaa", ["aa"]], ["aaa", ["aa"]], ["abab", ["ab", "a"]], ["b", ["a"]]],
   lambda r: [["a" * 3000 + "b", ["a", "aa", "aaa"]], ["ab" * 1500, ["a", "b", "ab"]]], category="algoritmos")
def word_break(s, words):
    ws = set(words)
    lens = sorted({len(w) for w in ws if w})
    ok = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        ok[i] = any(n <= i and ok[i - n] and s[i - n:i] in ws for n in lens)
    return ok[len(s)]


# ============================================================================================== matemáticas
@S(1, "def int_sqrt(n: int) -> int",
   "The largest integer r with r * r <= n, exact for very large n (floating point is not precise enough). Raise "
   "ValueError if n < 0.",
   [[17], [16]], lambda r: [r.choice((r.randint(-2, 200), r.randint(0, 10 ** 6), r.randint(0, 10 ** 40)))],
   [[0], [1], [2], [3], [-1], [10 ** 30], [10 ** 30 - 1], [2 ** 106], [(10 ** 20 + 1) ** 2 - 1]], errs=("ValueError",),
   category="matematicas")
def int_sqrt(n):
    import math
    if n < 0:
        raise ValueError(n)
    return math.isqrt(n)


@S(1, "def pow_mod(base: int, exp: int, mod: int) -> int",
   "(base ** exp) % mod for base >= 0, exp >= 0 and mod >= 1 (0 ** 0 is 1). It must be fast for exp up to 10**18.",
   [[2, 10, 1000], [3, 0, 7]], lambda r: [r.randint(0, 50), r.randint(0, 60), r.randint(1, 100)],
   [[0, 0, 1], [0, 0, 5], [5, 3, 1], [0, 5, 7], [10 ** 9, 10 ** 18, 10 ** 9 + 7]],
   lambda r: [[r.randint(2, 10 ** 9), 10 ** 18 + r.randint(0, 99), 998244353]], category="matematicas")
def pow_mod(base, exp, mod):
    return pow(base, exp, mod)


@S(1, "def polygon_area(points: list[list[int]]) -> float",
   "Area of the polygon with the given vertices [x, y] in order, by the shoelace formula: abs(sum of x_i * y_(i+1) - "
   "x_(i+1) * y_i over consecutive vertices, the last one followed by the first) / 2. Fewer than 3 points give 0.0.",
   [[[[0, 0], [4, 0], [4, 3]]], [[[0, 0], [1, 1]]]],
   lambda r: [[[r.randint(-5, 5), r.randint(-5, 5)] for _ in range(r.randint(0, 6))]],
   [[[]], [[[1, 1]]], [[[0, 0], [2, 0], [2, 2], [0, 2]]], [[[0, 0], [0, 2], [2, 2], [2, 0]]], [[[0, 0], [1, 0], [2, 0]]],
    [[[1, 1], [3, 1], [2, 4]]]], category="matematicas")
def polygon_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    s = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


# ============================================================================================== parsing
def _g_kp(r):
    s = _word(r, 3, "ab_1") or "x"
    for _ in range(r.randint(0, 3)):
        s += "." + (_word(r, 3, "ab") or "y") if r.random() < 0.5 else f"[{r.randint(0, 12)}]"
    if r.random() < 0.25:
        i = r.randint(0, len(s))
        s = s[:i] + r.choice(".[]-") + s[i:]
    return [s]


@S(2, "def parse_key_path(path: str) -> list",
   "Parse a path like 'users[2].name' into a list of keys: names become strings and [n] becomes the integer n "
   "('a.b[0][1]' -> ['a', 'b', 0, 1]). The path starts with a name; each later step is '.name' or '[n]'. Names are "
   "non-empty runs of ASCII letters, digits and '_'; n is one or more ASCII digits. Raise ValueError for anything "
   "else.",
   [["users[2].name"], ["a..b"]], _g_kp,
   [[""], ["a"], ["a[0]"], ["[0]"], ["a.[0]"], ["a[01]"], ["a[]"], ["a.b."], ["_x.y2[10]"], ["a[1"], ["a b"]],
   errs=("ValueError",), category="parsing")
def parse_key_path(path):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+|\[[0-9]+\])*", path):
        raise ValueError(path)
    return [int(m[1]) if m[1] is not None else m[2] for m in re.finditer(r"\[([0-9]+)\]|([A-Za-z0-9_]+)", path)]


@S(1, "def sum_numbers_in_text(text: str) -> int",
   "Sum of the integers written in text: maximal runs of ASCII digits, negative when a '-' comes right before the "
   "run ('a-3b10' -> 7).",
   [["a-3b10"], ["no numbers"]],
   lambda r: ["".join(r.choice(("a", " ", "-", "1", "2", "0", "9", "x", "--")) for _ in range(r.randint(0, 10)))],
   [[""], ["-"], ["--5"], ["007"], ["1-1"], ["a1b2c3"], ["\u0663"], ["-0"]], category="parsing")
def sum_numbers_in_text(text):
    import re
    return sum(int(m) for m in re.findall(r"-?[0-9]+", text))


def _g_date(r):
    y, m, d = r.choice((1900, 2000, 2023, 2024, r.randint(1, 9999))), r.randint(0, 13), r.randint(0, 32)
    s = f"{y:04d}-{m:02d}-{d:02d}"
    if r.random() < 0.15:
        s = r.choice((s.replace("-", "/"), s[:-1], " " + s, s + "x", f"{y}-{m}-{d}"))
    return [s]


@S(2, "def parse_iso_date(s: str) -> list[int]",
   "Parse a date 'YYYY-MM-DD' (exactly 4, 2 and 2 ASCII digits, year 0001-9999) into [year, month, day], checking "
   "that the day exists in the Gregorian calendar (leap years: divisible by 4, except years divisible by 100 but not "
   "by 400). Raise ValueError otherwise.",
   [["2024-02-29"], ["2023-02-29"]], _g_date,
   [["2000-02-29"], ["1900-02-29"], ["2024-12-31"], ["2024-13-01"], ["2024-00-10"], ["2024-04-31"], ["0000-01-01"],
    ["2024-1-01"], [""], ["2024-06-30"]], errs=("ValueError",), category="parsing")
def parse_iso_date(s):
    import re
    m = re.fullmatch(r"([0-9]{4})-([0-9]{2})-([0-9]{2})", s)
    if not m:
        raise ValueError(s)
    y, mo, d = map(int, m.groups())
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if y < 1 or not 1 <= mo <= 12 or not 1 <= d <= days[mo - 1]:
        raise ValueError(s)
    return [y, mo, d]


# ============================================================================================== seguridad
def _g_user(r):
    if r.random() < 0.1:
        return [r.choice(("admin", "root", "system", "Root", "admin1"))]
    return [r.choice("abz") + _word(r, r.choice((2, 6, 17)), "abz09_A-")]


@S(1, "def validate_username(s: str) -> bool",
   "A valid username has 3 to 16 characters, starts with a lowercase ASCII letter, contains only lowercase ASCII "
   "letters, digits and '_', does not end with '_' and is not a reserved name (admin, root or system).",
   [["alice_01"], ["Admin"]], lambda r: _bal(r, validate_username, _g_user),
   [[""], ["ab"], ["abc"], ["a" * 16], ["a" * 17], ["abc_"], ["1abc"], ["root"], ["root_1"], ["ab-c"], ["\u00f1andu"],
    ["a_b"], ["z9_z"], ["user_name_15chr"], ["b00"]], category="seguridad")
def validate_username(s):
    import re
    return (re.fullmatch(r"[a-z][a-z0-9_]{2,15}", s) is not None and not s.endswith("_")
            and s not in ("admin", "root", "system"))


@S(1, "def shell_quote(s: str) -> str",
   "Quote s for a POSIX shell: wrap it in single quotes and replace every single quote inside it with the five "
   "characters '\"'\"' (close the quotes, a double-quoted quote, reopen). The empty string becomes ''.",
   [["it's"], ["a b"]], lambda r: [_word(r, 8, "ab '\"$;")],
   [[""], ["'"], ["''"], ["$(rm -rf ~)"], ["\\"], ["a'b'c"]], category="seguridad")
def shell_quote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _g_rl(r):
    ts, t = [], 0
    for _ in range(r.randint(0, 9)):
        t += r.choice((0, 0, 1, 1, 2, 5))
        ts.append(t)
    return [ts, r.randint(1, 3), r.randint(1, 5)]


@S(2, "def rate_limit(timestamps: list[int], limit: int, window: int) -> list[bool]",
   "Requests arrive at the given times (seconds, non-decreasing). A request at time t is allowed if fewer than limit "
   "requests were ALLOWED in the window (t - window, t]; rejected requests do not count. Return one boolean per "
   "request.",
   [[[1, 2, 3, 4], 2, 3], [[], 1, 10]], _g_rl,
   [[[5, 5, 5], 2, 1], [[0, 10], 1, 10], [[0, 9], 1, 10], [[1, 2, 3], 1, 1], [[7], 0, 5], [[0, 1, 2, 3, 4], 2, 2]],
   category="seguridad")
def rate_limit(timestamps, limit, window):
    allowed, out = [], []
    for t in timestamps:
        ok = sum(1 for a in allowed if a > t - window) < limit
        out.append(ok)
        if ok:
            allowed.append(t)
    return out


# ============================================================================================== edge cases
@S(0, "def percent_change(old: float, new: float) -> float",
   "Percentage change from old to new, (new - old) / abs(old) * 100, rounded to 2 decimals. Raise ValueError if old "
   "is 0.",
   [[50, 75], [200, 50]],
   lambda r: [r.choice((0, r.randint(-100, 100), round(r.uniform(-10, 10), 2))), r.randint(-100, 100)],
   [[0, 5], [0, 0], [3, 3], [-4, 4], [3, 4], [0.001, 1]], errs=("ValueError",), category="edge_cases")
def percent_change(old, new):
    if old == 0:
        raise ValueError("old is 0")
    return round((new - old) / abs(old) * 100, 2)


@S(1, "def trim_mean(xs: list[int], k: int) -> float",
   "Mean of xs after removing its k smallest and its k largest values (repeated values count separately). Raise "
   "ValueError if k < 0 or if no value would remain.",
   [[[1, 2, 3, 4, 100], 1], [[5, 5], 1]], lambda r: [_ints(r, 7, -9, 9), r.choice((-1, 0, 1, 1, 2))],
   [[[], 0], [[7], 0], [[1, 2, 3], 1], [[1, 1, 1, 9], 1], [[2, 4], -1], [[3, 1, 2], 0]], errs=("ValueError",),
   category="edge_cases")
def trim_mean(xs, k):
    if k < 0 or len(xs) - 2 * k <= 0:
        raise ValueError((len(xs), k))
    ys = sorted(xs)[k:len(xs) - k]
    return sum(ys) / len(ys)


# ============================================================================================== optimización
def _slow_span(prices):
    out = []
    for i, p in enumerate(prices):
        j = i
        while j >= 0 and prices[j] <= p:
            j -= 1
        out.append(i - j)
    return out


@S(2, "def stock_span(prices: list[int]) -> list[int]",
   "For each day i, the number of consecutive days ending at day i (including it) whose price is <= prices[i]. It "
   "must be fast for hundreds of thousands of days.",
   [[[100, 80, 60, 70, 60, 75, 85]], [[]]], lambda r: [_ints(r, 10, 1, 9)],
   [[[5]], [[1, 2, 3]], [[3, 2, 1]], [[2, 2, 2]]],
   lambda r: [[list(range(1, 200001))], [[r.randint(1, 10 ** 6) for _ in range(200000)]]],
   category="optimizacion", slow=_slow_span)
def stock_span(prices):
    out, st = [], []
    for i, p in enumerate(prices):
        while st and prices[st[-1]] <= p:
            st.pop()
        out.append(i - st[-1] if st else i + 1)
        st.append(i)
    return out


def _slow_min_window(xs, target):
    best = 0
    for i in range(len(xs)):
        s = 0
        for j in range(i, len(xs)):
            s += xs[j]
            if s >= target:
                if not best or j - i + 1 < best:
                    best = j - i + 1
                break
    return best


@S(2, "def min_window_sum(xs: list[int], target: int) -> int",
   "Length of the shortest contiguous subarray of xs (all elements > 0) whose sum is at least target (target >= 1); "
   "0 if there is none. It must be fast for lists of hundreds of thousands of numbers.",
   [[[2, 3, 1, 2, 4, 3], 7], [[1, 1], 5]], lambda r: [_ints(r, 10, 1, 9), r.randint(1, 30)],
   [[[], 1], [[5], 5], [[5], 6], [[1, 1, 1], 3], [[10, 1], 1]],
   lambda r: [[[1] * 200000, 150000], [[r.randint(1, 10) for _ in range(200000)], 10 ** 6]],
   category="optimizacion", slow=_slow_min_window)
def min_window_sum(xs, target):
    best, s, lo = 0, 0, 0
    for hi, x in enumerate(xs):
        s += x
        while s >= target:
            if not best or hi - lo + 1 < best:
                best = hi - lo + 1
            s -= xs[lo]
            lo += 1
    return best


# ============================================================================================== refactoring
@S(1, "def tax_due(income: float) -> float",
   "Rewrite this function as tax_due(income) with clear code and exactly the same results:\n\n"
   "    def tax(i):\n        if i < 0:\n            raise ValueError('income')\n        t = 0\n        if i > 10000:\n"
   "            if i > 40000:\n                t = t + (i - 40000) * 0.4\n                t = t + 30000 * 0.2\n"
   "            else:\n                t = t + (i - 10000) * 0.2\n        return round(t, 2)\n",
   [[5000], [50000]], lambda r: [r.choice((r.randint(-100, 100000), round(r.uniform(0, 60000), 2), 10000, 40000))],
   [[0], [10000], [10000.01], [40000], [40001], [-1]], errs=("ValueError",), category="refactoring")
def tax_due(income):
    if income < 0:
        raise ValueError("income")
    t = 0
    if income > 40000:
        t = (income - 40000) * 0.4 + 30000 * 0.2
    elif income > 10000:
        t = (income - 10000) * 0.2
    return round(t, 2)


@S(1, "def rps_winner(a: str, b: str) -> str",
   "Rewrite this function as rps_winner(a, b) with simple logic and exactly the same behaviour:\n\n"
   "    def w(a, b):\n        if a == b:\n            if a == 'rock' or a == 'paper' or a == 'scissors':\n"
   "                return 'draw'\n        if a == 'rock':\n            if b == 'scissors':\n                return 'a'\n"
   "            if b == 'paper':\n                return 'b'\n        if a == 'paper':\n            if b == 'rock':\n"
   "                return 'a'\n            if b == 'scissors':\n                return 'b'\n"
   "        if a == 'scissors':\n            if b == 'paper':\n                return 'a'\n            if b == 'rock':\n"
   "                return 'b'\n        raise ValueError('move')\n",
   [["rock", "scissors"], ["paper", "paper"]],
   lambda r: [r.choice(("rock", "paper", "scissors") * 4 + ("lizard", "Rock")),
              r.choice(("rock", "paper", "scissors") * 4 + ("", "spock"))],
   [["scissors", "rock"], ["rock", "rock"], ["x", "x"], ["rock", "Rock"], ["paper", "rock"], ["scissors", "paper"]],
   errs=("ValueError",), category="refactoring")
def rps_winner(a, b):
    moves = ("rock", "paper", "scissors")
    if a not in moves or b not in moves:
        raise ValueError("move")
    if a == b:
        return "draw"
    beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    return "a" if beats[a] == b else "b"


# ============================================================================================== debugging
@S(0, "def last_index(xs: list[int], v: int) -> int",
   "Index of the last occurrence of v in xs, or -1 if v is not in xs. The implementation below has bugs; write a "
   "correct version:\n\n"
   "    def last_index(xs, v):\n        for i in range(len(xs) - 1, 0, -1):\n            if xs[i] == v:\n"
   "                return i\n        return 0\n",
   [[[1, 2, 1, 3], 1], [[4, 5], 6]], lambda r: [_ints(r, 7, 0, 4), r.randint(0, 4)],
   [[[], 1], [[7], 7], [[7], 8], [[3, 3, 3], 3], [[1, 2], 1]], category="debugging")
def last_index(xs, v):
    for i in range(len(xs) - 1, -1, -1):
        if xs[i] == v:
            return i
    return -1


def _g_matrix(r):
    rows, cols = r.randint(0, 4), r.randint(1, 4)
    return [[[r.randint(0, 9) for _ in range(cols)] for _ in range(rows)]]


@S(1, "def transpose(m: list[list[int]]) -> list[list[int]]",
   "Transpose a rectangular matrix (row i becomes column i); [] stays []. The implementation below has bugs; write a "
   "correct version:\n\n"
   "    def transpose(m):\n        n = len(m)\n        out = [[0] * n for _ in range(n)]\n        for i in range(n):\n"
   "            for j in range(n):\n                out[i][j] = m[i][j]\n        return out\n",
   [[[[1, 2, 3], [4, 5, 6]]], [[]]], _g_matrix,
   [[[[1]]], [[[1, 2]]], [[[1], [2]]], [[[1, 2], [3, 4]]]], category="debugging")
def transpose(m):
    return [list(col) for col in zip(*m)]


# ============================================================================================== comprensión
@S(2, "def bubble_passes(xs: list[int], k: int) -> list[int]",
   "Given this code:\n\n    for _ in range(k):\n        for i in range(len(xs) - 1):\n            if xs[i] > xs[i + 1]:\n"
   "                xs[i], xs[i + 1] = xs[i + 1], xs[i]\n\nReturn the list xs after running it. k can be as large "
   "as 10**9.",
   [[[3, 1, 2], 1], [[4, 3, 2, 1], 2]], lambda r: [_ints(r, 8, 0, 9), r.randint(0, 9)],
   [[[], 5], [[2, 1], 0], [[5, 4, 3, 2, 1], 3], [[1, 1, 0], 10 ** 9]],
   lambda r: [[[r.randint(0, 10 ** 6) for _ in range(800)], 10 ** 9]], category="comprension")
def bubble_passes(xs, k):
    xs = list(xs)
    for _ in range(min(k, len(xs))):
        swapped = False
        for i in range(len(xs) - 1):
            if xs[i] > xs[i + 1]:
                xs[i], xs[i + 1] = xs[i + 1], xs[i]
                swapped = True
        if not swapped:
            break
    return xs


@S(1, "def trace_counter(s: str) -> dict",
   "Given this code:\n\n    d = {}\n    for i, c in enumerate(s):\n        if c in d:\n            d[c] = d[c] * 2 + i\n"
   "        else:\n            d[c] = i\n\nReturn the final value of d for the given s.",
   [["abab"], [""]], lambda r: [_word(r, 10, "abc")],
   [["a"], ["aaa"], ["cba"], ["aabbcc"]], category="comprension")
def trace_counter(s):
    d = {}
    for i, c in enumerate(s):
        d[c] = d[c] * 2 + i if c in d else i
    return d


# ============================================================================================== especificación
@S(1, "def format_list(items: list[str]) -> str",
   "Join items in English: [] -> '', ['a'] -> 'a', ['a', 'b'] -> 'a and b', ['a', 'b', 'c'] -> 'a, b and c' (no "
   "comma before 'and').",
   [[["red", "green", "blue"]], [["x"]]], lambda r: [[_word(r, 4, "abc") for _ in range(r.randint(0, 5))]],
   [[[]], [["x", "yy"]], [["", ""]], [["a", "b", "c", "d"]], [["only"]]], category="especificacion")
def format_list(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


@S(0, "def truncate_words(text: str, n: int) -> str",
   "The first n words of text (words are separated by any whitespace) joined by single spaces, followed by '...' if "
   "some words were left out. Raise ValueError if n < 0.",
   [["the quick brown fox", 2], ["hi", 5]],
   lambda r: [" ".join(_word(r, 4, "ab") or "z" for _ in range(r.randint(0, 6))) + r.choice(("", " ", "\n")),
              r.randint(-1, 7)],
   [["", 0], ["a b", 0], ["  a   b  ", 1], ["a\tb\nc", 3], ["x", -1]], errs=("ValueError",),
   category="especificacion")
def truncate_words(text, n):
    if n < 0:
        raise ValueError(n)
    ws = text.split()
    return " ".join(ws[:n]) + ("..." if len(ws) > n else "")


# ============================================================================================== multi-paso
def _g_tourn(r):
    out = []
    for _ in range(r.randint(0, 6)):
        a, b = r.sample("ABCD", 2)
        out.append(f"{a} {r.randint(0, 3)}-{r.randint(0, 3)} {b}")
    return [out]


@S(2, "def tournament_table(results: list[str]) -> list[list]",
   "results are strings like 'A 2-1 B' (team names have no spaces). A win gives 3 points and a draw 1 to each team. "
   "Return [team, points, goal difference] for every team, sorted by points (descending), then goal difference "
   "(descending), then name.",
   [[["A 2-1 B", "B 0-0 C"]], [[]]], _g_tourn,
   [[["X 0-0 Y"]], [["A 5-0 B", "B 5-0 A"]], [["Z 1-0 A", "A 1-0 Z"]], [["A 10-9 B"]], [["B 0-2 A", "C 1-1 A"]]],
   category="multi_step")
def tournament_table(results):
    pts, gd = {}, {}
    for line in results:
        a, score, b = line.split()
        x, y = map(int, score.split("-"))
        for t in (a, b):
            pts.setdefault(t, 0)
            gd.setdefault(t, 0)
        gd[a] += x - y
        gd[b] += y - x
        if x > y:
            pts[a] += 3
        elif x < y:
            pts[b] += 3
        else:
            pts[a] += 1
            pts[b] += 1
    return sorted(([t, pts[t], gd[t]] for t in pts), key=lambda e: (-e[1], -e[2], e[0]))


def _g_lib(r):
    ev, day = [], 0
    for _ in range(r.randint(0, 7)):
        day += r.choice((0, 1, 5, 10, 15))
        ev.append([day, r.choice(("out", "in")), r.choice(("a", "b", "c"))])
    return [ev, day + r.choice((0, 5, 20))]


@S(2, "def library_overdue(events: list[list], today: int) -> list[str]",
   "events are [day, action, book] with non-decreasing days and action 'out' or 'in'. A book that goes out must be "
   "back within 14 days (an 'in' on day <= out day + 14 is on time). Return, sorted and without repetitions, the books "
   "that were returned late at least once or are still out and overdue on day today (today > out day + 14). An 'in' "
   "for a book that is not out, and an 'out' for a book that is already out, are ignored.",
   [[[[1, "out", "dune"], [2, "out", "emma"], [20, "in", "dune"]], 20], [[[5, "out", "a"], [19, "in", "a"]], 30]],
   _g_lib,
   [[[], 0], [[[0, "in", "x"]], 100], [[[0, "out", "x"]], 14], [[[0, "out", "x"]], 15],
    [[[0, "out", "x"], [3, "out", "x"], [16, "in", "x"]], 16], [[[0, "out", "x"], [14, "in", "x"], [15, "out", "x"]], 30]],
   category="multi_step")
def library_overdue(events, today):
    out, late = {}, set()
    for day, action, book in events:
        if action == "out" and book not in out:
            out[book] = day
        elif action == "in" and book in out and day > out.pop(book) + 14:
            late.add(book)
    late |= {b for b, d in out.items() if today > d + 14}
    return sorted(late)
