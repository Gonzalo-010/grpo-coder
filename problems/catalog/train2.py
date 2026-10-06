"""Catálogo v3, split train (63 problemas, las 14 categorías). Familias propias: ninguna se repite en val ni en test.
Las referencias importan dentro de la función: así también se pueden ejecutar como si fueran un candidato."""
from problems.problems import P, _bal, _ints, _word


def T(level, sig, doc, ex, gen, edge=(), large=None, errs=(), *, category, family=None, slow=None):
    def deco(fn):
        return P(level, sig, doc, ex, gen, edge, large, errs, split="train", category=category,
                 family=family or fn.__name__, slow=slow, source="catalog")(fn)
    return deco


# ============================================================================================== arrays
@T(0, "def running_sum(xs: list[int]) -> list[int]", "out[i] is the sum of xs[0..i] (inclusive).",
   [[[1, 2, 3, 4]], [[5, -5]]], lambda r: [_ints(r, 12, -9, 9)],
   [[[]], [[7]], [[0, 0]], [[-1, -2, -3]]], lambda r: [[[r.randint(-10, 10) for _ in range(200000)]]],
   category="arrays")
def running_sum(xs):
    out, acc = [], 0
    for x in xs:
        acc += x
        out.append(acc)
    return out


@T(1, "def find_peaks(xs: list[int]) -> list[int]",
   "Indices i (0 < i < len(xs) - 1) such that xs[i] is strictly greater than both neighbours, in increasing order.",
   [[[1, 3, 2, 4, 1]], [[1, 2, 3]]], lambda r: [_ints(r, 12, 0, 6)],
   [[[]], [[5]], [[1, 2]], [[2, 2, 2]], [[1, 3, 3, 1]], [[0, 1, 0, 1, 0]]], category="arrays")
def find_peaks(xs):
    return [i for i in range(1, len(xs) - 1) if xs[i] > xs[i - 1] and xs[i] > xs[i + 1]]


def _g_major(r):
    n = r.randint(0, 11)
    if r.random() < 0.5 and n:
        v = r.randint(-3, 3)
        xs = [v] * (n // 2 + 1) + [r.randint(-3, 3) for _ in range(n - n // 2 - 1)]
        r.shuffle(xs)
        return [xs]
    return [_ints(r, 11, -3, 3)]


@T(1, "def majority_element(xs: list[int]) -> int | None",
   "The value that appears MORE than len(xs) / 2 times, or None if there is no such value.",
   [[[3, 1, 3, 3, 2]], [[1, 2]]], _g_major,
   [[[]], [[4]], [[1, 1, 2, 2]], [[2, 2, 1]], [[5, 5, 5, 1, 1, 1]]],
   lambda r: [[[7] * 100001 + [r.randint(0, 9) for _ in range(100000)]]], category="arrays")
def majority_element(xs):
    from collections import Counter
    if not xs:
        return None
    v, c = Counter(xs).most_common(1)[0]
    return v if c * 2 > len(xs) else None


def _slow_window_max(xs, k):
    if k < 1:
        raise ValueError(k)
    return [max(xs[i:i + k]) for i in range(len(xs) - k + 1)]


@T(2, "def window_max(xs: list[int], k: int) -> list[int]",
   "Maximum of every contiguous window of length k, from left to right ([] if k > len(xs)). Raise ValueError if k < 1. "
   "It must be fast for lists of hundreds of thousands of numbers and large k.",
   [[[1, 3, -1, -3, 5, 3, 6, 7], 3], [[4, 2], 1]], lambda r: [_ints(r, 12, -9, 9), r.randint(0, 5)],
   [[[], 1], [[1], 1], [[2, 1], 3], [[5, 4, 3, 2], 2], [[1, 2], 0]],
   lambda r: [[[r.randint(-10 ** 6, 10 ** 6) for _ in range(200000)], 50000],
              [list(range(200000, 0, -1)), 100000]], ("ValueError",),
   category="optimizacion", slow=_slow_window_max)
def window_max(xs, k):
    from collections import deque
    if k < 1:
        raise ValueError(k)
    dq, out = deque(), []
    for i, x in enumerate(xs):
        while dq and xs[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(xs[dq[0]])
    return out


@T(0, "def cumulative_max(xs: list[int]) -> list[int]", "out[i] is the largest value among xs[0..i].",
   [[[1, 3, 2, 5, 4]], [[-2, -5]]], lambda r: [_ints(r, 12, -9, 9)],
   [[[]], [[3]], [[5, 4, 3]], [[-1, -1, 0]]], category="arrays")
def cumulative_max(xs):
    out = []
    for x in xs:
        out.append(x if not out or x > out[-1] else out[-1])
    return out


# ============================================================================================== strings
def _g_sentence(r):
    words = [_word(r, 6, "abcXYz") or "a" for _ in range(r.randint(0, 6))]
    return [r.choice((" ", "  ", "")).join(words) if words else r.choice(("", " "))]


@T(0, "def capitalize_words(s: str) -> str",
   "Uppercase the first letter of every word and lowercase the rest. Words are separated by single or multiple spaces, "
   "which are kept exactly as they are.",
   [["hello wORLD"], ["  a  b"]], _g_sentence, [[""], [" "], ["x"], ["ABC def"], ["a  b   c"]], category="strings")
def capitalize_words(s):
    out, start = [], True
    for ch in s:
        if ch == " ":
            out.append(ch)
            start = True
        else:
            out.append(ch.upper() if start else ch.lower())
            start = False
    return "".join(out)


@T(0, "def count_vowels(s: str) -> int", "Number of vowels (a, e, i, o, u, in either case) in s.",
   [["Hello World"], ["xyz"]], lambda r: [_word(r, 14, "aeiouAEIbcdXY ")],
   [[""], ["AEIOU"], ["bcd"], ["y"]], category="strings")
def count_vowels(s):
    return sum(c in "aeiouAEIOU" for c in s)


def _g_caesar(r):
    return [_word(r, 10, "abcxyzABCXYZ -!"), r.randint(-30, 30)]


@T(1, "def caesar_shift(s: str, k: int) -> str",
   "Shift every ASCII letter k positions in the alphabet, wrapping around and keeping its case (k may be negative or "
   "larger than 26). Other characters stay the same.",
   [["abc xyz", 2], ["Hello!", -1]], _g_caesar,
   [["", 5], ["z", 1], ["A", -1], ["aZ", 26], ["abc", 52], ["m-n", 13]], category="strings")
def caesar_shift(s, k):
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append(chr((ord(c) - 97 + k) % 26 + 97))
        elif "A" <= c <= "Z":
            out.append(chr((ord(c) - 65 + k) % 26 + 65))
        else:
            out.append(c)
    return "".join(out)


def _g_subseq(r):
    t = _word(r, 10, "abc")
    if r.random() < 0.5 and t:
        s = "".join(c for c in t if r.random() < 0.5)
    else:
        s = _word(r, 4, "abc")
    return [s, t]


@T(1, "def is_subsequence(s: str, t: str) -> bool",
   "True if s can be obtained from t by deleting zero or more characters without changing the order of the rest.",
   [["ace", "abcde"], ["aec", "abcde"]], lambda r: _bal(r, is_subsequence, _g_subseq),
   [["", ""], ["", "abc"], ["a", ""], ["abc", "abc"], ["aa", "a"], ["b", "ab"]],
   lambda r: [["ab" * 50000, "a" * 100000 + "b" * 100000]], category="strings")
def is_subsequence(s, t):
    i = 0
    for c in t:
        if i < len(s) and s[i] == c:
            i += 1
    return i == len(s)


@T(1, "def squeeze_spaces(s: str) -> str",
   "Replace every run of spaces with a single space and remove spaces at both ends. Other characters are kept.",
   [["  a   b  c "], ["no-spaces"]], lambda r: [r.choice(("", " ", "  ")) + "".join(
       _word(r, 3, "ab") + r.choice((" ", "   ", "")) for _ in range(r.randint(0, 5)))],
   [[""], ["   "], ["a"], [" a "], ["a\tb"]], category="strings")
def squeeze_spaces(s):
    out, prev = [], False
    for c in s.strip(" "):
        if c == " ":
            if not prev:
                out.append(c)
            prev = True
        else:
            out.append(c)
            prev = False
    return "".join(out)


# ============================================================================================== estructuras
def _g_freq(r):
    return [_ints(r, 14, -4, 4)]


@T(1, "def frequency_sort(xs: list[int]) -> list[int]",
   "Return xs sorted by how often each value appears (most frequent first); values with the same frequency go in "
   "increasing order. Repeated values stay together.",
   [[[1, 1, 2, 2, 2, 3]], [[4, 6, 4, 6]]], _g_freq,
   [[[]], [[7]], [[3, 1, 2]], [[-1, -1, 5, 5, 0]]], category="estructuras")
def frequency_sort(xs):
    from collections import Counter
    c = Counter(xs)
    return sorted(xs, key=lambda v: (-c[v], v))


def _g_invert(r):
    keys = r.sample("abcdefg", r.randint(0, 6))
    return [{k: r.randint(0, 3) for k in keys}]


@T(1, "def invert_mapping(d: dict) -> dict",
   "Invert a dict from str to int: return a dict that maps each value to the sorted list of keys that had it. JSON "
   "turns int keys into strings, so the result's keys must be the values written as strings.",
   [[{"a": 1, "b": 2, "c": 1}], [{}]], _g_invert,
   [[{"x": 0}], [{"a": 5, "b": 5}], [{"b": 1, "a": 1, "c": 2}]], category="estructuras")
def invert_mapping(d):
    out = {}
    for k in sorted(d):
        out.setdefault(str(d[k]), []).append(k)
    return out


def _g_counts_merge(r):
    def one():
        return {k: r.randint(-3, 5) for k in r.sample("abcd", r.randint(0, 4))}
    return [[one() for _ in range(r.randint(0, 3))]]


@T(1, "def merge_counts(dicts: list[dict]) -> dict",
   "Add up dicts of str -> int key by key. Keys whose total is 0 are left out of the result.",
   [[[{"a": 1, "b": 2}, {"a": 3}]], [[{"x": 1}, {"x": -1}]]], _g_counts_merge,
   [[[]], [[{}]], [[{"a": 0}]], [[{"a": 2}, {"b": -2}, {"a": -2}]]], category="estructuras")
def merge_counts(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def _g_window_ops(r):
    ops = []
    for _ in range(r.randint(0, 12)):
        ops.append(["add", r.randint(-5, 5)] if r.random() < 0.6 else ["avg"])
    return [r.randint(1, 4), ops]


@T(2, "def moving_average(k: int, ops: list[list]) -> list[float]",
   "Process ['add', x] and ['avg'] operations. 'avg' appends the mean of the last k added numbers (or of all of them if "
   "fewer than k were added; 0.0 if none). Return the list of averages. Raise ValueError if k < 1.",
   [[2, [["add", 1], ["add", 3], ["avg"], ["add", 5], ["avg"]]], [3, [["avg"]]]], _g_window_ops,
   [[1, []], [1, [["add", 4], ["avg"], ["add", -4], ["avg"]]], [5, [["add", 1], ["avg"]]], [0, [["avg"]]]],
   lambda r: [[1000, [["add", r.randint(-9, 9)] if i % 3 else ["avg"] for i in range(200000)]]],
   ("ValueError",), category="estructuras")
def moving_average(k, ops):
    from collections import deque
    if k < 1:
        raise ValueError(k)
    q, s, out = deque(), 0, []
    for op in ops:
        if op[0] == "add":
            q.append(op[1])
            s += op[1]
            if len(q) > k:
                s -= q.popleft()
        else:
            out.append(s / len(q) if q else 0.0)
    return out


# ============================================================================================== algoritmos
@T(1, "def insert_position(xs: list[int], target: int) -> int",
   "xs is sorted in ascending order (it may contain duplicates). Return the smallest index at which target could be "
   "inserted keeping the list sorted.",
   [[[1, 3, 5, 6], 5], [[1, 3, 5, 6], 2]], lambda r: [sorted(_ints(r, 12, -6, 6)), r.randint(-8, 8)],
   [[[], 3], [[1], 0], [[1], 2], [[2, 2, 2], 2], [[1, 2, 3], 4]],
   lambda r: [[list(range(0, 2 * 10 ** 6, 2)), 999999], [list(range(10 ** 6)), -5]], category="algoritmos")
def insert_position(xs, target):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@T(1, "def climb_ways(n: int) -> int",
   "Number of ways to climb n steps taking 1, 2 or 3 steps at a time (climb_ways(0) == 1). Raise ValueError if n < 0.",
   [[3], [4]], lambda r: [r.randint(-1, 40)], [[0], [1], [2], [-1], [30]], lambda r: [[5000]],
   ("ValueError",), category="algoritmos")
def climb_ways(n):
    if n < 0:
        raise ValueError(n)
    a, b, c = 1, 0, 0  # ways(i), ways(i-1), ways(i-2)
    for _ in range(n):
        a, b, c = a + b + c, a, b
    return a


def _g_knap(r):
    n = r.randint(0, 6)
    return [[r.randint(1, 9) for _ in range(n)], [r.randint(0, 20) for _ in range(n)], r.randint(0, 20)]


@T(2, "def knapsack(weights: list[int], values: list[int], cap: int) -> int",
   "0/1 knapsack: the largest total value of a subset of items (each used at most once) whose total weight is at most "
   "cap. weights and values have the same length. Raise ValueError if cap < 0.",
   [[[1, 3, 4, 5], [1, 4, 5, 7], 7], [[5], [10], 4]], _g_knap,
   [[[], [], 0], [[1], [1], 0], [[2, 2], [3, 3], 4], [[3], [5], -1]],
   lambda r: [[[r.randint(1, 50) for _ in range(200)], [r.randint(1, 100) for _ in range(200)], 2000]],
   ("ValueError",), category="algoritmos")
def knapsack(weights, values, cap):
    if cap < 0:
        raise ValueError(cap)
    best = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
    return best[cap]


def _g_jump(r):
    n = r.randint(1, 10)
    return [[r.choice((0, 0, 1, 2, 3)) for _ in range(n)]]


@T(2, "def can_reach_end(jumps: list[int]) -> bool",
   "You start at index 0; from index i you may move forward between 1 and jumps[i] positions. True if the last index "
   "can be reached. Raise ValueError if jumps is empty.",
   [[[2, 3, 1, 1, 4]], [[3, 2, 1, 0, 4]]], lambda r: _bal(r, can_reach_end, _g_jump),
   [[[0]], [[1, 0]], [[0, 1]], [[2, 0, 0]], [[]]],
   lambda r: [[[1] * 300000], [[2] * 150000 + [0, 0] + [1] * 1000]], ("ValueError",), category="algoritmos")
def can_reach_end(jumps):
    if not jumps:
        raise ValueError("empty")
    far = 0
    for i, j in enumerate(jumps):
        if i > far:
            return False
        far = max(far, i + j)
    return True


def _g_paths(r):
    rows, cols = r.randint(1, 5), r.randint(1, 5)
    return [[[1 if r.random() < 0.2 else 0 for _ in range(cols)] for _ in range(rows)]]


@T(2, "def grid_paths(grid: list[list[int]]) -> int",
   "Number of paths from the top-left to the bottom-right cell moving only right or down, where cells with 1 are "
   "blocked (0 if the start or the end is blocked). An empty grid has 0 paths.",
   [[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], [[[0, 1], [1, 0]]]], _g_paths,
   [[[]], [[[0]]], [[[1]]], [[[0, 0], [0, 0]]], [[[0, 0, 0]]]],
   lambda r: [[[[0] * 60 for _ in range(60)]]], category="algoritmos")
def grid_paths(grid):
    if not grid or not grid[0]:
        return 0
    cols = len(grid[0])
    ways = [0] * cols
    ways[0] = 1 if grid[0][0] == 0 else 0
    for row in grid:
        for j in range(cols):
            if row[j] == 1:
                ways[j] = 0
            elif j > 0:
                ways[j] += ways[j - 1]
    return ways[-1]


# ============================================================================================== matemáticas
@T(0, "def digit_sum(n: int) -> int", "Sum of the decimal digits of abs(n).",
   [[1234], [-56]], lambda r: [r.randint(-10 ** 6, 10 ** 6)], [[0], [9], [-1], [10 ** 18]], category="matematicas")
def digit_sum(n):
    return sum(int(c) for c in str(abs(n)))


def _g_kind(r):
    want = r.choice(("perfect", "abundant", "deficient", "error"))
    if want == "perfect":
        return [r.choice((6, 28, 496, 8128))]
    if want == "error":
        return [r.randint(-9, 0)]
    for _ in range(200):
        n = r.randint(1, 3000)
        if number_kind(n) == want:
            return [n]
    return [r.randint(1, 3000)]


@T(1, "def number_kind(n: int) -> str",
   "Compare n with the sum of its divisors smaller than itself (for 12: 1+2+3+4+6 = 16): return 'perfect' if they are "
   "equal, 'abundant' if the sum is larger and 'deficient' if it is smaller. Raise ValueError if n < 1.",
   [[6], [8]], _g_kind, [[1], [2], [12], [28], [945], [0], [-6], [8128], [9]], errs=("ValueError",),
   category="matematicas")
def number_kind(n):
    if n < 1:
        raise ValueError(n)
    s, i = (1 if n > 1 else 0), 2
    while i * i <= n:
        if n % i == 0:
            s += i + (n // i if i * i != n else 0)
        i += 1
    return "perfect" if s == n else "abundant" if s > n else "deficient"


@T(1, "def fib_mod(n: int, m: int) -> int",
   "The n-th Fibonacci number (fib(0) = 0, fib(1) = 1) modulo m. Raise ValueError if n < 0 or m < 1. It must be fast "
   "for n up to a few million.",
   [[10, 1000], [7, 5]], lambda r: [r.randint(-1, 90), r.randint(0, 50)],
   [[0, 7], [1, 1], [2, 10], [-1, 5], [5, 0], [90, 10 ** 9]], lambda r: [[3 * 10 ** 6, 10 ** 9 + 7]],
   ("ValueError",), category="matematicas")
def fib_mod(n, m):
    if n < 0 or m < 1:
        raise ValueError((n, m))
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % m
    return a % m


@T(1, "def prime_factors(n: int) -> list[int]",
   "Prime factorization of n >= 2 in ascending order, with repetitions (12 -> [2, 2, 3]). Raise ValueError if n < 2.",
   [[12], [97]], lambda r: [r.choice((r.randint(-3, 1), r.randint(2, 5000), r.randint(2, 10 ** 9)))],
   [[2], [1], [0], [1024], [999983], [600851475143]], errs=("ValueError",), category="matematicas")
def prime_factors(n):
    if n < 2:
        raise ValueError(n)
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


@T(1, "def binary_gap(n: int) -> int",
   "Length of the longest run of 0s surrounded by 1s on both sides in the binary representation of n (0 if none). "
   "Raise ValueError if n < 0.",
   [[9], [20]], lambda r: [r.choice((r.randint(-2, 64), r.randint(0, 2 ** 20)))],
   [[0], [1], [15], [32], [529], [-3]], errs=("ValueError",), category="matematicas")
def binary_gap(n):
    if n < 0:
        raise ValueError(n)
    best, cur, seen = 0, 0, False
    for b in bin(n)[2:]:
        if b == "1":
            if seen:
                best = max(best, cur)
            seen, cur = True, 0
        else:
            cur += 1
    return best


# ============================================================================================== parsing
def _g_intlist(r):
    nums = [str(r.randint(-50, 50)) for _ in range(r.randint(0, 5))]
    s = ",".join(r.choice(("", " ")) + x + r.choice(("", " ")) for x in nums)
    if r.random() < 0.15:
        s += r.choice((",", ",,", ", x", "1a", " - 3"))
    return [s]


@T(1, "def parse_int_list(s: str) -> list[int]",
   "Parse a comma-separated list of integers such as '1, -2,3'. Spaces around each number are ignored and an empty or "
   "blank string gives []. Raise ValueError if any item is not an integer (including empty items like in '1,,2').",
   [["1, -2,3"], [""]], _g_intlist, [["  "], ["7"], ["1,"], [",1"], ["1 2"], ["+4"], ["0, 0"]],
   errs=("ValueError",), category="parsing")
def parse_int_list(s):
    if not s.strip():
        return []
    out = []
    for part in s.split(","):
        p = part.strip()
        body = p[1:] if p[:1] in "+-" else p
        if not body or not all(c in "0123456789" for c in body):
            raise ValueError(part)
        out.append(int(p))
    return out


def _g_camel(r):
    parts = [r.choice(("get", "set", "html", "parse", "x", "id", "url")) for _ in range(r.randint(1, 3))]
    s = parts[0] + "".join(p.capitalize() if r.random() < 0.8 else p.upper() for p in parts[1:])
    return [s if r.random() < 0.85 else s.capitalize()]


@T(1, "def split_camel(s: str) -> list[str]",
   "Split a camelCase or PascalCase identifier into lowercase words. A new word starts at each uppercase letter that "
   "follows a lowercase letter, and at the last uppercase letter of a run that is followed by a lowercase letter "
   "('parseHTMLText' -> ['parse', 'html', 'text']).",
   [["getUserName"], ["parseHTMLText"]], _g_camel,
   [[""], ["x"], ["ABC"], ["Already"], ["aB"], ["XMLHttpRequest"]], category="parsing")
def split_camel(s):
    words, cur = [], ""
    for i, c in enumerate(s):
        if c.isupper() and cur and (cur[-1].islower() or (i + 1 < len(s) and s[i + 1].islower())):
            words.append(cur)
            cur = ""
        cur += c
    if cur:
        words.append(cur)
    return [w.lower() for w in words]


def _g_query(r):
    parts = []
    for _ in range(r.randint(0, 4)):
        k = r.choice("abc")
        parts.append(k + r.choice(("=", "=", "")) + _word(r, 3, "xy1+"))
    return ["&".join(parts) + r.choice(("", "&"))]


@T(1, "def parse_query(q: str) -> dict",
   "Parse a URL query string like 'a=1&b=x'. Pairs are separated by '&' and empty pairs are skipped. A pair without '=' "
   "maps the key to ''. '+' in keys and values means a space. If a key repeats, the last value wins.",
   [["a=1&b=x"], ["flag&n=2+3"]], _g_query,
   [[""], ["&&"], ["a="], ["a"], ["a=1&a=2"], ["a=b=c"]], category="parsing")
def parse_query(q):
    out = {}
    for pair in q.split("&"):
        if not pair:
            continue
        k, _, v = pair.partition("=")
        out[k.replace("+", " ")] = v.replace("+", " ")
    return out


def _g_clock(r):
    u = r.random()
    if u < 0.15:
        return [r.choice(("", "1:2", "61", "1:60", "x:10", "1:2:3:4", "12:5a"))]
    if u < 0.6:
        return [f"{r.randint(0, 9)}:{r.randint(0, 59):02d}"]
    return [f"{r.randint(0, 99)}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}"]


@T(1, "def clock_to_seconds(s: str) -> int",
   "Convert 'M:SS' or 'H:MM:SS' to seconds. H and M are one or more digits; MM and SS are exactly two digits between 00 "
   "and 59. Raise ValueError for anything else.",
   [["3:05"], ["1:00:00"]], _g_clock, [["0:00"], ["0:59"], ["10:00:00"], ["1:5"], ["1:60"], ["a:00"], [""]],
   errs=("ValueError",), category="parsing")
def clock_to_seconds(s):
    parts = s.split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() and p.isascii() for p in parts):
        raise ValueError(s)
    if any(len(p) != 2 or int(p) > 59 for p in parts[1:]):
        raise ValueError(s)
    total = 0
    for p in parts:
        total = total * 60 + int(p)
    return total


def _g_arith(r):
    toks = []
    for _ in range(r.randint(1, 5)):
        toks.append(r.choice((str(r.randint(0, 99)), "(", ")", "+", "-", "*", "/")))
    s = r.choice(("", " ")).join(toks)
    return [s + (r.choice(("x", "$", " 2.5")) if r.random() < 0.1 else "")]


@T(2, "def tokenize_expr(s: str) -> list[str]",
   "Split an arithmetic expression into tokens: non-negative integers (runs of digits), the operators + - * / and "
   "parentheses. Spaces separate nothing by themselves and are skipped. Raise ValueError for any other character.",
   [["3+42*(7-1)"], ["  12 / 4 "]], _g_arith, [[""], ["7"], ["((1))"], ["1 2"], ["a"], ["3.5"]],
   errs=("ValueError",), category="parsing")
def tokenize_expr(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == " ":
            i += 1
        elif c in "+-*/()":
            out.append(c)
            i += 1
        elif c in "0123456789":
            j = i
            while j < len(s) and s[j] in "0123456789":
                j += 1
            out.append(s[i:j])
            i = j
        else:
            raise ValueError(c)
    return out


# ============================================================================================== seguridad
def _g_fname(r):
    return [_word(r, 10, "ab./\\ :*?\"<>|-_") + r.choice(("", ".txt", "..", " "))]


@T(1, "def sanitize_filename(name: str) -> str",
   "Make a safe file name: replace each of the characters / \\ : * ? \" < > | with '_', remove leading and trailing "
   "spaces and dots, and return 'unnamed' if nothing is left.",
   [["my:file?.txt"], [" ../.. "]], _g_fname, [[""], ["..."], ["a b"], ["*"], [" .x. "], ["ok.txt"]],
   category="seguridad")
def sanitize_filename(name):
    out = "".join("_" if c in '/\\:*?"<>|' else c for c in name).strip(" .")
    return out or "unnamed"


def _g_card(r):
    digits = "".join(str(r.randint(0, 9)) for _ in range(r.randint(0, 19)))
    if r.random() < 0.4:
        digits = " ".join(digits[i:i + 4] for i in range(0, len(digits), 4))
    if r.random() < 0.1:
        digits += r.choice(("x", "-1"))
    return [digits]


@T(1, "def mask_card(s: str) -> str",
   "Mask a card number: keep only its digits (spaces are allowed and dropped), replace every digit except the last 4 "
   "with '*'. Raise ValueError if s contains anything other than ASCII digits and spaces, or has fewer than 4 digits.",
   [["4111 1111 1111 1234"], ["123456"]], _g_card, [[""], ["1234"], ["123"], ["12 34"], ["1234-5678"], ["00000"]],
   errs=("ValueError",), category="seguridad")
def mask_card(s):
    if any(c not in "0123456789 " for c in s):
        raise ValueError("bad character")
    d = s.replace(" ", "")
    if len(d) < 4:
        raise ValueError("too short")
    return "*" * (len(d) - 4) + d[-4:]


def _g_pwd(r):
    pool = "abcABC123!?"
    return [_word(r, 12, pool) if r.random() < 0.7 else r.choice(("aB3!aaaa", "password", "ABCdef12", "Aa1!", "aaaaaaaaaaaa1A!"))]


@T(1, "def is_strong_password(p: str) -> bool",
   "True if p has at least 8 characters and contains at least one lowercase ASCII letter, one uppercase ASCII letter, "
   "one ASCII digit and one character that is none of those.",
   [["Abcdef1!"], ["abcdefgh"]], lambda r: _bal(r, is_strong_password, _g_pwd),
   [[""], ["Aa1!Aa1!"], ["Aa1!Aa1"], ["AAAAAAA1!"], ["aaaaaaa1!"], ["Aaaaaaaa!"], ["Aaaaaaa1a"], ["Zz9 zz9 "],
    ["P@ssw0rd"], ["12345678aA_"], ["\u00e9\u00e9\u00e9\u00e9Aa1!"]], category="seguridad")
def is_strong_password(p):
    lower = any("a" <= c <= "z" for c in p)
    upper = any("A" <= c <= "Z" for c in p)
    digit = any("0" <= c <= "9" for c in p)
    other = any(not ("a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9") for c in p)
    return len(p) >= 8 and lower and upper and digit and other


@T(0, "def escape_html(s: str) -> str",
   "Escape text for HTML: & -> &amp;, < -> &lt;, > -> &gt;, \" -> &quot;, ' -> &#39;. Everything else is unchanged.",
   [["<a href='x'>&</a>"], ["plain"]], lambda r: [_word(r, 12, "ab<>&\"' ")],
   [[""], ["&&"], ["&amp;"], ["\"'"]], category="seguridad")
def escape_html(s):
    rep = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}
    return "".join(rep.get(c, c) for c in s)


# ============================================================================================== edge cases
@T(0, "def safe_mean(xs: list[float]) -> float | None", "Arithmetic mean of xs, or None if xs is empty.",
   [[[1, 2, 3, 4]], [[]]], lambda r: [_ints(r, 8, -20, 20)], [[[5]], [[-1, 1]], [[0, 0, 1]]], category="edge_cases")
def safe_mean(xs):
    return sum(xs) / len(xs) if xs else None


@T(1, "def first_unique_char(s: str) -> int",
   "Index of the first character that appears exactly once in s, or -1 if there is none.",
   [["leetcode"], ["aabb"]], lambda r: [_word(r, 10, "abcd")], [[""], ["a"], ["aa"], ["abab"], ["aabbc"]],
   lambda r: [["".join(r.choice("abcdefghij") for _ in range(100000)) + "z"]], category="edge_cases")
def first_unique_char(s):
    from collections import Counter
    c = Counter(s)
    return next((i for i, ch in enumerate(s) if c[ch] == 1), -1)


@T(0, "def last_word_length(s: str) -> int",
   "Length of the last word of s; words are maximal runs of non-space characters. 0 if there are no words.",
   [["Hello World"], ["  fly me   to the moon  "]], _g_sentence, [[""], ["   "], ["a"], ["a "], ["ab  cd"]],
   category="edge_cases")
def last_word_length(s):
    words = s.split(" ")
    words = [w for w in words if w]
    return len(words[-1]) if words else 0


@T(0, "def argmax(xs: list[int]) -> int",
   "Index of the largest value (the first one if it appears several times). Raise ValueError if xs is empty.",
   [[[3, 7, 7, 1]], [[-5]]], lambda r: [_ints(r, 10, -5, 5)], [[[]], [[1, 2]], [[2, 1]], [[0, 0, 0]], [[-3, -1, -1]]],
   errs=("ValueError",), category="edge_cases")
def argmax(xs):
    if not xs:
        raise ValueError("empty")
    best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best]:
            best = i
    return best


# ============================================================================================== optimización
def _slow_subarray_sum(xs, k):
    c = 0
    for i in range(len(xs)):
        s = 0
        for j in range(i, len(xs)):
            s += xs[j]
            c += s == k
    return c


@T(2, "def count_subarrays_sum(xs: list[int], k: int) -> int",
   "Number of non-empty contiguous subarrays of xs whose sum is exactly k. It must be fast for lists of tens of "
   "thousands of numbers.",
   [[[1, 1, 1], 2], [[1, -1, 0], 0]], lambda r: [_ints(r, 10, -3, 3), r.randint(-3, 3)],
   [[[], 0], [[0], 0], [[5], 5], [[1, 2, 3], 3], [[0, 0, 0], 0]],
   lambda r: [[[r.randint(-2, 2) for _ in range(40000)], 3], [[0] * 30000, 0]],
   category="optimizacion", slow=_slow_subarray_sum)
def count_subarrays_sum(xs, k):
    seen, s, c = {0: 1}, 0, 0
    for x in xs:
        s += x
        c += seen.get(s - k, 0)
        seen[s] = seen.get(s, 0) + 1
    return c


def _slow_longest_run(xs):
    best = 0
    for x in xs:
        n = 1
        while x + n in xs:
            n += 1
        best = max(best, n)
    return best


@T(2, "def longest_consecutive(xs: list[int]) -> int",
   "Length of the longest run of consecutive integers (v, v+1, v+2, ...) that all appear in xs, in any order. It must "
   "be fast for lists of tens of thousands of numbers.",
   [[[100, 4, 200, 1, 3, 2]], [[0, 0, -1]]], lambda r: [_ints(r, 12, -8, 8)],
   [[[]], [[5]], [[1, 1, 1]], [[3, 1, 2]], [[-2, 0, 2]]],
   lambda r: [[list(range(40000, 0, -1))], [[r.randint(-10 ** 9, 10 ** 9) for _ in range(30000)] + list(range(5000))]],
   category="optimizacion", slow=_slow_longest_run)
def longest_consecutive(xs):
    s, best = set(xs), 0
    for x in s:
        if x - 1 not in s:
            n = 1
            while x + n in s:
                n += 1
            best = max(best, n)
    return best


def _slow_nearby(xs, k):
    for i in range(len(xs)):
        for j in range(i + 1, min(len(xs), i + k + 1)):
            if xs[i] == xs[j]:
                return True
    return False


def _g_nearby(r):
    return [_ints(r, 10, 0, 6), r.randint(0, 4)]


@T(2, "def has_close_duplicate(xs: list[int], k: int) -> bool",
   "True if there are two indices i < j with xs[i] == xs[j] and j - i <= k. It must be fast even for large k.",
   [[[1, 2, 3, 1], 3], [[1, 2, 3, 1], 2]], lambda r: _bal(r, has_close_duplicate, _g_nearby),
   [[[], 1], [[1, 1], 0], [[1, 1], 1], [[1, 2, 1], 1], [[5, 5], 100], [[7, 0, 7], 2], [[2, 9, 9, 4], 1],
    [[3, 1, 2, 3], 3]],
   lambda r: [[list(range(60000)) + [0], 60000], [list(range(60000)) + [0], 59999]],
   category="optimizacion", slow=_slow_nearby)
def has_close_duplicate(xs, k):
    last = {}
    for i, x in enumerate(xs):
        if x in last and i - last[x] <= k:
            return True
        last[x] = i
    return False


def _slow_pair_diff(xs, d):
    return sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs)) if abs(xs[i] - xs[j]) == d)


@T(2, "def count_pairs_diff(xs: list[int], d: int) -> int",
   "Number of index pairs i < j with abs(xs[i] - xs[j]) == d (d >= 0). It must be fast for tens of thousands of numbers.",
   [[[1, 5, 3, 4, 2], 2], [[1, 1, 1], 0]], lambda r: [_ints(r, 12, -5, 5), r.randint(0, 4)],
   [[[], 1], [[3], 0], [[1, 2], 1], [[2, 2, 2, 2], 0], [[0, 10], 5]],
   lambda r: [[[r.randint(0, 30) for _ in range(30000)], 3], [[r.randint(-10 ** 6, 10 ** 6) for _ in range(30000)], 0]],
   category="optimizacion", slow=_slow_pair_diff)
def count_pairs_diff(xs, d):
    from collections import Counter
    c = Counter(xs)
    if d == 0:
        return sum(v * (v - 1) // 2 for v in c.values())
    return sum(v * c.get(x + d, 0) for x, v in c.items())


# ============================================================================================== refactoring
def _g_tri(r):
    u = r.random()
    if u < 0.15:
        s = [r.randint(-2, 0), r.randint(1, 6), r.randint(1, 6)]
    elif u < 0.4:
        a = r.randint(1, 6)
        s = [a, a, a] if r.random() < 0.4 else [a, a, r.randint(1, 2 * a + 1)]
    else:
        s = [r.randint(1, 9) for _ in range(3)]
    r.shuffle(s)
    return s


@T(1, "def triangle_kind(a: int, b: int, c: int) -> str",
   "Rewrite this function as triangle_kind(a, b, c) with simple logic and exactly the same behaviour:\n\n"
   "    def t(a, b, c):\n        if a <= 0 or b <= 0 or c <= 0:\n            raise ValueError('side')\n"
   "        if a + b <= c or a + c <= b or b + c <= a:\n            return 'none'\n        if a == b:\n"
   "            if b == c:\n                return 'equilateral'\n            return 'isosceles'\n"
   "        if b == c or a == c:\n            return 'isosceles'\n        return 'scalene'\n",
   [[3, 3, 3], [3, 4, 5]], _g_tri,
   [[1, 1, 2], [2, 2, 3], [0, 1, 1], [5, 3, 3], [4, 5, 4], [1, 2, 10]], errs=("ValueError",), category="refactoring")
def triangle_kind(a, b, c):
    if min(a, b, c) <= 0:
        raise ValueError("side")
    x, y, z = sorted((a, b, c))
    if x + y <= z:
        return "none"
    if x == z:
        return "equilateral"
    return "isosceles" if x == y or y == z else "scalene"


@T(1, "def shipping_cost(weight: float, express: bool) -> float",
   "Rewrite this function as shipping_cost(weight, express) with clear code and exactly the same results:\n\n"
   "    def cost(w, e):\n        c = 0\n        if w <= 0:\n            raise ValueError('weight')\n"
   "        if w <= 1:\n            c = 5\n        else:\n            if w <= 5:\n                c = 5 + (w - 1) * 2\n"
   "            else:\n                c = 13 + (w - 5) * 1.5\n        if e == True:\n            c = c * 2\n"
   "        return round(c, 2)\n",
   [[0.5, False], [7, True]], lambda r: [r.choice((r.randint(-1, 12), round(r.uniform(0, 12), 2))), r.random() < 0.5],
   [[1, False], [5.0, False], [5.5, True], [0, False], [1.01, True], [100, False]], errs=("ValueError",),
   category="refactoring")
def shipping_cost(weight, express):
    if weight <= 0:
        raise ValueError("weight")
    if weight <= 1:
        c = 5
    elif weight <= 5:
        c = 5 + (weight - 1) * 2
    else:
        c = 13 + (weight - 5) * 1.5
    return round(c * 2 if express else c, 2)


@T(1, "def bmi_class(weight_kg: float, height_m: float) -> str",
   "Rewrite this function as bmi_class(weight_kg, height_m) with flat, readable logic and exactly the same behaviour:\n\n"
   "    def f(w, h):\n        if h <= 0 or w <= 0:\n            raise ValueError('input')\n        b = w / (h * h)\n"
   "        if b < 18.5:\n            return 'under'\n        else:\n            if b < 25:\n                return 'normal'\n"
   "            else:\n                if b < 30:\n                    return 'over'\n        return 'obese'\n",
   [[70, 1.75], [50, 1.80]], lambda r: [r.choice((r.randint(-1, 150), round(r.uniform(30, 140), 1))),
                                        r.choice((round(r.uniform(1.4, 2.1), 2), r.choice((0, -1.7))))],
   [[1, 1], [18.5, 1], [25, 1], [30, 1], [0, 1.6], [60, 0]], errs=("ValueError",), category="refactoring")
def bmi_class(weight_kg, height_m):
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("input")
    b = weight_kg / (height_m * height_m)
    if b < 18.5:
        return "under"
    if b < 25:
        return "normal"
    return "over" if b < 30 else "obese"


@T(1, "def normalize_scores(xs: list[float]) -> list[float]",
   "Rewrite this function as normalize_scores(xs) with clear code and the same results:\n\n"
   "    def n(xs):\n        out = []\n        if len(xs) == 0:\n            return out\n        lo = min(xs)\n"
   "        hi = max(xs)\n        for x in xs:\n            if hi == lo:\n                out.append(0.0)\n"
   "            else:\n                out.append((x - lo) / (hi - lo))\n        return out\n",
   [[[1, 2, 3]], [[5, 5]]], lambda r: [_ints(r, 8, -10, 10)], [[[]], [[4]], [[-2, 2]], [[0, 10, 5]]],
   category="refactoring")
def normalize_scores(xs):
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]


# ============================================================================================== debugging
@T(1, "def count_words(text: str) -> dict",
   "Count how many times each word appears (words are separated by whitespace and compared in lowercase). The "
   "implementation below has bugs; write a correct version:\n\n"
   "    def count_words(text):\n        counts = {}\n        for w in text.split(' '):\n            if w in counts:\n"
   "                counts[w] += 1\n            counts[w] = 1\n        return counts\n",
   [["the cat The"], [""]], lambda r: [r.choice((" ", "  ", "\t")).join(r.choice(("a", "A", "b", "cat", "Cat"))
                                                                         for _ in range(r.randint(0, 7)))],
   [["  "], ["x"], ["a a a"], ["A\ta\nA"]], category="debugging")
def count_words(text):
    out = {}
    for w in text.lower().split():
        out[w] = out.get(w, 0) + 1
    return out


def _g_bsearch(r):
    xs = sorted(r.sample(range(-20, 20), r.randint(0, 9)))
    return [xs, r.choice(xs) if xs and r.random() < 0.65 else r.randint(-21, 21)]


@T(1, "def binary_search(xs: list[int], target: int) -> int",
   "xs is sorted in ascending order with distinct values. Return the index of target or -1. The implementation below "
   "has bugs; write a correct version:\n\n"
   "    def binary_search(xs, target):\n        lo, hi = 0, len(xs)\n        while lo < hi:\n"
   "            mid = (lo + hi) // 2\n            if xs[mid] < target:\n                lo = mid\n"
   "            else:\n                hi = mid\n        return lo\n",
   [[[1, 3, 5, 7], 5], [[1, 3, 5, 7], 4]], _g_bsearch,
   [[[], 1], [[1], 1], [[1], 2], [[1, 2], 2], [[1, 2], 0]],
   lambda r: [[list(range(0, 3 * 10 ** 6, 3)), 2999997], [list(range(10 ** 6)), -1]], category="debugging")
def binary_search(xs, target):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


@T(1, "def without_negatives(xs: list[int]) -> list[int]",
   "Return a new list without the negative numbers, keeping the order of the rest; xs must not be modified. The "
   "implementation below has bugs; write a correct version:\n\n"
   "    def without_negatives(xs):\n        for x in xs:\n            if x < 0:\n                xs.remove(x)\n"
   "        return xs\n",
   [[[1, -2, -3, 4]], [[-1, -1]]], lambda r: [_ints(r, 10, -4, 4)], [[[]], [[0]], [[-5]], [[-1, -2, 3, -4, -5]]],
   category="debugging")
def without_negatives(xs):
    return [x for x in xs if x >= 0]


@T(1, "def factorial(n: int) -> int",
   "n! for n >= 0 (0! = 1). Raise ValueError if n < 0. The implementation below has bugs; write a correct version:\n\n"
   "    def factorial(n):\n        result = 0\n        for i in range(1, n):\n            result *= i\n        return result\n",
   [[5], [0]], lambda r: [r.randint(-2, 30)], [[1], [2], [-1], [20], [60]], errs=("ValueError",),
   category="debugging")
def factorial(n):
    if n < 0:
        raise ValueError(n)
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


# ============================================================================================== comprensión
@T(1, "def trace_alternating(xs: list[int]) -> int",
   "Given this function:\n\n    def f(xs):\n        total = 0\n        for i, x in enumerate(xs):\n"
   "            if i % 2 == 0:\n                total += x\n            else:\n                total -= 2 * x\n"
   "        return total\n\nWrite trace_alternating(xs) returning the same value as f(xs).",
   [[[1, 2, 3]], [[5]]], lambda r: [_ints(r, 10, -9, 9)], [[[]], [[0, 1]], [[-1, -1, -1, -1]]],
   category="comprension")
def trace_alternating(xs):
    return sum(x if i % 2 == 0 else -2 * x for i, x in enumerate(xs))


@T(1, "def halving_steps(n: int) -> int",
   "Given this code:\n\n    steps = 0\n    while n > 1:\n        if n % 2 == 0:\n            n = n // 2\n"
   "        else:\n            n = n - 1\n        steps += 1\n\nReturn the final value of steps for the given n.",
   [[10], [1]], lambda r: [r.randint(-3, 10 ** 6)], [[0], [2], [3], [7], [-5], [2 ** 40]], category="comprension")
def halving_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else n - 1
        steps += 1
    return steps


@T(1, "def build_string(n: int) -> str",
   "Given this code:\n\n    s = ''\n    for i in range(n):\n        if i % 3 == 0:\n            s = s + 'a'\n"
   "        elif i % 3 == 1:\n            s = 'b' + s\n        else:\n            s = s[::-1]\n\n"
   "Return the final value of s for the given n.",
   [[3], [4]], lambda r: [r.randint(0, 25)], [[0], [1], [2], [6]], category="comprension")
def build_string(n):
    s = ""
    for i in range(n):
        if i % 3 == 0:
            s += "a"
        elif i % 3 == 1:
            s = "b" + s
        else:
            s = s[::-1]
    return s


@T(2, "def multiples_total(n: int) -> int",
   "Given this code:\n\n    s = 0\n    for i in range(1, n + 1):\n        if i % 3 == 0 or i % 5 == 0:\n"
   "            s += i\n\nReturn the final value of s for the given n. It must be fast for n up to 10**12.",
   [[10], [0]], lambda r: [r.choice((r.randint(-3, 60), r.randint(0, 10 ** 6)))], [[1], [3], [5], [15], [-4], [16]],
   lambda r: [[10 ** 12], [10 ** 12 - 7]], category="comprension")
def multiples_total(n):
    def tri(k):  # suma de los múltiplos de k en 1..n
        m = n // k
        return k * m * (m + 1) // 2
    return 0 if n < 1 else tri(3) + tri(5) - tri(15)


# ============================================================================================== especificación
@T(1, "def ordinal(n: int) -> str",
   "The number with its English ordinal suffix: 1st, 2nd, 3rd, 4th... 11th, 12th and 13th use 'th', as do 111th, 112th "
   "and 113th; 21st, 22nd, 23rd, 101st... Raise ValueError if n < 0.",
   [[1], [12]], lambda r: [r.choice((r.randint(-1, 30), r.randint(90, 130), r.randint(0, 10 ** 6)))],
   [[0], [2], [3], [11], [13], [21], [111], [112], [1003], [-1]], errs=("ValueError",), category="especificacion")
def ordinal(n):
    if n < 0:
        raise ValueError(n)
    if n % 100 in (11, 12, 13):
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }".replace(" ", "")


def _g_money(r):
    return [r.choice((r.randint(-10 ** 7, 10 ** 7) / 100, r.randint(-999, 999) / 100, float(r.randint(0, 10 ** 6))))]


@T(1, "def format_money(x: float) -> str",
   "Format an amount with exactly 2 decimals (rounded as Python's round), commas every 3 digits in the integer part, "
   "and negative amounts in parentheses without a minus sign: 1234.5 -> '1,234.50', -0.5 -> '(0.50)'.",
   [[1234.5], [-1234567.891]], _g_money, [[0.0], [-0.0], [999.999], [1000], [-5], [0.004]], category="especificacion")
def format_money(x):
    x = round(x, 2)
    s = f"{abs(x):,.2f}"
    return f"({s})" if x < 0 else s


@T(2, "def headline(s: str) -> str",
   "Title-case a headline: words are separated by single spaces; every word is capitalized (first letter upper, rest "
   "lower) except the small words a, an, the, of, in, on, and, or, to, which are lowercase unless they are the first "
   "or the last word.",
   [["the lord of the rings"], ["a tale OF two cities"]],
   lambda r: [" ".join(r.choice(("the", "a", "of", "and", "in", "war", "PEACE", "cat", "x")) for _ in range(r.randint(0, 6)))],
   [[""], ["of"], ["THE END"], ["war and peace"], ["in the end of it"]], category="especificacion")
def headline(s):
    small = {"a", "an", "the", "of", "in", "on", "and", "or", "to"}
    words = s.split(" ") if s else []
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        out.append(lw if lw in small and 0 < i < len(words) - 1 else lw[:1].upper() + lw[1:])
    return " ".join(out)


@T(1, "def plural(word: str, n: int) -> str",
   "Return '<n> <word>' with word in plural when n != 1: words ending in s, x, z, ch or sh add 'es'; words ending in a "
   "consonant followed by y change y to 'ies'; any other word adds 's'. Words are lowercase.",
   [["box", 2], ["city", 1]], lambda r: [r.choice(("box", "city", "day", "dish", "cat", "bus", "quiz", "church", "toy",
                                                   "fly")), r.randint(0, 3)],
   [["cat", 0], ["day", 2], ["fly", 2], ["bus", 1], ["match", 5]], category="especificacion")
def plural(word, n):
    if n == 1:
        return f"{n} {word}"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        w = word + "es"
    elif len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        w = word[:-1] + "ies"
    else:
        w = word + "s"
    return f"{n} {w}"


# ============================================================================================== multi-step
def _g_orders(r):
    stock = {k: r.randint(0, 5) for k in r.sample("abc", r.randint(1, 3))}
    orders = [[r.choice("abcd"), r.randint(-1, 4)] for _ in range(r.randint(0, 7))]
    return [stock, orders]


@T(2, "def fulfill_orders(stock: dict, orders: list[list]) -> list",
   "stock maps item -> units. Process orders [item, qty] in order: an order is filled only if the item exists, qty >= 1 "
   "and there are at least qty units (then they are removed); otherwise it is rejected and changes nothing. Return "
   "[the remaining stock (a new dict), the number of filled orders, the indices of rejected orders].",
   [[{"a": 3, "b": 1}, [["a", 2], ["b", 2], ["a", 1]]], [{"a": 1}, []]], _g_orders,
   [[{}, [["a", 1]]], [{"a": 0}, [["a", 0]]], [{"a": 2}, [["a", 2], ["a", 1]]], [{"x": 5}, [["x", -1], ["x", 5]]]],
   category="multi_step")
def fulfill_orders(stock, orders):
    left, filled, rejected = dict(stock), 0, []
    for i, (item, qty) in enumerate(orders):
        if item in left and qty >= 1 and left[item] >= qty:
            left[item] -= qty
            filled += 1
        else:
            rejected.append(i)
    return [left, filled, rejected]


@T(2, "def text_stats(text: str) -> dict",
   "Statistics of a text: {'chars': number of characters, 'words': number of whitespace-separated words, 'lines': "
   "number of lines (0 for '', otherwise newlines + 1), 'longest': the longest word (the first one on ties, '' if no "
   "words)}.",
   [["hello big world\nok"], [""]], lambda r: ["\n".join(" ".join(_word(r, 5, "abc") or "a" for _ in range(r.randint(0, 3)))
                                                     for _ in range(r.randint(1, 3)))],
   [["\n"], ["a"], ["  "], ["ab\ncd\n"], ["xx yy"]], category="multi_step")
def text_stats(text):
    words = text.split()
    longest = ""
    for w in words:
        if len(w) > len(longest):
            longest = w
    return {"chars": len(text), "words": len(words), "lines": text.count("\n") + 1 if text else 0, "longest": longest}


def _g_sessions(r):
    """Casi siempre el evento "natural" (in si está fuera, out si está dentro): la mayoría de listas tienen sesiones."""
    evs, t, inside = [], r.randint(0, 600), set()
    for _ in range(r.randint(0, 8)):
        t += r.choice((0, 5, 30, 61, 120))
        if t >= 24 * 60:
            break
        u = r.choice(("ana", "bo", "cy"))
        kind = ("out" if u in inside else "in") if r.random() < 0.8 else r.choice(("in", "out"))
        (inside.add if kind == "in" else inside.discard)(u)
        evs.append(f"{t // 60:02d}:{t % 60:02d} {u} {kind}")
    return [evs]


@T(2, "def session_totals(events: list[str]) -> dict",
   "events are 'HH:MM user in' or 'HH:MM user out' in chronological order (same day). Return a dict user -> total "
   "minutes logged in, counting only complete in -> out pairs. An 'in' while the user is already in restarts the "
   "session (the earlier 'in' is forgotten); an 'out' without an open session is ignored. Users with no complete "
   "session are left out.",
   [[["09:00 ana in", "09:30 ana out", "10:00 bo in"]], [[]]], _g_sessions,
   [[["08:00 x out"]], [["08:00 x in", "08:00 x out"]], [["08:00 x in", "08:10 x in", "08:15 x out"]],
    [["07:59 a in", "08:01 a out", "09:00 a in", "10:00 a out", "10:00 a out"]],
    [["00:00 a in", "23:59 a out", "23:59 b in"]]], category="multi_step")
def session_totals(events):
    start, total = {}, {}
    for e in events:
        clock, user, kind = e.split()
        h, m = clock.split(":")
        t = int(h) * 60 + int(m)
        if kind == "in":
            start[user] = t
        elif user in start:
            total[user] = total.get(user, 0) + t - start.pop(user)
    return total


def _g_grades(r):
    rows = []
    for _ in range(r.randint(0, 7)):
        rows.append([r.choice(("ana", "bo", "cy")), r.randint(-5, 105)])
    return [rows]


@T(2, "def grade_report(rows: list[list]) -> dict",
   "rows are [student, score]. Scores outside 0..100 are ignored. Return a dict student -> [average score rounded to 1 "
   "decimal, letter], where the letter uses the unrounded average: 'A' if it is >= 90, 'B' >= 80, 'C' >= 70 and 'F' "
   "otherwise. Students with no valid scores are left out.",
   [[[["ana", 90], ["bo", 70], ["ana", 80]]], [[]]], _g_grades,
   [[[["x", -1]]], [[["x", 100], ["x", 0]]], [[["a", 89.95]]], [[["a", 70], ["b", 69]]],
    [[["a", 90], ["b", 80], ["b", 80], ["c", 101]]]], category="multi_step")
def grade_report(rows):
    acc = {}
    for name, score in rows:
        if 0 <= score <= 100:
            acc.setdefault(name, []).append(score)
    out = {}
    for name, xs in acc.items():
        avg = sum(xs) / len(xs)
        letter = "A" if avg >= 90 else "B" if avg >= 80 else "C" if avg >= 70 else "F"
        out[name] = [round(avg, 1), letter]
    return out


# ============================================================================================== extra (arrays / strings)
@T(0, "def pairwise_diffs(xs: list[int]) -> list[int]", "out[i] = xs[i + 1] - xs[i] for every consecutive pair.",
   [[[1, 4, 9, 16]], [[3]]], lambda r: [_ints(r, 12, -9, 9)], [[[]], [[1, 1]], [[5, 2]], [[-1, 0, 1]]],
   category="arrays")
def pairwise_diffs(xs):
    return [b - a for a, b in zip(xs, xs[1:])]


@T(1, "def common_elements(a: list[int], b: list[int]) -> list[int]",
   "Values that appear in both lists, without repetitions, in the order of their first appearance in a.",
   [[[3, 1, 2, 3], [2, 3, 5]], [[1], []]], lambda r: [_ints(r, 10, 0, 6), _ints(r, 10, 0, 6)],
   [[[], []], [[1, 1], [1]], [[1, 2], [3, 4]], [[5, 4, 5], [4, 5, 5]]],
   lambda r: [[list(range(100000)), list(range(50000, 150000))]], category="arrays")
def common_elements(a, b):
    sb, seen, out = set(b), set(), []
    for x in a:
        if x in sb and x not in seen:
            seen.add(x)
            out.append(x)
    return out
