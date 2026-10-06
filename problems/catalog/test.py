"""Catálogo, split test (16 problemas, las 14 categorías). Ni se entrena ni se usa para elegir nada: sólo
`train.py eval --split test`. Familias propias, distintas de las de train y val."""
from problems.problems import P, _bal, _ints, _word


def S(level, sig, doc, ex, gen, edge=(), large=None, errs=(), *, category, family, slow=None):
    return P(level, sig, doc, ex, gen, edge, large, errs, split="test", category=category, family=family, slow=slow,
             source="catalog")


# ---------- algoritmos ----------
@S(3, "def lcs_length(a: str, b: str) -> int",
   "Length of the longest common subsequence of a and b (characters in the same order, not necessarily adjacent).",
   [["abcde", "ace"], ["abc", "def"]], lambda r: [_word(r, 10, "abc"), _word(r, 10, "abc")],
   [["", ""], ["abc", ""], ["abc", "abc"], ["abc", "def"], ["aaaa", "aa"]],
   lambda r: [["".join(r.choice("abcd") for _ in range(1000)), "".join(r.choice("abcd") for _ in range(1000))]],
   category="algoritmos", family="lcs_length")
def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ch == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _g_islands(r):
    rows, cols = r.randint(1, 6), r.randint(1, 6)
    return [["".join("1" if r.random() < 0.5 else "0" for _ in range(cols)) for _ in range(rows)]]


@S(2, "def count_islands(grid: list[str]) -> int",
   "grid is a list of equal-length strings of '1' (land) and '0' (water). Count the islands: groups of land cells "
   "connected horizontally or vertically.",
   [[["110", "010", "001"]], [["0"]]], _g_islands,
   [[[]], [["0"]], [["1"]], [["11", "01"]], [["101", "010", "101"]]],
   lambda r: [[["1" * 400 for _ in range(400)]], [["".join(r.choice("10") for _ in range(400)) for _ in range(400)]]],
   category="algoritmos", family="count_islands")
def count_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and not seen[r][c]:
                count += 1
                stack = [(r, c)]
                seen[r][c] = True
                while stack:
                    y, x = stack.pop()
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < rows and 0 <= nx < cols and grid[ny][nx] == "1" and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
    return count


# ---------- arrays ----------
def _g_square(r):
    n = r.randint(0, 5)
    return [[[r.randint(-9, 9) for _ in range(n)] for _ in range(n)]]


@S(1, "def rotate_matrix(m: list[list[int]]) -> list[list[int]]",
   "Return a new square matrix equal to m rotated 90 degrees clockwise.",
   [[[[1, 2], [3, 4]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], _g_square,
   [[[]], [[[1]]], [[[1, 2], [3, 4]]]], category="arrays", family="rotate_matrix")
def rotate_matrix(m):
    n = len(m)
    return [[m[n - 1 - j][i] for j in range(n)] for i in range(n)]


# ---------- strings ----------
def _scramble(r, s):
    cs = list(s)
    r.shuffle(cs)
    return "".join(c.upper() if r.random() < 0.3 else c for c in cs) + r.choice(("", " ", "!", ", "))


def _anagram_pair(r):
    w = _word(r, 6, "abcd")
    return [w, _scramble(r, w) if r.random() < 0.6 else _word(r, 6, "abcd")]


@S(1, "def is_anagram(a: str, b: str) -> bool",
   "True if a and b are anagrams of each other when case is ignored and every character that is not a letter is "
   "ignored.",
   [["Listen", "Silent!"], ["abc", "abd"]],
   lambda r: _bal(r, is_anagram, _anagram_pair),
   [["", ""], ["a", "A"], ["ab", "a"], ["a b!", "B,A"], ["abc", "abd"]], category="strings", family="is_anagram")
def is_anagram(a, b):
    def norm(s):
        return sorted(c.lower() for c in s if c.isalpha())
    return norm(a) == norm(b)


# ---------- estructuras de datos ----------
@S(2, "def prefix_counts(words: list[str], prefixes: list[str]) -> list[int]",
   "For each prefix, how many of the words (counting repetitions) start with it. Return the counts in the order of "
   "prefixes. It must be fast for tens of thousands of words and prefixes.",
   [[["apple", "app", "bat"], ["ap", "b", "c"]], [["a"], [""]]],
   lambda r: [[_word(r, 5, "ab") for _ in range(r.randint(0, 10))], [_word(r, 3, "ab") for _ in range(r.randint(0, 6))]],
   [[[], ["a"]], [["a", "ab"], [""]], [["abc"], ["abcd"]], [["ab", "ab"], ["ab", "a", "b"]]],
   lambda r: [[["".join(r.choice("abc") for _ in range(20)) for _ in range(20000)],
               ["".join(r.choice("abc") for _ in range(r.randint(1, 6))) for _ in range(20000)]]],
   category="estructuras", family="prefix_counts")
def prefix_counts(words, prefixes):
    counts = {}
    for w in words:
        for k in range(len(w) + 1):
            counts[w[:k]] = counts.get(w[:k], 0) + 1
    return [counts.get(p, 0) for p in prefixes]


# ---------- matemáticas ----------
def _g_gcd(r):
    f = r.choice((1, 2, 3, 4, 6, 10))
    return [[f * r.randint(-12, 12) for _ in range(r.randint(0, 6))]]


@S(1, "def gcd_list(xs: list[int]) -> int",
   "Greatest common divisor of all the integers in xs (never negative). The gcd of numbers that are all 0 is 0. Raise "
   "ValueError if xs is empty.",
   [[[12, -18, 30]], [[7]]], _g_gcd, [[[]], [[0]], [[0, 0]], [[-4, 6]], [[-7]], [[0, 5]]], errs=("ValueError",),
   category="matematicas", family="gcd_list")
def gcd_list(xs):
    if not xs:
        raise ValueError("empty")
    g = 0
    for x in xs:
        a, b = g, abs(x)
        while b:
            a, b = b, a % b
        g = a
    return g


# ---------- parsing ----------
def _g_csv(r):
    fields = []
    for _ in range(r.randint(1, 4)):
        w = _word(r, 4, "ab ")
        if r.random() < 0.3:
            w += r.choice((",", '"', ""))
            fields.append('"' + w.replace('"', '""') + '"')
        else:
            fields.append(w)
    line = ",".join(fields)
    u = r.random()
    if u < 0.08:
        line += ',"abc'
    elif u < 0.15:
        line = '"x"y,' + line
    return [line]


@S(2, "def parse_csv_line(line: str) -> list[str]",
   "Split one CSV line into fields separated by commas. A field may be wrapped in double quotes; then it may contain "
   "commas, and a double quote inside it is written as two double quotes. Spaces are kept. Raise ValueError for an "
   "unterminated quoted field or for any character other than a comma right after a closing quote.",
   [['a,"b,c",d'], ['"say ""hi""",x']], _g_csv,
   [[""], ["a,b"], ['"a,b",c'], ['"a""b"'], ['"abc'], ['"a"x,b'], ["a,"], [",,"]], errs=("ValueError",),
   category="parsing", family="parse_csv_line")
def parse_csv_line(line):
    fields, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            buf = []
            while True:
                if i >= n:
                    raise ValueError("unterminated quote")
                if line[i] == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        buf.append('"')
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    buf.append(line[i])
                    i += 1
            fields.append("".join(buf))
            if i < n and line[i] != ",":
                raise ValueError("text after closing quote")
        else:
            j = line.find(",", i)
            j = n if j < 0 else j
            fields.append(line[i:j])
            i = j
        if i >= n:
            return fields
        i += 1


def _g_version(r):
    def v():
        return ".".join(str(r.randint(0, 12)) for _ in range(r.randint(1, 4)))
    a = v()
    b = a if r.random() < 0.2 else (a + ".0" if r.random() < 0.15 else v())
    if r.random() < 0.12:
        a = r.choice(("1..2", "1.a", "", "1.", ".1", "1.-1"))
    return [a, b]


@S(1, "def version_compare(a: str, b: str) -> int",
   "Compare version strings such as '1.2.10' and '1.10': split them on '.', compare the parts as numbers from left to "
   "right, and treat missing parts as 0. Return -1, 0 or 1. Raise ValueError if some part is empty or is not made "
   "only of ASCII digits.",
   [["1.2.10", "1.10"], ["1", "1.0"]], _g_version,
   [["1", "1.0"], ["2", "1.9.9"], ["1.a", "1"], ["", "1"], ["1.", "1"], ["0.0.1", "0.0.01"]], errs=("ValueError",),
   category="parsing", family="version_compare")
def version_compare(a, b):
    def parse(v):
        parts = v.split(".")
        if any(not p or not all(c in "0123456789" for c in p) for p in parts):
            raise ValueError(v)
        return [int(p) for p in parts]
    x, y = parse(a), parse(b)
    n = max(len(x), len(y))
    x += [0] * (n - len(x))
    y += [0] * (n - len(y))
    return (x > y) - (x < y)


# ---------- seguridad ----------
def _g_redact(r):
    words = []
    for _ in range(r.randint(0, 6)):
        u = r.random()
        if u < 0.35:
            local = _word(r, 3, "ab.").strip(".") or "a"
            words.append(local + "@" + (_word(r, 3, "xy") or "x") + "." + r.choice(("com", "org", "io", "c")))
        elif u < 0.5:
            words.append(r.choice(("a@b.c", "@x.com", "me@", "x@y..com", "u.v@ex-ample.org", "a_b@c-d.ef.gh")))
        else:
            words.append(_word(r, 5, "abc") or "z")
    return [" ".join(words) + r.choice(("", ".", "!"))]


@S(2, "def redact_emails(text: str) -> str",
   "Replace every e-mail address in text with '[email]'. An address is: one or more letters, digits, '.', '_' or '-'; "
   "then '@'; then one or more labels made of letters, digits or '-', each followed by a dot; then a final label of at "
   "least 2 letters (for example a.b-c@mail.example.org). Only ASCII letters and digits count.",
   [["mail me at ana.p@example.com today"], ["no address here"]], _g_redact,
   [[""], ["a@b.co"], ["mail me: x.y-z@sub.example.org!"], ["a@b.c"], ["@b.com"], ["u@v.com and w@x.org"]],
   category="seguridad", family="redact_emails")
def redact_emails(text):
    import re
    return re.sub(r"[A-Za-z0-9._-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}", "[email]", text)


# ---------- edge cases ----------
@S(1, "def safe_divide_all(pairs: list[list[int]]) -> list",
   "For each pair [a, b] return a / b as a float, or None when b is 0.",
   [[[[1, 2], [3, 0]]], [[]]], lambda r: [[[r.randint(-5, 5), r.randint(-3, 3)] for _ in range(r.randint(0, 6))]],
   [[[]], [[[1, 0]]], [[[0, 5]]], [[[-3, 2]]], [[[0, 0]]]], category="edge_cases", family="safe_divide_all")
def safe_divide_all(pairs):
    return [None if b == 0 else a / b for a, b in pairs]


# ---------- optimización ----------
def _l_unique(r):
    """Casos grandes que obligan a la versión O(n^2) a hacer ~n^2/2 pasos de verdad: bloques LARGOS de caracteres
    todos distintos. (Con caracteres aleatorios de un alfabeto de 3000, la ventana sin repetidos mide ~70 por el
    efecto cumpleaños y la versión ingenua es en la práctica O(70 n): en una CPU rápida terminaba antes del timeout.)"""
    pool = list(range(0x100, 0xD800))  # 55040 code points válidos (sin sustitutos), disjuntos del ruido 'abc'
    r.shuffle(pool)
    noise = lambda n: "".join(r.choice("abc") for _ in range(n))
    return [["".join(map(chr, pool[:50000]))],                                   # 50000 distintos: ~1.25e9 pasos
            [noise(20000) + "".join(map(chr, pool[-40000:])) + noise(20000)]]    # bloque de 40000 entre ruido


def _slow_longest_unique_substring(s):
    best = 0
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.add(s[j])
            best = max(best, j - i + 1)
    return best


@S(2, "def longest_unique_substring(s: str) -> int",
   "Length of the longest substring of s whose characters are all different. It must be fast for strings of "
   "hundreds of thousands of characters.",
   [["abcabcbb"], ["bbbbb"]], lambda r: [_word(r, 15, "abcd")],
   [[""], ["a"], ["aaaa"], ["abba"], ["dvdf"]],
   _l_unique, category="optimizacion", family="longest_unique_substring", slow=_slow_longest_unique_substring)
def longest_unique_substring(s):
    last, start, best = {}, 0, 0
    for i, ch in enumerate(s):
        if ch in last and last[ch] >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best


# ---------- refactoring ----------
def _g_text(r):
    vocab = ("the", "cat", "dog", "The", "Cat", "a", "sat")
    words = [r.choice(vocab) + r.choice(("", "", ".", ",", "!")) for _ in range(r.randint(0, 12))]
    return [" ".join(words), r.randint(0, 5)]


@S(2, "def top_words(text: str, n: int) -> list[list]",
   "Rewrite this function as top_words(text, n) with clear code and exactly the same behaviour:\n\n"
   "    def tw(text, n):\n        counts = {}\n        for w in text.lower().split():\n"
   "            w = w.strip('.,;:!?')\n            if w == '':\n                continue\n"
   "            if w in counts:\n                counts[w] = counts[w] + 1\n            else:\n"
   "                counts[w] = 1\n        items = []\n        for k in counts:\n"
   "            items.append([k, counts[k]])\n        items.sort(key=lambda kv: (-kv[1], kv[0]))\n"
   "        return items[:n]\n",
   [["the cat. The dog!", 2], ["a a b", 1]], _g_text,
   [["", 3], ["a a b", 1], ["B b. a!", 5], ["...", 2], ["x y", 0]], category="refactoring", family="top_words")
def top_words(text, n):
    counts = {}
    for w in text.lower().split():
        w = w.strip(".,;:!?")
        if w:
            counts[w] = counts.get(w, 0) + 1
    return sorted(([k, v] for k, v in counts.items()), key=lambda kv: (-kv[1], kv[0]))[:n]


# ---------- debugging ----------
@S(1, "def merge_sorted(a: list[int], b: list[int]) -> list[int]",
   "Merge two ascending lists into one ascending list, keeping duplicates. The implementation below has a bug; write a "
   "correct version:\n\n"
   "    def merge(a, b):\n        out = []\n        i = j = 0\n        while i < len(a) and j < len(b):\n"
   "            if a[i] <= b[j]:\n                out.append(a[i])\n                i += 1\n            else:\n"
   "                out.append(b[j])\n                j += 1\n        return out\n",
   [[[1, 3, 5], [2, 4]], [[], [1]]], lambda r: [sorted(_ints(r, 8, -10, 10)), sorted(_ints(r, 8, -10, 10))],
   [[[], []], [[1], []], [[], [2]], [[1, 3], [2]], [[1, 1], [1]]],
   lambda r: [[sorted(r.randint(-10 ** 6, 10 ** 6) for _ in range(100000)),
               sorted(r.randint(-10 ** 6, 10 ** 6) for _ in range(100000))]], category="debugging",
   family="merge_sorted")
def merge_sorted(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out + a[i:] + b[j:]


# ---------- comprensión ----------
@S(1, "def loop_count(n: int, m: int) -> int",
   "Given this code:\n\n    count = 0\n    for i in range(n):\n        for j in range(i, m):\n"
   "            if (i + j) % 3 == 0:\n                count += 1\n\n"
   "Return the final value of count for the given n and m.",
   [[3, 3], [0, 5]], lambda r: [r.randint(0, 40), r.randint(0, 40)], [[0, 5], [5, 0], [1, 1], [3, 3], [6, 2]],
   category="comprension", family="loop_count")
def loop_count(n, m):
    count = 0
    for i in range(n):
        for j in range(i, m):
            if (i + j) % 3 == 0:
                count += 1
    return count


# ---------- especificación ----------
@S(1, "def interleave_chunks(a: str, b: str, k: int) -> str",
   "Build a string by alternately taking k characters from a and k characters from b, starting with a; when one of "
   "them runs out, append the rest of the other. Raise ValueError if k < 1.",
   [["abcdef", "12", 2], ["abc", "12", 1]],
   lambda r: [_word(r, 8, "abc"), _word(r, 8, "123"), r.randint(0, 4) if r.random() < 0.1 else r.randint(1, 4)],
   [["", "", 1], ["abc", "", 2], ["", "xy", 1], ["ab", "xy", 0], ["ab", "xy", 5]], errs=("ValueError",),
   category="especificacion", family="interleave_chunks")
def interleave_chunks(a, b, k):
    if k < 1:
        raise ValueError(k)
    out, i = [], 0
    while i < len(a) or i < len(b):
        out.append(a[i:i + k])
        out.append(b[i:i + k])
        i += k
    return "".join(out)


# ---------- multi-step ----------
def _g_ledger(r):
    names = r.sample("abc", r.randint(0, 3))
    accounts = {n: r.randint(0, 20) for n in names}
    pool = names + ["z"]
    txs = []
    for _ in range(r.randint(0, 10)):
        kind, amt = r.choice(("deposit", "withdraw", "transfer", "transfer")), r.randint(-2, 15)
        if kind == "transfer":
            txs.append([kind, r.choice(pool), r.choice(pool), amt])
        else:
            txs.append([kind, r.choice(pool), amt])
    return [accounts, txs]


@S(3, "def bank_ledger(accounts: dict, txs: list[list]) -> list",
   "Apply transactions to accounts (a dict name -> balance). Transactions are ['deposit', name, amount], "
   "['withdraw', name, amount] and ['transfer', src, dst, amount]. A transaction is rejected (and has no effect) if "
   "an account does not exist, the amount is not positive, or a withdraw or transfer would leave a negative balance. "
   "Return [final balances (a new dict), list of the indices of the rejected transactions].",
   [[{"a": 10, "b": 0}, [["transfer", "a", "b", 4], ["withdraw", "b", 5]]], [{}, []]], _g_ledger,
   [[{}, []], [{"a": 5}, [["withdraw", "a", 6]]], [{"a": 5, "b": 0}, [["transfer", "a", "b", 5]]],
    [{"a": 1}, [["deposit", "a", 0]]], [{"a": 1}, [["deposit", "x", 3]]], [{"a": 3}, [["transfer", "a", "a", 3]]]],
   category="multi_step", family="bank_ledger")
def bank_ledger(accounts, txs):
    bal = dict(accounts)
    rejected = []
    for i, t in enumerate(txs):
        kind, amt, names = t[0], t[-1], t[1:-1]
        ok = amt > 0 and all(n in bal for n in names)
        if ok and kind == "deposit" and len(t) == 3:
            bal[t[1]] += amt
        elif ok and kind == "withdraw" and len(t) == 3 and bal[t[1]] >= amt:
            bal[t[1]] -= amt
        elif ok and kind == "transfer" and len(t) == 4 and bal[t[1]] >= amt:
            bal[t[1]] -= amt
            bal[t[2]] += amt
        else:
            rejected.append(i)
    return [bal, rejected]
