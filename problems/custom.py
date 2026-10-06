"""Tareas libres (panel o CLI): firma + descripción + tests `assert f(...) == valor`.

Se comportan como un Problem para score(): los asserts son los casos visibles (salen en el prompt, como los
ejemplos del catálogo) y no hay ocultos ni generados. Sólo se aceptan argumentos y valores literales, que se leen
con ast.literal_eval en el proceso confiable: nada del texto de los tests se ejecuta fuera del sandbox.
"""
import ast, json, re

from problems.problems import Case
from sandbox.harness import enc


def parse_tests(text, entry):
    cases = []
    try:
        body = ast.parse(text).body
    except (SyntaxError, RecursionError, MemoryError) as e:
        raise ValueError(f"los tests no son Python válido: {getattr(e, 'msg', type(e).__name__)}") from None
    for node in body:
        ok = (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
              and isinstance(node.test.ops[0], ast.Eq) and isinstance(node.test.left, ast.Call)
              and isinstance(node.test.left.func, ast.Name) and node.test.left.func.id == entry
              and not node.test.left.keywords)
        if not ok:
            raise ValueError(f"sólo se admiten líneas `assert {entry}(...) == valor`: {ast.unparse(node)[:80]}")
        try:
            args = [ast.literal_eval(x) for x in node.test.left.args]
            want = enc(ast.literal_eval(node.test.comparators[0]))
            json.dumps(args)  # los argumentos viajan como JSON al sandbox: un set o bytes fallaría allí, no aquí
        except (ValueError, TypeError, RecursionError, MemoryError):
            raise ValueError(f"argumentos y valor esperado deben ser literales JSON (números, textos, listas, dicts): "
                             f"{ast.unparse(node)[:80]}") from None
        cases.append(Case(args, want, None))
    if not cases:
        raise ValueError("hace falta al menos un assert")
    return cases


class Task:
    level = 0

    def __init__(self, sig, doc, tests):
        m = re.match(r"\s*def\s+([A-Za-z_]\w*)\s*\(", sig)
        if not m:
            raise ValueError("la firma debe empezar por `def nombre(`")
        self.entry = m.group(1)
        self.id = "task:" + self.entry  # nunca coincide con un id del catálogo (improve lee los intentos por id)
        self.sig, self.doc = sig.strip().rstrip(":"), doc.strip()
        self.cases = parse_tests(tests, self.entry)

    def visible(self):
        return self.cases

    def hidden(self):
        return []

    def generated(self, seed, n):
        return []

    def prompt(self):
        ex = [f"# {self.entry}({', '.join(map(repr, c.args))}) -> {json.loads(c.exp)!r}" for c in self.cases]
        return f'{self.sig}:\n    """{self.doc}"""\n' + "\n".join(ex)
