"""Benchmark del agente: mini-repos con un bug real (mutante que falla tests de un problema train) y sus tests.

Éxito = los tests pasan al final SIN haber tocado el fichero de tests. Los tests se escriben como funciones test_*
(pytest) y además el fichero se ejecuta solo (`python test_solution.py`), así sirve sin pytest instalado.
"""
import json, os, shutil, time

from common import new_run_dir, opt
from problems import repair
from problems.problems import select

TEST_FILE, SRC_FILE = "test_solution.py", "solution.py"
# Comando de tests del benchmark: el fichero de tests se ejecuta solo. NO el de la config (pytest), que puede no estar
# instalado: en la primera ejecución en la RTX 5070 todos los repos fallaban con "No module named pytest" (0/20 en
# las dos configuraciones, aunque el modelo hubiera arreglado el bug).
TEST_CMD = f"python {TEST_FILE}"


def _test_source(entry, cases):
    out = ["import json", "import math", "", f"from solution import {entry}", "", "",
           "def _eq(a, b):",
           "    if isinstance(a, float) or isinstance(b, float):",
           "        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)",
           "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):",
           "        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))",
           "    if isinstance(a, dict) and isinstance(b, dict):",
           "        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)",
           "    return a == b", ""]
    for i, c in enumerate(cases):
        args = json.dumps(c.args)
        out += ["", f"def test_case_{i}():", f"    args = json.loads({args!r})"]
        if c.exc:
            out += ["    try:", f"        {entry}(*args)", f"    except {c.exc}:", "        return",
                    f"    raise AssertionError('expected {c.exc}')"]
        else:
            out += [f"    want = json.loads({c.exp!r})", f"    got = {entry}(*args)",
                    "    assert _eq(json.loads(json.dumps(got)), want), f'got {got!r}, want {want!r}'"]
    out += ["", "", "if __name__ == '__main__':", "    import sys", "    fails = 0",
            "    tests = [f for n, f in sorted(globals().items()) if n.startswith('test_')]", "    for f in tests:",
            "        try:", "            f()", "        except Exception as e:", "            fails += 1",
            "            print('FAILED', f.__name__, '-', type(e).__name__ + ':', e)",
            "    print(f'{len(tests) - fails} passed, {fails} failed')", "    sys.exit(1 if fails else 0)", ""]
    return "\n".join(out)


def make_repo(path, task):
    """Escribe el mini-repo de una RepairTask: solution.py con el bug y test_solution.py con casos visibles,
    bordes y aleatorios (los esperados salen de la referencia). Devuelve el texto de la tarea."""
    os.makedirs(path, exist_ok=True)
    cases = task.visible() + task.hidden()[:len(task.base.edge)] + task.generated(5, 6)
    with open(os.path.join(path, SRC_FILE), "w") as f:
        f.write(task.buggy.rstrip() + "\n")
    with open(os.path.join(path, TEST_FILE), "w") as f:
        f.write(_test_source(task.entry, cases))
    return (f"The function `{task.entry}` in {SRC_FILE} has a bug and the tests in {TEST_FILE} fail. "
            f"Fix {SRC_FILE} so that all the tests pass. Do not modify {TEST_FILE}. "
            f"Specification: {task.base.doc.splitlines()[0]}")


def bench_tasks(n, seed=0):
    base = select()  # los 19 originales (train)
    per = max(1, -(-n // len(base)))  # n > 19: varios bugs por problema
    tasks = repair.tasks(base, per_problem=per, seed=seed)
    return sorted(tasks, key=lambda t: (int(t.id.rsplit(":", 1)[1]), t.id))[:n]


def run_bench(engine, cfg, n=10, seed=0, test_cmd=None, iters=None, root=None):
    """Ejecuta el agente sobre n mini-repos. Devuelve y guarda (bench.json) éxito, iteraciones, repeticiones."""
    from agent.loop import Session
    out_dir = root or new_run_dir(cfg, "agent_bench")
    try:
        import torch
    except ImportError:  # tests con un motor doble
        torch = None
    rows = []
    for i, t in enumerate(bench_tasks(n, seed)):
        if torch is not None:
            torch.manual_seed(seed * 1_000_003 + i)  # cada repo arranca del mismo RNG en cualquier config: A/B pareado
        repo = os.path.join(out_dir, t.id.replace(":", "_"), "repo")
        task_text = make_repo(repo, t)
        t0, s = time.time(), None
        try:
            s = Session(repo, task_text, cfg, test_cmd or TEST_CMD, root=os.path.join(out_dir, "sessions"))
            s.run(engine, iters)
            tampered = TEST_FILE in s.changed
            ok = s.status == "passed" and not tampered
            rows.append(dict(id=t.id, bug=t.desc, solved=ok, tampered=tampered, iters=len(s.iters),
                             repeats=sum(bool(it.get("repeat")) for it in s.iters),
                             reverts=sum(bool(it.get("reverted")) for it in s.iters),
                             stopped=next((it["stopped"] for it in s.iters if it.get("stopped")), None),
                             secs=round(time.time() - t0, 1)))
        except Exception as e:  # un repo que rompe no debe tumbar el benchmark entero
            rows.append(dict(id=t.id, bug=t.desc, solved=False, error=repr(e)[:200]))
        finally:
            if s is not None and hasattr(s, "work"):
                shutil.rmtree(s.work, ignore_errors=True)
    rep = dict(n=len(rows), success=round(sum(r["solved"] for r in rows) / len(rows), 4) if rows else None,
               mean_iters=round(sum(r.get("iters", 0) for r in rows) / len(rows), 2) if rows else None,
               tampered=sum(bool(r.get("tampered")) for r in rows), repeats=sum(r.get("repeats", 0) for r in rows),
               reverts=sum(r.get("reverts", 0) for r in rows), secs=round(sum(r.get("secs", 0) for r in rows), 1),
               rollback=bool(opt(cfg, "agent.rollback", True)), patience=opt(cfg, "agent.patience", 0),
               test_cmd=test_cmd or TEST_CMD, model=cfg.model,
               parse_version=2,  # 2: bloque sin cabecera al fichero de la tarea y plantilla rechazada (agent/edits.py)
               repos=rows, dir=out_dir)
    with open(os.path.join(out_dir, "bench.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep
