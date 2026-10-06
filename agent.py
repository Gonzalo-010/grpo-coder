"""Agente de código local sobre un repositorio. Trabaja en una copia; el repo sólo cambia si aceptas el diff.

python agent.py RUTA_REPO "tarea" [--test "python -m pytest -x -q"] [--iters 4] [--checkpoint DIR] [--config F]
python agent.py --bench 10 [--checkpoint DIR] [--test "python test_solution.py"]   # benchmark de mini-repos
"""
import argparse

from common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?")
    ap.add_argument("task", nargs="?")
    ap.add_argument("--bench", type=int, metavar="N", help="benchmark: N mini-repos con un bug real (agent/bench.py)")
    ap.add_argument("--bench-seed", type=int, default=0)
    ap.add_argument("--test", help="comando de tests, sin shell (defecto: agent.test_cmd)")
    ap.add_argument("--iters", type=int, help="rondas editar -> testear (defecto: agent.iters)")
    ap.add_argument("--checkpoint", help="checkpoint a usar (defecto: modelo base)")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from agent.loop import Session
    from model.infer import Engine

    if a.bench:
        from agent.bench import run_bench
        rep = run_bench(Engine(cfg, a.checkpoint), cfg, a.bench, a.bench_seed, a.test, a.iters)
        print({k: v for k, v in rep.items() if k != "repos"})
        return
    if not a.repo or not a.task:
        ap.error("hacen falta RUTA_REPO y \"tarea\" (o --bench N)")

    s = Session(a.repo, a.task, cfg, a.test)
    print(f"copia de trabajo: {s.work}")
    eng = Engine(cfg, a.checkpoint)

    def show(it):
        if it is None:
            print(f"tests iniciales: {'pasan' if s.baseline['passed'] else 'fallan'} (exit {s.baseline['code']})")
            return
        print(f"iteración {it['i']}: {', '.join(it['files']) or 'sin cambios'} -> tests "
              f"{'pasan' if it['passed'] else 'fallan'} ({it['secs']}s)")
        for r in it["rejected"]:
            print("  edición rechazada:", r)

    s.run(eng, a.iters, show)
    if not s.last_diff:
        print("el agente no propuso cambios")
        s.reject()
        return
    print(s.last_diff)
    try:
        answer = input(f"¿Aplicar estos cambios a {s.repo}? [s/N] ")
    except EOFError:  # sin terminal (stdin cerrado): no se aplica nada
        answer = ""
    if answer.strip().lower() == "s":
        s.accept()
        print("aplicado")
    else:
        s.reject()
        print("descartado")


if __name__ == "__main__":
    main()
