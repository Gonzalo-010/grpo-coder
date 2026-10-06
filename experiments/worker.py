"""Cola de trabajos del proyecto: ejecuta, de uno en uno, los comandos del proyecto que se dejen en runs/jobs/queue/.

python -m experiments.worker [--idle-hours 8]

Para qué: lanzar tests y experimentos largos en la GPU sin tener que estar delante (p. ej. los que prepara un
asistente que sólo puede escribir en la carpeta del proyecto). Lo arrancas tú y lo paras con Ctrl+C.

Seguridad: sólo ejecuta el intérprete con el que lo arrancaste sobre scripts de ESTE proyecto, de una lista cerrada
(train.py, tests/run_tests.py y unos pocos módulos -m), sin shell, con cwd en el proyecto y variables de entorno de una
lista cerrada. Cada trabajo deja su salida en runs/jobs/done/<id>.log y su resultado en runs/jobs/done/<id>.json.
Un trabajo {"stop": true} cierra la cola. Latido en runs/jobs/worker.json.

Formato de un trabajo (runs/jobs/queue/<id>.json, se ejecutan por orden de nombre):
  {"argv": ["tests/run_tests.py"], "timeout_s": 7200, "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}
"""
from __future__ import annotations

import argparse, json, os, signal, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, "runs", "jobs")
SCRIPTS = {"train.py", "tests/run_tests.py", "app.py", "agent.py"}
MODULES = {"experiments.run", "experiments.report", "experiments.calibrate", "problems.validate",
           "evaluation.compare"}
ENV_OK = {"CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "ALLOW_UNSAFE_SANDBOX", "HF_HUB_OFFLINE",
          "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"}


def validate(job):
    """argv permitido o ValueError. Devuelve (argv, env, timeout)."""
    if not isinstance(job, dict):
        raise ValueError("el trabajo no es un objeto JSON")
    argv = job.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and "\0" not in a for a in argv):
        raise ValueError("argv debe ser una lista de textos")
    if argv[0] == "-m":
        if len(argv) < 2 or argv[1] not in MODULES:
            raise ValueError(f"módulo no permitido: {argv[1:2]}")
    elif argv[0] not in SCRIPTS:
        raise ValueError(f"script no permitido: {argv[0]}")
    env = job.get("env") or {}
    if not isinstance(env, dict) or any(k not in ENV_OK or not isinstance(v, str) for k, v in env.items()):
        raise ValueError(f"variables de entorno no permitidas: {sorted(set(env) - ENV_OK)}")
    timeout = float(job.get("timeout_s", 6 * 3600))
    return argv, env, timeout


def gpu_snapshot():
    try:
        from rollouts.monitor import snapshot
        s = snapshot()
        return {k: s.get(k) for k in ("used_mb", "total_mb", "util", "power_w", "temp_c")}
    except Exception as e:
        return {"error": repr(e)}


def _write(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cola de trabajos del proyecto (ver docstring)")
    ap.add_argument("--idle-hours", type=float, default=8.0, help="se cierra tras tantas horas sin trabajos")
    a = ap.parse_args(argv)
    sys.path.insert(0, ROOT)
    queue, done = os.path.join(JOBS, "queue"), os.path.join(JOBS, "done")
    os.makedirs(queue, exist_ok=True)
    os.makedirs(done, exist_ok=True)
    beat = os.path.join(JOBS, "worker.json")
    state = dict(pid=os.getpid(), python=sys.executable, started=time.time(), current=None)
    stop = {"f": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("f", True))
    import importlib.util
    state["torch"] = importlib.util.find_spec("torch") is not None
    print(f"cola de trabajos en {queue} (Ctrl+C para salir); intérprete: {sys.executable}", flush=True)
    if not state["torch"]:
        print("AVISO: este intérprete no tiene torch: los tests de GPU se saltarán y el entrenamiento no arrancará. "
              "Activa antes el entorno (venv/conda) con el que entrenas.", flush=True)
    idle_since = time.time()
    try:
        while not stop["f"]:
            _write(beat, dict(state, last_seen=time.time()))
            names = sorted(n for n in os.listdir(queue) if n.endswith(".json"))
            if not names:
                if time.time() - idle_since > a.idle_hours * 3600:
                    print("sin trabajos: cierre por inactividad", flush=True)
                    break
                time.sleep(5)
                continue
            name = names[0]
            jid = name[:-5]
            src = os.path.join(queue, name)
            try:
                with open(src) as f:
                    job = json.load(f)
            except (OSError, ValueError) as e:
                time.sleep(1)  # puede estar a medio escribir
                try:
                    with open(src) as f:
                        job = json.load(f)
                except (OSError, ValueError):
                    os.replace(src, os.path.join(done, name))
                    _write(os.path.join(done, f"{jid}.result.json"), dict(id=jid, error=f"JSON ilegible: {e!r}"))
                    continue
            os.replace(src, os.path.join(done, name))
            if job.get("stop"):
                print("trabajo stop: cierre", flush=True)
                _write(os.path.join(done, f"{jid}.result.json"), dict(id=jid, stopped=True, t=time.time()))
                break
            try:
                args, env, timeout = validate(job)
            except ValueError as e:
                print(f"[{jid}] rechazado: {e}", flush=True)
                _write(os.path.join(done, f"{jid}.result.json"), dict(id=jid, rejected=str(e)))
                continue
            cmd = [sys.executable, "-u", *args]
            print(f"[{jid}] {' '.join(cmd)}", flush=True)
            state["current"] = jid
            _write(beat, dict(state, last_seen=time.time()))
            t0, gpu0 = time.time(), gpu_snapshot()
            log = os.path.join(done, f"{jid}.log")
            rc = None
            with open(log, "w") as out:
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env=dict(os.environ, **env))
                while rc is None:
                    try:
                        rc = p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        _write(beat, dict(state, last_seen=time.time()))
                        if time.time() - t0 > timeout:
                            p.send_signal(signal.SIGINT)  # train.py guarda checkpoint al recibir SIGINT
                            try:
                                rc = p.wait(timeout=300)
                            except subprocess.TimeoutExpired:
                                p.kill()
                                rc = p.wait()
                            rc = f"timeout({rc})"
            _write(os.path.join(done, f"{jid}.result.json"),
                   dict(id=jid, argv=args, env=env, rc=rc, start=t0, end=time.time(), secs=round(time.time() - t0, 1),
                        gpu_before=gpu0, gpu_after=gpu_snapshot()))
            print(f"[{jid}] rc={rc} ({time.time() - t0:.0f}s)", flush=True)
            state["current"] = None
            idle_since = time.time()
    except KeyboardInterrupt:
        print("\ncola detenida", flush=True)
    finally:
        _write(beat, dict(state, last_seen=time.time(), stopped=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
