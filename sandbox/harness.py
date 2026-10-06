"""Corre DENTRO del sandbox (sólo stdlib). Job por stdin; resultados por un fd privado.

El candidato nunca ve valores esperados: recibe entradas y devuelve salidas; la comparación
ocurre fuera, en el proceso confiable. stdout/stderr del candidato van a /dev/null.
"""
import builtins, hashlib, json, os, random, resource, signal, sys, time

BLOCKED = ("open", "exec", "eval", "compile", "input", "breakpoint", "help", "exit", "quit",
           "copyright", "credits", "license")


def enc(x):
    """Valor -> JSON canónico (str). tuple -> lista, set -> lista ordenada; otro tipo -> TypeError."""
    def dflt(o):
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        raise TypeError("unserializable: " + type(o).__name__)
    return json.dumps(x, default=dflt, separators=(",", ":"))


class _Timeout(BaseException):  # BaseException: `except Exception` del candidato no la traga
    pass


def _alarm(*_):
    raise _Timeout()


def _msg(e):
    try:
        return str(e)[:120]
    except BaseException:
        return ""


def _line(e):
    """Línea del código del candidato donde se originó `e` (el marco más interno de <candidate>), o None."""
    n, tb = None, e.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<candidate>":
            n = tb.tb_lineno
        tb = tb.tb_next
    return n


def main():
    job = json.loads(sys.stdin.buffer.read())
    lim = job["limits"]
    res = os.fdopen(os.dup(1), "w", buffering=1)  # canal privado de resultados
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    sys.stdin = sys.stdout = sys.stderr = open(os.devnull, "r+")

    def emit(d):
        try:
            res.write(json.dumps(d, separators=(",", ":")) + "\n")
        except Exception:
            pass

    for name, v in (("RLIMIT_AS", lim["mem_mb"] << 20), ("RLIMIT_CPU", lim["cpu_s"]),
                    ("RLIMIT_FSIZE", lim["fsize_mb"] << 20), ("RLIMIT_NOFILE", lim["nofile"]),
                    ("RLIMIT_CORE", 0), ("RLIMIT_NPROC", 0)):  # NPROC=0: sin fork. Sólo válido si el proceso
                    # carece de CAP_SYS_RESOURCE: bajo bwrap se cumple (cap bounding set vacío por defecto;
                    # ver --cap-drop ALL en executor.py); bajo el backend rlimit corriendo como root real, NO
                    # se aplica (por eso rlimit exige sandbox.allow_unsafe=true: es defensa parcial, no aislamiento).
        try:
            resource.setrlimit(getattr(resource, name), (v, v))
        except (ValueError, OSError):
            emit({"warn": name})

    allowed = frozenset(job["allowed"])
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in allowed:
            raise ImportError("blocked import: " + name)
        return real_import(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in vars(builtins).items() if k not in BLOCKED}
    safe["__import__"] = guarded
    ns = {"__name__": "candidate", "__builtins__": safe}
    sys.setrecursionlimit(4000)
    signal.signal(signal.SIGALRM, _alarm)
    ct = float(lim["case_timeout_s"])

    try:
        signal.setitimer(signal.ITIMER_REAL, ct)
        exec(compile(job["code"], "<candidate>", "exec"), ns)
        fn = ns[job["entry"]]
        if not callable(fn):
            raise TypeError("entry not callable")
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        emit({"fatal": type(e).__name__ + ": " + _msg(e), "line": _line(e)})
        res.flush()
        os._exit(0)
    signal.setitimer(signal.ITIMER_REAL, 0)

    n_to = 0
    for i, args in enumerate(job["cases"]):
        random.seed(0)
        t0 = time.perf_counter()
        try:
            signal.setitimer(signal.ITIMER_REAL, ct)
            r = fn(*args)
            signal.setitimer(signal.ITIMER_REAL, 0)
            s = enc(r)
            rec = {"i": i}
            if len(s) <= 2048:
                rec["out"] = s
            else:  # salidas grandes: sólo hash (el padre compara contra el hash del esperado)
                rec["h"] = hashlib.sha256(s.encode()).hexdigest()
        except _Timeout as e:
            rec = {"i": i, "exc": "Timeout", "line": _line(e)}
            n_to += 1
        except BaseException as e:
            rec = {"i": i, "exc": type(e).__name__, "mro": [c.__name__ for c in type(e).__mro__],
                   "msg": _msg(e), "line": _line(e)}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        rec["t"] = round(time.perf_counter() - t0, 5)
        emit(rec)
        if n_to >= lim["max_timeouts"]:
            emit({"aborted": "timeouts"})
            break

    emit({"done": 1, "cpu": round(time.process_time(), 4),
          "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)})
    res.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
