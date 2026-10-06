"""Ejecución aislada de código no confiable (Linux/WSL2).

bwrap : namespaces (user, pid, net, ipc, uts), root tmpfs con sólo /usr en lectura, sin red, sin acceso al proyecto.
rlimit: sólo límites de recursos. NO aísla filesystem ni red -> sólo con sandbox.allow_unsafe: true.
Siempre: proceso separado en su propia sesión, kill del grupo por timeout, tmp borrado al terminar.
"""
import json, os, shutil, signal, subprocess, sys, tempfile, time
from dataclasses import dataclass, field

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness.py")
PY = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
MSG = ("bubblewrap no disponible o sin permisos de user namespaces: sudo apt install bubblewrap "
       "(o sandbox.allow_unsafe: true bajo tu responsabilidad; el backend rlimit no aísla FS ni red)")


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class Run:
    status: str                       # ok | timeout | load_error | crash:<rc>
    cases: list = field(default_factory=list)   # un dict del harness por caso, o None si no llegó a ejecutarse
    fatal: str = ""
    cpu_s: float = 0.0
    rss_mb: float = 0.0
    wall_s: float = 0.0
    fatal_line: int | None = None     # línea del candidato del error de carga, si se conoce


def _argv(b, tail):
    if b == "rlimit":
        return [PY, "-I", "-S", "-B", *tail]
    return [shutil.which("bwrap"), "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL",  # explícito: bwrap ya lo hace por defecto, pero no confiamos en defaults implícitos
            "--ro-bind", "/usr", "/usr", "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", HARNESS, "/sandbox/harness.py", "--chdir", "/tmp",
            "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "PYTHONHASHSEED", "0",
            PY, "-I", "-S", "-B", *tail]


def raw(b, snippet, timeout=10):
    """Corre `python -c snippet` con la misma configuración del sandbox (para tests de aislamiento)."""
    return subprocess.run(_argv(b, ["-c", snippet]), capture_output=True, text=True, timeout=timeout,
                          env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL)


_cache = {}


def backend(sb):
    key = (sb["backend"], bool(sb["allow_unsafe"]))
    if key in _cache:
        return _cache[key]
    b = None
    if sb["backend"] in ("auto", "bwrap") and shutil.which("bwrap"):
        try:
            r = raw("bwrap", "print(40+2)")
            b = "bwrap" if r.returncode == 0 and r.stdout.strip() == "42" else None
        except Exception:
            b = None
    if b is None:
        if not sb["allow_unsafe"]:
            raise SandboxUnavailable(MSG)
        b = "rlimit"
    _cache[key] = b
    return b


def _kill(p):
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        p.wait()


def _parse(path, n):
    recs, done, fatal, fline = [None] * n, None, "", None
    with open(path, "rb") as f:
        data = f.read(4 << 20)
    for line in data.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        i = d.get("i")
        if isinstance(i, int) and 0 <= i < n:
            if recs[i] is None:
                recs[i] = d
        elif "done" in d:
            done = d
        elif "fatal" in d:
            fatal = str(d["fatal"])
            fline = d.get("line") if isinstance(d.get("line"), int) else None
    return recs, done, fatal, fline


def run_cases(code, entry, cases, limits, allowed, b):
    """Ejecuta `entry(*args)` para cada args en `cases` dentro del sandbox. Devuelve Run."""
    if not cases:
        return Run("ok")
    with tempfile.TemporaryDirectory(prefix="sbx_") as td:
        jp, op = os.path.join(td, "job.json"), os.path.join(td, "out.jsonl")
        with open(jp, "w") as f:
            json.dump({"code": code, "entry": entry, "cases": cases, "limits": dict(limits),
                       "allowed": list(allowed)}, f)
        tail = ["/sandbox/harness.py"] if b == "bwrap" else [HARNESS]
        t0, timed_out = time.perf_counter(), False
        with open(jp, "rb") as fin, open(op, "wb") as fout:
            p = subprocess.Popen(_argv(b, tail), stdin=fin, stdout=fout, stderr=subprocess.DEVNULL,
                                 cwd=td if b == "rlimit" else "/", start_new_session=True, close_fds=True,
                                 env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
            try:
                p.wait(timeout=limits["timeout_s"])
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                _kill(p)
        wall = time.perf_counter() - t0
        recs, done, fatal, fline = _parse(op, len(cases))
    status = "timeout" if timed_out else "ok" if done else "load_error" if fatal else f"crash:{p.returncode}"
    return Run(status, recs, fatal, (done or {}).get("cpu", 0.0), (done or {}).get("rss_mb", 0.0), wall, fline)
