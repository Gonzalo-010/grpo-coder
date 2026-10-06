"""Comandos arbitrarios dentro del sandbox (los tests de un repo, para el agente).

bwrap: la misma base que executor.py (sin red, /usr de sólo lectura, /tmp en tmpfs) más la COPIA de trabajo en /work
(lectura-escritura) y el entorno Python actual en sólo lectura, para que pytest y las dependencias existan. Nada más
del home ni del proyecto es visible. Un lanzador aplica rlimits y hace exec del comando.
rlimit: sólo límites, sin aislamiento (exige sandbox.allow_unsafe, como en executor.py).
"""
import os, shutil, subprocess, sys, tempfile, time
from dataclasses import dataclass

from sandbox.executor import PY, _kill, backend

LAUNCH = ("import os, resource, sys\n"
          "for n, v in ((resource.RLIMIT_AS, int(sys.argv[1]) << 20), (resource.RLIMIT_CPU, int(sys.argv[2])),\n"
          "             (resource.RLIMIT_FSIZE, 64 << 20), (resource.RLIMIT_NPROC, 1024), (resource.RLIMIT_CORE, 0)):\n"
          "    resource.setrlimit(n, (v, v))\n"
          "os.execvp(sys.argv[3], sys.argv[3:])\n")


@dataclass
class CmdResult:
    code: int | None  # None = timeout
    out: str
    secs: float

    @property
    def ok(self):
        return self.code == 0


VENV = "/venv"  # punto de montaje del entorno Python dentro del sandbox


def _env_dirs():
    """(origen, destino) del Python actual si vive fuera de /usr, en sólo lectura. El venv va en /venv: montado en su
    ruta real (p. ej. /home/<usuario>/.venvs/x), bwrap crearía /home/<usuario> dentro del sandbox. Python lo reconoce
    igual (busca pyvenv.cfg junto al ejecutable). Su base, si no es la de /usr, va en su misma ruta: los enlaces
    bin/python del venv apuntan a ella."""
    prefix, base = os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)
    out = [] if prefix.startswith("/usr") else [(prefix, VENV)]
    return out + ([(base, base)] if base != prefix and not base.startswith("/usr") else [])


def _argv(b, workdir, cmd, mem_mb, cpu_s):
    launch = [PY, "-c", LAUNCH, str(mem_mb), str(cpu_s)]
    if b == "rlimit":
        return launch + cmd
    args = [shutil.which("bwrap"), "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--ro-bind", "/usr", "/usr", "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for src, dst in _env_dirs():
        args += ["--ro-bind", src, dst]
    args += ["--bind", os.path.realpath(workdir), "/work", "--chdir", "/work", "--clearenv",
             "--setenv", "PATH", f"{VENV}/bin:/usr/bin", "--setenv", "HOME", "/tmp",
             "--setenv", "PYTHONDONTWRITEBYTECODE", "1"]
    return args + launch + cmd


def _trim(s, cap):
    return s if len(s) <= cap else s[:cap // 4] + "\n[... salida recortada ...]\n" + s[-(3 * cap // 4):]


def run(cmd, workdir, sb, timeout_s, mem_mb, cap=60_000):
    """Ejecuta `cmd` (lista, sin shell) en `workdir` dentro del sandbox. Salida combinada y recortada."""
    b = backend(sb)
    env = {"PATH": f"{os.path.join(sys.prefix, 'bin')}:/usr/bin:/bin", "HOME": workdir, "PYTHONDONTWRITEBYTECODE": "1"}
    t0 = time.perf_counter()
    with tempfile.TemporaryFile() as out:
        p = subprocess.Popen(_argv(b, workdir, list(cmd), int(mem_mb), int(timeout_s) + 1), stdin=subprocess.DEVNULL,
                             stdout=out, stderr=subprocess.STDOUT, cwd=workdir if b == "rlimit" else "/",
                             start_new_session=True, close_fds=True, env=env)
        try:
            code = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            code = None
        finally:
            _kill(p)
        out.seek(0)
        text = out.read().decode(errors="replace")
    if code is None:
        text += f"\n[timeout: más de {timeout_s}s]"
    return CmdResult(code, _trim(text, cap), round(time.perf_counter() - t0, 2))
