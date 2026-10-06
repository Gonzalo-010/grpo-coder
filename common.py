import json, os, platform, random, sys, time, warnings

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
# Cómo se carga un checkpoint para inferencia (model/infer.py). 2: el adapter LoRA se aplica SIN fusionar, igual que
# en el entrenamiento. 1 (código original hasta v3h): se fusionaba en los pesos bf16 y el redondeo atenuaba la
# actualización y le añadía ruido. Las evaluaciones de checkpoints con versión 1 no miden exactamente el modelo
# entrenado: experiments/run.py las repite.
INFERENCE_VERSION = 2
# ast.parse()/compile() del código que GENERA el modelo emite SyntaxWarning (p. ej. "\\w" fuera de un raw string)
# con fichero "<unknown>": ruido en los logs que no dice nada del proyecto. Los avisos de nuestros módulos se ven.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="<unknown>")


class D(dict):
    """dict con acceso por atributo; una clave inexistente falla (no devuelve None)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


def _wrap(x):
    if isinstance(x, dict):
        return D({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def read_config(path=None, _chain=()):
    """YAML -> dict. `extends: otro.yaml` (relativo al fichero) hereda de otra config y sólo pisa lo que declara."""
    path = os.path.abspath(path or os.path.join(ROOT, "config.yaml"))
    if path in _chain:
        raise ValueError(f"`extends` en ciclo: {' -> '.join(map(os.path.basename, _chain + (path,)))}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: vacío o no es un mapa YAML de claves")
    base = cfg.pop("extends", None)
    return _merge(read_config(os.path.join(os.path.dirname(path), base), _chain + (path,)), cfg) if base else cfg


def load_config(path=None, **over):
    cfg = read_config(path)
    for k, v in over.items():
        if v is not None:
            cfg[k] = v
    return _wrap(cfg)


def workers(cfg):
    w = cfg.sandbox.workers
    return max(1, (os.cpu_count() or 2) // 2) if w == "auto" else int(w)


def is_training(pid):
    """¿Es `pid` un intérprete Python ejecutando train.py? Que el PID exista no basta (se reutilizan) y que un
    argumento acabe en train.py tampoco (un editor con el fichero abierto)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            args = f.read().split(b"\0")
    except OSError:
        return False
    return os.path.basename(args[0]).startswith(b"python") and any(a.endswith(b"train.py") for a in args[1:])


def training_pids():
    """Entrenamientos vivos en la máquina (lanzados desde el panel o desde una terminal), salvo este proceso."""
    return [int(p) for p in os.listdir("/proc") if p.isdigit() and int(p) != os.getpid() and is_training(int(p))]


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 2),
                        capability=list(torch.cuda.get_device_capability(0)),
                        arch_list=torch.cuda.get_arch_list())
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def new_run_dir(cfg, name):
    base = os.path.join(ROOT, "runs", name, time.strftime("%Y%m%d-%H%M%S"))
    d, i = base, 1
    while True:  # creación atómica: dos runs en el mismo segundo nunca comparten directorio (-2, -3, ...)
        try:
            os.makedirs(d)
            break
        except FileExistsError:
            i += 1
            d = f"{base}-{i}"
    with open(os.path.join(d, "config.yaml"), "w") as f:  # cada run guarda con qué config se creó
        yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False)
    with open(os.path.join(d, "env.json"), "w") as f:
        json.dump(env_info(), f, indent=1)
    return d


def opt(cfg, path, default=None):
    """Clave opcional del config ('fix.pool'): default si falta o es null. Para las opciones nuevas, que deben
    tener un valor seguro aunque el config.yaml del usuario sea anterior a ellas."""
    cur = cfg
    for k in path.split("."):
        try:
            cur = cur[k]
        except (KeyError, TypeError, IndexError):
            return default
    return default if cur is None else cur
