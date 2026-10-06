"""Checkpoints: guardado atómico, verificable y autodescriptivo.

Escritura: todo va a <root>/.tmp_<step>/, se escribe MANIFEST.json (tamaño + sha256 de cada fichero) el ÚLTIMO, se
sincroniza a disco y se renombra a <root>/ckpt_<step>/. Un ckpt_* existe sólo si se escribió entero, y si un fichero
se corrompe después (disco, copia a medias, /mnt/c de WSL), verify() lo detecta y la reanudación cae al anterior.
best/ y LATEST también se reemplazan de forma atómica (antes best/ se borraba y se copiaba: un corte a mitad lo dejaba
a medias o sin best/).

meta.json (format 2): step, config_hash, métricas, arquitectura (capas, parámetros totales/entrenables, bloques
crecidos), eventos de crecimiento (model/growth.py), linaje (de qué checkpoint/run viene), hash del código, entorno y
la configuración completa. Un checkpoint dice exactamente qué modelo contiene y con qué se entrenó.

Contenido: adapter (peft) o modelo completo, growth.safetensors (bloques crecidos), optim.pt (AdamW con sus grupos),
rng.pt (python/numpy/torch/cuda), improve.json (currículo de ese step, si lo hay).
El bucle de recovery (probar el más reciente y caer al anterior) vive en training/loop.py: necesita envolver también
la carga del modelo.
"""
import hashlib, json, os, random, shutil, time

import yaml

FORMAT = 2
_MANIFEST = "MANIFEST.json"


class CorruptCheckpoint(RuntimeError):
    pass


def _rng_state():
    import torch
    s = dict(py=random.getstate(), torch=torch.get_rng_state())
    try:
        import numpy as np
        s["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _rng_restore(s):
    import torch
    random.setstate(s["py"])
    t = s["torch"]
    torch.set_rng_state(t.cpu() if t.device.type != "cpu" else t)
    if "numpy" in s:
        import numpy as np
        np.random.set_state(s["numpy"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _sha_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _fsync_dir(d):
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_manifest(d):
    files = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if name == _MANIFEST or not os.path.isfile(p):
            continue
        try:  # a disco antes del manifiesto y del rename
            with open(p, "rb+") as f:
                os.fsync(f.fileno())
        except OSError:
            pass
        files[name] = dict(bytes=os.path.getsize(p), sha256=_sha_file(p))
    with open(os.path.join(d, _MANIFEST), "w") as f:
        json.dump(dict(format=FORMAT, files=files), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(d)
    return files


def verify(path, deep=True):
    """Comprueba el manifiesto (si existe: los checkpoints anteriores a format 2 no lo tienen y se aceptan).
    deep=False sólo mira tamaños (rápido)."""
    mp = os.path.join(path, _MANIFEST)
    if not os.path.exists(os.path.join(path, "meta.json")):
        raise CorruptCheckpoint(f"{path}: sin meta.json")
    if not os.path.exists(mp):
        return False
    with open(mp) as f:
        man = json.load(f)
    for name, info in man.get("files", {}).items():
        p = os.path.join(path, name)
        if not os.path.isfile(p):
            raise CorruptCheckpoint(f"{path}: falta {name}")
        if os.path.getsize(p) != info["bytes"]:
            raise CorruptCheckpoint(f"{path}: {name} tiene {os.path.getsize(p)} bytes, el manifiesto dice {info['bytes']}")
        if deep and _sha_file(p) != info["sha256"]:
            raise CorruptCheckpoint(f"{path}: {name} no coincide con su sha256")
    return True


def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _replace_dir(src_tmp, dst):
    """dst <- src_tmp de forma que en todo momento exista uno completo (dst o dst.old)."""
    old = dst + ".old"
    if os.path.exists(old):
        shutil.rmtree(old)
    if os.path.exists(dst):
        os.replace(dst, old)
    os.replace(src_tmp, dst)
    shutil.rmtree(old, ignore_errors=True)
    _fsync_dir(os.path.dirname(dst))


def recover(root):
    """Tras un corte en _replace_dir: si falta best/ pero quedó best.old/, se restaura."""
    for name in ("best",):
        d, old = os.path.join(root, name), os.path.join(root, name + ".old")
        if not os.path.exists(d) and os.path.isdir(old):
            os.replace(old, d)


def code_hash(root=None):
    """Huella del código del proyecto (.py), para saber con qué versión se creó un checkpoint."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h, n = hashlib.sha256(), 0
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in ("runs", "__pycache__", ".git", "tests"))
        for name in sorted(files):
            if name.endswith(".py"):
                p = os.path.join(d, name)
                h.update(os.path.relpath(p, root).encode())
                with open(p, "rb") as f:
                    h.update(f.read())
                n += 1
    return dict(sha256=h.hexdigest()[:16], files=n)


def save(root, step, model, optimizer, metrics, lora_enabled, config_hash, is_best=False, extra=None, files=None):
    """Escribe <root>/.tmp_<step>, su manifiesto y renombra a <root>/ckpt_<step>. extra: claves de meta.json
    (arch, growth, lineage, config, env...). files: {nombre: función(ruta)} para ficheros adicionales."""
    import torch

    from model.policy import save_policy
    os.makedirs(root, exist_ok=True)
    final = os.path.join(root, f"ckpt_{step:07d}")
    tmp = os.path.join(root, f".tmp_{step:07d}")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    save_policy(model, tmp, lora_enabled)
    torch.save(optimizer.state_dict(), os.path.join(tmp, "optim.pt"))
    torch.save(_rng_state(), os.path.join(tmp, "rng.pt"))
    for name, fn in (files or {}).items():
        fn(os.path.join(tmp, name))
    meta = dict(format=FORMAT, step=step, time=time.time(), config_hash=config_hash, metrics=metrics,
                lora_enabled=lora_enabled)
    meta.update(extra or {})
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    write_manifest(tmp)
    _replace_dir(tmp, final)  # ckpt_NNNNNNN/ sólo existe si se escribió entero
    _atomic_write(os.path.join(root, "LATEST"), os.path.basename(final))
    if is_best:
        tb = os.path.join(root, ".tmp_best")
        if os.path.exists(tb):
            shutil.rmtree(tb)
        shutil.copytree(final, tb)
        _replace_dir(tb, os.path.join(root, "best"))
    return final


def candidates(root):
    """Rutas de checkpoints completos, más recientes primero (LATEST, si existe y apunta a un dir real, va
    primero; el resto por nombre descendente). ckpt_NNNNNNN/ sólo existe tras el rename atómico -> completo."""
    if not os.path.isdir(root):
        return []
    recover(root)
    ordered = []
    latest = os.path.join(root, "LATEST")
    if os.path.exists(latest):
        p = os.path.join(root, open(latest).read().strip())
        if os.path.isdir(p):
            ordered.append(p)
    rest = sorted((d for d in os.listdir(root) if d.startswith("ckpt_") and not d.endswith(".old")), reverse=True)
    ordered += [os.path.join(root, d) for d in rest if os.path.join(root, d) not in ordered]
    return ordered


def read_meta(path):
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def config_hash(cfg):
    """Sólo lo que fija el estado guardado: modelo base y dtype, el adapter (al reanudar, peft toma r/alpha/
    target_modules del adapter_config.json del checkpoint, no de la config), lr/weight_decay (optimizer.load_state_dict
    restaura los guardados) y, si está activo, el crecimiento (cambia la arquitectura). Cambiar lo demás (max_steps,
    frecuencias, generación, reward, niveles...) no impide reanudar. Sin crecimiento el hash es el de siempre: los runs
    existentes se siguen reanudando."""
    g = cfg["grpo"]
    d = dict(model=cfg["model"], dtype=cfg["dtype"], lora=cfg["lora"], lr=g["lr"], weight_decay=g["weight_decay"])
    gr = cfg.get("growth") or {}
    if gr.get("enabled"):
        d["growth"] = {k: gr.get(k) for k in ("method", "schedule", "zero_modules", "lr", "weight_decay")}
    return _sha(d)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def meta_hash(path):
    """config_hash del checkpoint, comparable con config_hash(). Los anteriores a acotar el hash guardaban el de
    la config completa: si la config.yaml de su run reproduce exactamente ese hash, se recalcula desde ella."""
    h = read_meta(path).get("config_hash")
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(path)), "config.yaml")) as f:
            run_cfg = yaml.safe_load(f)
        if _sha(run_cfg) == h:
            return config_hash(run_cfg)
    except (OSError, KeyError, TypeError, AttributeError, yaml.YAMLError):
        pass
    return h


def load_optim_rng(path, optimizer, map_location):
    import torch
    optimizer.load_state_dict(torch.load(os.path.join(path, "optim.pt"), map_location=map_location))
    _rng_restore(torch.load(os.path.join(path, "rng.pt"), map_location="cpu", weights_only=False))


def claim_run(run):
    """Un solo entrenamiento por run: dos procesos reanudando el mismo run (terminal + panel) se pisarían
    checkpoints, rotación y LATEST. Un train.pid de un proceso muerto (o de un PID reutilizado) se reclama."""
    from common import is_training
    path = os.path.join(run, "train.pid")
    try:
        with open(path) as f:
            pid = int(f.read().strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid and pid != os.getpid() and is_training(pid):
        raise RuntimeError(f"el proceso {pid} ya está entrenando en {run}: páralo antes o usa --fresh para un run nuevo")
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return path


def release_run(path):
    try:
        with open(path) as f:
            mine = f.read().strip() == str(os.getpid())
        if mine:
            os.remove(path)
    except OSError:
        pass


def find_run_dir(root_runs, config_hash):
    """Último run de phase2 con al menos un checkpoint de esta config_hash, o None (arrancar uno nuevo)."""
    if not os.path.isdir(root_runs):
        return None
    for name in sorted(os.listdir(root_runs), reverse=True):
        for path in candidates(os.path.join(root_runs, name, "checkpoints")):
            try:
                if meta_hash(path) == config_hash:
                    return os.path.join(root_runs, name)
            except Exception:
                continue
    return None
