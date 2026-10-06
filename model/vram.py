"""Presupuesto de VRAM del proceso.

En Windows (también bajo WSL2) el driver no da OOM cuando la VRAM se acaba: pagina memoria de la GPU a la RAM del
sistema ("sysmem fallback") y todo sigue, 10-30x más lento. Medido en runs/phase2/20261002-234419, steps 64-70: VRAM al
99 % (12,04-12,18 de 12,23 GB, ~1 GB de otras aplicaciones de Windows), GPU al 100 % de uso pero a ~55 W (frente a
85-125 W) y 17 tok/s en vez de ~450, sin ningún OOM.

Defensa: el asignador de PyTorch de ESTE proceso se limita (torch.cuda.set_per_process_memory_fraction) a lo que estaba
libre al arrancar menos un margen. Pasarse da un OOM normal, que los autotuners y el backoff ya manejan, en vez de una
degradación silenciosa. Lo que otras aplicaciones ocupen DESPUÉS de arrancar no se puede impedir: eso lo detecta
pressure() (y se registra por step).
"""
from __future__ import annotations

from common import opt

GiB = 1 << 30
_STATE = {"budget": None, "total": None, "others": None}


def apply_budget(cfg):
    """Fija el tope del asignador. vram.budget_gb: auto (libre al arrancar - vram.headroom_gb) | número | off.
    Llamar antes de cargar el modelo. Devuelve un dict para el log (o None sin CUDA)."""
    import torch
    if not torch.cuda.is_available():
        return None
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    mine = torch.cuda.memory_reserved()
    v = opt(cfg, "vram.budget_gb", "auto")
    head = float(opt(cfg, "vram.headroom_gb", 0.6)) * GiB
    others = max(0, total - free - mine)
    if v in ("off", False, 0):
        _STATE.update(budget=None, total=total, others=others)
        return dict(total_gb=round(total / GiB, 2), others_gb=round(others / GiB, 2), budget_gb=None)
    budget = (free + mine - head) if v == "auto" else float(v) * GiB
    budget = int(max(GiB, min(budget, total)))
    torch.cuda.set_per_process_memory_fraction(budget / total)
    _STATE.update(budget=budget, total=total, others=others)
    return dict(total_gb=round(total / GiB, 2), others_gb=round(others / GiB, 2), budget_gb=round(budget / GiB, 2),
                headroom_gb=round(head / GiB, 2))


def free_bytes():
    """Lo que este proceso aún puede reservar: lo libre en la GPU, acotado por el presupuesto."""
    import torch
    free, _ = torch.cuda.mem_get_info()
    if _STATE["budget"]:
        free = min(free, _STATE["budget"] - torch.cuda.memory_reserved())
    return max(0, int(free))


def budget():
    return _STATE["budget"]


def pressure(window, ref_power=None, margin_mb=300):
    """Señales de paginación de VRAM en una ventana de muestras del monitor (dicts con used_mb/total_mb/util/power_w).
    vram_full: la GPU estuvo a menos de margin_mb de llenarse. busy_lowpower: uso medio >= 90 % con una potencia media
    < 60 % de ref_power (la típica de los steps sanos; sin ella, el máximo de la ventana): la GPU espera a la memoria
    (paginando) en vez de calcular, como en los steps 64-70 del run de referencia. Puro Python (testeable)."""
    used = [r["used_mb"] for r in window if r.get("used_mb") is not None]
    tot = [r["total_mb"] for r in window if r.get("total_mb")]
    full = bool(used and tot and max(used) >= max(tot) - margin_mb)
    util = [r["util"] for r in window if r.get("util") is not None]
    pw = [r["power_w"] for r in window if r.get("power_w") is not None]
    u_mean = sum(util) / len(util) if util else None
    p_mean = sum(pw) / len(pw) if pw else None
    ref = ref_power or (max(pw) if pw else None)
    busy_low = bool(u_mean is not None and p_mean is not None and ref and u_mean >= 90 and p_mean < 0.6 * ref)
    return dict(vram_full=full, busy_lowpower=busy_low, vram_used_max_mb=max(used) if used else None,
                util_mean=round(u_mean, 1) if u_mean is not None else None,
                power_mean_w=round(p_mean, 1) if p_mean is not None else None)
