"""Optimizador con grupos con nombre: "policy" (LoRA o full-FT) y un grupo "growth<k>" por evento de crecimiento
(bloques nuevos, con su propio lr y warmup desde el step en que nacen).

El orden y la composición de los grupos son deterministas a partir del modelo, así que optim.pt se recarga sobre un
modelo reconstruido (mismos eventos de crecimiento) sin ambigüedad. El lr se recalcula en cada step a partir de
base_lr y el step (sin estado de scheduler que guardar: reanudar en el step s da exactamente el mismo lr).
"""
from common import opt


def _growth_cfg(cfg):
    return dict(lr=float(opt(cfg, "growth.lr", 5e-5)), wd=float(opt(cfg, "growth.weight_decay", 0.0)),
                warmup=int(opt(cfg, "growth.warmup_steps", 0)))


def _block_params(model, ev):
    from model import growth
    return growth.grown_parameters(model, event=ev.get("event", 0))


def build(model, cfg, events=()):
    """AdamW con el grupo de la política y uno por evento de crecimiento ya aplicado (en orden)."""
    import torch
    from model import growth
    grown = {id(p) for p in growth.grown_parameters(model)}
    pol = [p for p in model.parameters() if p.requires_grad and id(p) not in grown]
    groups = [dict(params=pol, lr=float(cfg.grpo.lr), weight_decay=float(cfg.grpo.weight_decay), name="policy",
                   base_lr=float(cfg.grpo.lr), start_step=0, warmup=int(opt(cfg, "grpo.warmup_steps", 0)))]
    g = _growth_cfg(cfg)
    for k, ev in enumerate(events):
        groups.append(dict(params=_block_params(model, ev), lr=g["lr"], weight_decay=g["wd"], name=f"growth{k}",
                           base_lr=g["lr"], start_step=int(ev.get("step", 0)), warmup=g["warmup"]))
    covered = sum(len(gr["params"]) for gr in groups)
    if covered != sum(1 for p in model.parameters() if p.requires_grad):
        raise RuntimeError("optimizador: los grupos no cubren exactamente los parámetros entrenables")
    return torch.optim.AdamW([gr for gr in groups if gr["params"]])


def add_growth_group(optimizer, model, cfg, ev, k):
    g = _growth_cfg(cfg)
    optimizer.add_param_group(dict(params=_block_params(model, ev), lr=0.0 if g["warmup"] else g["lr"],
                                   weight_decay=g["wd"], name=f"growth{k}", base_lr=g["lr"],
                                   start_step=int(ev.get("step", 0)), warmup=g["warmup"]))


def lr_factor(step, start, warmup):
    """Warmup lineal desde `start` durante `warmup` steps (el primer step ya actualiza con lr/warmup)."""
    if warmup <= 0:
        return 1.0
    return min(1.0, max(0, step - start + 1) / warmup)


def set_lr(optimizer, step):
    """lr de cada grupo para este step. Devuelve {nombre: lr}."""
    out = {}
    for g in optimizer.param_groups:
        if "base_lr" in g:
            g["lr"] = g["base_lr"] * lr_factor(step, g.get("start_step", 0), g.get("warmup", 0))
        out[g.get("name", "group")] = g["lr"]
    return out
