"""¿Cabe el modelo en la GPU? Se estima antes de cargar nada: el modelo se instancia en el device 'meta' a partir
de su config.json (sin pesos ni RAM). Sin cuantización: bf16/fp16 = 2 bytes por parámetro.
"""


def need_bytes(n_params, training, lora):
    w = 2 * n_params
    if not training or lora:
        return w  # LoRA: la referencia es el mismo modelo con los adapters apagados; adapters y AdamW son ~MB
    return 5 * w  # full-FT: pesos + copia de referencia + gradientes + AdamW (m, v)


def decide(need, total):
    """'ok' | 'checkpointing' (se recalculan activaciones en el backward) | 'too_big'.
    El resto de la VRAM es para KV cache de generación y activaciones. 0.4: 0.5B no lo activa ni en full-FT (~38 %,
    comportamiento validado intacto) y 3B con LoRA sí (~47 %), donde sin él las activaciones no caben."""
    if need > 0.8 * total:
        return "too_big"
    return "checkpointing" if need > 0.4 * total else "ok"


def n_params(model_name, per_layer=False):
    """Parámetros del modelo (y, con per_layer, los de una capa de decoder), sin descargar pesos."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    conf = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(conf)
    total = sum(p.numel() for p in m.parameters())
    if not per_layer:
        return total
    layer = sum(p.numel() for p in m.get_decoder().layers[0].parameters())
    return total, layer


def growth_layers(cfg):
    """Capas nuevas que añadirá el crecimiento configurado (0 si está apagado)."""
    from common import opt
    if not opt(cfg, "growth.enabled", False):
        return 0
    sched = opt(cfg, "growth.schedule", None) or [dict(new_layers=opt(cfg, "growth.new_layers", 4))]
    return sum(int(e.get("new_layers", 0)) for e in sched)


def check(cfg, training):
    """Falla pronto y con un mensaje claro si no cabe. Devuelve la decisión de decide()."""
    import torch
    if not torch.cuda.is_available():
        return "ok"
    try:
        n, per_layer = n_params(cfg.model, per_layer=True)
    except Exception as e:  # sin config en caché ni red, arquitectura que no instancia en meta...: sólo avisar
        print(f"      [fit] no se pudo estimar el tamaño de {cfg.model} ({e!r}); se intenta cargar igualmente")
        return "ok"
    total = torch.cuda.get_device_properties(0).total_memory
    need = need_bytes(n, training, cfg.lora.enabled)
    grown = growth_layers(cfg) * per_layer  # bloques nuevos en fp32 (+ gradiente + AdamW m, v si se entrenan)
    need += grown * (16 if training else 4)
    d = decide(need, total)
    msg = (f"{cfg.model}: {n / 1e9:.2f}B params, ~{need / 2**30:.1f} GB para "
           f"{'entrenar' if training else 'inferencia'} de {total / 2**30:.1f} GB")
    if d == "too_big":
        hint = "activa lora.enabled o usa un modelo menor" if training and not cfg.lora.enabled else "usa un modelo menor"
        raise RuntimeError(f"{msg}: no cabe sin cuantización ({hint})")
    print(f"      [fit] {msg} -> {d}")
    return d
