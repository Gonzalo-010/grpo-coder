"""Policy para RL: modelo base (+ crecimiento opcional, model/growth.py) + LoRA (peft) o full-FT.

load_policy devuelve (model, tok, ref_ctx). `with ref_ctx() as ref_model:` da el modelo de la referencia del KL:
* LoRA: el propio `model` con los adapters desactivados y los bloques crecidos en bypass = el modelo base exacto, sin
  una segunda copia en VRAM (torchtune usa el mismo patrón para DPO).
* full-FT: una copia congelada del modelo base (sin crecimiento), que sí ocupa VRAM.

LoRA sin dropout: en RL on-policy el gradiente debe ser el de la política que muestreó (el trainer además pone p=0 en
cualquier Dropout, también en adapters antiguos guardados con lora_dropout > 0).

resume_from=<dir de un checkpoint>: repite los eventos de crecimiento de su meta.json, carga growth.safetensors y el
adapter con PeftModel.from_pretrained (la vía documentada para adapters ya entrenados).
events=[...]: eventos de crecimiento a aplicar ANTES de añadir LoRA (crecimiento en el step 0 de un run nuevo); así
LoRA se crea sólo sobre las capas originales.
"""
import json, os
from contextlib import contextmanager

import torch

from model import fit, growth
from model.loader import load as load_base


@contextmanager
def _lora_ref_ctx(model):
    with model.disable_adapter(), growth.bypass(model), torch.no_grad():
        yield model


@contextmanager
def _full_ref_ctx(ref_model):
    with torch.no_grad():
        yield ref_model


def _load_full_state(model, path, map_location):
    sd_path = os.path.join(path, "model.safetensors")
    if os.path.exists(sd_path):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(sd_path, device=str(map_location)))
    else:
        model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin"), map_location=map_location))


def checkpoint_events(path):
    """Eventos de crecimiento registrados en un checkpoint ([] si no creció o es anterior a esta versión)."""
    try:
        with open(os.path.join(path, "meta.json")) as f:
            return list(json.load(f).get("growth") or [])
    except (OSError, ValueError):
        return []


def load_policy(cfg, device="cuda", resume_from=None, events=None):
    big = fit.check(cfg, training=True) == "checkpointing"  # falla aquí, antes de cargar, si no cabe
    model, tok = load_base(cfg, device)
    events = checkpoint_events(resume_from) if resume_from else list(events or [])
    if events:
        growth.replay(model, events)
        if resume_from:
            growth.load(model, os.path.join(resume_from, "growth.safetensors"))
    if big:  # activaciones recalculadas en el backward; en generación (eval) no se aplica
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not cfg.lora.enabled:
        ref, _ = load_base(cfg, device)
        ref.eval().requires_grad_(False)
        if resume_from:
            _load_full_state(model, resume_from, device)
        return model, tok, (lambda: _full_ref_ctx(ref))

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    if resume_from:
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=0.0,
                        target_modules=_targets(cfg.lora.target_modules), bias="none",
                        **growth.lora_layer_kwargs(model))
        model = get_peft_model(model, lc)
    growth._sync_peft(model)  # peft congela todo lo que no es adapter: los bloques crecidos se entrenan enteros
    return model, tok, (lambda: _lora_ref_ctx(model))


def _targets(tm):
    return tm if isinstance(tm, str) else list(tm)  # "all-linear" (peft) para arquitecturas con otros nombres


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def save_policy(model, path, lora_enabled):
    """LoRA: sólo el adapter (MB) + los bloques crecidos (growth.safetensors). full-FT: todo el modelo."""
    model.save_pretrained(path, safe_serialization=True)
    growth.save(model, os.path.join(path, "growth.safetensors"))
