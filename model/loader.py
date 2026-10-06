"""Carga del modelo. Sólo bf16/fp16: no existe ruta de cuantización en este proyecto."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name):
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    try:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
    except KeyError:
        raise ValueError(f"dtype '{name}' no soportado (bf16 | fp16 | fp32 | auto)") from None


def load(cfg, device="cuda"):
    dt = resolve_dtype(cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    tok.padding_side = "left"  # generación por lotes con padding a la izquierda
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dt)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dt)
    if getattr(model.config, "quantization_config", None):
        raise RuntimeError(f"{cfg.model} es un checkpoint cuantizado; usa uno bf16/fp16")
    if next(model.parameters()).dtype != dt:  # algunos loaders ignoran dtype en silencio
        model = model.to(dt)
    model = model.to(device).eval()
    if cfg.context_length > model.config.max_position_embeddings:
        raise ValueError("context_length > max_position_embeddings del modelo")
    return model, tok
