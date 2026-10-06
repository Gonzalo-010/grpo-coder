"""Modelo para inferencia (eval, fix, agente, panel): base o base + checkpoint.

Un checkpoint se reconstruye igual que al reanudar: eventos de crecimiento de su meta.json -> growth.safetensors ->
adapter LoRA, que se aplica SIN fusionar (common.INFERENCE_VERSION = 2): la misma cuenta que hace el entrenamiento
(pesos base bf16 + rama LoRA en fp32), así que se evalúa exactamente la política entrenada.

Por qué no se fusiona (merge_and_unload, lo que hacía el código original): la actualización de LoRA es del orden de
la resolución de bf16 de los pesos. En el checkpoint final del brazo A (lr 1e-5, 40 steps), |ΔW| mediano es 5e-6 a
1e-5, frente a medio ulp de bf16 ≈ 6e-5 para |W| ≈ 0,02. Simulando la fusión con esos ΔW y pesos N(0, σ),
σ = 0,01-0,04: el 65-96 % de los pesos no cambia, de ΔW sólo sobrevive el 11-77 % en su dirección y se añade ruido de
redondeo del 37-63 % de |ΔW|: los pesos fusionados no son los entrenados. En la práctica el efecto medido fue nulo
(re-evaluación con las mismas semillas: A Δpass@1 = 0,000 con greedy idéntico en los 50 problemas de val; U2 +0,006,
p = 0,12), pero se deja sin fusionar para que la evaluación sea exacta por construcción y no favorezca a los brazos
con más lr (mayor ΔW, menos atenuación). Sin fusionar, ΔW·x se suma a las activaciones de cada token, igual que al
entrenar. Coste: la rama LoRA añade kernels por paso de generación (evaluar un checkpoint: 155 s -> 291 s en A).
"""
import gc, os

import torch

from model import fit, growth
from model.loader import load
from rollouts.generate import Generator


def _attach(model, path):
    from model.policy import _load_full_state, checkpoint_events
    events = checkpoint_events(path)
    if events:
        growth.replay(model, events)
        gp = os.path.join(path, "growth.safetensors")
        if os.path.exists(gp):
            growth.load(model, gp)
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        from peft import PeftModel
        return PeftModel.from_pretrained(model, path)  # sin merge_and_unload: ver el docstring
    _load_full_state(model, path, model.device)
    return model


class Engine:
    def __init__(self, cfg, checkpoint=None, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        fit.check(cfg, training=False)
        model, self.tok = load(cfg, device)
        self.m = (_attach(model, checkpoint) if checkpoint else model).eval()
        self.m.requires_grad_(False)
        self.cfg, self.checkpoint = cfg, checkpoint
        self.arch = growth.arch(self.m)

    def generator(self):
        return Generator(self.m, self.tok, self.cfg)

    def chat(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, text):
        return len(self.tok(text).input_ids)

    def complete(self, prompt, max_new, temperature=None):
        """Respuesta libre (agente): sin prefill ni parada en el fence de código. Mismos parámetros de muestreo que
        en entrenamiento (sin top_k ni repetition_penalty: los defaults de Qwen los activarían)."""
        g = self.cfg.generation
        t = g.temperature if temperature is None else temperature
        enc = self.tok(prompt, return_tensors="pt").to(self.m.device)
        if enc["input_ids"].shape[1] + max_new > self.m.config.max_position_embeddings:
            raise ValueError("prompt + max_new_tokens supera el contexto del modelo")
        kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id, repetition_penalty=1.0)
        kw.update(dict(do_sample=True, temperature=t, top_p=g.top_p, top_k=0) if t > 0 else dict(do_sample=False))
        self.m.eval()
        with torch.inference_mode():
            out = self.m.generate(**enc, **kw)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def close(self):
        self.m = None
        gc.collect()  # ciclos de referencias (hooks, generation_config...) retendrían la VRAM del modelo
        torch.cuda.empty_cache()
