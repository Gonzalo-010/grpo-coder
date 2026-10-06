"""Crecimiento del modelo en profundidad con bloques identidad (block expansion: LLaMA Pro, Wu et al. 2024,
arXiv:2401.02415; preservación de función como en Net2Net, Chen et al. 2016).

Un bloque nuevo es una COPIA de la capa original que lo precede con las proyecciones que escriben en el residual
(o_proj de la atención y down_proj del MLP) a cero: su salida es x + 0 + 0 = x, así que el modelo crecido calcula los
mismos logits, bit a bit, que el original (test: t_growth_identity). El gradiente llega a o_proj/down_proj desde el
primer step (depende de las activaciones internas, que no son cero) y, en cuanto dejan de ser cero, al resto del bloque.

Lo que hace falta tocar para que transformers >= 5 (Qwen2, Llama...) lo acepte, y que se comprueba en los tests:
* config.layer_types y config.num_hidden_layers: Qwen2Model recorre layers[:num_hidden_layers] y elige la máscara por
  layer_types[i]; DynamicCache(config=...) crea una entrada de caché KV por elemento de layer_types.
* self_attn.layer_idx de TODAS las capas se renumera: la caché KV se indexa por él.
* RoPE no cambia (Qwen2Model calcula cos/sin una vez y se los pasa a cada capa); los pesos atados (embed/lm_head) tampoco.
* Los bloques nuevos viven en fp32 (pesos maestros: en bf16 una actualización de Adam de ~1e-4 sobre un peso de ~0.05
  se pierde al redondear) y convierten entrada/salida al dtype del modelo. Con delta 0, bf16 -> fp32 -> bf16 es exacto.
* bypass(model): los bloques nuevos devuelven su entrada sin calcular nada = el modelo original exacto. Es la
  referencia del KL junto con LoRA desactivado (si no, la "referencia" incluiría los bloques entrenados).
* LoRA sólo en las capas originales (peft: layers_to_transform con sus índices NUEVOS, layers_pattern="layers"): los
  bloques nuevos se entrenan enteros. El adapter guardado lleva esos índices, así que recargar un checkpoint es: cargar
  el modelo base -> repetir los eventos de crecimiento de meta.json -> cargar growth.safetensors -> cargar el adapter.

Un evento de crecimiento (lo que se guarda en meta.json["growth"]) es un dict:
  {"method": "depth", "step": s, "after": [índices de capas ORIGINALES tras las que se inserta], "zero": [...],
   "layers_before": L, "layers_after": L', "new": [índices nuevos], "params_added": n}
"""
from __future__ import annotations

from contextlib import contextmanager

try:
    import torch
    from torch import nn
except ImportError:
    torch = nn = None

DEFAULT_ZERO = ("o_proj", "down_proj")  # salidas al residual en Qwen2/Llama/Mistral


# ---------------------------------------------------------------------------------------------------------- plan
def plan_positions(n_layers, n_total):
    """Tras qué capas originales van n_total bloques repartidos de forma uniforme (LLaMA Pro: uno cada L/n capas,
    el último tras la última capa). 24 capas, 4 bloques -> [5, 11, 17, 23]."""
    if n_total < 1:
        return []
    return sorted({(i + 1) * n_layers // n_total - 1 for i in range(n_total)})


def _bitrev(i, bits):
    return int(format(i, f"0{bits}b")[::-1], 2) if bits else 0


def schedule_positions(n_layers, sizes):
    """Reparte las posiciones de plan_positions(n_layers, sum(sizes)) entre eventos sucesivos de forma intercalada
    (orden de bits invertidos): cada evento cubre todo el modelo, no sólo un tramo. sizes = bloques por evento."""
    total = sum(sizes)
    pos = plan_positions(n_layers, total)
    bits = max(1, (total - 1).bit_length())
    order = sorted(range(len(pos)), key=lambda i: (_bitrev(i, bits), i))
    out, k = [], 0
    for n in sizes:
        out.append(sorted(pos[i] for i in order[k:k + n]))
        k += n
    return out


# ---------------------------------------------------------------------------------------------------------- acceso
def unwrap(model):
    """El CausalLM de transformers debajo de un PeftModel (o el propio modelo)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def decoder(model):
    return unwrap(model).get_decoder()


def text_config(model):
    cfg = unwrap(model).config
    return cfg.get_text_config(decoder=True) if hasattr(cfg, "get_text_config") else cfg


def is_grown(layer):
    return bool(getattr(type(layer), "_grown", False))


def grown_blocks(model):
    return [(i, l) for i, l in enumerate(decoder(model).layers) if is_grown(l)]


def original_indices(model):
    return [i for i, l in enumerate(decoder(model).layers) if not is_grown(l)]


def grown_parameters(model, event=None):
    """Parámetros de los bloques crecidos (de un evento concreto si se indica)."""
    return [p for _, b in grown_blocks(model) if event is None or getattr(b, "growth_event", 0) == event
            for p in b.parameters()]


def count_params(model):
    """(total, entrenables) contando una sola vez los tensores compartidos (embeddings atados)."""
    seen, total, train = set(), 0, 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        train += p.numel() if p.requires_grad else 0
    return total, train


def arch(model):
    """Descripción de la arquitectura para meta.json."""
    c = text_config(model)
    total, train = count_params(model)
    return dict(model_type=getattr(c, "model_type", None), layers=len(decoder(model).layers),
                hidden=getattr(c, "hidden_size", None), intermediate=getattr(c, "intermediate_size", None),
                heads=getattr(c, "num_attention_heads", None), kv_heads=getattr(c, "num_key_value_heads", None),
                vocab=getattr(c, "vocab_size", None), tied=bool(getattr(c, "tie_word_embeddings", False)),
                grown=[i for i, _ in grown_blocks(model)], params_total=total, params_trainable=train)


# ---------------------------------------------------------------------------------------------------------- bloque
_CLASSES = {}


def _cast(x, dt):
    if torch.is_tensor(x) and x.is_floating_point() and x.dtype != dt:
        return x.to(dt)
    if isinstance(x, tuple):
        return tuple(_cast(v, dt) for v in x)
    return x


def grown_class(cls):
    """Subclase de la capa de decoder del modelo con `bypass` y cómputo en el dtype de sus propios pesos."""
    if getattr(cls, "_grown", False):
        return cls
    if cls not in _CLASSES:
        def forward(self, hidden_states, *args, **kwargs):
            if self.bypass:
                return hidden_states
            dt = hidden_states.dtype
            wdt = self.growth_dtype
            if dt == wdt:
                return cls.forward(self, hidden_states, *args, **kwargs)
            args = tuple(_cast(a, wdt) for a in args)
            kwargs = {k: _cast(v, wdt) if k in ("attention_mask", "position_embeddings") else v for k, v in kwargs.items()}
            out = cls.forward(self, hidden_states.to(wdt), *args, **kwargs)
            if torch.is_tensor(out):
                return out.to(dt)
            return (out[0].to(dt),) + tuple(out[1:])

        _CLASSES[cls] = type(f"Grown{cls.__name__}", (cls,), {"forward": forward, "_grown": True, "bypass": False,
                                                              "growth_dtype": torch.float32})
    return _CLASSES[cls]


def _base_state(layer):
    """state_dict de la capa sin LoRA (base_layer.* -> *), para copiar una capa ya envuelta por peft."""
    out = {}
    for k, v in layer.state_dict().items():
        if "lora_" in k or ".lora" in k:
            continue
        out[k.replace(".base_layer.", ".")] = v
    return out


def _zero(block, names):
    hit = 0
    for name, mod in block.named_modules():
        if name.split(".")[-1] in names and isinstance(mod, nn.Linear):
            nn.init.zeros_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
            hit += 1
    if not hit:
        raise ValueError(f"ningún módulo {list(names)} en la capa: growth.zero_modules no encaja con esta arquitectura")
    return hit


def make_block(layer, cfg, src_idx, zero=DEFAULT_ZERO, dtype=None):
    """Bloque identidad: copia de `layer` (sin LoRA) con las salidas al residual a cero, en `dtype` (fp32).
    Se construye limpio con la clase de capa del modelo (config, layer_idx=src_idx: el índice de la capa fuente, válido
    en config.layer_types; luego se renumera) y se le cargan los pesos base de la fuente."""
    dtype = dtype or torch.float32
    base_cls = type(layer).__mro__[1] if is_grown(layer) else type(layer)
    ref = next(layer.parameters())
    try:
        with torch.device("meta"):  # sin inicialización aleatoria: no consume RNG (los runs con y sin crecimiento
            blk = base_cls(cfg, src_idx)  # siguen el mismo flujo aleatorio) y no gasta tiempo en pesos que se pisan
    except TypeError as e:
        raise NotImplementedError(f"growth: la capa {base_cls.__name__} no se construye como (config, layer_idx): {e}")
    blk = blk.to_empty(device=ref.device).to(dtype)
    missing, unexpected = blk.load_state_dict(_base_state(layer), strict=False)
    if unexpected or missing:
        raise RuntimeError(f"copia de capa incompleta: faltan {list(missing)[:5]} sobran {list(unexpected)[:5]}")
    src_bufs = dict(layer.named_buffers())  # buffers no persistentes (no van en el state_dict): copiarlos tal cual
    for name, buf in blk.named_buffers():
        if name in src_bufs:
            buf.copy_(src_bufs[name])
        elif name not in _base_state(layer):
            raise RuntimeError(f"growth: buffer {name} sin valor de origen")
    _zero(blk, zero)
    blk.__class__ = grown_class(base_cls)
    blk.bypass = False
    blk.growth_dtype = dtype
    for attr in ("gradient_checkpointing", "_gradient_checkpointing_func"):  # gradient checkpointing de HF
        if hasattr(layer, attr):
            setattr(blk, attr, getattr(layer, attr))
    blk.train(layer.training)
    blk.requires_grad_(True)
    return blk


def _renumber(layers):
    for i, layer in enumerate(layers):
        for mod in layer.modules():
            if hasattr(mod, "layer_idx") and isinstance(getattr(mod, "layer_idx"), int):
                mod.layer_idx = i


# ---------------------------------------------------------------------------------------------------------- crecer
def apply_event(model, ev, dtype=None):
    """Aplica un evento de crecimiento en profundidad (in place) y lo devuelve completado (new, layers_after,
    params_added). Sirve para crecer por primera vez y para repetir el crecimiento al recargar un checkpoint.
    ev["after"]: índices de capas ORIGINALES (0..L-1); un bloque nuevo va detrás de esa capa y de los bloques que
    ya la siguieran (orden estable entre eventos)."""
    if ev.get("method", "depth") != "depth":
        raise ValueError(f"método de crecimiento desconocido: {ev.get('method')}")
    dec, cfg = decoder(model), text_config(model)
    layers = list(dec.layers)
    types = list(getattr(cfg, "layer_types", None) or [])
    orig = [i for i, l in enumerate(layers) if not is_grown(l)]
    after = sorted(int(a) for a in ev["after"])
    if any(not 0 <= a < len(orig) for a in after):
        raise ValueError(f"growth: capas originales fuera de rango {after} (hay {len(orig)})")
    zero = tuple(ev.get("zero") or DEFAULT_ZERO)
    k_ev = 1 + max((getattr(b, "growth_event", -1) for _, b in grown_blocks(model)), default=-1)
    new_layers, new_types, new_idx = [], [], []
    i = 0
    while i < len(layers):
        layer, t, src = layers[i], (types[i] if types else None), i
        new_layers.append(layer)
        new_types.append(t)
        i += 1
        if is_grown(layer):
            continue
        while i < len(layers) and is_grown(layers[i]):  # bloques de eventos anteriores tras esta capa
            new_layers.append(layers[i])
            new_types.append(types[i] if types else None)
            i += 1
        for _ in range(after.count(orig.index(src))):
            blk = make_block(layer, cfg, src, zero, dtype)
            blk.growth_event = k_ev  # a qué evento pertenece (sus índices cambian si luego crece más)
            new_idx.append(len(new_layers))
            new_layers.append(blk)
            new_types.append(t)
    dec.layers = nn.ModuleList(new_layers)
    _renumber(dec.layers)
    if types:
        cfg.layer_types = new_types
    cfg.num_hidden_layers = len(new_layers)
    added = sum(p.numel() for i in new_idx for p in dec.layers[i].parameters())
    ev.update(method="depth", event=k_ev, after=after, zero=list(zero), layers_before=len(layers),
              layers_after=len(new_layers), new=new_idx, params_added=added)
    _sync_peft(model)
    return ev


def _sync_peft(model):
    """Si el modelo ya tiene LoRA: su config guardada debe apuntar a las capas originales con los índices nuevos (al
    recargar, peft inyecta LoRA sólo ahí) y los bloques nuevos deben ser entrenables (peft congela todo lo demás)."""
    pc = getattr(model, "peft_config", None)
    if not pc:
        return
    idx = original_indices(model)
    for c in pc.values():
        if getattr(c, "target_modules", None) is not None and not isinstance(c.target_modules, str):
            c.layers_to_transform = idx
            c.layers_pattern = "layers"
    for p in grown_parameters(model):
        p.requires_grad_(True)


def lora_layer_kwargs(model):
    """kwargs para LoraConfig si el modelo ya creció antes de añadir LoRA."""
    if not grown_blocks(model):
        return {}
    return dict(layers_to_transform=original_indices(model), layers_pattern="layers")


@contextmanager
def bypass(model):
    """Bloques crecidos desactivados (identidad sin cómputo): el modelo original exacto."""
    blocks = [b for _, b in grown_blocks(model)]
    prev = [b.bypass for b in blocks]
    for b in blocks:
        b.bypass = True
    try:
        yield model
    finally:
        for b, p in zip(blocks, prev):
            b.bypass = p


# ---------------------------------------------------------------------------------------------------------- estado
def state_dict(model):
    """Pesos de los bloques crecidos: {"<índice>.<clave>": tensor}. Los índices son los finales (tras todos los eventos)."""
    out = {}
    for i, b in grown_blocks(model):
        for k, v in b.state_dict().items():
            out[f"{i}.{k}"] = v.detach().contiguous()
    return out


def load_state_dict(model, sd):
    blocks = dict(grown_blocks(model))
    by = {}
    for k, v in sd.items():
        i, rest = k.split(".", 1)
        by.setdefault(int(i), {})[rest] = v
    if set(by) != set(blocks):
        raise RuntimeError(f"growth.safetensors no encaja: bloques guardados {sorted(by)} vs modelo {sorted(blocks)}")
    for i, d in by.items():
        blocks[i].load_state_dict(d, strict=True)


def save(model, path):
    from safetensors.torch import save_file
    sd = state_dict(model)
    if sd:
        save_file({k: v.cpu() for k, v in sd.items()}, path)
    return bool(sd)


def load(model, path):
    from safetensors.torch import load_file
    dev = next(model.parameters()).device
    load_state_dict(model, load_file(path, device=str(dev)))


def replay(model, events, dtype=None):
    """Repite los eventos de crecimiento de un checkpoint sobre el modelo base recién cargado."""
    for ev in events:
        apply_event(model, {k: v for k, v in ev.items() if k in ("method", "after", "zero", "step")}, dtype)
    return model
