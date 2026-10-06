"""Log-probabilidades por token de la RESPUESTA sin materializar los logits del vocabulario entero.

Antes: model(ids).logits -> (B, T, V) en bf16, copia fp32 y los intermedios de cross_entropy con sus gradientes, para
TODAS las posiciones (prompt incluido, que luego se enmascaraba). Con V = 151 936, micro_bs = 4 y ~600 tokens son
~5 GB transitorios por micro-batch más otros tantos del forward de referencia: es lo que llevaba el pico a 10,4 GB
y, con ~1 GB ocupado por Windows, a paginar VRAM en la RAM del sistema (steps 25x más lentos sin OOM).

Ahora:
1. El decoder da los estados ocultos (B, T, H) (use_cache=False: en training no hace falta caché KV).
2. Se seleccionan SÓLO las posiciones que predicen tokens de respuesta: N = suma de longitudes de respuesta.
3. La cabeza (lm_head) se aplica por trozos de filas; con gradiente, cada trozo va dentro de
   torch.utils.checkpoint: logits fp32 -> logsumexp -> logp del token elegido, y se descartan; el backward los
   recalcula (coste: repetir la matmul de la cabeza, ~10 % de un forward+backward de 0.5B).
Pico de la cabeza: un trozo (filas x V x 4 bytes x ~3), independiente de B y T.

Todo es "plano": el resultado es un vector (N,) en el orden item a item y, dentro de cada item, token a token.
"""
from __future__ import annotations

try:
    import torch
    from torch.utils.checkpoint import checkpoint
except ImportError:  # la aritmética de posiciones se puede usar/testear sin torch
    torch = None


def lm_parts(model):
    """(decoder, cabeza) de un CausalLM de transformers, también envuelto por PEFT (LoRA vive dentro del decoder)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.get_decoder(), base.get_output_embeddings()


def flat_positions(prompt_lens, resp_lens):
    """Índices (fila, posición del estado oculto, posición del token) de cada token de respuesta, padding a la derecha.
    El token en la posición p lo predice el estado oculto de p - 1."""
    rows, hid, tgt = [], [], []
    for b, (n, m) in enumerate(zip(prompt_lens, resp_lens)):
        if m and n < 1:
            raise ValueError("una respuesta necesita al menos un token de prompt delante")
        rows += [b] * m
        hid += range(n - 1, n - 1 + m)
        tgt += range(n, n + m)
    return rows, hid, tgt


def chunk_rows(vocab, chunk_mb):
    """Filas por trozo para que logits fp32 + intermedios (~3 copias) quepan en chunk_mb."""
    return max(16, int(chunk_mb * 2**20 / (vocab * 4 * 3)))


def _head_logp(head, h, tgt, inv_t, want_entropy):
    logits = head(h).float()
    if inv_t != 1.0:
        logits = logits * inv_t
    lse = torch.logsumexp(logits, dim=-1)
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lse
    if not want_entropy:
        return lp
    with torch.no_grad():  # entropía sólo como diagnóstico: sin gradiente
        p = torch.softmax(logits, dim=-1)
        ent = lse - (p * logits).sum(-1)
    return lp, ent


def token_logps(model, input_ids, attention_mask, prompt_lens, resp_lens, temperature=1.0, chunk_mb=512,
                entropy=False):
    """logp (N,) fp32 de cada token de respuesta bajo `model` a `temperature`, y su entropía (N,) si entropy=True.
    El gradiente fluye si está habilitado y algún parámetro lo requiere (LoRA, bloques crecidos...)."""
    dec, head = lm_parts(model)
    hidden = dec(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    rows, hid, tgt = flat_positions(prompt_lens, resp_lens)
    dev = hidden.device
    if not rows:
        empty = hidden.new_zeros(0, dtype=torch.float32)
        return (empty, empty.clone()) if entropy else empty
    r = torch.tensor(rows, device=dev)
    h = hidden[r, torch.tensor(hid, device=dev)]                   # (N, H)
    y = input_ids[r, torch.tensor(tgt, device=dev)]                 # (N,)
    inv_t = 1.0 / float(temperature) if temperature and temperature > 0 else 1.0
    step = chunk_rows(head.weight.shape[0] if hasattr(head, "weight") else 151936, chunk_mb)
    grad = torch.is_grad_enabled() and h.requires_grad
    lps, ents = [], []
    for s in range(0, h.shape[0], step):
        args = (head, h[s:s + step], y[s:s + step], inv_t, entropy)
        out = checkpoint(_head_logp, *args, use_reentrant=False) if grad else _head_logp(*args)
        if entropy:
            lps.append(out[0])
            ents.append(out[1])
        else:
            lps.append(out)
    lp = torch.cat(lps)
    return (lp, torch.cat(ents)) if entropy else lp


def full_token_logps(model, input_ids, attention_mask, temperature=1.0):
    """Referencia ingenua (todas las posiciones, logits completos): sólo para tests de equivalencia."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :].float()
    if temperature and temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    return torch.log_softmax(logits, -1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
