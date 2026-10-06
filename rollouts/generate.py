"""Generación por lotes con UNA sola copia del modelo. Backoff ante OOM. Autotuner de concurrencia.

Concurrencia == nº de secuencias por llamada a generate (el batch de generación). Los N rollouts de un problema
son N secuencias del mismo prompt; se agrupan en lotes de `chunk` entre todos los problemas pendientes.

resp_ids contiene SÓLO tokens muestreados por el modelo: un EOS elegido por el modelo entra (también es una decisión
que se entrena), pero el relleno que generate() pone tras una parada por stop string ("```") no. Antes ese token de
relleno (<|endoftext|>) entraba en la respuesta y GRPO lo entrenaba como si el modelo lo hubiera elegido.
stats["steps"] cuenta posiciones decodificadas (llamadas al modelo por secuencia): ms/step = coste de un paso de
decodificación del lote, independiente de la longitud de las respuestas; un salto brusco delata paginación de VRAM.
"""
import gc, re, time
from dataclasses import dataclass

try:
    import torch
except ImportError:  # la lógica de backoff se puede testear sin torch
    torch = None


@dataclass
class Sample:
    text: str
    code: str
    ntok: int
    closed: bool          # terminó solo (fence de cierre o EOS) antes del límite
    prompt_ids: list = None   # tokens reales del prompt (sin padding)
    resp_ids: list = None     # tokens reales generados (sin padding; incluye el EOS de cierre si lo hubo)


def is_oom(e):
    if torch is not None and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


STOP = "```"


def response_length(row, pad_id, eos_ids, stopped):
    """Nº de tokens MUESTREADOS en una fila de generate() (sin el prompt): hasta el primer EOS/relleno. Un EOS entra
    (lo eligió el modelo); el relleno no. Si pad_id también es un EOS (Qwen: <|endoftext|>), se distingue por
    `stopped`: tras una parada por stop string el siguiente pad es relleno; si no hubo stop string, lo eligió el modelo."""
    for k, t in enumerate(row):
        if t in eos_ids and not (t == pad_id and stopped):
            return k + 1
        if t == pad_id:
            return k
    return len(row)


def build_prompt(tok, system, user, prefill):
    s = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                tokenize=False, add_generation_prompt=True)
    assert isinstance(s, str), "apply_chat_template(tokenize=False) debe devolver str"
    return s + "```python\n" if prefill else s


def extract_code(text, prefilled):
    if prefilled:
        return text.split("```", 1)[0]
    m = re.search(r"```(?:python|py)?\n(.*?)(?:```|$)", text, re.S)
    return m.group(1) if m else text


class Generator:
    def __init__(self, model, tok, cfg):
        g = cfg.generation
        self.m, self.tok, self.ctx = model, tok, int(cfg.context_length)
        self.chunk, self.max_new = int(g.chunk), int(cfg.max_new_tokens)
        self.floor, self.margin = int(g.min_new_tokens_floor), float(cfg.vram_margin)
        self.prefilled = bool(cfg.prompt.prefill_fence)
        t = float(g.temperature)
        # on-policy: sin top_k, sin repetition_penalty (los defaults de Qwen los activarían)
        self.sampling = (dict(do_sample=True, temperature=t, top_p=float(g.top_p), top_k=0, repetition_penalty=1.0)
                         if t > 0 else dict(do_sample=False))
        self.stats = dict(tokens=0, samples=0, seconds=0.0, oom=0, steps=0, calls=0)
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos = set(eos if isinstance(eos, (list, tuple, set)) else [eos] if eos is not None else [])
        self.eos_ids = {e for e in eos | {tok.eos_token_id} if e is not None}

    def _sampling(self, temperature=None):
        """Parámetros de muestreo. None = los de siempre (self.sampling, sin cambios); un override por llamada
        cambia sólo la temperatura (<= 0: greedy)."""
        if temperature is None:
            return self.sampling
        t = float(temperature)
        return (dict(do_sample=True, temperature=t, top_p=float(self.sampling.get("top_p", 1.0)), top_k=0,
                     repetition_penalty=1.0) if t > 0 else dict(do_sample=False, repetition_penalty=1.0))
        # greedy puro: sin repetition_penalty (el generation_config de Qwen lo activaría también sin muestreo)

    def _gen(self, texts, full=False, temperature=None):
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.m.device)
        plen = enc["input_ids"].shape[1]
        if plen + self.max_new > self.ctx:
            raise ValueError(f"prompt({plen}) + max_new_tokens({self.max_new}) > context_length({self.ctx})")
        kw = dict(self._sampling(temperature), max_new_tokens=self.max_new, pad_token_id=self.tok.pad_token_id)
        if full:  # peor caso para el autotuner: sin parada anticipada
            kw["min_new_tokens"] = self.max_new
        else:     # parada anticipada en el fence de cierre: cero tokens de explicación
            kw.update(stop_strings=[STOP], tokenizer=self.tok)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.m.generate(**enc, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[:, plen:]
        self.stats["steps"] += gen.shape[1]
        self.stats["calls"] += 1
        rows = gen.tolist()
        outs = self.tok.batch_decode(gen, skip_special_tokens=True)
        pad = self.tok.pad_token_id
        ntok = [response_length(r, pad, self.eos_ids, STOP in t) for r, t in zip(rows, outs)]
        pids = [enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist() for i in range(len(texts))]
        return [Sample(t, extract_code(t, self.prefilled), n, STOP in t or n < self.max_new, p, r[:n])
                for t, n, p, r in zip(outs, ntok, pids, rows)], dt

    def _backoff(self):
        """VRAM insuficiente -> menos secuencias concurrentes -> (si ya es 1) menor longitud -> continuar."""
        self.stats["oom"] += 1
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.chunk > 1:
            self.chunk = max(1, self.chunk // 2)
        elif self.max_new > self.floor:
            self.max_new = max(self.floor, int(self.max_new * 0.75))
        else:
            raise RuntimeError("OOM incluso con chunk=1 y max_new_tokens mínimo")

    def stream(self, prompts, n, temperature=None):
        """Genera n muestras por prompt. Produce (índice_prompt, Sample) según termina cada lote.
        eval() explícito: no confiar en que quien llamó dejó el modelo en el modo correcto (p.ej. tras un
        GrpoTrainer.step(), que usa .train() para su propio forward -- ver training/grpo.py). Sin esto,
        el dropout de LoRA quedaría activo durante la generación, añadiendo ruido no controlado a cada
        rollout posterior al primer training step."""
        self.m.eval()
        items = sorted(((i, p) for i, p in enumerate(prompts) for _ in range(n)), key=lambda x: len(x[1]))
        pos = 0
        while pos < len(items):
            batch, oom = items[pos:pos + self.chunk], False
            try:
                texts = [p for _, p in batch]
                res, dt = self._gen(texts) if temperature is None else self._gen(texts, temperature=temperature)
            except Exception as e:
                if not is_oom(e):
                    raise
                oom = True
            if oom:  # fuera del except: libera las referencias del traceback antes del reintento
                self._backoff()
                continue
            pos += len(batch)
            self.stats["tokens"] += sum(s.ntok for s in res)
            self.stats["samples"] += len(res)
            self.stats["seconds"] += dt
            for (i, _), s in zip(batch, res):
                yield i, s

    def tune(self, prompt, max_bs=512, start=1):
        """Duplica las secuencias concurrentes (1,2,4,...) midiendo tokens/s y VRAM extra del peor caso.
        Se detiene al pasar el margen de VRAM, al hacer OOM o cuando doblar rinde <5%. Fija self.chunk.
        start > 1 (re-ajuste tras crecer el modelo): prueba ese tamaño y, si ya no cabe, la mitad, y así; no vuelve a
        barrer desde 1 (cada prueba genera max_new tokens: el barrido completo costaba ~85 s)."""
        from model.vram import free_bytes
        self.m.eval()  # ver nota en stream(): no asumir el modo en que quedó el modelo
        down = start > 1
        rows, best, bs = [], None, max(1, int(start))
        while 1 <= bs <= max_bs:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free0, base, oom = free_bytes(), torch.cuda.memory_reserved(), False
            try:
                res, dt = self._gen([prompt] * bs, full=True)
            except Exception as e:
                if not is_oom(e):
                    raise
                oom = True
            if oom:
                rows.append(dict(bs=bs, oom=True))
            else:
                extra = torch.cuda.max_memory_reserved() - base
                tps = sum(s.ntok for s in res) / dt
                rows.append(dict(bs=bs, tok_s=round(tps, 1), extra_mb=extra >> 20, s=round(dt, 2)))
            if oom or extra > self.margin * free0:
                if down and best is None:  # re-ajuste: el tamaño anterior ya no cabe -> la mitad
                    bs //= 2
                    continue
                break
            if best and tps < best[0] * 1.05:
                break
            best, bs = (tps, bs), bs * 2
            if down:  # re-ajuste: el primero que cabe (los mayores ya se descartaron en el ajuste inicial)
                break
        gc.collect()
        torch.cuda.empty_cache()
        if best:
            self.chunk = best[1]
        return rows
