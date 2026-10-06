"""GRPO (Shao et al. 2024, arXiv:2402.03300) sobre tokens de respuesta "planos".

Un step = un lote de rollouts (G respuestas por prompt) -> ventaja por grupo -> U pasadas (grpo.updates_per_batch) sobre
el lote, cada una partida en M minibatches (grpo.minibatches); cada minibatch es UN optimizer.step (acumulando los
micro-batches que caben en VRAM).
* U*M = 1 (por defecto): una actualización por lote. logp_old = logp_new.detach(): el ratio es 1 exacto y el clip no
  actúa; el gradiente es el de REINFORCE con baseline de grupo + KL. No hay pasada extra.
* U*M > 1 (PPO): logp_old se calcula ANTES de la primera actualización (pasada sin gradiente) y el ratio/clip actúan de
  verdad en las siguientes. approx_kl_old mide cuánto se movió la política dentro del lote.
Correcciones respecto a la versión anterior (medibles en tests):
* logp a la temperatura de muestreo en TODOS los modos (grpo.logp_temperature: generation, como TRL/verl): el
  ratio y el gradiente se refieren a la distribución que generó las muestras. Antes U=1 usaba T=1 y U>1 T=0.8: los dos
  modos optimizaban objetivos distintos y no eran comparables.
* Sin dropout en ningún modo (lora.dropout: 0.0 por defecto): con dropout el gradiente es el de una política distinta
  de la que muestreó, y en U*M>1 movería el ratio sin que cambien los pesos.
* La referencia (adapters apagados + bloques crecidos en bypass) se calcula una vez por lote.
* Logits sólo de la respuesta y por trozos (training/logprob.py): el pico de VRAM ya no depende de V x T.
* OOM: se reduce el micro-batch y se rehace sólo el minibatch en curso (las actualizaciones ya aplicadas son
  coherentes); un gradiente no finito descarta ESE optimizer.step.
"""
from __future__ import annotations

import gc, math, statistics as st, time
from dataclasses import dataclass

try:
    import torch
except ImportError:
    torch = None

from common import opt
from rollouts.generate import is_oom


def group_advantages(items, eps=1e-4, norm="std"):
    """items: [(group_key, reward), ...] -> [advantage, ...] en el mismo orden. Puro Python (sin torch).
    std: A_i = (r_i - media)/std (GRPO original). mean: A_i = r_i - media (Dr. GRPO, arXiv:2503.20783: sin el sesgo de
    dificultad de dividir por la std). batch: A_i = (r_i - media del grupo) / std de TODOS los rewards del lote (Lite PPO,
    arXiv:2508.08221): quita el sesgo de la std por grupo sin cambiar la escala global (con `mean` la ventaja es ~3-10x
    menor y el término KL, que no se escala, pesa más). Grupo sin varianza -> A = 0 para todo el grupo (no hay señal)."""
    if norm not in ("std", "mean", "batch"):
        raise ValueError(f"grpo.adv_norm desconocido: {norm} (std | mean | batch)")
    groups = {}
    for k, r in items:
        groups.setdefault(k, []).append(r)
    stats = {k: (st.fmean(v), st.pstdev(v)) for k, v in groups.items()}
    if norm == "mean":
        return [0.0 if stats[k][1] < eps else r - stats[k][0] for k, r in items]
    if norm == "batch":
        sd = st.pstdev([r for _, r in items]) if len(items) > 1 else 0.0
        return [0.0 if stats[k][1] < eps or sd < eps else (r - stats[k][0]) / sd for k, r in items]
    return [0.0 if stats[k][1] < eps else (r - stats[k][0]) / stats[k][1] for k, r in items]


def zero_var_keys(items, eps=1e-4):
    """Claves de grupo cuyo reward no varía (ventaja 0 para todo el grupo)."""
    groups = {}
    for k, r in items:
        groups.setdefault(k, []).append(r)
    return {k for k, v in groups.items() if st.pstdev(v) < eps}


def kl_k3(logp_new, logp_ref):
    """D_KL(pi_theta || pi_ref) por token, estimador k3 (Schulman): exp(d) - d - 1 con d = logp_ref - logp_new. >= 0."""
    d = logp_ref - logp_new
    return torch.exp(d) - d - 1


def logp_temperature(cfg):
    """Temperatura a la que se calculan los logp del ratio, del KL y del gradiente."""
    v = opt(cfg, "grpo.logp_temperature", "generation")
    t = float(cfg["generation"]["temperature"]) if v == "generation" else float(v)
    return t if t > 0 else 1.0


@dataclass
class RollItem:
    prompt_ids: list
    resp_ids: list
    advantage: float


@dataclass
class StepStats:
    loss: float = 0.0
    kl: float = 0.0             # KL(policy || ref) k3 medio por token de respuesta
    clip_frac: float = 0.0      # fracción de tokens con el ratio fuera de [1-eps, 1+eps] (0 si U*M = 1)
    grad_norm: float = 0.0      # norma pre-clip, media de los optimizer.step del lote
    tokens: int = 0             # tokens de respuesta del lote
    microbatches: int = 0
    oom: int = 0
    skipped: int = 0            # optimizer.step descartados por gradiente no finito
    # --- aditivos ---
    updates: int = 0            # optimizer.step aplicados
    approx_kl: float = 0.0      # KL(old || new) k3 medio en la última pasada (cuánto se movió la política en el lote)
    ratio_max: float = 1.0
    entropy: float = 0.0        # entropía media por token de la política (nats, a la temperatura de logp)
    logp: float = 0.0           # log-prob media por token de las respuestas, antes de actualizar (SFT: -NLL)
    t_ref: float = 0.0
    t_old: float = 0.0
    t_update: float = 0.0


def _pad_right(batch, pad_id, device):
    L = max(len(b.prompt_ids) + len(b.resp_ids) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n, m = len(b.prompt_ids), len(b.resp_ids)
        ids[i, :n] = torch.tensor(b.prompt_ids)
        ids[i, n:n + m] = torch.tensor(b.resp_ids)
        attn[i, :n + m] = 1
    return ids.to(device), attn.to(device), [len(b.prompt_ids) for b in batch], [len(b.resp_ids) for b in batch]


def grpo_loss(lp_new, lp_old, lp_ref, adv, clip_eps, kl_coef, denom):
    """Pérdida PPO-clip + KL k3 sobre tokens planos (N,). denom = tokens del minibatch completo: sumando la pérdida
    de sus micro-batches se obtiene un token-mean exacto del minibatch."""
    ratio = torch.exp(lp_new - lp_old)
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    kl = kl_k3(lp_new, lp_ref)
    loss = -(surr - kl_coef * kl).sum() / denom
    with torch.no_grad():
        d = lp_old - lp_new.detach()
        stats = dict(kl=kl.detach().sum().item(), clip=((ratio.detach() - 1).abs() > clip_eps).float().sum().item(),
                     approx_kl=(torch.exp(d) - d - 1).sum().item(),
                     ratio_max=ratio.detach().max().item() if ratio.numel() else 1.0)
    return loss, stats


class GrpoTrainer:
    def __init__(self, model, ref_ctx, tok, optimizer, cfg):
        self.m, self.ref_ctx, self.tok, self.opt, self.cfg = model, ref_ctx, tok, optimizer, cfg
        g = cfg.grpo
        self.eps, self.kl_coef, self.max_norm = float(g.clip_eps), float(g.kl_coef), float(g.max_grad_norm)
        self.margin = float(cfg.vram_margin)
        self.updates = max(1, int(opt(cfg, "grpo.updates_per_batch", 1)))
        self.minibatches = max(1, int(opt(cfg, "grpo.minibatches", 1)))
        self.temp = logp_temperature(cfg)
        self.chunk_mb = float(opt(cfg, "grpo.head_chunk_mb", 512))
        self.micro_bs = 4 if g.micro_bs == "auto" else int(g.micro_bs)
        self.nograd_scale = 1 if g.micro_bs != "auto" else 4  # micro-batches de las pasadas sin gradiente
        self.ref_len = None  # longitud con la que se ajustó micro_bs (tune): presupuesto = micro_bs * ref_len tokens
        self.pad_id = tok.pad_token_id
        for mod in model.modules():  # RL on-policy: sin dropout (también en adapters guardados con lora_dropout > 0)
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

    # ------------------------------------------------------------------ utilidades
    def _params(self):
        return [p for p in self.m.parameters() if p.requires_grad]

    def _micro(self, items, scale=1):
        """Micro-batches de items ordenados por longitud. Con tune(): presupuesto de micro_bs * ref_len tokens con
        padding (más items si son cortos); sin tune: micro_bs items como siempre. scale > 1 para las pasadas sin
        gradiente (no guardan activaciones: caben varias veces más tokens)."""
        bs = self.micro_bs * scale
        out, cur, budget = [], [], bs * self.ref_len if self.ref_len else None
        for it in items:
            L = len(it.prompt_ids) + len(it.resp_ids)
            longest = max([L] + [len(c.prompt_ids) + len(c.resp_ids) for c in cur])
            full = (len(cur) >= bs) if budget is None else ((len(cur) + 1) * longest > budget)
            if cur and full:
                out.append(cur)
                cur = []
            cur.append(it)
        return out + ([cur] if cur else [])

    def _logps(self, model, batch, entropy=False):
        from training.logprob import token_logps
        ids, attn, pl, rl = _pad_right(batch, self.pad_id, self.m.device)
        return token_logps(model, ids, attn, pl, rl, self.temp, self.chunk_mb, entropy=entropy)

    def _oom(self, stats):
        stats.oom += 1
        self.opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        if self.micro_bs <= 1:
            raise RuntimeError("OOM en training incluso con micro_bs=1")
        self.micro_bs = max(1, self.micro_bs // 2)

    def _no_grad_pass(self, items, stats, ref):
        """logp (y entropía de la política) por item, sin gradiente: [(N_i,) ...] en el orden de items."""
        while True:
            out, ents, oom = [], [], False
            try:
                with torch.no_grad():
                    for mb in self._micro(items, scale=self.nograd_scale):
                        if ref:
                            with self.ref_ctx() as rm:
                                lp = self._logps(rm, mb)
                        else:
                            lp, en = self._logps(self.m, mb, entropy=True)
                            ents.append(en)
                        out += list(torch.split(lp, [len(b.resp_ids) for b in mb]))
            except Exception as e:
                if not is_oom(e):
                    raise
                oom = True
            if not oom:
                return out, (torch.cat(ents) if ents else None)
            self._oom(stats)  # fuera del except: sin referencias del traceback

    # ------------------------------------------------------------------ step
    def step(self, items):
        """Deja el modelo en eval() al salir pase lo que pase (la generación nunca hereda train())."""
        stats = StepStats()
        if not items:
            return stats
        items = sorted(items, key=lambda b: len(b.prompt_ids) + len(b.resp_ids))
        index = {id(b): i for i, b in enumerate(items)}
        stats.tokens = sum(len(b.resp_ids) for b in items)
        total = self.updates * self.minibatches
        applied = 0
        self.m.eval()  # sin dropout también en las pasadas sin gradiente
        try:
            t0 = time.perf_counter()
            ref, _ = self._no_grad_pass(items, stats, ref=True)
            stats.t_ref = time.perf_counter() - t0
            old, ent = None, None
            if total > 1:
                t0 = time.perf_counter()
                old, ent = self._no_grad_pass(items, stats, ref=False)
                stats.t_old = time.perf_counter() - t0
            t0 = time.perf_counter()
            self.m.train()
            losses, norms, kl_sum, clip_sum, akl_sum, last_tokens, ent_sum, lp_sum = [], [], 0.0, 0.0, 0.0, 0, 0.0, 0.0
            for ep in range(self.updates):
                mbs = [items[k::self.minibatches] for k in range(self.minibatches)]  # reparto por longitud, sin RNG
                for mb in mbs:
                    if not mb:
                        continue
                    denom = max(1, sum(len(b.resp_ids) for b in mb))
                    while True:
                        self.opt.zero_grad(set_to_none=True)
                        part = dict(loss=0.0, kl=0.0, clip=0.0, akl=0.0, rmax=1.0, ent=0.0, micro=0, lp=0.0)
                        oom = False
                        try:
                            for micro in self._micro(mb):
                                want_ent = old is None  # U*M = 1: la entropía sale del propio forward con gradiente
                                res = self._logps(self.m, micro, entropy=want_ent)
                                lp_new, en = res if want_ent else (res, None)
                                idx = [index[id(b)] for b in micro]
                                lp_ref = torch.cat([ref[i] for i in idx])
                                lp_old = torch.cat([old[i] for i in idx]) if old is not None else lp_new.detach()
                                adv = torch.cat([lp_new.new_full((len(items[i].resp_ids),), items[i].advantage)
                                                 for i in idx])
                                loss, s = grpo_loss(lp_new, lp_old, lp_ref, adv, self.eps, self.kl_coef, denom)
                                loss.backward()
                                part["loss"] += loss.item()
                                part["kl"] += s["kl"]
                                part["clip"] += s["clip"]
                                part["akl"] += s["approx_kl"]
                                part["rmax"] = max(part["rmax"], s["ratio_max"])
                                part["ent"] += en.sum().item() if en is not None else 0.0
                                part["lp"] += lp_new.detach().sum().item()
                                part["micro"] += 1
                        except Exception as e:
                            if not is_oom(e):
                                raise
                            oom = True
                        if oom:
                            self._oom(stats)
                            continue
                        break
                    gn = float(torch.nn.utils.clip_grad_norm_(self._params(), self.max_norm))
                    if math.isfinite(gn):
                        self.opt.step()
                        applied += 1
                    else:  # aplicarlo escribiría NaN en los pesos, y de ahí en todos los checkpoints siguientes
                        stats.skipped += 1
                    self.opt.zero_grad(set_to_none=True)
                    losses.append(part["loss"])
                    norms.append(gn)
                    stats.microbatches += part["micro"]
                    stats.ratio_max = max(stats.ratio_max, part["rmax"])
                    clip_sum += part["clip"]
                    if ep == self.updates - 1:
                        kl_sum += part["kl"]
                        akl_sum += part["akl"]
                        last_tokens += denom
                    ent_sum += part["ent"]
                    if ep == 0:
                        lp_sum += part["lp"]
            stats.t_update = time.perf_counter() - t0
            stats.updates = applied
            stats.loss = sum(losses) / len(losses) if losses else 0.0
            finite = [n for n in norms if math.isfinite(n)]
            stats.grad_norm = sum(finite) / len(finite) if finite else float("nan")
            stats.kl = kl_sum / max(1, last_tokens)
            stats.approx_kl = akl_sum / max(1, last_tokens) if total > 1 else 0.0
            stats.clip_frac = clip_sum / max(1, stats.tokens * self.updates) if total > 1 else 0.0
            stats.logp = lp_sum / max(1, stats.tokens)
            stats.entropy = (ent.mean().item() if ent is not None and ent.numel() else
                             ent_sum / max(1, stats.tokens * self.updates))
            return stats
        except Exception as e:
            e.updates_done = applied  # train.py: el modelo ya lleva `applied` actualizaciones de ESTE step
            raise
        finally:
            self.m.eval()

    # ------------------------------------------------------------------ autotuner
    def tune(self, prompt_ids, resp_len, max_bs=64):
        """Mayor micro_bs que cabe en el PEOR caso: prompt más largo + max_new_tokens de respuesta (antes se medía con
        una muestra real, a menudo corta: luego las respuestas largas pasaban del presupuesto). Mide el forward+backward
        de la política (el pico del step; las pasadas sin gradiente gastan mucho menos)."""
        from model.vram import free_bytes
        filler = self.tok.eos_token_id if self.tok.eos_token_id is not None else self.pad_id
        item = RollItem(list(prompt_ids), [filler] * int(resp_len), 1.0)
        self.ref_len = len(item.prompt_ids) + len(item.resp_ids)
        self.m.train()
        try:
            rows, best, bs = [], None, 1
            while bs <= max_bs:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                free0, base = free_bytes(), torch.cuda.memory_reserved()
                batch = [item] * bs
                self.opt.zero_grad(set_to_none=True)
                oom = False
                try:
                    lp = self._logps(self.m, batch)
                    loss = -(lp * 1.0).sum() / max(1, lp.numel())
                    loss.backward()
                except Exception as e:
                    if not is_oom(e):
                        raise
                    oom = True
                self.opt.zero_grad(set_to_none=True)
                if oom:
                    rows.append(dict(bs=bs, oom=True))
                    break
                extra = torch.cuda.max_memory_reserved() - base
                rows.append(dict(bs=bs, tokens=bs * self.ref_len, extra_mb=extra >> 20))
                if extra > self.margin * free0:
                    break
                best, bs = bs, bs * 2
            gc.collect()
            torch.cuda.empty_cache()
            if best:
                self.micro_bs = best
            return rows
        finally:
            self.m.eval()
