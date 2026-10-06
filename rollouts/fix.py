"""Generate -> test -> fix.

engine v1 (por defecto, el bucle de siempre): cada intento fallido vuelve al modelo con su código y el feedback,
hasta resolver o agotar las rondas; todas las cadenas pendientes de una ronda van en un único collect().
engine v2: k candidatos por ronda, memoria de candidatos (AST normalizado: un repetido no se ejecuta), detección de
la misma firma de error, reinicio desde cero si se atasca, paciencia, presupuesto y parada al pasar todos los tests
PERMITIDOS. La selección nunca usa los tests ocultos: sólo visibles + pool de contraejemplos (rollouts/solve.py).
fix.feedback: basic (el mensaje de siempre) | counterexample (rewards/feedback.py, feedback_v2).
compare(): A/B basic vs counterexample sobre exactamente los mismos primeros intentos fallidos y las mismas semillas.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from common import opt, seed_all, workers
from rewards.feedback import build as build_feedback
from rollouts.collect import code_key, collect
from rollouts.generate import build_prompt

ASK = "Fix the function. Reply with the complete corrected code in one Python code block."


@dataclass
class Attempt:
    problem: str
    chain: int
    round: int
    code: str
    reward: float
    solved: bool
    compiled: bool
    error: str | None
    feedback: str | None  # lo que verá el modelo en la ronda siguiente (None si resolvió o no se le pidió)
    ntok: int
    # --- aditivos ---
    mode: str = "basic"            # feedback con el que se construyó `feedback`
    cand: int = 0                  # candidato dentro de la ronda (v2)
    strategy: str = "first"        # first | repair | restart
    repeat: bool = False           # AST idéntico a otro candidato de la cadena: no se volvió a ejecutar
    allowed: float | None = None   # fracción de tests permitidos (visibles + pool) que pasa (v2)
    selected: bool = True          # candidato elegido en su ronda (v2; en v1 todos)
    sig: str | None = None         # firma del error: tipo | categoría del caso | línea
    truncated: bool = False
    line: int | None = None
    fb_source: str | None = None   # de dónde sale el feedback v2: visible | pool | large | static | none


def fix_prompt(tok, cfg, p, code, fb):
    msgs = [{"role": "system", "content": cfg.prompt.system}, {"role": "user", "content": p.prompt()},
            {"role": "assistant", "content": f"```python\n{code.strip()}\n```"},
            {"role": "user", "content": f"{fb}\n{ASK}"}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s + "```python\n" if cfg.prompt.prefill_fence else s


def prompt_for(gen, cfg, p, code=None, fb=None):
    plain = build_prompt(gen.tok, cfg.prompt.system, p.prompt(), cfg.prompt.prefill_fence)
    if code is None:
        return plain
    s = fix_prompt(gen.tok, cfg, p, code, fb)
    # un intento previo enorme no debe tumbar el lote entero (Generator exige prompt + max_new <= contexto)
    return s if len(gen.tok(s).input_ids) + gen.max_new <= gen.ctx else plain


def signature(r):
    """Firma del error: dos intentos con la misma firma fallan de la misma manera."""
    if r.solved:
        return None
    fc = r.fail_case or {}
    return f"{(r.error or '').split(':')[0]}|{fc.get('category', '')}|{r.line or ''}"


def _fb_seed(seed, k, rnd):
    return seed * 7919 + k * 104729 + rnd  # semilla del pool de contraejemplos: reproducible e independiente


def _attempt(p, k, rnd, code, ntok, r, cfg, mode, seed, truncated, want_feedback=True, **kw):
    fb, info = (build_feedback(p, code, r, cfg, mode, _fb_seed(seed, k, rnd), truncated) if want_feedback
                else (None, {}))
    return Attempt(p.id, k, rnd, code, r.reward, r.solved, r.compiled, r.error, fb, ntok, mode=mode,
                   truncated=truncated, line=r.line, sig=signature(r), fb_source=info.get("source"), **kw)


def first_attempts(gen, probs, cfg, n=1, seed=0):
    """Primer intento de cada cadena (probs x n): [(código, Result, ntok, truncado)]. Base común del A/B."""
    out = []
    for items in collect(gen, list(probs), n, cfg, seed):
        out += [(s.code, r, s.ntok, not getattr(s, "closed", True)) for s, r in items]
    return out


def run(gen, probs, cfg, rounds, n=1, seed=0, on_attempt=None, feedback_mode=None, first=None, engine=None):
    """rounds = correcciones tras el primer intento; n cadenas independientes por problema.
    Devuelve las cadenas (listas de Attempt) en el orden probs x n. on_attempt(attempt, result) por cada intento.
    first: primeros intentos ya hechos (ver first_attempts), para comparar feedbacks sobre los mismos fallos."""
    mode = feedback_mode or opt(cfg, "fix.feedback", "basic")
    if (engine or opt(cfg, "fix.engine", "v1")) == "v2":
        return run_v2(gen, probs, cfg, rounds, n, seed, on_attempt, mode, first)
    chains = [(p, k, []) for p in probs for k in range(n)]
    start = 0
    if first is not None:
        for (p, k, att), (code, r, ntok, trunc) in zip(chains, first):
            a = _attempt(p, k, 0, code, ntok, r, cfg, mode, seed, trunc)
            att.append(a)
            if on_attempt:
                on_attempt(a, r)
        start = 1
    for rnd in range(start, rounds + 1):
        todo = [c for c in chains if not c[2] or not c[2][-1].solved]
        if not todo:
            break
        prompts = [prompt_for(gen, cfg, p, att[-1].code, att[-1].feedback) if att else prompt_for(gen, cfg, p)
                   for p, _, att in todo]
        results = collect(gen, [p for p, _, _ in todo], 1, cfg, seed, prompts=prompts)
        for (p, k, att), [(s, r)] in zip(todo, results):
            a = _attempt(p, k, rnd, s.code, s.ntok, r, cfg, mode, seed, not getattr(s, "closed", True),
                         strategy="repair" if att else "first")
            att.append(a)
            if on_attempt:
                on_attempt(a, r)
    return [att for _, _, att in chains]


class _Chain:
    def __init__(self, p, k):
        self.p, self.k, self.atts, self.seen = p, k, [], {}
        self.best, self.best_r, self.stall, self.restarts, self.used, self.done = None, None, 0, 0, 0, None
        self.sigs = Counter()


def run_v2(gen, probs, cfg, rounds, n=1, seed=0, on_attempt=None, feedback_mode=None, first=None):
    """Motor v2 (ver docstring del módulo). Devuelve las cadenas (listas de Attempt, todos los candidatos)."""
    from rollouts.solve import allowed_score, sample, score_many
    mode = feedback_mode or opt(cfg, "fix.feedback", "basic")
    k_cand = max(1, int(opt(cfg, "fix.candidates", 4)))
    patience, max_restarts = int(opt(cfg, "fix.patience", 2)), int(opt(cfg, "fix.restarts", 1))
    budget, pool_n = int(opt(cfg, "fix.budget", 16)), int(opt(cfg, "fix.pool", 64))
    temp = opt(cfg, "fix.temperature", None)
    chains = [_Chain(p, k) for p in probs for k in range(n)]

    for rnd in range(rounds + 1):
        active = [c for c in chains if c.done is None]
        if not active:
            break
        jobs = []  # (cadena, prompt, estrategia, nº candidatos)
        for ch in active:
            if rnd == 0:
                jobs.append((ch, None if first is not None else prompt_for(gen, cfg, ch.p), "first", 1))
                continue
            stuck = ch.stall >= patience or (ch.best is not None and ch.sigs[ch.best.sig] >= 2 and ch.stall > 0)
            if stuck and ch.restarts >= max_restarts:
                ch.done = "patience"
                continue
            room = budget - ch.used
            if room <= 0:
                ch.done = "budget"
                continue
            if stuck:
                ch.restarts += 1
                ch.stall = 0
                ch.sigs.clear()
                jobs.append((ch, prompt_for(gen, cfg, ch.p), "restart", min(k_cand, room)))
            else:
                if ch.best.feedback is None:  # feedback sólo del candidato que se va a reparar
                    ch.best.feedback = build_feedback(ch.p, ch.best.code, ch.best_r, cfg, mode,
                                                      _fb_seed(seed, ch.k, ch.best.round), ch.best.truncated)[0]
                jobs.append((ch, prompt_for(gen, cfg, ch.p, ch.best.code, ch.best.feedback), "repair",
                             min(k_cand, room)))
        if not jobs:
            break
        if rnd == 0 and first is not None:
            gens = [[(code, ntok, trunc)] for code, _, ntok, trunc in first]
            pre = {id(ch): r for ch, (_, r, _, _) in zip(chains, first)}
        else:
            flat = [(j, pr) for j, (_, pr, _, m) in enumerate(jobs) for _ in range(m)]
            outs = sample(gen, [pr for _, pr in flat], 1, temp if rnd > 0 else None)
            gens = [[] for _ in jobs]
            for (j, _), [s] in zip(flat, outs):
                gens[j].append((s.code, s.ntok, not getattr(s, "closed", True)))
            pre = {}
        new, dups, pending = [], [], set()  # (cadena, cand, código, ntok, truncado, estrategia, clave)
        for (ch, _, strategy, _), items in zip(jobs, gens):
            for cand, (code, ntok, trunc) in enumerate(items):
                key = code_key(code)
                ch.used += 1
                item = (ch, cand, code, ntok, trunc, strategy, key)
                if key in ch.seen or (id(ch), key) in pending:  # repetido (antes o en esta ronda): no se ejecuta
                    dups.append(item)
                else:
                    pending.add((id(ch), key))
                    new.append(item)
        fresh = [it for it in new if id(it[0]) not in pre]
        scored = {id(it): r for it, r in zip(fresh, score_many([(it[0].p, it[2]) for it in fresh], cfg, seed))}
        with ThreadPoolExecutor(workers(cfg)) as ex:  # tests permitidos de todos los candidatos nuevos, en paralelo
            allowed = list(ex.map(lambda it: allowed_score(it[0].p, it[2], cfg, seed, pool_n), new))
        for it, al in zip(new, allowed):
            ch, cand, code, ntok, trunc, strategy, key = it
            r = pre[id(ch)] if id(ch) in pre else scored[id(it)]
            r.truncated = trunc
            a = _attempt(ch.p, ch.k, rnd, code, ntok, r, cfg, mode, seed, trunc, want_feedback=False, cand=cand,
                         strategy=strategy, selected=False, allowed=al)
            ch.seen[key] = (a, r)
            ch.atts.append(a)
            if on_attempt:
                on_attempt(a, r)
        for ch, cand, code, ntok, trunc, strategy, key in dups:  # heredan las puntuaciones del original
            a0, r0 = ch.seen[key]
            att = Attempt(ch.p.id, ch.k, rnd, code, a0.reward, a0.solved, a0.compiled, a0.error, None, ntok,
                          mode=mode, cand=cand, strategy=strategy, repeat=True, allowed=a0.allowed,
                          selected=False, sig=a0.sig, truncated=trunc, line=a0.line)
            ch.atts.append(att)
            if on_attempt:
                on_attempt(att, r0)
        for ch in {id(j[0]): j[0] for j in jobs}.values():
            cands = [a for a in ch.atts if a.round == rnd and not a.repeat]
            if not cands:
                ch.stall += 1
                continue
            pick = max(cands, key=lambda a: (a.allowed, -len(a.code), -a.cand))
            pick.selected = True
            ch.sigs[pick.sig] += 1
            if ch.best is None or pick.allowed > ch.best.allowed:
                ch.best, ch.best_r, ch.stall = pick, ch.seen[code_key(pick.code)][1], 0
            else:
                ch.stall += 1
            if ch.best.allowed >= 1.0:
                ch.done = "allowed_pass"  # pasa todo lo comprobable: parar (solved se mide aparte, con ocultos)
    for ch in chains:
        if ch.best is not None and ch.best.feedback is None and not ch.best.solved:
            ch.best.feedback = build_feedback(ch.p, ch.best.code, ch.best_r, cfg, mode,
                                              _fb_seed(seed, ch.k, ch.best.round), ch.best.truncated)[0]
    return [ch.atts for ch in chains]


def final_attempt(chain):
    """Respuesta final de una cadena: v1 = último intento; v2 = el mejor por tests permitidos (el más temprano)."""
    sel = [a for a in chain if a.selected and a.allowed is not None]
    if not sel:
        return chain[-1]
    return max(sel, key=lambda a: (a.allowed, -a.round, -a.cand))


def compare(gen, probs, cfg, rounds, n=1, seed=0, on_attempt=None, engine=None):
    """A/B reproducible basic vs counterexample: mismos primeros intentos (luego los mismos fallos), mismas semillas
    de muestreo en cada rama; sólo cambia el feedback. Devuelve {modo: {"summary", "chains"}}."""
    seed_all(seed)
    firsts = first_attempts(gen, probs, cfg, n, seed)
    out = {}
    for mode in ("basic", "counterexample"):
        seed_all(seed + 1)
        chains = run(gen, probs, cfg, rounds, n, seed, on_attempt, feedback_mode=mode, first=firsts, engine=engine)
        out[mode] = dict(summary=summary(chains), chains=chains)
    return out


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def _best_until(chain, r):
    sel = [a for a in chain if a.selected and a.allowed is not None and a.round <= r]
    return max(sel, key=lambda a: (a.allowed, -a.round, -a.cand)) if sel else None


def summary(chains):
    """solved@r = fracción de cadenas resueltas con hasta r correcciones (solved@0 = pass@1 sin feedback).
    fix_rate = de las que fallaron al principio, cuántas acaban con una respuesta final resuelta (v2: la elegida con
    tests permitidos; oracle_any = cota superior si se pudiera elegir mirando los ocultos)."""
    chains = [c for c in chains if c]
    if not chains:
        return {}
    v2 = any(a.allowed is not None for c in chains for a in c)
    s = {"chains": len(chains), "attempts": sum(len(c) for c in chains), "engine": "v2" if v2 else "v1"}
    for r in range(max(a.round for c in chains for a in c) + 1):
        if v2:
            s[f"solved@{r}"] = _mean(bool(b and b.solved) for b in (_best_until(c, r) for c in chains))
        else:
            s[f"solved@{r}"] = _mean(any(a.solved for a in c[:r + 1]) for c in chains)
    failed = [c for c in chains if not c[0].solved]
    final = {id(c): final_attempt(c) for c in chains}
    s["fix_rate"] = _mean(final[id(c)].solved for c in failed)
    repairs = [a for c in failed for a in c[1:]]
    s.update(reward_first=_mean(c[0].reward for c in chains), reward_last=_mean(final[id(c)].reward for c in chains),
             compile_first=_mean(c[0].compiled for c in chains), compile_last=_mean(final[id(c)].compiled for c in chains),
             errors_last=dict(Counter(final[id(c)].error for c in chains if not final[id(c)].solved)),
             reward_gain=_mean(final[id(c)].reward - c[0].reward for c in failed),
             oracle_any=_mean(any(a.solved for a in c) for c in failed),
             success_per_candidate=_mean(a.solved for a in repairs if not a.repeat),
             candidates=_mean(len(c) - 1 for c in failed))
    keys = [[code_key(a.code) for a in c] for c in failed]
    s["repetition"] = _mean(a.repeat or keys[j][i] in keys[j][:i]
                            for j, c in enumerate(failed) for i, a in enumerate(c) if i > 0)
    s["same_error"] = _mean(a.sig is not None and a.sig == c[0].sig for c in failed for a in c[1:])
    by = {}
    for c in failed:
        by.setdefault((c[0].error or "?").split(":")[0], []).append(final[id(c)].solved)
    s["fix_by_error"] = {k: dict(n=len(v), fix_rate=_mean(v)) for k, v in sorted(by.items())}
    if v2:
        s["restarts"] = _mean(sum(a.strategy == "restart" and a.cand == 0 for a in c) for c in failed)
    return s
