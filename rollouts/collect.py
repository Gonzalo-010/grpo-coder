"""Generación + reward, compartido entre evaluación (Fase 1) y recolección de datos de entrenamiento (Fase 2).

Caché de score (rollouts.score_cache, activa por defecto): el mismo código para el mismo problema (id + versión),
semilla y configuración del evaluador sólo entra una vez al sandbox, también si llega a la vez desde dos hilos.
La clave usa el texto EXACTO del código (la normalización más estricta: nunca mezcla dos códigos distintos); la
normalización por AST se usa para detectar soluciones repetidas (Fix v2, agente, diversidad), no para el score.
El reward es determinista para un mismo (código, tests), así que reutilizarlo no cambia el entrenamiento.
"""
import ast, copy, json, threading, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from common import opt, workers
from rollouts.generate import build_prompt
from rewards.reward import score


def code_key(code, positions=False):
    """Código normalizado por AST (ignora comentarios y formato). positions=True conserva línea/columna de cada nodo:
    es la clave de la caché, para que la línea de error de un Result siga siendo válida para quien lo reutiliza."""
    try:
        return ast.dump(ast.parse(code), include_attributes=positions)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "raw:" + code


def evaluator_sig(cfg):
    """Todo lo de la configuración que puede cambiar el Result de score()."""
    return json.dumps([cfg["reward_weights"], cfg["reward"], cfg["sandbox"], cfg["sandbox_limits"]], sort_keys=True,
                      default=str)


class ScoreCache:
    """LRU segura con hilos y deduplicación en vuelo: si dos hilos piden la misma clave, uno ejecuta y el otro espera."""

    def __init__(self, size=4096):
        self.size, self.d, self.lock, self.pending = size, OrderedDict(), threading.Lock(), {}
        self.hits = self.misses = 0

    def get(self, key, fn):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(self.d[key])
            ev = self.pending.get(key)
            owner = ev is None
            if owner:
                ev = self.pending[key] = threading.Event()
        if not owner:
            ev.wait()
            return self.get(key, fn)  # ya está en la caché (o el dueño falló y ahora ejecutamos nosotros)
        try:
            r = fn()
            with self.lock:
                self.misses += 1
                if not str(getattr(r, "error", "") or "").startswith("crash"):  # un crash puede ser transitorio
                    self.d[key] = r
                    while len(self.d) > self.size:
                        self.d.popitem(last=False)
            return copy.deepcopy(r)
        finally:
            with self.lock:
                self.pending.pop(key, None)
            ev.set()

    def invalidate(self, problem_id=None):
        """Borra las entradas de un problema (o todas). Cambiar la versión del problema ya invalida por clave."""
        with self.lock:
            for k in [k for k in self.d if problem_id is None or k[1] == problem_id]:
                del self.d[k]

    def stats(self):
        with self.lock:
            return dict(hits=self.hits, misses=self.misses, size=len(self.d))


CACHE = ScoreCache()


def cached_score(p, code, cfg, seed=0):
    """score() con caché. Lee `score` del módulo en cada llamada (los tests lo sustituyen) y lo incluye en la clave."""
    fn = score
    version = getattr(p, "version", None)
    if not opt(cfg, "rollouts.score_cache", True) or version is None:  # sin versión (tarea del panel): sin caché
        return fn(p, code, cfg, seed)
    CACHE.size = int(opt(cfg, "rollouts.cache_size", 4096))
    key = (fn, p.id, version, seed, code, evaluator_sig(cfg))
    return CACHE.get(key, lambda: fn(p, code, cfg, seed))


def _score_sample(p, s, cfg, seed):
    r = cached_score(p, s.code, cfg, seed)
    r.truncated = not getattr(s, "closed", True)
    return r


TIMING = {"gen_wall": 0.0, "score_wait": 0.0}  # acumulado del proceso: el bucle de entrenamiento lee diferencias


def collect(gen, probs, n, cfg, seed=0, prompts=None, temperature=None):
    """gen.stream(...) mientras un pool de hilos puntúa en el sandbox lotes anteriores (solapado).
    Devuelve una lista paralela a `probs`: para cada problema, [(Sample, Result), ...] de sus n rollouts.
    `prompts` (paralela a probs) sustituye al prompt del problema, p. ej. con un intento previo y su feedback.
    `temperature` (opcional) sustituye a la de generation sólo en esta llamada (<= 0: greedy).
    TIMING: gen_wall = tiempo hasta que termina la generación (con el sandbox solapado); score_wait = lo que se espera
    después al sandbox. Si score_wait crece, el cuello de botella es la CPU/el sandbox, no la GPU."""
    prompts = prompts or [build_prompt(gen.tok, cfg.prompt.system, p.prompt(), cfg.prompt.prefill_fence) for p in probs]
    pool, pend = ThreadPoolExecutor(workers(cfg)), [[] for _ in probs]
    stream = gen.stream(prompts, n) if temperature is None else gen.stream(prompts, n, temperature=temperature)
    t0 = time.perf_counter()
    for i, s in stream:
        pend[i].append((s, pool.submit(_score_sample, probs[i], s, cfg, seed)))
    t1 = time.perf_counter()
    out = [[(s, f.result()) for s, f in items] for items in pend]
    pool.shutdown()
    TIMING["gen_wall"] += t1 - t0
    TIMING["score_wait"] += time.perf_counter() - t1
    return out
