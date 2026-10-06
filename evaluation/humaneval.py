"""Benchmark externo: HumanEval (164 problemas; OpenAI, licencia MIT) ejecutado en el sandbox del proyecto.

python train.py humaneval [--checkpoint DIR] [--n 8] [--chunk 128] [--out DIR] [--data FICHERO] [--limit N]

Para qué: val son 50 problemas de NUESTRO catálogo, de las mismas categorías que train. HumanEval dice si lo aprendido
sirve fuera de él. Se evalúa como en el artículo original (Chen et al. 2021, arXiv 2107.03374): programa = prompt del
problema + respuesta del modelo + test del problema + check(función); pasa si termina sin error. Se ejecuta con el
mismo aislamiento que el reward (bwrap: sin red, sin acceso al proyecto, /usr de sólo lectura) más límites de memoria,
CPU, ficheros y procesos, y un timeout. El código del modelo corre con __name__ distinto de "__main__" (un bloque
`if __name__ == "__main__":` con input() no lo tumba).

Datos: runs/data/HumanEval.jsonl.gz. Si no está, se descarga una vez de github.com/openai/human-eval (desde tu
máquina) y se guarda su sha256 en runs/data/HumanEval.jsonl.gz.sha256; después se comprueba en cada uso.

Prompt: el system prompt del proyecto y, como usuario, una instrucción corta + el prompt de HumanEval (firma +
docstring). La respuesta arranca en ```python (prefill), igual que al entrenar. El modelo escribe la función completa;
el programa pone antes el prompt original (imports y funciones auxiliares que algunos problemas definen ahí) y la
definición del modelo la sustituye.

Salida: JSON con las mismas filas que train.py eval (id, n, c, pass_at, greedy): se compara pareado por problema con
python -m evaluation.compare A.json B.json.
"""
from __future__ import annotations

import gzip, hashlib, json, os, subprocess, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
USER = ("Complete the following Python function. Write the whole function, with any imports it needs, in one code "
        "block.\n\n```python\n{prompt}```")
_LIMITS = """import resource as _r
for _n, _v in (("RLIMIT_AS", {mem}), ("RLIMIT_CPU", {cpu}), ("RLIMIT_FSIZE", {fsize}), ("RLIMIT_NPROC", 0),
               ("RLIMIT_CORE", 0)):
    try:
        _r.setrlimit(getattr(_r, _n), (_v, _v))
    except (ValueError, OSError):
        pass
del _r, _n, _v
"""


def default_path():
    from common import ROOT
    return os.path.join(ROOT, "runs", "data", "HumanEval.jsonl.gz")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path=None, download=True):
    """Lista de problemas {task_id, prompt, test, entry_point, ...} y el sha256 del fichero."""
    path = path or default_path()
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        import urllib.request
        with urllib.request.urlopen(URL, timeout=60) as r:
            data = r.read()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
        with open(path + ".sha256", "w") as f:
            f.write(_sha(data) + "\n")
    with open(path, "rb") as f:
        data = f.read()
    sha = _sha(data)
    if os.path.exists(path + ".sha256"):
        want = open(path + ".sha256").read().strip()
        if want != sha:
            raise ValueError(f"{path}: sha256 {sha} != {want} (fichero cambiado o corrupto)")
    text = gzip.decompress(data).decode() if path.endswith(".gz") else data.decode()
    recs = [json.loads(x) for x in text.splitlines() if x.strip()]
    for r in recs:
        missing = {"task_id", "prompt", "test", "entry_point"} - set(r)
        if missing:
            raise ValueError(f"{r.get('task_id')}: faltan campos {sorted(missing)}")
    return recs, sha


def program(rec, code):
    """Programa completo que se ejecuta: prompt original + código del modelo + test + check(entry_point)."""
    return f"{rec['prompt']}\n{code}\n\n{rec['test']}\n\ncheck({rec['entry_point']})\n"


def snippet(rec, code, lim):
    body = program(rec, code)
    return (_LIMITS.format(mem=int(lim["mem_mb"]) << 20, cpu=int(lim["cpu_s"]), fsize=int(lim["fsize_mb"]) << 20)
            + f"_ns = {{'__name__': 'candidate'}}\nexec(compile({body!r}, '<humaneval>', 'exec'), _ns)\n")


def _error(stderr):
    lines = [x for x in (stderr or "").strip().splitlines() if x.strip()]
    if not lines:
        return "error"
    last = lines[-1].split(":")[0].strip()
    return last if last.isidentifier() else "error"


def run_one(rec, code, cfg, backend, timeout):
    """(pasa, tipo de error) de un código para un problema, en el sandbox."""
    from sandbox import executor
    lim = dict(cfg.sandbox_limits)
    try:
        p = executor.raw(backend, snippet(rec, code, lim), timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if p.returncode == 0:
        return True, "ok"
    return False, _error(p.stderr)


def prompts_for(tok, cfg, recs):
    from rollouts.generate import build_prompt
    return [build_prompt(tok, cfg.prompt.system, USER.format(prompt=r["prompt"]), cfg.prompt.prefill_fence)
            for r in recs]


def score_all(recs, samples, cfg, backend, timeout):
    """samples: lista paralela a recs con listas de Sample. Devuelve listas paralelas de (pasa, error)."""
    from common import workers
    flat = [(i, s) for i, ss in enumerate(samples) for s in ss]
    with ThreadPoolExecutor(workers(cfg)) as pool:
        res = list(pool.map(lambda x: run_one(recs[x[0]], x[1].code, cfg, backend, timeout), flat))
    out = [[] for _ in recs]
    for (i, _), r in zip(flat, res):
        out[i].append(r)
    return out


def report(recs, samples, results, greedy, n, ks, seed, B):
    from evaluation.evaluate import _mean, bootstrap_ci
    from evaluation.metrics import pass_at_k
    rows = []
    for j, (rec, ss, rs) in enumerate(zip(recs, samples, results)):
        c = sum(ok for ok, _ in rs)
        rows.append(dict(id=rec["task_id"], n=len(rs), c=c,
                         pass_at={str(k): round(pass_at_k(len(rs), c, k), 4) for k in ks},
                         greedy=None if greedy is None else bool(greedy[j][0]),
                         errors=dict(Counter(e for _, e in rs)), ntok=_mean(s.ntok for s in ss),
                         trunc=_mean(float(not s.closed) for s in ss)))
    rep = dict(split="humaneval", problems=len(rows), n=n, ks=ks, seed=seed, rows=rows,
               greedy_enabled=greedy is not None)
    for k in ks:
        vals = [r["pass_at"][str(k)] for r in rows]
        rep[f"pass@{k}"] = _mean(vals)
        rep[f"ci_pass@{k}"] = bootstrap_ci(vals, B, seed)
    if greedy is not None:
        g = [float(r["greedy"]) for r in rows]
        rep.update(greedy=_mean(g), ci_greedy=bootstrap_ci(g, B, seed))
    allr = [e for rs in results for _, e in rs]
    rep["error_rates"] = {e: round(v / len(allr), 4) for e, v in Counter(allr).items()} if allr else {}
    rep["trunc_rate"] = _mean(r["trunc"] for r in rows)
    rep["ntok"] = _mean(r["ntok"] for r in rows)
    return rep


def run(cfg, a):
    from common import INFERENCE_VERSION, ROOT, opt
    from evaluation.evaluate import rng_guard
    from model.infer import Engine
    from sandbox import executor
    recs, sha = load(a.data)
    if a.limit:
        recs = recs[:a.limit]
    backend = executor.backend(cfg.sandbox)
    n = int(a.n)
    ks = [k for k in (1, 5, 10, 16) if k <= n] or [1]
    seed = int(opt(cfg, "eval.seed", 999983) if a.seed is None else a.seed)
    B = int(opt(cfg, "eval.bootstrap", 1000))
    timeout = float(a.timeout)
    eng = Engine(cfg, a.checkpoint)
    gen = eng.generator()
    gen.chunk = int(getattr(a, "chunk", None) or 128)  # la generación está limitada por la sobrecarga por paso: lotes
    # grandes salen casi gratis; mismo valor para todos los modelos -> muestras comparables (un OOM lo reduce solo)
    prompts = prompts_for(eng.tok, cfg, recs)
    t0, st0 = time.time(), dict(gen.stats)
    with rng_guard(seed):
        samples = [[] for _ in recs]
        for i, s in gen.stream(prompts, n):
            samples[i].append(s)
        greedy_s = [[] for _ in recs]
        for i, s in gen.stream(prompts, 1, temperature=0.0):
            greedy_s[i].append(s)
    t_gen = time.time() - t0
    results = score_all(recs, samples, cfg, backend, timeout)
    greedy = [rs[0] for rs in score_all(recs, greedy_s, cfg, backend, timeout)]
    rep = report(recs, samples, results, greedy, n, ks, seed, B)
    rep.update(checkpoint=a.checkpoint, model=cfg.model, inference=INFERENCE_VERSION, data_sha256=sha,
               sandbox=backend, timeout_s=timeout, gen_s=round(t_gen, 1), wall_s=round(time.time() - t0, 1),
               gen_tokens=gen.stats.get("tokens", 0) - st0.get("tokens", 0), arch=getattr(eng, "arch", None))
    out = a.out or os.path.join(ROOT, "runs", "humaneval",
                                "base" if not a.checkpoint else os.path.basename(os.path.normpath(a.checkpoint)))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "humaneval.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"humaneval: {len(recs)} problemas, n={n}: " + "  ".join(f"pass@{k}={rep[f'pass@{k}']:.3f}" for k in ks)
          + f"  greedy={rep.get('greedy')}  IC95 pass@1={rep['ci_pass@1']}  errores={rep['error_rates']}")
    print(f"guardado en {path}")
    eng.close()
    return rep
