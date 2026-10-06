"""Benchmark de generación: ¿cuánto se gana con CUDA graphs (caché estática + torch.compile de transformers)?

python train.py genbench [--checkpoint DIR] [--problems 6] [--reps 3]

Contexto: en los experimentos la generación es ~68 % del step y va a ~47 ms por paso de decodificación con 48
secuencias, cuando un modelo de 0,5B necesita unos pocos ms de GPU por paso: el coste es lanzar ~1 600 kernels por
paso desde Python (más la rama LoRA sin fusionar). Esto mide, con el mismo lote que un step de entrenamiento
(`problems` problemas de train x num_rollouts) y la política con LoRA sin fusionar:
  eager_lora   generate() tal como se usa ahora
  eager_base   lo mismo con el adapter desactivado (coste de la rama LoRA)
  static_lora  caché estática: transformers compila el paso de decodificación (torch.compile, CUDA graphs)
Cada modo decodifica exactamente max_new_tokens pasos (sin parada anticipada) para que ms/paso sea comparable; la
primera llamada de cada modo es calentamiento (en static incluye la compilación, que se informa aparte). Además:
greedy de static frente a eager en los mismos prompts (fracción de salidas idénticas) y la primera llamada con OTRO
lote (otras longitudes de prompt: ¿recompila?). Sólo mide: no cambia nada del entrenamiento. Resultado en
runs/genbench/<fecha>.json.
"""
from __future__ import annotations

import json, os, random, time, traceback


def _batch_texts(tok, cfg, probs, n):
    from rollouts.generate import build_prompt
    return [build_prompt(tok, cfg.prompt.system, p.prompt(), cfg.prompt.prefill_fence) for p in probs for _ in range(n)]


def run(cfg, a):
    import torch

    from common import ROOT, env_info, opt, seed_all
    from model.policy import load_policy
    from problems.problems import select
    from rollouts.generate import Generator

    if not torch.cuda.is_available():
        raise SystemExit("genbench necesita GPU")
    seed_all(cfg.seed)
    pool = select(sets=tuple(opt(cfg, "dataset.sets", ["core", "catalog"])), splits=("train",))
    rng = random.Random(cfg.seed)
    pick = rng.sample(pool, min(len(pool), 2 * a.problems))
    first, other = pick[:a.problems], pick[a.problems:]
    n = int(cfg.num_rollouts)
    model, tok, _ = load_policy(cfg, "cuda", resume_from=a.checkpoint)
    model.eval()
    gen = Generator(model, tok, cfg)
    texts, texts2 = _batch_texts(tok, cfg, first, n), _batch_texts(tok, cfg, other, n)
    gen.chunk = max(len(texts), len(texts2))
    out = dict(env=env_info(), checkpoint=a.checkpoint, seqs=len(texts), max_new=gen.max_new, reps=a.reps, modes={})

    def call(txts, greedy=False):
        torch.cuda.synchronize()
        s0 = gen.stats["steps"]
        res, dt = gen._gen(txts, full=not greedy, temperature=0.0 if greedy else None)
        return res, dt, gen.stats["steps"] - s0

    def measure(name, ctx_factory, static=False):
        gc_ = model.generation_config
        prev = getattr(gc_, "cache_implementation", None)
        row = {}
        try:
            if static:
                gc_.cache_implementation = "static"
            with ctx_factory():
                torch.cuda.reset_peak_memory_stats()
                _, dt0, _ = call(texts)  # calentamiento (en static: compilación + captura)
                row["first_call_s"] = round(dt0, 2)
                times = []
                for _ in range(a.reps):
                    res, dt, steps = call(texts)
                    times.append((dt, steps, sum(s.ntok for s in res)))
                row["ms_per_step"] = round(1000 * sum(t for t, _, _ in times) / sum(s for _, s, _ in times), 2)
                row["tok_s"] = round(sum(k for _, _, k in times) / sum(t for t, _, _ in times), 1)
                row["call_s"] = [round(t, 2) for t, _, _ in times]
                row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                if static:
                    _, d1, _ = call(texts2)  # otras longitudes de prompt: ¿recompila?
                    _, d2, _ = call(texts2)
                    row["other_batch_first_s"], row["other_batch_second_s"] = round(d1, 2), round(d2, 2)
                    row["compile_s_est"] = round(dt0 - sum(t for t, _, _ in times) / len(times), 1)
        except Exception as e:  # p. ej. torch.compile sin soporte para esta versión de Python
            row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            row["trace"] = traceback.format_exc(limit=4)[-1500:]
        finally:
            gc_.cache_implementation = prev
        out["modes"][name] = row
        print(f"      [genbench] {name}: {json.dumps({k: v for k, v in row.items() if k != 'trace'})}", flush=True)
        return row

    from contextlib import nullcontext
    measure("eager_lora", nullcontext)
    measure("eager_base", model.disable_adapter if hasattr(model, "disable_adapter") else nullcontext)
    st = measure("static_lora", nullcontext, static=True)
    if "error" not in st:  # greedy: ¿el paso compilado da las mismas salidas que el eager?
        try:
            eager, _, _ = call(texts, greedy=True)
            model.generation_config.cache_implementation = "static"
            comp, _, _ = call(texts, greedy=True)
            same = sum(x.text == y.text for x, y in zip(eager, comp))
            out["greedy_identical"] = round(same / len(eager), 3)
        except Exception as e:
            out["greedy_identical_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            model.generation_config.cache_implementation = None
    e, s = out["modes"].get("eager_lora", {}), out["modes"].get("static_lora", {})
    if e.get("ms_per_step") and s.get("ms_per_step"):
        out["speedup_static_vs_eager"] = round(e["ms_per_step"] / s["ms_per_step"], 2)
    b = out["modes"].get("eager_base", {})
    if e.get("ms_per_step") and b.get("ms_per_step"):
        out["lora_overhead"] = round(e["ms_per_step"] / b["ms_per_step"], 2)
    d = os.path.join(ROOT, "runs", "genbench")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, time.strftime("%Y%m%d-%H%M%S") + ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"      [genbench] speedup static/eager={out.get('speedup_static_vs_eager')} "
          f"coste LoRA={out.get('lora_overhead')} greedy idéntico={out.get('greedy_identical')} -> {path}")
    return out
