"""Destilación por rechazo (expert iteration / ReST-EM, arXiv 2312.06585; STaR, arXiv 2203.14465).

python train.py distill [--teacher MODELO] [--k 8] [--per-problem 4] [--epochs 2] [--batch 16] [--lr 1e-4] [--run DIR]

1. Un profesor (por defecto el propio modelo base; p. ej. Qwen/Qwen2.5-Coder-1.5B-Instruct) genera k soluciones por
   problema de TRAIN; el sandbox se queda con las que resuelven todos los tests (como el reward normal) y se guardan
   hasta `per_problem` distintas por problema (teacher_solutions.jsonl). val/test nunca entran.
2. El alumno (config.model con LoRA) hace SFT sobre esas soluciones: es el mismo trainer de GRPO con ventaja +1, KL 0
   y log-probs a T=1, lo que con una actualización por lote es exactamente la NLL media por token.
3. Deja un checkpoint normal (formato 2) que sirve para `train.py eval --checkpoint` y para `phase2 --init-from`.

Por qué: con RL el pass@1 sube pero pass@16 no se mueve (afina lo que el modelo ya sabía); DeepSeek-R1 (arXiv
2501.12948) concluye que destilar un modelo más capaz en uno pequeño da mejores resultados que el RL a gran escala
sobre el pequeño. Con profesor = el propio modelo es la línea base "RFT" (sólo refuerza lo que ya acierta).
"""
from __future__ import annotations

import copy, json, os, random, re, time


def _norm_code(code):
    """Clave para no repetir la misma solución con otro espaciado o comentarios."""
    lines = [re.sub(r"\s+#.*$", "", ln).rstrip() for ln in code.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#"))


def select_solutions(probs, results, per_problem):
    """De los rollouts del profesor, hasta per_problem soluciones distintas que resuelven por problema."""
    out = []
    for p, rs in zip(probs, results):
        seen = set()
        for s, r in rs:
            if not r.solved:
                continue
            key = _norm_code(s.code)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(problem=p.id, code=s.code, reward=r.reward))
            if len(seen) >= per_problem:
                break
    return out


def sft_items(tok, cfg, probs_by_id, sols):
    """RollItems (prompt del alumno, respuesta = la solución tal como la habría escrito tras el prefill) con ventaja +1."""
    from rollouts.generate import build_prompt
    from training.grpo import RollItem
    items = []
    prefill = bool(cfg.prompt.prefill_fence)
    for s in sols:
        p = probs_by_id[s["problem"]]
        prompt = build_prompt(tok, cfg.prompt.system, p.prompt(), prefill)
        resp = (s["code"].rstrip("\n") + "\n```") if prefill else f"```python\n{s['code'].rstrip()}\n```"
        items.append(RollItem(tok(prompt).input_ids, tok(resp, add_special_tokens=False).input_ids, 1.0))
    return items


def sft_config(cfg):
    """Config del trainer para SFT: KL 0, una pasada, log-probs a T=1 (la NLL estándar)."""
    c = copy.deepcopy(cfg)
    c["grpo"].update(kl_coef=0.0, updates_per_batch=1, minibatches=1, logp_temperature=1.0)
    return c


def run(cfg, a):
    import torch

    from common import env_info, new_run_dir, opt, seed_all
    from model import growth as gr
    from model.infer import Engine
    from model.policy import load_policy
    from rollouts.collect import collect
    from sandbox import executor
    from training import checkpoint as ck
    from training import optim as topt
    from training.grpo import GrpoTrainer
    from training.loop import train_pool

    seed_all(cfg.seed)
    executor.backend(cfg.sandbox)
    probs = train_pool(cfg, a)
    if not probs:
        raise SystemExit("ningún problema de train seleccionado")
    run_dir = a.run or new_run_dir(cfg, "distill")
    os.makedirs(run_dir, exist_ok=True)
    teacher = a.teacher or cfg.model
    sol_path = os.path.join(run_dir, "teacher_solutions.jsonl")
    t0 = time.time()
    if os.path.exists(sol_path):  # reanudar: las soluciones del profesor ya están
        sols = [json.loads(x) for x in open(sol_path) if x.strip()]
        print(f"      [distill] {len(sols)} soluciones del profesor reutilizadas de {sol_path}")
    else:
        tcfg = copy.deepcopy(cfg)
        tcfg["model"] = teacher
        eng = Engine(tcfg)
        gen = eng.generator()
        if torch.cuda.is_available():  # concurrencia de generación medida (el defecto de 32 secuencias es lento)
            from rollouts.generate import build_prompt
            longest = max(probs, key=lambda p: len(p.prompt()))
            gen.tune(build_prompt(eng.tok, tcfg.prompt.system, longest.prompt(), tcfg.prompt.prefill_fence))
        print(f"      [distill] profesor {teacher}: {a.k} muestras x {len(probs)} problemas de train "
              f"(chunk={gen.chunk})")
        results = collect(gen, probs, a.k, tcfg, cfg.seed)
        eng.close()
        del eng
        torch.cuda.empty_cache()
        sols = select_solutions(probs, results, a.per_problem)
        solved = len({s["problem"] for s in sols})
        print(f"      [distill] {len(sols)} soluciones distintas que resuelven, de {solved}/{len(probs)} problemas "
              f"({time.time() - t0:.0f}s)")
        with open(sol_path + ".tmp", "w") as f:
            f.writelines(json.dumps(s) + "\n" for s in sols)
        os.replace(sol_path + ".tmp", sol_path)
    if not sols:
        raise SystemExit("el profesor no resolvió ningún problema: nada que destilar")

    scfg = sft_config(cfg)
    scfg["grpo"]["lr"] = float(a.lr)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, ref_ctx = load_policy(scfg, dev)
    optimizer = topt.build(model, scfg)
    trainer = GrpoTrainer(model, ref_ctx, tok, optimizer, scfg)
    items = sft_items(tok, scfg, {p.id: p for p in probs}, sols)
    rng = random.Random(cfg.seed)
    steps_per_epoch = max(1, -(-len(items) // a.batch))
    log_path = os.path.join(run_dir, "distill.jsonl")
    step = 0
    with open(log_path, "a") as log:
        for ep in range(a.epochs):
            order = items[:]
            rng.shuffle(order)
            for b in range(steps_per_epoch):
                batch = order[b * a.batch:(b + 1) * a.batch]
                if not batch:
                    continue
                st = trainer.step(batch)
                # st.loss es la pérdida sustituta de GRPO (ratio x ventaja = -1 con U=1): no informa; la NLL real es
                # -logp medio por token de las soluciones, medido antes de la actualización (T=1)
                row = dict(step=step, epoch=ep, nll=-st.logp, tokens=st.tokens, grad_norm=st.grad_norm,
                           skipped=st.skipped, oom=st.oom, t=time.time())
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"      [distill] epoch {ep} step {step}: nll={-st.logp:.4f} tokens={st.tokens} "
                      f"grad_norm={st.grad_norm:.3f}")
                step += 1
    meta = dict(arch=gr.arch(model), growth=[], code=ck.code_hash(),
                config=json.loads(json.dumps(scfg, default=str)), env=env_info(),
                lineage=dict(run=os.path.abspath(run_dir), parent=None, distill=dict(
                    teacher=teacher, k=a.k, per_problem=a.per_problem, epochs=a.epochs, batch=a.batch, lr=a.lr,
                    solutions=len(sols), problems=len({s["problem"] for s in sols}), train_problems=len(probs))))
    path = ck.save(os.path.join(run_dir, "checkpoints"), max(0, step - 1), model, optimizer,
                   dict(distill_steps=step, solutions=len(sols)), bool(opt(cfg, "lora.enabled", True)),
                   ck.config_hash(cfg), extra=meta)
    print(f"      [distill] checkpoint -> {path}")
    print(f"      siguiente: python train.py eval --split val --checkpoint {path} --n 16")
    return path
