"""Experimentos comparables: varios brazos con la misma config base, dataset, semilla, presupuesto y evaluación.

python -m experiments.run PLAN [--arms A B] [--steps N] [--dry-run] [--no-report]

PLAN = experiments/plans/<PLAN>.yaml (o una ruta). Cada brazo:
  runs/exp/<plan>/<brazo>/config.yaml   config completa (base + common + brazo), sin `extends`
  runs/exp/<plan>/<brazo>/              run de phase2 (checkpoints, metrics.jsonl, gpu.jsonl, train.log)
  runs/exp/<plan>/<brazo>/eval_<split>.json   evaluación final (mismas semillas para todos los brazos)
  runs/exp/<plan>/base_<modelo>/eval_<split>.json   modelo de partida (referencia común)
Cada brazo corre en un proceso aparte (la VRAM se libera entre brazos). Relanzar el mismo comando continúa: los brazos
terminados se saltan y uno interrumpido se reanuda desde su último checkpoint. Al final: report.md / report.json.
"""
from __future__ import annotations

import argparse, copy, json, os, re, subprocess, sys, time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import INFERENCE_VERSION, _merge, read_config  # noqa: E402


def load_plan(name):
    path = name if os.path.exists(name) else os.path.join(ROOT, "experiments", "plans", f"{name}.yaml")
    with open(path) as f:
        plan = yaml.safe_load(f)
    plan.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return plan


# claves del brazo que no son config (sin chocar con secciones de config.yaml como `checkpoint`): de qué checkpoint
# parte el entrenamiento / qué checkpoint ya entrenado se evalúa sin entrenar
ARM_KEYS = ("init_from", "eval_checkpoint")


def arm_config(plan, arm, steps=None):
    base = read_config(os.path.join(ROOT, plan["base"]) if plan.get("base") else None)
    cfg = _merge(base, copy.deepcopy(plan.get("common") or {}))
    cfg = _merge(cfg, copy.deepcopy({k: v for k, v in (plan["arms"][arm] or {}).items() if k not in ARM_KEYS}))
    if steps:
        cfg["grpo"]["max_steps"] = int(steps)
    cfg.pop("extends", None)
    return cfg


def slug(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def _status(plan_dir, **kw):
    path = os.path.join(plan_dir, "status.json")
    st = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                st = json.load(f)
        except ValueError:
            st = {}
    st.update(kw, t=time.time())
    with open(path + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(path + ".tmp", path)


def _run(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return p.returncode


def resolve_ckpt(path):
    """Ruta de un checkpoint: la propia si tiene meta.json; si es un run (o su carpeta checkpoints/), el más reciente
    completo. Relativa a la raíz del proyecto."""
    from training import checkpoint as ck
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if os.path.exists(os.path.join(p, "meta.json")):
        return p
    for root in (os.path.join(p, "checkpoints"), p):
        cands = ck.candidates(root)
        if cands:
            return cands[0]
    raise FileNotFoundError(f"no hay ningún checkpoint en {path}")


def trained(d, max_steps):
    """¿Terminó el entrenamiento del brazo? (checkpoint del último step)."""
    ck = os.path.join(d, "checkpoints", f"ckpt_{max_steps - 1:07d}")
    return os.path.exists(os.path.join(ck, "meta.json"))


def final_ckpt(d, which, max_steps):
    if which == "best" and os.path.isdir(os.path.join(d, "checkpoints", "best")):
        return os.path.join(d, "checkpoints", "best")
    return os.path.join(d, "checkpoints", f"ckpt_{max_steps - 1:07d}")


def evaluate(cfg_path, ckpt, split, n, out, log):
    """Evalúa salvo que ya exista la evaluación de ESE checkpoint (con --steps mayor el último checkpoint cambia y
    la evaluación anterior no sirve) hecha con la carga de inferencia actual. Una evaluación de checkpoint con otra
    versión (1 = LoRA fusionado en bf16, ver model/infer.py) se conserva como eval_<split>.inference<v>.json (el informe
    la compara con la nueva) y se repite."""
    path = os.path.join(out, f"eval_{split}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                prev = json.load(f)
        except ValueError:
            prev = {}
        same = (prev.get("checkpoint") is None and ckpt is None) or (
            prev.get("checkpoint") and ckpt and os.path.normpath(prev["checkpoint"]) == os.path.normpath(ckpt))
        if same and prev.get("n") == n:
            if ckpt is None or prev.get("inference") == INFERENCE_VERSION:  # el modelo base no lleva LoRA
                return 0
            old = os.path.join(out, f"eval_{split}.inference{prev.get('inference') or 1}.json")
            if not os.path.exists(old):
                os.replace(path, old)
    cmd = [sys.executable, "-u", "train.py", "eval", "--config", cfg_path, "--split", split, "--n", str(n), "--out", out,
           "--checkpoint", ckpt or "base"]
    return _run(cmd, log)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Brazos comparables de un experimento (ver docstring)")
    ap.add_argument("plan")
    ap.add_argument("--arms", nargs="*", help="sólo estos brazos (por defecto todos, en orden)")
    ap.add_argument("--steps", type=int, help="sobrescribe grpo.max_steps de todos los brazos")
    ap.add_argument("--dry-run", action="store_true", help="sólo escribe las configs y muestra los comandos")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    plan = load_plan(a.plan)
    plan_dir = os.path.join(ROOT, "runs", "exp", plan["name"])
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, "plan.yaml"), "w") as f:
        yaml.safe_dump(dict(plan, steps_override=a.steps), f, sort_keys=False, allow_unicode=True)
    fe = plan.get("final_eval") or {}
    splits, n, which = fe.get("splits", ["val", "test"]), int(fe.get("n", 16)), fe.get("checkpoint", "last")
    arms = a.arms or list(plan["arms"])
    failed = []
    for arm in arms:
        d = os.path.join(plan_dir, arm)
        os.makedirs(d, exist_ok=True)
        cfg = arm_config(plan, arm, a.steps)
        cfg_path = os.path.join(d, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        steps = int(cfg["grpo"]["max_steps"])
        log = os.path.join(d, "train.log")
        extra = plan["arms"][arm] or {}
        train_cmd = [sys.executable, "-u", "train.py", "phase2", "--config", cfg_path, "--run", d]
        if extra.get("init_from"):  # rama: parte de los pesos de otro checkpoint (p. ej. el destilado)
            try:
                train_cmd += ["--init-from", resolve_ckpt(extra["init_from"])]
            except FileNotFoundError as e:
                print(f"[{arm}] {e}")
                failed.append(f"{arm}:init_from")
                continue
        if a.dry_run:
            print(" ".join(train_cmd))
            continue
        # referencia: el modelo de partida de este brazo, una vez por modelo
        bdir = os.path.join(plan_dir, f"base_{slug(cfg['model'])}")
        os.makedirs(bdir, exist_ok=True)
        bcfg = os.path.join(bdir, "config.yaml")
        if not os.path.exists(bcfg):
            with open(bcfg, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        for sp in splits:
            _status(plan_dir, current=f"base_{slug(cfg['model'])}:{sp}")
            if evaluate(bcfg, None, sp, n, bdir, os.path.join(bdir, "eval.log")):
                failed.append(f"base:{sp}")
        if extra.get("eval_checkpoint"):  # brazo sólo de evaluación: un checkpoint ya entrenado (p. ej. destilación)
            try:
                ck_eval = resolve_ckpt(extra["eval_checkpoint"])
            except FileNotFoundError as e:
                print(f"[{arm}] {e}")
                failed.append(f"{arm}:eval_checkpoint")
                continue
            for sp in splits:
                _status(plan_dir, current=f"{arm}:eval_{sp}")
                if evaluate(cfg_path, ck_eval, sp, n, d, log):
                    failed.append(f"{arm}:eval_{sp}")
            _status(plan_dir, **{f"{arm}_done": True})
            print(f"[{arm}] listo (sólo evaluación de {ck_eval})")
            continue
        if not trained(d, steps):
            _status(plan_dir, current=f"{arm}:train", **{f"{arm}_started": time.time()})
            t0 = time.time()
            rc = _run(train_cmd, log)
            _status(plan_dir, **{f"{arm}_train_rc": rc, f"{arm}_train_s": time.time() - t0})
            if rc != 0 or not trained(d, steps):
                print(f"[{arm}] el entrenamiento falló (rc={rc}); ver {log}")
                failed.append(f"{arm}:train")
                continue
        ck = final_ckpt(d, which, steps)
        for sp in splits:
            _status(plan_dir, current=f"{arm}:eval_{sp}")
            if evaluate(cfg_path, ck, sp, n, d, log):
                failed.append(f"{arm}:eval_{sp}")
        _status(plan_dir, **{f"{arm}_done": True})
        print(f"[{arm}] listo: {d}")
    _status(plan_dir, current=None, failed=failed)
    if not a.dry_run and not a.no_report:
        from experiments.report import build
        rep = build(plan_dir)
        print(rep["markdown"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
