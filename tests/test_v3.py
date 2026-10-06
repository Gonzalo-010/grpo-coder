"""Tests de la v3 (los registra tests/run_tests.py). Tres niveles:
* sin torch: lógica pura (siempre se ejecutan).
* torch en CPU con modelos Qwen2 MÍNIMOS creados al vuelo (sin descargas ni GPU): logprobs por trozos, GRPO,
  crecimiento, checkpoints y un phase2 de extremo a extremo con crecimiento y reanudación.
* GPU con el modelo real (memoria de los logprobs, identidad del crecimiento en 0.5B).
SKIP = no verificado en esta máquina (falta torch/peft/GPU), nunca cuenta como éxito.
"""
import json, math, os, random, shutil, subprocess, sys, tempfile, time, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import load_config, opt  # noqa: E402

TESTS = []


class Skip(Exception):
    pass


def test(name):
    def deco(f):
        TESTS.append((name, f))
        return f
    return deco


def _torch():
    try:
        import torch
    except ImportError:
        raise Skip("torch no instalado")
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise Skip("transformers no instalado")
    return torch


def _peft():
    _torch()
    try:
        import peft  # noqa: F401
    except ImportError:
        raise Skip("peft no instalado")


# ====================================================================================================== sin torch
@test("v3 generación: resp_ids sólo con tokens MUESTREADOS (el relleno tras parar en ``` no entra; un EOS elegido sí)")
def t3_response_length():
    from rollouts.generate import response_length as rl
    PAD, END = 151643, 151645  # Qwen: <|endoftext|> (pad y también EOS) y <|im_end|>
    eos = {PAD, END}
    assert rl([10, 11, 12, PAD, PAD], PAD, eos, stopped=True) == 3      # stop string: el pad no se muestreó
    assert rl([10, 11, END, PAD], PAD, eos, stopped=False) == 3         # <|im_end|> elegido por el modelo
    assert rl([10, 11, PAD, PAD], PAD, eos, stopped=False) == 3         # <|endoftext|> elegido (pad == eos)
    assert rl([10, 11, 12], PAD, eos, stopped=False) == 3               # max_new_tokens agotado
    assert rl([7, 0, 0], 0, {5}, stopped=True) == 1 and rl([7, 5, 0], 0, {5}, stopped=True) == 2


@test("v3 vram: detector de paginación con los datos reales del run 20261002-234419 (sano vs steps 64-70)")
def t3_vram_pressure():
    from model.vram import pressure
    sano = [dict(used_mb=12059, total_mb=12227, util=55, power_w=119.4), dict(used_mb=12129, total_mb=12227, util=33,
                                                                              power_w=77.3)]
    paginando = [dict(used_mb=12042, total_mb=12227, util=100, power_w=55.6),
                 dict(used_mb=12042, total_mb=12227, util=98, power_w=58.6)]
    assert not pressure(sano, ref_power=100)["busy_lowpower"]
    assert pressure(paginando, ref_power=100)["busy_lowpower"] and pressure(paginando)["vram_full"]
    assert pressure([], ref_power=100) == dict(vram_full=False, busy_lowpower=False, vram_used_max_mb=None,
                                               util_mean=None, power_mean_w=None)


@test("v3 telemetría: un step 3x más lento que la mediana se marca; la paginación no contamina la referencia")
def t3_telemetry_judge():
    from training.loop import Telemetry
    t = Telemetry(types.SimpleNamespace())
    for _ in range(8):
        v = t.judge(20.0, 450.0, {"power_mean_w": 100.0}, {})
        assert not v["slow"] and not v["vram_paging"]
    v = t.judge(90.0, 450.0, {"power_mean_w": 100.0}, {})
    assert v["slow"] and v["secs_median"] == 20.0
    v = t.judge(359.0, 17.0, {"power_mean_w": 55.0}, {"vram_full": True, "busy_lowpower": False})
    assert v["vram_paging"], v  # tok/s < 35 % de la mediana con la VRAM llena
    assert statistics_median(t.secs) == 20.0  # los steps anómalos no entran en la mediana


def statistics_median(xs):
    import statistics
    return statistics.median(xs)


@test("v3 monitor: resumen de ventana y recuento de procesos zombie (hijos sin recoger)")
def t3_monitor():
    from rollouts import monitor as m
    rows = [dict(t=1, used_mb=100, util=50, power_w=80, cpu_pct=10, ram_avail_mb=5000, swap_used_mb=0),
            dict(t=2, used_mb=300, util=90, power_w=60, cpu_pct=30, ram_avail_mb=4000, swap_used_mb=12)]
    s = m.summarize(m.window(rows, 1.5, 3))
    assert s["samples"] == 1 and s["vram_used_max_mb"] == 300 and s["swap_used_max_mb"] == 12
    s = m.summarize(rows)
    assert s["util_mean"] == 70.0 and s["ram_avail_min_mb"] == 4000
    if not os.path.isdir("/proc"):
        raise Skip("sin /proc")
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 5
    while m.zombies() < 1 and time.time() < deadline:  # termina pero nadie la recoge: zombie
        time.sleep(0.05)
    assert m.zombies() >= 1
    p.wait()
    assert m.zombies() == 0


@test("v3 crecimiento: posiciones uniformes y crecimiento progresivo intercalado (cada evento cubre todo el modelo)")
def t3_growth_plan():
    from model.growth import plan_positions, schedule_positions
    assert plan_positions(24, 4) == [5, 11, 17, 23] and plan_positions(24, 0) == []
    assert schedule_positions(24, [4]) == [[5, 11, 17, 23]]
    ev = schedule_positions(24, [2, 2])
    assert ev == [[5, 17], [11, 23]] and sorted(sum(ev, [])) == plan_positions(24, 4)
    from training.loop import growth_plan
    cfg = load_config()
    cfg["growth"].update(enabled=True, schedule=[{"step": 0, "new_layers": 2}, {"step": 20, "new_layers": 2}])
    plan = growth_plan(cfg, 24)
    assert [(e["step"], e["after"]) for e in plan] == [(0, [5, 17]), (20, [11, 23])], plan
    cfg["growth"]["enabled"] = False
    assert growth_plan(cfg, 24) == []


@test("v3 optimizador: warmup lineal por grupo desde el step en que nace (sin estado de scheduler)")
def t3_lr_factor():
    from training.optim import lr_factor, set_lr
    assert lr_factor(0, 0, 0) == 1.0 and lr_factor(0, 0, 4) == 0.25 and lr_factor(3, 0, 4) == 1.0
    assert lr_factor(19, 20, 5) == 0.0 and lr_factor(20, 20, 5) == 0.2 and lr_factor(30, 20, 5) == 1.0
    opt_ = types.SimpleNamespace(param_groups=[dict(name="policy", base_lr=1e-5, start_step=0, warmup=0, lr=0),
                                               dict(name="growth0", base_lr=5e-5, start_step=10, warmup=5, lr=0),
                                               dict(lr=3e-5)])  # grupo de un optim.pt antiguo: se respeta su lr
    lrs = set_lr(opt_, 11)
    assert lrs["policy"] == 1e-5 and abs(lrs["growth0"] - 2e-5) < 1e-12 and opt_.param_groups[2]["lr"] == 3e-5


@test("v3 checkpoint: manifiesto sha256 detecta corrupción y truncado; best/ se reemplaza sin quedar a medias")
def t3_checkpoint_manifest():
    from training import checkpoint as ck
    root = tempfile.mkdtemp()
    d = os.path.join(root, "ckpt_0000003")
    os.makedirs(d)
    for name, data in (("meta.json", json.dumps({"step": 3})), ("optim.pt", "x" * 1000), ("adapter.bin", "y" * 500)):
        with open(os.path.join(d, name), "w") as f:
            f.write(data)
    ck.write_manifest(d)
    assert ck.verify(d) is True
    with open(os.path.join(d, "optim.pt"), "r+") as f:  # mismo tamaño, contenido cambiado
        f.write("z")
    try:
        ck.verify(d)
        raise AssertionError("corrupción no detectada")
    except ck.CorruptCheckpoint:
        pass
    assert ck.verify(d, deep=False)  # el modo rápido sólo mira tamaños
    with open(os.path.join(d, "adapter.bin"), "w") as f:  # truncado
        f.write("y")
    for deep in (True, False):
        try:
            ck.verify(d, deep=deep)
            raise AssertionError("truncado no detectado")
        except ck.CorruptCheckpoint:
            pass
    legacy = os.path.join(root, "ckpt_0000001")  # anterior a format 2: sin manifiesto, se acepta
    os.makedirs(legacy)
    open(os.path.join(legacy, "meta.json"), "w").write("{}")
    assert ck.verify(legacy) is False
    # best/: corte simulado justo después de apartar el viejo -> recover lo restaura
    best = os.path.join(root, "best")
    os.makedirs(best)
    open(os.path.join(best, "meta.json"), "w").write('{"step": 1}')
    os.replace(best, best + ".old")
    ck.recover(root)
    assert json.load(open(os.path.join(best, "meta.json")))["step"] == 1
    tmp = os.path.join(root, ".tmp_best")
    os.makedirs(tmp)
    open(os.path.join(tmp, "meta.json"), "w").write('{"step": 2}')
    ck._replace_dir(tmp, best)
    assert json.load(open(os.path.join(best, "meta.json")))["step"] == 2 and not os.path.exists(best + ".old")
    assert all(not os.path.basename(p).endswith(".old") for p in ck.candidates(root))
    shutil.rmtree(root)


@test("v3 checkpoint: el hash de la config cubre el crecimiento sólo si está activo; code_hash estable")
def t3_config_hash_growth():
    from common import read_config
    from training import checkpoint as ck
    base = read_config()
    h = ck.config_hash(base)
    off = json.loads(json.dumps(base))
    off["growth"]["lr"] = 1.0
    assert ck.config_hash(off) == h  # apagado: sus claves no cuentan
    on = json.loads(json.dumps(base))
    on["growth"]["enabled"] = True
    h_on = ck.config_hash(on)
    assert h_on != h
    on["growth"]["schedule"] = [{"step": 0, "new_layers": 2}]
    assert ck.config_hash(on) != h_on
    c1, c2 = ck.code_hash(), ck.code_hash()
    assert c1 == c2 and c1["files"] > 20


@test("v3 GRPO: micro-batches por presupuesto de tokens (más items si son cortos) y sin presupuesto = micro_bs")
def t3_micro_batching():
    from training.grpo import GrpoTrainer, RollItem
    t = object.__new__(GrpoTrainer)
    t.micro_bs, t.ref_len = 2, None
    items = [RollItem([0] * 5, [0] * n, 0.0) for n in (1, 2, 3, 4, 5)]
    assert [len(b) for b in t._micro(items)] == [2, 2, 1]
    t.ref_len = 10  # presupuesto 2*10 = 20 tokens con padding
    sizes = [[len(i.prompt_ids) + len(i.resp_ids) for i in b] for b in t._micro(items)]
    assert sizes == [[6, 7], [8, 9], [10]], sizes
    short = [RollItem([0] * 2, [0] * 2, 0.0) for _ in range(9)]  # 4 tokens: caben 5 por micro-batch
    assert [len(b) for b in t._micro(short)] == [5, 4]
    assert sum(len(b) for b in t._micro(items + short)) == 14


@test("v3 agente: una edición que empeora los tests se REVIERTE (vuelve el mejor estado), la paciencia corta y "
      "agent.rollback: false lo desactiva")
def t3_agent_rollback():
    import agent.loop as al
    from agent.loop import Session
    from sandbox.command import CmdResult
    repo = tempfile.mkdtemp()
    with open(os.path.join(repo, "calc.py"), "w") as f:
        f.write("def add(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a + b\n")
    outs = {"a - b": 1, "a + b": 1}

    def fake_run(cmd, work, sb, timeout, mem):  # pasan los tests de las funciones correctas (2 tests)
        src = open(os.path.join(work, "calc.py")).read()
        ok = ("return a + b\n\ndef mul" in src) + ("return a * b" in src)
        txt = f"{ok} passed, {2 - ok} failed" if ok < 2 else "2 passed"
        return CmdResult(0 if ok == 2 else 1, f"FAILED test_calc.py::t - AssertionError\n{txt}", 0.1)
    responses = iter([
        "FILE: calc.py\n```python\ndef add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a + b\n```\n",  # 0->1
        "FILE: calc.py\n```python\ndef add(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a / b\n```\n",  # 1->0
        "FILE: calc.py\n```python\ndef add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n```\n",  # 1->2
    ])
    prompts = []

    class Eng:
        def chat(self, msgs):
            return "\n".join(m["content"] for m in msgs)

        def n_tokens(self, text):
            return len(text) // 4

        def complete(self, prompt, max_new, temperature=None):
            prompts.append(prompt)
            return next(responses)
    real = al.command.run
    al.command.run = fake_run
    try:
        cfg = load_config()
        cfg["agent"]["patience"] = 0
        s = Session(repo, "arregla calc", cfg, root=tempfile.mkdtemp())
        s.run(Eng(), 4)
    finally:
        al.command.run = real
    assert [it["reverted"] for it in s.iters] == [False, True, False], [it["reverted"] for it in s.iters]
    assert "made the tests WORSE" in prompts[2] and s.status == "passed"
    src = open(os.path.join(s.work, "calc.py")).read()
    assert "return a * b" in src and "a / b" not in src
    # paciencia: dos rondas sin mejorar -> para antes de agotar las iteraciones
    responses = iter(["FILE: calc.py\n```python\ndef add(a, b):\n    return 0\n```\n",
                      "FILE: calc.py\n```python\ndef add(a, b):\n    return 1\n```\n"] * 5)
    al.command.run = lambda *a, **k: CmdResult(1, "FAILED t - x\n0 passed, 2 failed", 0.1)
    try:
        cfg["agent"]["patience"] = 2
        s = Session(repo, "arregla calc", cfg, root=tempfile.mkdtemp())
        s.run(Eng(), 6)
    finally:
        al.command.run = real
    assert len(s.iters) == 2 and s.iters[-1].get("stopped") == "patience", [it.get("stopped") for it in s.iters]
    # agent.rollback: false (comportamiento anterior, para el A/B del benchmark): la edición que empeora se queda
    responses = iter([
        "FILE: calc.py\n```python\ndef add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a + b\n```\n",
        "FILE: calc.py\n```python\ndef add(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a / b\n```\n"])
    al.command.run = fake_run
    try:
        cfg["agent"].update(patience=0, rollback=False)
        s = Session(repo, "arregla calc", cfg, root=tempfile.mkdtemp())
        s.run(Eng(), 2)
    finally:
        al.command.run = real
    assert [it["reverted"] for it in s.iters] == [False, False] and "a / b" in open(os.path.join(s.work, "calc.py")).read()


@test("v3 control de reward espurio: por defecto no cambia nada; con reward.control=random, Bernoulli(0.5) determinista "
      "por step y sin tocar los rewards originales")
def t3_control_rewards():
    from training.loop import control_rewards
    cfg = load_config()
    keyed = [("a", 0.3), ("a", 0.9), ("b", 0.1), ("b", 0.1)]
    assert control_rewards(keyed, cfg, 0) is keyed
    cfg["reward"]["control"] = "random"
    many = [r for s in range(200) for _, r in control_rewards(keyed, cfg, s)]
    assert set(many) == {0.0, 1.0} and 0.4 < sum(many) / len(many) < 0.6
    assert control_rewards(keyed, cfg, 7) == control_rewards(keyed, cfg, 7) != control_rewards(keyed, cfg, 8)
    assert keyed == [("a", 0.3), ("a", 0.9), ("b", 0.1), ("b", 0.1)]


@test("v3 destilación (puro): deduplicado de soluciones, selección de las que resuelven, config SFT; brazos con "
      "init_from/checkpoint fuera de la config y resolución del checkpoint de un run")
def t3_distill_helpers():
    import types
    from experiments.run import ARM_KEYS, arm_config, resolve_ckpt
    from training.distill import _norm_code, select_solutions, sft_config
    assert _norm_code("def f(x):\n    # suma\n    return x+1  # fin\n\n") == _norm_code("def f(x):\n    return x+1")
    P = types.SimpleNamespace
    probs = [P(id="a"), P(id="b"), P(id="c")]
    R = lambda solved: P(solved=solved, reward=1.0 if solved else 0.2)
    S = lambda code: P(code=code)
    res = [[(S("def f():\n    return 1"), R(True)), (S("def f():\n    return 1  # igual"), R(True)),
            (S("def f():\n    return 2"), R(True)), (S("def f():\n    return 3"), R(True))],
           [(S("x"), R(False))], [(S("def g():\n    return 0"), R(True))]]
    sols = select_solutions(probs, res, per_problem=2)
    assert [(x["problem"], x["code"].split()[-1]) for x in sols] == [("a", "1"), ("a", "2"), ("c", "0")], sols
    c = sft_config(load_config())
    assert c["grpo"]["kl_coef"] == 0.0 and c["grpo"]["updates_per_batch"] == 1 and c["grpo"]["logp_temperature"] == 1.0
    plan = dict(arms={"KD": {"eval_checkpoint": "runs/x", "init_from": "runs/y", "grpo": {"lr": 3e-5},
                             "checkpoint": {"every_steps": 5}}})
    cfg = arm_config(plan, "KD")  # `checkpoint` (sección de config) sigue siendo config; las claves del brazo no
    assert "eval_checkpoint" not in cfg and "init_from" not in cfg and cfg["grpo"]["lr"] == 3e-5
    assert cfg["checkpoint"]["every_steps"] == 5 and set(ARM_KEYS) == {"init_from", "eval_checkpoint"}
    d = tempfile.mkdtemp()
    try:
        ckd = os.path.join(d, "checkpoints", "ckpt_0000003")
        os.makedirs(ckd)
        open(os.path.join(ckd, "meta.json"), "w").write("{}")
        assert resolve_ckpt(d) == ckd and resolve_ckpt(ckd) == ckd
        try:
            resolve_ckpt(os.path.join(d, "no_existe"))
            raise AssertionError("debía fallar")
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(d)


@test("v3 destilación (CPU, modelo mínimo): el profesor genera, el sandbox filtra, SFT del alumno con LoRA y "
      "checkpoint que carga train.py eval")
def t3_distill_cpu():
    _torch()
    _peft()
    import argparse
    import rollouts.collect as rc
    from model.infer import Engine
    from rewards.reward import Result
    from training import checkpoint as ck
    from training.distill import run as run_distill
    cfg = tiny_cfg()
    cfg["dataset"]["levels"] = [0]
    run_dir = tempfile.mkdtemp(prefix="distill_")
    real = rc.score
    calls = {"n": 0}

    def fake(p, code, cfg_, seed=0):  # la mitad "resuelve": hay soluciones que destilar
        calls["n"] += 1
        return Result(1.0, True, True) if calls["n"] % 2 else Result(0.1, True, False)
    rc.score = fake
    try:
        a = argparse.Namespace(teacher=None, k=2, per_problem=2, epochs=1, batch=2, lr=1e-3, run=run_dir,
                               levels=None, problems=["clamp", "abs_sum"])
        path = run_distill(cfg, a)
        sols = [json.loads(x) for x in open(os.path.join(run_dir, "teacher_solutions.jsonl"))]
        assert sols and all(s["problem"] in ("clamp", "abs_sum") for s in sols), sols
        rows = [json.loads(x) for x in open(os.path.join(run_dir, "distill.jsonl"))]
        assert rows and all(r["tokens"] > 0 and r["skipped"] == 0 and r["nll"] > 0 for r in rows), rows
        meta = ck.read_meta(path)
        assert meta["lineage"]["distill"]["solutions"] == len(sols) and meta["format"] == 2, meta["lineage"]
        ck.verify(path)
        eng = Engine(cfg, path, device="cpu")  # el checkpoint se carga como cualquier otro
        assert eng.arch["layers"] == 4
        # reanudar: las soluciones del profesor se reutilizan (no se vuelve a generar)
        n0 = calls["n"]
        run_distill(cfg, a)
        assert calls["n"] == n0
    finally:
        rc.score = real
        shutil.rmtree(run_dir, ignore_errors=True)


@test("v3 HumanEval (CPU, modelo mínimo): train.py humaneval genera, ejecuta en el sandbox y escribe filas pareables")
def t3_humaneval_cpu():
    _torch()
    _peft()
    import argparse, gzip
    from evaluation.humaneval import run as run_he
    cfg = tiny_cfg()
    d = tempfile.mkdtemp(prefix="he_")
    try:
        recs = [dict(task_id=f"T/{i}", entry_point="f", prompt=f'def f(x):\n    """Devuelve x + {i}."""\n',
                     test=f"def check(candidate):\n    assert candidate(1) == {1 + i}\n") for i in range(2)]
        data = os.path.join(d, "he.jsonl.gz")
        with open(data, "wb") as f:
            f.write(gzip.compress("".join(json.dumps(r) + "\n" for r in recs).encode()))
        a = argparse.Namespace(data=data, limit=None, n=2, seed=None, timeout=5.0, checkpoint=None, out=d, chunk=8)
        rep = run_he(cfg, a)
        saved = json.load(open(os.path.join(d, "humaneval.json")))
        assert saved["problems"] == 2 and [r["n"] for r in saved["rows"]] == [2, 2], saved
        assert saved["split"] == "humaneval" and "pass@1" in saved and saved["greedy"] is not None
        assert rep["inference"] == saved["inference"] and len(saved["data_sha256"]) == 64
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("v3 experimentos: config de cada brazo = base + common + brazo; informe pareado sobre runs sintéticos")
def t3_experiments():
    from experiments.report import build
    from experiments.run import arm_config, load_plan
    plan = load_plan("main")
    a, b, c = (arm_config(plan, k) for k in ("A", "B", "C"))
    assert a["grpo"]["max_steps"] == b["grpo"]["max_steps"] == c["grpo"]["max_steps"] == 40
    assert not a["growth"]["enabled"] and b["growth"]["enabled"] and c["model"].endswith("1.5B-Instruct")
    assert a["seed"] == b["seed"] == c["seed"] and a["eval"]["seed"] == c["eval"]["seed"]
    assert arm_config(plan, "A", steps=7)["grpo"]["max_steps"] == 7
    assert arm_config(plan, "U2")["grpo"]["updates_per_batch"] == 2 and arm_config(plan, "A_lr2")["grpo"]["lr"] == 2e-5
    assert plan["final_eval"]["splits"] == ["val"]  # test no se usa para elegir entre brazos
    # informe: dos brazos sintéticos con evaluaciones finales
    d = tempfile.mkdtemp()
    import yaml
    with open(os.path.join(d, "plan.yaml"), "w") as f:
        yaml.safe_dump(dict(name="sint", arms={"A": {}, "B": {"x": 1}}, final_eval={"splits": ["val"]}), f)

    def rep(vals):
        return dict(split="val", n=4, ks=[1], problems=len(vals), **{"pass@1": sum(vals) / len(vals)},
                    rows=[dict(id=f"p{i}", pass_at={"1": v}, greedy=v > 0.5) for i, v in enumerate(vals)])
    for arm, vals, secs, rw in (("A", [0.0, 0.25, 0.5, 0.0], 20.0, 0.5), ("B", [0.5, 0.5, 1.0, 0.25], 25.0, 0.6)):
        os.makedirs(os.path.join(d, arm))
        with open(os.path.join(d, arm, "config.yaml"), "w") as f:
            yaml.safe_dump({"model": "m"}, f)
        with open(os.path.join(d, arm, "metrics.jsonl"), "w") as f:
            for s_ in range(4):
                grow = 5.0 if arm == "B" and s_ == 0 else None  # el crecimiento no cuenta en s/step
                probs = ["p0", "p1"] if s_ != 3 or arm == "A" else ["p0", "p9"]  # step 3: problemas distintos
                f.write(json.dumps(dict(step=s_, loss=0.1, secs=secs + (grow or 0), t_growth=grow, reward=rw + s_ / 100,
                                        problems=probs, solved=4, n=8, kl=0.01)) + "\n")
        json.dump(rep(vals), open(os.path.join(d, arm, "eval_val.json"), "w"))
    r = build(d)
    c_ = r["comparisons"]["B_vs_A_val"]["metrics"]["pass@1"]
    assert abs(c_["delta"] - 0.375) < 1e-9 and c_["wins"] == 4 and r["arms"]["B"]["step_s_p50"] == 25.0, r["comparisons"]
    assert r["arms"]["B"]["growth_s"] == 5.0 and r["arms"]["B"]["wall_train_s"] == 105.0, r["arms"]["B"]
    tr = r["train_reward"]["B_vs_A"]["all"]  # sólo los 3 steps con los mismos problemas forman pareja
    assert tr["steps"] == 3 and tr["common"] == 4 and abs(tr["delta"] - 0.1) < 1e-9 and tr["wins"] == 3, tr
    assert "B_vs_A_val" in r["markdown"] and "Reward de train pareado" in r["markdown"]
    assert os.path.exists(os.path.join(d, "report.md"))
    shutil.rmtree(d)


# ============================================================================================ torch en CPU (mínimos)
_TINY = {}
CORPUS = ["def f(x):\n    return x + 1\n", "```python\ndef add(a, b):\n    return a + b\n```", "for i in range(10): print(i)",
          "if x > 0:\n    y = [1, 2, 3]\nelse:\n    y = {}\n", "Reply with one Python code block defining the function.",
          "assert two_sum([2, 7, 11, 15], 9) == [0, 1]", "class A:\n    pass\n", "while n != 1: n = n // 2"]


def tiny_dir():
    """Modelo Qwen2 mínimo (aleatorio, 4 capas) + tokenizer BPE de bytes entrenado al vuelo, en un dir local:
    carga por el camino real (model.loader, peft, checkpoints) sin descargar nada."""
    torch = _torch()
    if "dir" in _TINY:
        return _TINY["dir"]
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    d = tempfile.mkdtemp(prefix="tiny_qwen2_")
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(CORPUS * 20, trainers.BpeTrainer(
        vocab_size=320, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                         "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=384, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True,
                      pad_token_id=tok.pad_token_id, eos_token_id=[tok.eos_token_id, tok.pad_token_id])
    m = Qwen2ForCausalLM(cfg)
    with torch.no_grad():  # pesos algo mayores que la init por defecto: distribuciones no uniformes (tests más finos)
        for p in m.parameters():
            if p.dim() > 1:
                p.mul_(4.0)
    m.generation_config.eos_token_id = [tok.eos_token_id, tok.pad_token_id]
    m.generation_config.pad_token_id = tok.pad_token_id
    m.save_pretrained(d)
    _TINY.update(dir=d, tok=tok)
    return d


def tiny_cfg(**over):
    cfg = load_config()
    cfg["model"] = tiny_dir()
    cfg["context_length"] = 1024
    cfg["max_new_tokens"] = 24
    cfg["generation"]["min_new_tokens_floor"] = 8
    cfg["sandbox"]["allow_unsafe"] = True  # sin bwrap también; con bwrap se usa bwrap (backend auto)
    for k, v in over.items():
        d = cfg
        *path, last = k.split("__")
        for p in path:
            d = d[p]
        d[last] = v
    return cfg


def tiny_model(dtype="float32", layers=None):
    torch = _torch()
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(tiny_dir(), dtype=getattr(torch, dtype))
    return m.eval()


def _batch(torch, specs, vocab=384, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = max(n + m for n, m in specs)
    ids = torch.zeros((len(specs), L), dtype=torch.long)
    attn = torch.zeros_like(ids)
    for i, (n, m) in enumerate(specs):
        ids[i, :n + m] = torch.randint(3, vocab, (n + m,), generator=g)
        attn[i, :n + m] = 1
    return ids, attn, [n for n, _ in specs], [m for _, m in specs]


@test("v3 experimentos: una evaluación de checkpoint hecha con LoRA fusionado (inferencia v1) se guarda aparte y se "
      "repite; la del modelo base y las de la versión actual no; el informe compara las dos del mismo checkpoint")
def t3_reeval_inference():
    import yaml
    import experiments.run as er
    from common import INFERENCE_VERSION
    from experiments.report import build
    d = tempfile.mkdtemp()
    calls, real = [], er._run
    er._run = lambda cmd, log: calls.append(cmd) or 0
    try:
        ck_ = os.path.join(d, "ckpt")
        json.dump(dict(checkpoint=ck_, n=4), open(os.path.join(d, "eval_val.json"), "w"))  # sin "inference" = v1
        assert er.evaluate("c.yaml", ck_, "val", 4, d, os.path.join(d, "log")) == 0 and len(calls) == 1
        assert os.path.exists(os.path.join(d, "eval_val.inference1.json"))
        json.dump(dict(checkpoint=ck_, n=4, inference=INFERENCE_VERSION), open(os.path.join(d, "eval_val.json"), "w"))
        er.evaluate("c.yaml", ck_, "val", 4, d, os.path.join(d, "log"))
        assert len(calls) == 1  # versión actual: no se repite
        b = os.path.join(d, "base")
        os.makedirs(b)
        json.dump(dict(checkpoint=None, n=4), open(os.path.join(b, "eval_val.json"), "w"))
        er.evaluate("c.yaml", None, "val", 4, b, os.path.join(b, "log"))
        assert len(calls) == 1  # el modelo base no lleva LoRA: su evaluación sigue valiendo
    finally:
        er._run = real
        shutil.rmtree(d)
    # informe: el mismo checkpoint evaluado con las dos cargas -> comparación pareada "sin fusionar vs fusionado"
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, "plan.yaml"), "w") as f:
            yaml.safe_dump(dict(name="sint", arms={"A": {}}, final_eval={"splits": ["val"]}), f)
        os.makedirs(os.path.join(d, "A"))

        def rep(vals, **kw):
            return dict(split="val", n=4, ks=[1], problems=len(vals), checkpoint="/x/ckpt_0000039",
                        **{"pass@1": sum(vals) / len(vals)}, **kw,
                        rows=[dict(id=f"p{i}", pass_at={"1": v}, greedy=v > 0.5) for i, v in enumerate(vals)])
        json.dump(rep([0.0, 0.25, 0.5, 0.0]), open(os.path.join(d, "A", "eval_val.inference1.json"), "w"))
        json.dump(rep([0.25, 0.5, 0.5, 0.25], inference=INFERENCE_VERSION), open(os.path.join(d, "A", "eval_val.json"), "w"))
        r = build(d)
        c = r["comparisons"]["A_sinfusionar_vs_fusionado_val"]
        assert abs(c["metrics"]["pass@1"]["delta"] - 0.1875) < 1e-9, c
    finally:
        shutil.rmtree(d)


@test("v3 informe: un step marcado lento sólo por el crecimiento (t_growth) no cuenta como lento; uno lento de verdad sí")
def t3_report_growth_slow():
    from experiments.report import arm_summary
    d = tempfile.mkdtemp()
    try:
        rows = [dict(step=0, loss=0.1, secs=20.0), dict(step=1, loss=0.1, secs=69.0, t_growth=48.0, slow=True,
                                                         secs_median=20.0),
                dict(step=2, loss=0.1, secs=90.0, t_growth=5.0, slow=True, secs_median=20.0),  # 85 s sin crecer
                dict(step=3, loss=0.1, secs=70.0, slow=True, secs_median=20.0)]
        with open(os.path.join(d, "metrics.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        s_ = arm_summary(d)
        assert s_["slow_steps"] == 2 and s_["growth_s"] == 53.0, s_
    finally:
        shutil.rmtree(d)


@test("v3 HumanEval: programa = prompt + código + test + check(); en el sandbox pasa lo correcto y falla lo incorrecto, "
      "el bucle infinito (timeout) y la sintaxis rota; un bloque __main__ con input() no lo tumba; datos con sha256")
def t3_humaneval_sandbox():
    import gzip
    from evaluation import humaneval as he
    from evaluation.compare import compare
    from sandbox import executor
    cfg = load_config()
    cfg["sandbox"]["allow_unsafe"] = True  # sin bwrap (CI) usa rlimit; con bwrap, bwrap
    b = executor.backend(cfg.sandbox)
    rec = dict(task_id="T/0", entry_point="add",
               prompt='from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    """Suma dos enteros."""\n',
               test="METADATA = {}\n\n\ndef check(candidate):\n    assert candidate(2, 3) == 5\n"
                    "    assert candidate(-1, 1) == 0\n")
    ok = lambda code, t=5.0: he.run_one(rec, code, cfg, b, t)
    assert ok("def add(a, b):\n    return a + b\n") == (True, "ok")
    assert ok("def add(a, b):\n    return a - b\n") == (False, "AssertionError")
    assert ok("def add(a, b):\n    return a +\n") == (False, "SyntaxError")
    assert ok("def add(a, b):\n    while True:\n        pass\n", 2.0) == (False, "timeout")
    assert ok("def add(a, b):\n    return a + b\n\nif __name__ == '__main__':\n    print(add(int(input()), 1))\n")[0]
    # datos: jsonl.gz local, sha256 guardado y comprobado
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "he.jsonl.gz")
        with open(path, "wb") as f:
            f.write(gzip.compress((json.dumps(rec) + "\n").encode()))
        recs, sha = he.load(path, download=False)
        assert [r["task_id"] for r in recs] == ["T/0"] and len(sha) == 64
        with open(path + ".sha256", "w") as f:
            f.write("0" * 64)
        try:
            he.load(path, download=False)
            raise AssertionError("debía fallar: sha256 distinto")
        except ValueError:
            pass
    finally:
        shutil.rmtree(d)
    # informe: pass@k insesgado, greedy y filas comparables con evaluation.compare

    class S:
        def __init__(self, n):
            self.ntok, self.closed = n, True
    recs2 = [dict(rec, task_id=f"T/{i}") for i in range(3)]
    samples = [[S(10)] * 4 for _ in recs2]
    res_a = [[(True, "ok")] * 4, [(False, "AssertionError")] * 4, [(True, "ok"), (False, "timeout")] * 2]
    res_b = [[(True, "ok")] * 4, [(True, "ok")] * 4, [(True, "ok")] * 4]
    ra = he.report(recs2, samples, res_a, [(True, "ok"), (False, "x"), (True, "ok")], 4, [1, 2], 0, 100)
    rb = he.report(recs2, samples, res_b, [(True, "ok")] * 3, 4, [1, 2], 0, 100)
    assert abs(ra["pass@1"] - 0.5) < 1e-9 and abs(ra["greedy"] - 2 / 3) < 1e-3 and rb["pass@1"] == 1.0, ra
    assert ra["error_rates"]["timeout"] == round(2 / 12, 4)
    c = compare(ra, rb)
    assert abs(c["metrics"]["pass@1"]["delta"] - 0.5) < 1e-9, c["metrics"]["pass@1"]


@test("v3 logprobs (CPU): sólo-respuesta por trozos == log_softmax completo, también el gradiente y con T=0.8")
def t3_logprob_equivalence():
    torch = _torch()
    from training.logprob import full_token_logps, token_logps
    m = tiny_model("float32")
    for p in m.parameters():
        p.requires_grad_(True)
    ids, attn, pl, rl = _batch(torch, [(5, 7), (9, 3), (2, 11), (6, 0)])
    for temp in (1.0, 0.8):
        full = full_token_logps(m, ids, attn, temp)
        want = torch.cat([full[i, n - 1:n - 1 + k] for i, (n, k) in enumerate(zip(pl, rl))])
        got = token_logps(m, ids, attn, pl, rl, temp, chunk_mb=0.01)  # trozos de 16 filas: varios trozos
        assert got.shape == want.shape and torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
        w = torch.randn(want.shape, generator=torch.Generator().manual_seed(1))
        ga = torch.autograd.grad((want * w).sum(), [m.lm_head.weight, m.model.layers[0].mlp.up_proj.weight])
        gb = torch.autograd.grad((got * w).sum(), [m.lm_head.weight, m.model.layers[0].mlp.up_proj.weight])
        for a, b in zip(ga, gb):
            assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    lp, ent = token_logps(m, ids, attn, pl, rl, 1.0, entropy=True)
    assert lp.shape == ent.shape and (ent >= 0).all() and (ent <= math.log(384) + 1e-4).all()


def _lora_trainer(cfg, lr=0.05, sgd=False, **trainer_over):
    torch = _torch()
    _peft()
    from model.policy import load_policy, trainable_params
    from training import optim as topt
    from training.grpo import GrpoTrainer
    model, tok, ref_ctx = load_policy(cfg, "cpu")
    opt_ = torch.optim.SGD(trainable_params(model), lr=lr) if sgd else topt.build(model, cfg)
    for g in opt_.param_groups:
        g["lr"] = g["base_lr"] = lr
    tr = GrpoTrainer(model, ref_ctx, tok, opt_, cfg)
    for k, v in trainer_over.items():
        setattr(tr, k, v)
    return model, tok, ref_ctx, opt_, tr


def _items(tok, n=6, seed=0):
    from training.grpo import RollItem
    r = random.Random(seed)
    out = []
    for i in range(n):
        p = tok(f"<|im_start|>user\ndef f{i}(x): return x + {i}<|im_end|>\n<|im_start|>assistant\n").input_ids
        resp = [r.randrange(3, 300) for _ in range(r.randint(2, 9))]
        out.append(RollItem(p, resp, [1.0, -1.0, 0.5, -0.5, 0.0, 1.5][i % 6]))
    return out


def _lora_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@test("v3 GRPO (CPU, LoRA real): U=1 ratio 1 exacto; U=2 referencia una vez, ratio/clip/approx_kl reales; lr=0 -> nada se mueve")
def t3_grpo_step_cpu():
    torch = _torch()
    cfg = tiny_cfg()
    model, tok, ref_ctx, opt_, tr = _lora_trainer(cfg, lr=0.0)
    calls = [0]
    real_ctx = tr.ref_ctx

    def counting():
        calls[0] += 1
        return real_ctx()
    tr.ref_ctx = counting
    items = _items(tok)
    w0 = _lora_params(model)
    st = tr.step(items)
    assert st.updates == 1 and st.clip_frac == 0.0 and st.approx_kl == 0.0 and st.ratio_max == 1.0, st
    assert all(torch.equal(w0[n], p) for n, p in model.named_parameters() if p.requires_grad)  # lr = 0
    assert st.kl == 0.0 or st.kl < 1e-9, st.kl  # LoRA B=0: política == referencia
    assert calls[0] >= 1 and math.isfinite(st.loss) and st.entropy > 0 and st.tokens == sum(len(i.resp_ids) for i in items)
    assert st.logp < 0 and math.isfinite(st.logp), st.logp  # log-prob media por token (la NLL de la destilación)
    # lr grande, U=2: la segunda pasada ve el ratio moverse
    for g in opt_.param_groups:
        g["lr"] = 0.05
    tr.updates, calls[0] = 2, 0
    st = tr.step(items)
    n_micro = len(tr._micro(sorted(items, key=lambda b: len(b.prompt_ids) + len(b.resp_ids)), tr.nograd_scale))
    assert calls[0] == n_micro, (calls[0], n_micro)  # referencia: una pasada, no una por actualización
    assert st.updates == 2 and st.approx_kl > 0 and st.ratio_max > 1.0, st
    assert not all(torch.equal(w0[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert model.training is False  # el trainer deja el modelo en eval()
    # M=2 minibatches: dos optimizer.step por pasada
    tr.updates, tr.minibatches = 1, 2
    st = tr.step(items)
    assert st.updates == 2 and st.microbatches >= 2, st


@test("v3 GRPO (CPU): OOM en un micro-batch -> micro_bs a la mitad y el minibatch se rehace (mismo resultado)")
def t3_grpo_oom_cpu():
    torch = _torch()
    import training.logprob as lpm
    cfg = tiny_cfg(dtype="fp32")  # fp32 + SGD: dos particiones en micro-batches dan el mismo update salvo redondeo
    torch.manual_seed(5)
    m1, tok, _, _, tr1 = _lora_trainer(cfg, lr=0.5, sgd=True, micro_bs=4)
    torch.manual_seed(5)
    m2, _, _, _, tr2 = _lora_trainer(cfg, lr=0.5, sgd=True, micro_bs=4)
    items = _items(tok, 8)
    m1_0 = _lora_params(m1)
    tr1.step(items)
    real, n = lpm.token_logps, [0]

    def flaky(model, *a, **k):
        if torch.is_grad_enabled():
            n[0] += 1
            if n[0] == 2:  # segundo micro-batch con gradiente, con el primero ya acumulado
                raise RuntimeError("CUDA out of memory (simulado)")
        return real(model, *a, **k)
    lpm.token_logps = flaky
    try:
        st = tr2.step(items)
    finally:
        lpm.token_logps = real
    assert st.oom == 1 and tr2.micro_bs == 2 and st.updates == 1, st
    p1, p2 = _lora_params(m1), _lora_params(m2)
    assert any(not torch.equal(p1[k], m1_0[k]) for k in p1), "sin actualización: el test no prueba nada"
    assert all(torch.allclose(p1[k], p2[k], atol=1e-6) for k in p1), "el OOM cambió el gradiente aplicado"


@test("v3 GRPO (CPU): gradiente NaN -> ese optimizer.step no se aplica (pesos intactos) y queda contado")
def t3_grpo_nonfinite_cpu():
    torch = _torch()
    import training.grpo as g
    cfg = tiny_cfg()
    model, tok, _, _, tr = _lora_trainer(cfg, lr=0.05)
    w0, real = _lora_params(model), g.grpo_loss

    def nan_loss(*a, **k):
        loss, st = real(*a, **k)
        return loss * float("nan"), st
    g.grpo_loss = nan_loss
    try:
        st = tr.step(_items(tok))
    finally:
        g.grpo_loss = real
    assert st.skipped == 1 and st.updates == 0
    assert all(torch.equal(w0[n], p) for n, p in model.named_parameters() if p.requires_grad), "NaN en los pesos"


@test("v3 GRPO (CPU): con los pesos quietos (lr=0) y U=2 el ratio es exactamente 1 en todas las pasadas (sin dropout)")
def t3_grpo_ratio_exact_cpu():
    torch = _torch()
    cfg = tiny_cfg()
    model, tok, _, _, tr = _lora_trainer(cfg, lr=0.0, updates=2)
    for mod in model.modules():
        assert not isinstance(mod, torch.nn.Dropout) or mod.p == 0.0  # el trainer apaga cualquier dropout
    st = tr.step(_items(tok))
    assert st.updates == 2 and st.approx_kl == 0.0 and st.ratio_max == 1.0 and st.clip_frac == 0.0, st


@test("v3 GRPO (CPU): un fallo a mitad de un step con U=2 dice cuántas actualizaciones ya aplicó (para el checkpoint)")
def t3_grpo_partial_cpu():
    torch = _torch()
    import training.grpo as g
    cfg = tiny_cfg()
    model, tok, _, _, tr = _lora_trainer(cfg, lr=0.05, updates=2)
    w0, real, calls = _lora_params(model), g.grpo_loss, [0]
    n_micro = len(tr._micro(sorted(_items(tok), key=lambda b: len(b.prompt_ids) + len(b.resp_ids))))

    def boom(*a, **k):
        calls[0] += 1
        if calls[0] == n_micro + 1:  # primera llamada de la segunda pasada
            raise RuntimeError("fallo simulado")
        return real(*a, **k)
    g.grpo_loss = boom
    try:
        tr.step(_items(tok))
        raise AssertionError("no propagó el fallo")
    except RuntimeError as e:
        assert getattr(e, "updates_done", None) == 1, getattr(e, "updates_done", None)
    finally:
        g.grpo_loss = real
    assert not all(torch.equal(w0[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert model.training is False


@test("v3 GRPO (CPU): logits enormes (pesos x2000) -> pérdida y KL finitos (sin exp desbordado)")
def t3_grpo_extreme_logits_cpu():
    torch = _torch()
    cfg = tiny_cfg(dtype="fp32")
    model, tok, _, _, tr = _lora_trainer(cfg, lr=0.0, updates=2)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lm_head" in n or "embed_tokens" in n:
                p.mul_(2000.0)
    st = tr.step(_items(tok))
    assert math.isfinite(st.loss) and math.isfinite(st.kl) and math.isfinite(st.approx_kl), st


@test("v3 crecimiento (CPU): bloques identidad -> logits BIT-idénticos (bf16 y fp32), greedy idéntico con caché KV, "
      "config/caché coherentes, bypass = original, gradiente sólo en o_proj/down_proj al nacer")
def t3_growth_identity_cpu():
    torch = _torch()
    from model import growth as gr
    for dt in ("bfloat16", "float32"):
        base = tiny_model(dt)
        grown = tiny_model(dt)
        ids, attn, _, _ = _batch(torch, [(12, 0), (7, 0)])
        with torch.no_grad():
            want = base(input_ids=ids, attention_mask=attn).logits
        ev = gr.apply_event(grown, {"after": [0, 3]})
        assert ev["layers_before"] == 4 and ev["layers_after"] == 6 and ev["new"] == [1, 5], ev
        c = grown.config
        assert c.num_hidden_layers == 6 and len(c.layer_types) == 6
        assert [l.self_attn.layer_idx for l in grown.model.layers] == list(range(6))
        with torch.no_grad():
            got = grown(input_ids=ids, attention_mask=attn).logits
        assert torch.equal(want, got), f"{dt}: el crecimiento cambió la función ({(want - got).abs().max()})"
        g_ids = torch.tensor([[5, 9, 22, 41]])
        a1 = base.generate(g_ids, max_new_tokens=12, do_sample=False, pad_token_id=0)
        a2 = grown.generate(g_ids, max_new_tokens=12, do_sample=False, pad_token_id=0)
        assert torch.equal(a1, a2), "greedy con caché KV distinto tras crecer (¿layer_idx/layer_types?)"
        assert gr.arch(grown)["grown"] == [1, 5] and gr.count_params(grown)[0] > gr.count_params(base)[0]
    # gradiente al nacer: sólo las proyecciones a cero reciben gradiente distinto de 0
    m = tiny_model("float32")
    gr.apply_event(m, {"after": [1]})
    blk = m.model.layers[2]
    out = m(input_ids=ids, attention_mask=attn).logits
    out.float().pow(2).mean().backward()
    assert blk.self_attn.o_proj.weight.grad.abs().sum() > 0 and blk.mlp.down_proj.weight.grad.abs().sum() > 0
    assert blk.self_attn.q_proj.weight.grad.abs().sum() == 0 and blk.mlp.up_proj.weight.grad.abs().sum() == 0
    # tras moverse, bypass devuelve exactamente el modelo original
    with torch.no_grad():
        blk.self_attn.o_proj.weight.normal_(0, 0.5)
        moved = m(input_ids=ids, attention_mask=attn).logits
        with gr.bypass(m):
            orig = m(input_ids=ids, attention_mask=attn).logits
    ref = tiny_model("float32")
    with torch.no_grad():
        want = ref(input_ids=ids, attention_mask=attn).logits
    assert not torch.equal(moved, want) and torch.equal(orig, want)
    # crecer dos veces: el segundo evento va detrás de los bloques del primero y los índices se renumeran
    gr.apply_event(m, {"after": [1, 2]})
    assert gr.arch(m)["grown"] == [2, 3, 5] and [getattr(b, "growth_event") for _, b in gr.grown_blocks(m)] == [0, 1, 1]


@test("v3 crecimiento + LoRA (CPU): LoRA sólo en capas originales, crecer a mitad de run actualiza peft, guardar y "
      "recargar (replay) da logits idénticos y el optimizador por grupos se recarga")
def t3_growth_peft_roundtrip_cpu():
    torch = _torch()
    _peft()
    from model import growth as gr
    from model.policy import load_policy
    from training import checkpoint as ck
    from training import optim as topt
    cfg = tiny_cfg(growth__enabled=True, growth__schedule=[{"step": 0, "new_layers": 1}, {"step": 3, "new_layers": 1}])
    cfg["dtype"] = "bf16"
    ev0 = {"after": [1], "step": 0}
    model, tok, ref_ctx = load_policy(cfg, "cpu", events=[ev0])  # crece ANTES de LoRA (step 0)
    names = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    assert names and not any(".layers.2." in n for n in names), names  # el bloque crecido (índice 2) sin LoRA
    assert all(p.requires_grad for p in gr.grown_parameters(model))
    events = [{"method": "depth", "after": [1], "step": 0, "event": 0}]
    gr_ev = gr.grown_blocks(model)
    events[0].update(new=[i for i, _ in gr_ev])
    opt_ = topt.build(model, cfg, events)
    assert [g["name"] for g in opt_.param_groups] == ["policy", "growth0"]
    # crecimiento a mitad de run sobre el PeftModel
    ev1 = gr.apply_event(model, {"after": [3], "step": 3})
    topt.add_growth_group(opt_, model, cfg, ev1, 1)
    events.append(ev1)
    assert model.peft_config["default"].layers_to_transform == gr.original_indices(model)
    # un paso de optimización para que todo (LoRA y bloques) tenga valores no triviales
    ids, attn, _, _ = _batch(torch, [(10, 0)])
    model.train()
    model(input_ids=ids, attention_mask=attn).logits.float().pow(2).mean().backward()
    for g in opt_.param_groups:
        g["lr"] = 0.01
    opt_.step()
    opt_.zero_grad()
    model.eval()
    with torch.no_grad():
        want = model(input_ids=ids, attention_mask=attn).logits
        with ref_ctx() as r:
            ref_logits = r(input_ids=ids, attention_mask=attn).logits
    base = tiny_model("bfloat16")
    with torch.no_grad():
        assert torch.equal(ref_logits, base(input_ids=ids, attention_mask=attn).logits)  # referencia = base exacto
    d = tempfile.mkdtemp()
    path = ck.save(d, 4, model, opt_, {"x": 1}, True, "h", extra=dict(growth=events, arch=gr.arch(model)))
    assert ck.verify(path) and os.path.exists(os.path.join(path, "growth.safetensors"))
    m2, _, _ = load_policy(cfg, "cpu", resume_from=path)
    with torch.no_grad():
        got = m2(input_ids=ids, attention_mask=attn).logits
    assert torch.equal(want, got), (want - got).abs().max()
    opt2 = topt.build(m2, cfg, events)
    ck.load_optim_rng(path, opt2, "cpu")
    assert [g["name"] for g in opt2.param_groups] == ["policy", "growth0", "growth1"]
    assert set(opt_.state_dict()["state"]) == set(opt2.state_dict()["state"])
    from model.infer import Engine
    eng = Engine(cfg, path, device="cpu")  # inferencia: replay + LoRA SIN fusionar (common.INFERENCE_VERSION)
    with torch.no_grad():
        inf = eng.m(input_ids=ids, attention_mask=attn).logits
    # la misma cuenta que el entrenamiento: idéntico bit a bit (fusionar LoRA en bf16 lo cambiaba; ver model/infer.py)
    assert torch.equal(inf, want), (inf.float() - want.float()).abs().max()
    assert eng.arch["layers"] == 6, eng.arch
    shutil.rmtree(d)


@test("v3 inferencia (CPU): un checkpoint LoRA se usa SIN fusionar (Engine = la política entrenada, bit a bit); "
      "fusionarlo en bf16 con un ΔW mucho menor que la resolución de los pesos (~1e-5) los deja casi sin cambiar")
def t3_inference_unmerged_cpu():
    torch = _torch()
    _peft()
    from peft import PeftModel
    from model import growth as gr
    from model.infer import Engine
    from model.policy import load_policy
    from training import checkpoint as ck
    from training import optim as topt
    cfg = tiny_cfg()
    cfg["dtype"] = "bf16"
    model, tok, _ = load_policy(cfg, "cpu")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():  # B pequeño: ΔW = escala·B·A del orden del medido en el brazo A tras 40 steps
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 1e-5)
    model.eval()
    ids, attn, _, _ = _batch(torch, [(12, 0), (7, 0)])
    with torch.no_grad():
        want = model(input_ids=ids, attention_mask=attn).logits
    d = tempfile.mkdtemp()
    try:
        path = ck.save(d, 0, model, topt.build(model, cfg), {}, True, "h", extra=dict(arch=gr.arch(model)))
        eng = Engine(cfg, path, device="cpu")
        assert hasattr(eng.m, "peft_config"), type(eng.m)  # adapter sin fusionar
        with torch.no_grad():
            got = eng.m(input_ids=ids, attention_mask=attn).logits
        assert torch.equal(got, want), (got.float() - want.float()).abs().max()
        # lo que hacía la carga antigua: merge_and_unload sobre pesos bf16
        merged = PeftModel.from_pretrained(tiny_model("bfloat16"), path).merge_and_unload()
        base = tiny_model("bfloat16")
        w_m = merged.model.layers[0].self_attn.q_proj.weight
        w_b = base.model.layers[0].self_attn.q_proj.weight
        same = (w_m == w_b).float().mean().item()
        assert same > 0.9, same  # > 90 % de los pesos no cambian al fusionar
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("v3 phase2 de extremo a extremo (CPU, modelo mínimo): crecimiento progresivo, checkpoint con arquitectura, "
      "reanudación con replay, eval pareada contra la base y train.py eval sobre el checkpoint crecido")
def t3_phase2_e2e_cpu():
    torch = _torch()
    _peft()
    tiny_dir()
    run = tempfile.mkdtemp(prefix="e2e_")
    cfg_path = os.path.join(run, "config.yaml")
    cfg = tiny_cfg(growth__enabled=True, growth__schedule=[{"step": 0, "new_layers": 1}, {"step": 2, "new_layers": 1}])
    cfg["num_rollouts"] = 4
    cfg["dataset"]["levels"] = [0]
    cfg["grpo"].update(problems_per_step=2, max_steps=4, micro_bs=4)
    cfg["checkpoint"].update(every_steps=2, eval_every_steps=2, keep_last=3)
    cfg["eval"].update(during_train=["val"], during_train_n=2, ks=[1, 2], bootstrap=50, n=2)
    import yaml
    with open(cfg_path, "w") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg)), f)
    runner = r'''
import random, sys
sys.path.insert(0, sys.argv[1])
import rollouts.collect as rc
from rewards.reward import Result
rng = random.Random(0)
rc.score = lambda p, code, cfg, seed=0: Result(rng.random(), True, False)  # reward con varianza: hay gradiente
import train
sys.argv = ["train.py"] + sys.argv[2:]
train.main()
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0")

    def go(*args, stop_at=None):
        return subprocess.run([sys.executable, "-c", runner, ROOT, *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=1200, env=env)
    cache = os.path.join(ROOT, "runs", "eval_cache")
    before = set(os.listdir(cache)) if os.path.isdir(cache) else set()
    try:
        cfg2 = dict(cfg)
        cfg2["grpo"] = dict(cfg["grpo"], max_steps=2)  # primera tanda: 2 steps, luego se reanuda hasta 4
        with open(cfg_path, "w") as f:
            yaml.safe_dump(json.loads(json.dumps(cfg2)), f)
        r = go("phase2", "--config", cfg_path, "--run", run)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        with open(cfg_path, "w") as f:
            yaml.safe_dump(json.loads(json.dumps(cfg)), f)
        r = go("phase2", "--config", cfg_path, "--run", run)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "reanudado desde" in r.stdout and "[growth] step 2" in r.stdout, r.stdout[-2000:]
        rows = [json.loads(x) for x in open(os.path.join(run, "metrics.jsonl"))]
        steps = [x for x in rows if "loss" in x and x.get("step", -1) >= 0]
        assert [x["step"] for x in steps] == [0, 1, 2, 3], [x["step"] for x in steps]
        assert [x["step"] for x in rows if x.get("event") == "growth"] == [0, 2]
        assert any(x.get("event") == "eval_base" for x in rows)
        assert all("val_delta_pass@1" in x for x in rows if x.get("event") == "eval_split")
        assert all(x["updates"] == 1 for x in steps) and all(x["t_update"] > 0 for x in steps)
        meta = json.load(open(os.path.join(run, "checkpoints", "ckpt_0000003", "meta.json")))
        assert meta["format"] == 2 and meta["arch"]["layers"] == 6 and len(meta["growth"]) == 2, meta["arch"]
        assert meta["arch"]["grown"] and meta["lineage"]["run"] == run and meta["code"]["files"] > 20
        r = go("eval", "--config", cfg_path, "--split", "val", "--n", "2", "--out", run, "--checkpoint",
               os.path.join(run, "checkpoints", "ckpt_0000003"))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        ev = json.load(open(os.path.join(run, "eval_val.json")))
        assert ev["arch"]["layers"] == 6 and ev["problems"] > 0
    finally:
        shutil.rmtree(run, ignore_errors=True)
        for f in (set(os.listdir(cache)) if os.path.isdir(cache) else set()) - before:
            os.remove(os.path.join(cache, f))


# ================================================================================================= GPU (modelo real)
def _cuda():
    torch = _torch()
    if not torch.cuda.is_available():
        raise Skip("CUDA no disponible")
    return torch


@test("v3 GPU: memoria del step de entrenamiento, logits completos (antes) vs sólo-respuesta por trozos (ahora)")
def t3_gpu_logprob_memory():
    torch = _cuda()
    from model.loader import load
    from training.logprob import full_token_logps, token_logps
    cfg = load_config()
    m, tok = load(cfg)
    for p in m.parameters():
        p.requires_grad_(False)
    m.model.layers[-1].mlp.down_proj.weight.requires_grad_(True)  # algo con gradiente, como LoRA
    ids, attn, pl, rl = _batch(torch, [(200, 384)] * 4, vocab=150000)
    ids, attn = ids.cuda(), attn.cuda()
    peaks = {}
    for name in ("full", "flat"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        m.train()
        if name == "full":
            lp = full_token_logps(m, ids, attn)
            lp = torch.cat([lp[i, n - 1:n - 1 + k] for i, (n, k) in enumerate(zip(pl, rl))])
        else:
            lp = token_logps(m, ids, attn, pl, rl)
        lp.sum().backward()
        peaks[name] = (torch.cuda.max_memory_allocated() - base) / 2**30
        m.zero_grad(set_to_none=True)
        del lp
    m.eval()
    print(f"      micro-batch 4 x (200+384) tokens: pico extra logits completos={peaks['full']:.2f} GB, "
          f"sólo respuesta por trozos={peaks['flat']:.2f} GB ({peaks['full'] / max(peaks['flat'], 1e-9):.1f}x menos)")
    assert peaks["flat"] < 0.6 * peaks["full"], peaks
    del m
    torch.cuda.empty_cache()


@test("v3 GPU: crecimiento sobre Qwen2.5-Coder real: logits bit-idénticos y misma respuesta greedy")
def t3_gpu_growth_real():
    torch = _cuda()
    from model import growth as gr
    from model.loader import load
    from rollouts.generate import build_prompt
    from problems.problems import PROBLEMS
    cfg = load_config()
    m, tok = load(cfg)
    p = build_prompt(tok, cfg.prompt.system, PROBLEMS["two_sum"].prompt(), True)
    enc = tok(p, return_tensors="pt").to(m.device)
    with torch.inference_mode():
        want = m(**enc).logits
        g1 = m.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
    n0 = gr.count_params(m)[0]
    gr.apply_event(m, {"after": gr.plan_positions(len(m.model.layers), 4)})
    with torch.inference_mode():
        got = m(**enc).logits
        g2 = m.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
    n1 = gr.count_params(m)[0]
    print(f"      params {n0 / 1e6:.1f}M -> {n1 / 1e6:.1f}M (+{(n1 - n0) / 1e6:.1f}M en 4 bloques), capas "
          f"{len(m.model.layers)}, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GB")
    assert torch.equal(want, got) and torch.equal(g1, g2)
    del m
    torch.cuda.empty_cache()
