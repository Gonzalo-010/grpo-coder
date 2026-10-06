"""Validación del dataset (CLI: python -m problems.validate [--sets core catalog] [--ids ...]).

Por problema: metadatos, determinismo, referencia correcta (pasa todos sus casos y no usa nada que el scorer marcaría
como unsafe), diversidad de salidas, tests suficientes, mutation score, que ninguna entrada oculta aparezca en el
prompt ni en el pool de feedback, y que la versión lenta de los problemas de optimización falle en los casos grandes.
Global: ids/referencias/prompts sin duplicados y ninguna familia repartida entre splits.
Los 19 originales son el baseline intacto: sus avisos se informan pero no cuentan como error.
"""
from __future__ import annotations

import argparse, ast, inspect, re, sys, textwrap
from collections import defaultdict

from problems import repair
from problems.problems import ALL, CATEGORIES, SPLITS, _key, baseline, select

MIN_MUTATION = 0.8    # fracción mínima de mutantes detectados (algunos mutantes son equivalentes)
MAX_BASELINE = 0.6    # el mejor predictor constante no puede acertar más que esto
MIN_HIDDEN, MIN_GENERATED = 12, 16


def _norm_ref(p):
    fn = ast.parse(p.ref_source()).body[0]
    fn.name = "f"
    return ast.dump(fn)


def _shown(s, prompt):
    """s aparece en el prompt como literal completo, no como trozo de otro (\"0, 1.7\" está dentro de \"70, 1.75\")."""
    return re.search(r"(?<![\w.])" + re.escape(s) + r"(?![\w.])", prompt) is not None


def check_problem(p, cfg=None, mutation=True, timeout=2.0):
    """Lista de errores (vacía = válido) y dict de métricas del problema."""
    from rewards.reward import unsafe
    errs, info = [], {}
    if p.split not in SPLITS:
        errs.append(f"split inválido: {p.split}")
    if p.category not in CATEGORIES:
        errs.append(f"categoría inválida: {p.category}")
    if not p.family or not isinstance(p.family, str):
        errs.append("sin familia")
    if p.level not in range(5):
        errs.append(f"nivel inválido: {p.level}")
    if not p.version:
        errs.append("sin versión")
    if p.category == "optimizacion" and (p.slow is None or not p.large):
        errs.append("optimización sin versión lenta o sin casos grandes")

    vis, hid, gen = p.visible(), p.hidden(), p.generated(0, 24)
    info.update(visible=len(vis), hidden=len(hid), generated=len(gen), edge=len(p.edge))
    if not vis or len(hid) < MIN_HIDDEN or len(gen) < MIN_GENERATED or not p.edge:
        errs.append(f"tests insuficientes: visibles={len(vis)} ocultos={len(hid)} generados={len(gen)} bordes={len(p.edge)}")
    # ocultos: el conjunto fijo tal cual. Generados: lo que importa es la distribución del generador (el reward saca
    # casos nuevos con otra semilla en cada step), y con 24 muestras un generador 50/50 da >= 62 % un tercio de las
    # veces; con 200 el error es de +-3.5 puntos.
    info["baseline"] = round(max(baseline(hid), baseline(p.generated(1, 200))), 3)
    if info["baseline"] > MAX_BASELINE:
        errs.append(f"salidas poco diversas: un predictor constante acierta {info['baseline']:.0%}")

    again = [p._case(c.args, strict=False) for c in hid + gen]  # determinismo: misma salida en una 2.ª llamada
    if any(a is None or (a.exp, a.exc) != (c.exp, c.exc) for a, c in zip(again, hid + gen)):
        errs.append("referencia no determinista")

    allowed = set(cfg.sandbox.allowed_imports) if cfg else None
    if allowed is not None and unsafe(ast.parse(p.ref_source()), allowed):
        errs.append("la referencia usa algo que el scorer marcaría como unsafe")
    ok = repair.run_batch([(p.ref_source(), p.entry)], vis + hid + gen, timeout)[0]
    if not ok["ok"]:
        errs.append(f"la referencia falla su caso {ok['first']} ({ok['why']})")

    hidden_keys = {_key(c.args) for c in hid}
    if hidden_keys & {_key(c.args) for c in vis}:
        errs.append("un caso oculto coincide con un ejemplo visible del prompt")
    prompt = p.prompt()
    leaks = [c.args for c in hid if len(_key(c.args)) >= 8 and _shown(repr(c.args)[1:-1], prompt)]
    if leaks:
        errs.append(f"entradas ocultas visibles en el prompt: {str(leaks[0])[:60]}")
    for seed in (0, 1, 2):
        pool = p.feedback_cases(seed) + p.timeout_cases(seed)
        if hidden_keys & {_key(c.args) for c in pool}:
            errs.append("el pool de feedback contiene una entrada oculta")
            break

    if p.slow is not None and p.large:
        kinds = p.hidden_kinds()
        big = [c for c, k in zip(hid, kinds) if k == "large"]
        small = [c for c, k in zip(hid, kinds) if k != "large"] + vis
        src = textwrap.dedent(inspect.getsource(p.slow))
        r_small, r_big = repair.run_batch([(src, p.slow.__name__)], small, timeout)[0], \
            repair.run_batch([(src, p.slow.__name__)], big, timeout)[0]
        if not r_small["ok"]:
            errs.append("la versión lenta de referencia es incorrecta en casos pequeños")
        if r_big["ok"]:
            errs.append("los casos grandes no detectan la solución O(n^2)")
        info["slow_rejected"] = not r_big["ok"]

    if mutation:
        muts = repair.mutants(p.ref_source())
        kill = repair.killed(p, [c for _, c in muts], timeout) if muts else []
        surv = [c for (_, c), k in zip(muts, kill) if not k]
        # superviviente que también pasa 200 casos aleatorios extra (otra semilla): probablemente equivalente
        extra = repair.run_batch([(c, p.entry) for c in surv], p.generated(7, 200), repair.SMALL_TIMEOUT) if surv else []
        equiv = sum(r["ok"] for r in extra)
        info.update(mutants=len(muts), killed=sum(kill), equivalent=equiv,
                    mutation_raw=round(sum(kill) / len(kill), 3) if kill else None)
        denom = len(kill) - equiv
        info["mutation_score"] = round(sum(kill) / denom, 3) if denom > 0 else None
        if denom >= 5 and sum(kill) / denom < MIN_MUTATION:
            errs.append(f"mutation score bajo: {sum(kill)}/{denom} (sin {equiv} probablemente equivalentes)")
    return errs, info


def check_global(probs):
    errs = []
    fam = defaultdict(set)
    for p in probs:
        fam[p.family].add(p.split)
    errs += [f"la familia {f} está en varios splits: {sorted(s)}" for f, s in fam.items() if len(s) > 1]
    for what, key in (("referencia", _norm_ref), ("prompt", lambda p: p.prompt()), ("firma+doc", lambda p: (p.sig, p.doc))):
        seen = {}
        for p in probs:
            k = key(p)
            if k in seen:
                errs.append(f"{what} duplicada: {seen[k]} y {p.id}")
            seen[k] = p.id
    return errs


def similar_pairs(probs, thr_ref=0.9, thr_doc=0.85):
    """Pares de problemas de splits DISTINTOS con referencia (AST normalizado) o enunciado casi iguales: posible
    fuga entre splits que las familias no recogen (una variante no declarada). Es un aviso, no un error."""
    import difflib
    ref = {p.id: _norm_ref(p) for p in probs}
    doc = {p.id: " ".join(p.doc.lower().split()) for p in probs}

    def sim(x, y, thr):
        m = difflib.SequenceMatcher(None, x, y, autojunk=False)
        return m.ratio() if m.real_quick_ratio() >= thr and m.quick_ratio() >= thr else 0.0
    ps, out = sorted(probs, key=lambda p: p.id), []
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            if a.split == b.split:
                continue
            r1, r2 = sim(ref[a.id], ref[b.id], thr_ref), sim(doc[a.id], doc[b.id], thr_doc)
            if r1 >= thr_ref or r2 >= thr_doc:
                out.append((a.id, b.id, round(r1, 3), round(r2, 3)))
    return out


def _safe_check(p, cfg, mutation):
    """Un problema que revienta al construir sus casos es un error de ESE problema, no de toda la validación."""
    try:
        return check_problem(p, cfg, mutation)
    except Exception as e:  # noqa: BLE001
        return [f"excepción al validar: {e!r}"[:200]], {}


def validate(probs, cfg=None, mutation=True, workers=4):
    """{id: (errores, info)} + errores globales. Los problemas core sólo avisan (baseline intacto)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip([p.id for p in probs], ex.map(lambda p: _safe_check(p, cfg, mutation), probs)))
    return res, check_global(probs)


def main(argv=None):
    from common import load_config
    ap = argparse.ArgumentParser(description="Valida el dataset de problemas")
    ap.add_argument("--sets", nargs="*", default=["core", "catalog"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--no-mutation", action="store_true")
    a = ap.parse_args(argv)
    probs = select(ids=a.ids, sets=tuple(a.sets)) if a.ids else [p for p in ALL.values() if p.source in a.sets]
    missing = sorted(set(a.ids or []) - {p.id for p in probs})
    if missing:
        print("ids desconocidos (o fuera de --sets):", ", ".join(missing))
        return 2
    res, glob = validate(probs, load_config(), not a.no_mutation)
    bad = 0
    for p in sorted(probs, key=lambda p: (p.source, p.split, p.level, p.id)):
        errs, info = res[p.id]
        hard = errs and p.source != "core"
        bad += bool(hard)
        mark = "ERROR" if hard else "aviso" if errs else "ok"
        print(f"[{mark:5s}] {p.source:7s} {p.split:5s} L{p.level} {p.category:14s} {p.id:26s} "
              f"mut={info.get('mutation_score')} base={info.get('baseline')} {'; '.join(errs)}")
    for e in glob:
        print("[ERROR] global:", e)
    for x, y, r1, r2 in similar_pairs(probs):
        print(f"[aviso] parecidos entre splits: {x} ~ {y} (referencia {r1}, enunciado {r2})")
    print(f"{len(probs)} problemas, {bad} con errores, {len(glob)} errores globales")
    return 1 if bad or glob else 0


if __name__ == "__main__":
    sys.exit(main())
