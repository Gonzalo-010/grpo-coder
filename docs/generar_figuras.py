"""Figuras de docs/INVESTIGACION.md a partir de los resultados en runs/ (reproducible).

python docs/generar_figuras.py [--runs runs] [--out docs/figuras]

Necesita matplotlib (pip install matplotlib); no toca el entrenamiento. Lee runs/exp/main (report.json, eval_val.json,
metrics.jsonl de cada brazo), runs/humaneval/*/humaneval.json, runs/genbench/*.json y runs/agent_bench/*/bench.json.
Lo que falte se omite (la figura correspondiente no se genera).

Estilo: superficie clara, tinta en grises, rejilla de una línea fina, marcas finas (líneas de 2 px, marcadores de
8 px con anillo del color de fondo, barras con el extremo de datos redondeado y separación de 2 px entre segmentos),
paleta categórica validada (azul, naranja, aguamarina, amarillo; contraste < 3:1 en los dos últimos -> siempre con
etiqueta directa y tabla en el documento) y gris para la referencia.
"""
from __future__ import annotations

import argparse, glob, json, os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categórica validada (orden fijo)
REF = "#a8a7a1"  # referencia / modelo base (gris de énfasis)
DPI = 200
PX = 72 / 96  # 1 px CSS en puntos


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if n in names:
            return n
    return "sans-serif"


plt.rcParams.update({
    "font.family": _font(), "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 10, "legend.frameon": False, "legend.fontsize": 8.5,
})


def fmt(x, d=2):
    """Coma decimal y redondeo del valor decimal mostrado (0,2925 -> 0,293), no del binario (0,29249...)."""
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q:.{d}f}".replace(".", ",").replace("-", "−")


def comma_axis(ax, axis="x", d=2):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: fmt(v, d))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def clean(ax, grid="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1 * PX)
    ax.tick_params(length=0, pad=4)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1 * PX, linestyle="-")
        ax.set_axisbelow(True)


def label(ax, x, y, s, **kw):
    """Etiqueta de valor sobre un fondo del color de la superficie, para que la rejilla no la atraviese."""
    kw.setdefault("va", "center")
    kw.setdefault("color", INK2)
    return ax.text(x, y, s, zorder=6, bbox=dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none"), **kw)


def _px(ax):
    """Píxeles CSS (1/96 in) por unidad de datos en x e y."""
    ax.figure.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    k = 96 / ax.figure.dpi
    return abs(x1 - x0) * k, abs(y1 - y0) * k


def hbar(ax, y, left, width, height, color, round_end=True, r_px=4, max_px=22):
    """Barra horizontal: base recta, extremo de datos redondeado (4 px) si round_end; grosor <= max_px."""
    if width <= 0:
        return
    pxx, pxy = _px(ax)
    height = min(height, max_px / pxy)
    rx, ry = min(r_px / pxx, width / 2), min(r_px / pxy, height / 2)
    x0, x1, y0, y1 = left, left + width, y - height / 2, y + height / 2
    if not round_end:
        verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    else:
        verts = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1),
                 (x0, y1), (x0, y0)]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
                 Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def dot(ax, x, y, color, filled=True, size=10, z=5):
    """Punto de >= 8 px con el mismo diámetro relleno o hueco. Relleno: disco del color de la serie con un anillo de
    2 px del color de fondo por fuera (lo separa de la línea que cruza). Hueco: borde de 2 px del color de la serie."""
    if filled:
        ax.plot([x], [y], marker="o", markersize=(size + 4) * PX, linestyle="none", zorder=z - 0.5,
                markerfacecolor=SURFACE, markeredgecolor=SURFACE, markeredgewidth=0)
        ax.plot([x], [y], marker="o", markersize=size * PX, linestyle="none", zorder=z, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=2 * PX)
    else:
        ax.plot([x], [y], marker="o", markersize=size * PX, linestyle="none", zorder=z, markerfacecolor=SURFACE,
                markeredgecolor=color, markeredgewidth=2 * PX)


def header(fig, title, handles=None, ncols=None, top=None):
    """Título arriba a la izquierda y, debajo, la leyenda en una fila (nunca encima del gráfico ni del título)."""
    h_in = fig.get_size_inches()[1]
    t_y = 1 - 0.12 / h_in
    fig.suptitle(title, x=0.01, y=t_y, ha="left", va="top", fontsize=10.5, fontweight="semibold", color=INK)
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, t_y - 0.30 / h_in),
                   ncols=ncols or len(handles), fontsize=8.5, handletextpad=0.5, columnspacing=1.6,
                   handlelength=1.6, handleheight=0.8)
    fig.tight_layout(rect=(0, 0, 1, top if top is not None else 1 - (0.5 if handles else 0.3) / h_in))


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ", path)
    return path


# ------------------------------------------------------------------------------------------------------------ datos
def load(runs):
    d = {"runs": runs}
    main = os.path.join(runs, "exp", "main")
    rep = _json(os.path.join(main, "report.json")) or {}
    d["report"] = rep
    d["arms"] = rep.get("arms", {})
    d["base"] = {("0.5B" if "0.5B" in k else "1.5B"): (v.get("val") or {}) for k, v in (rep.get("base") or {}).items()}
    d["cmp"] = rep.get("comparisons", {})
    d["metrics"] = {}
    for a in d["arms"]:
        rows = _jsonl(os.path.join(main, a, "metrics.jsonl"))
        d["metrics"][a] = [r for r in rows if "loss" in r and isinstance(r.get("step"), int) and r["step"] >= 0]
    d["he"] = {}
    for p in sorted(glob.glob(os.path.join(runs, "humaneval", "*", "humaneval.json"))):
        d["he"][os.path.basename(os.path.dirname(p))] = _json(p)
    gb = sorted(glob.glob(os.path.join(runs, "genbench", "*.json")))
    d["genbench"] = [_json(p) for p in gb]
    d["bench"] = [_json(p) for p in sorted(glob.glob(os.path.join(runs, "agent_bench", "*", "bench.json")))]
    return d


def _json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _jsonl(p):
    out = []
    try:
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


LABEL = {"A": "A · GRPO, lr 1e-5", "U2": "U2 · 2 pasadas por lote", "A_lr2": "A_lr2 · lr 2e-5", "lr5": "lr5 · lr 5e-5",
         "Abat": "Abat · ventaja std del lote", "DS": "DS · muestreo dinámico", "BIN": "BIN · recompensa binaria",
         "RND": "RND · recompensa aleatoria", "B": "B · crecimiento +4 bloques", "KD": "KD · destilación del 1.5B",
         "RFT": "RFT · SFT con soluciones propias", "KD_RL": "KD_RL · KD + GRPO", "C": "C · 1.5B + GRPO"}
ORDER_05 = ["A", "U2", "A_lr2", "lr5", "Abat", "DS", "BIN", "RND", "B", "KD", "RFT", "KD_RL"]


# --------------------------------------------------------------------------------------------------------- figuras
def fig_calidad(d, out):
    """pass@1 en val (n=16) con IC 95 % por brazo; 0.5B en azul, 1.5B en naranja; huecos = modelos base."""
    rows = [("base 0.5B (sin entrenar)", d["base"].get("0.5B"), S1, False)]
    for a in ORDER_05:
        f = ((d["arms"].get(a) or {}).get("final") or {}).get("val")
        if f and f.get("pass@1") is not None:
            rows.append((LABEL[a], f, S1, True))
    rows.append(("base 1.5B (sin entrenar)", d["base"].get("1.5B"), S2, False))
    f = ((d["arms"].get("C") or {}).get("final") or {}).get("val")
    if f:
        rows.append((LABEL["C"], f, S2, True))
    rows = [r for r in rows if r[1] and r[1].get("pass@1") is not None]
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.2))
    clean(ax, "x")
    ys = list(range(len(rows)))[::-1]
    for y, (name, f, c, trained) in zip(ys, rows):
        lo, hi = f.get("ci_pass1") or (None, None)
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=c, linewidth=2 * PX, solid_capstyle="round", zorder=4)
        dot(ax, f["pass@1"], y, c, filled=trained)
        label(ax, (hi if hi is not None else f["pass@1"]) + 0.012, y, fmt(f["pass@1"], 3), va="center", ha="left",
                fontsize=8.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    b05 = d["base"].get("0.5B", {}).get("pass@1")
    if b05:
        ax.axvline(b05, color=REF, linewidth=1 * PX, zorder=1)
    ax.set_xlim(0, max(0.75, max((r[1].get("ci_pass1") or [0, 0])[1] or 0 for r in rows) + 0.08))
    comma_axis(ax, "x", 1)
    ax.set_xlabel("pass@1 en val (50 problemas, 16 muestras por problema)")
    header(fig, "Figura 2. Calidad final de cada brazo (IC 95 % bootstrap por problema)", _size_handles())
    return save(fig, out, "fig2_calidad_val.png")


def _size_handles():
    return [Line2D([], [], marker="o", color=S1, linestyle="none", markersize=8 * PX, label="0.5B"),
            Line2D([], [], marker="o", color=S2, linestyle="none", markersize=8 * PX, label="1.5B"),
            Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=INK2, linestyle="none",
                   markersize=8 * PX, label="modelo base, sin entrenar (punto hueco)")]


def fig_forest(d, out):
    """Δpass@1 pareado contra la base 0.5B (brazos 0.5B); relleno = significativo (IC sin 0 y signos p<0,05)."""
    rows = []
    for a in ORDER_05:
        c = d["cmp"].get(f"{a}_vs_base_val")
        if c:
            m = c["metrics"]["pass@1"]
            rows.append((a, m, c.get("verdict") == "mejora"))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.3))
    clean(ax, "x")
    ys = list(range(len(rows)))[::-1]
    for y, (a, m, sig) in zip(ys, rows):
        lo, hi = m["ci"]
        ax.plot([lo, hi], [y, y], color=S1, linewidth=2 * PX, solid_capstyle="round", zorder=4)
        dot(ax, m["delta"], y, S1, filled=sig)
        label(ax, hi + 0.006, y, f"{'+' if m['delta'] >= 0 else ''}{fmt(m['delta'], 3)}  (+{m['wins']}/−{m['losses']}, "
                f"p={fmt(m['p_sign'], 3)})", va="center", ha="left", fontsize=8, color=INK2)
    ax.axvline(0, color=AXIS, linewidth=1 * PX, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[r[0]] for r in rows])
    ax.set_xlim(min(-0.05, min(r[1]["ci"][0] for r in rows) - 0.01), max(r[1]["ci"][1] for r in rows) + 0.11)
    comma_axis(ax, "x", 2)
    ax.set_xlabel("Δ pass@1 frente al modelo base 0.5B (pareado por problema, val)")
    header(fig, "Figura 3. ¿Mejora de verdad? Diferencia pareada con IC 95 % y test de signos",
           [Line2D([], [], marker="o", color=S1, linestyle="none", markersize=8 * PX,
                   label="significativo (IC sin 0 y p < 0,05)"),
            Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=S1, linestyle="none",
                   markersize=8 * PX, label="no significativo")])
    return save(fig, out, "fig3_delta_pareado.png")


def _passk(f):
    ks = [k for k in (1, 5, 10, 16) if f.get(f"pass@{k}") is not None]
    return ks, [f[f"pass@{k}"] for k in ks]


def fig_passk(d, out):
    """pass@k (k=1..16): el RL sube pass@1 pero no pass@16; el modelo mayor sube toda la curva."""
    fin = lambda a: ((d["arms"].get(a) or {}).get("final") or {}).get("val")
    panels = [("0.5B", [("base 0.5B", d["base"].get("0.5B"), REF), ("A_lr2", fin("A_lr2"), S1), ("B", fin("B"), S2),
                        ("KD_RL", fin("KD_RL"), S3)]),
              ("1.5B", [("base 1.5B", d["base"].get("1.5B"), REF), ("C", fin("C"), S4)])]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, (title, series) in zip(axs, panels):
        clean(ax, "y")
        live = [(n, f, c) for n, f, c in series if f]
        for j, (name, f, c) in enumerate(live):
            ks, vs = _passk(f)
            off = (j - (len(live) - 1) / 2) * 0.05  # pequeño desplazamiento: series con el mismo valor no se tapan
            xs = [i + off for i in range(len(ks))]
            ax.plot(xs, vs, color=c, linewidth=2 * PX, solid_joinstyle="round", solid_capstyle="round", zorder=4)
            for x, v in zip(xs, vs):
                dot(ax, x, v, c, size=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["pass@1", "pass@5", "pass@10", "pass@16"])
        ax.set_xlim(-0.35, 3.35)
        ax.set_title(title, fontsize=9.5)
        comma_axis(ax, "y", 1)
        # sin etiquetas al final: las líneas convergen en pass@16 y chocarían; identifica la leyenda
    axs[0].set_ylabel("tasa de acierto en val")
    axs[0].set_ylim(0, 0.9)
    header(fig, "Figura 4. ¿Afina o amplía? pass@k (probabilidad de acertar con k intentos)",
           [Line2D([], [], color=REF, linewidth=2 * PX, label="modelo base"),
            Line2D([], [], color=S1, linewidth=2 * PX, label="A_lr2"), Line2D([], [], color=S2, linewidth=2 * PX, label="B"),
            Line2D([], [], color=S3, linewidth=2 * PX, label="KD_RL"),
            Line2D([], [], color=S4, linewidth=2 * PX, label="C (1.5B)")])
    return save(fig, out, "fig4_pass_at_k.png")


def _rolling(xs, w=8):
    out = []
    for i in range(len(xs)):
        win = [x for x in xs[max(0, i - w + 1):i + 1] if x is not None]
        out.append(sum(win) / len(win) if win else None)
    return out


def fig_dinamica(d, out):
    """Recompensa de train (media móvil de 8 pasos), entropía y KL por paso: tres paneles, un eje cada uno."""
    series = [("A", REF), ("A_lr2", S1), ("U2", S2), ("lr5", S3), ("B", S4)]
    series = [(a, c) for a, c in series if d["metrics"].get(a)]
    if not series:
        return None
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.2))
    specs = [("reward", "recompensa media del lote (media móvil de 8 pasos)", 2), ("entropy", "entropía (nats/token)", 2),
             ("kl", "KL a la referencia (por token)", 3)]
    for ax, (key, title, dec) in zip(axs, specs):
        clean(ax, "y")
        for a, c in series:
            rows = d["metrics"][a]
            xs = [r["step"] for r in rows]
            ys = _rolling([r.get(key) for r in rows]) if key == "reward" else [r.get(key) for r in rows]
            if key != "reward":
                ys = _rolling(ys, 4)
            ax.plot(xs, ys, color=c, linewidth=2 * PX, solid_capstyle="round", zorder=4 if a != "A" else 3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("paso")
        comma_axis(ax, "y", dec)
    handles = [Line2D([], [], color=c, linewidth=2 * PX, label=LABEL[a]) for a, c in series]
    header(fig, "Figura 5. Dinámica del entrenamiento (40 pasos, mismos problemas por paso en todos los brazos)",
           handles)
    return save(fig, out, "fig5_dinamica.png")


def fig_coste(d, out):
    """Segundos medios por paso, por fase (barras apiladas horizontales, separación de 2 px)."""
    arms = [a for a in ["A", "U2", "A_lr2", "lr5", "Abat", "DS", "BIN", "RND", "B", "KD_RL", "C"]
            if len(d["metrics"].get(a) or []) >= 30]
    if not arms:
        return None
    parts = [("generación", S1), ("espera del sandbox", S2), ("actualización (ref + update)", S3), ("resto", S4)]
    vals = []
    for a in arms:
        rows = d["metrics"][a]
        mean = lambda k: sum((r.get(k) or 0) for r in rows) / len(rows)
        gen, wait = mean("t_gen"), mean("t_score_wait")
        upd = mean("t_ref") + mean("t_old") + mean("t_update")
        tot = sum((r.get("secs") or 0) - (r.get("t_growth") or 0) for r in rows) / len(rows)
        vals.append((gen, wait, upd, max(0.0, tot - gen - wait - upd), tot))
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(arms) + 1.3))
    clean(ax, "x")
    ax.set_xlim(0, max(v[-1] for v in vals) * 1.18)
    ys = list(range(len(arms)))[::-1]
    ax.set_ylim(-0.7, len(arms) - 0.3)
    pxx, _ = _px(ax)
    gap = 2 / pxx
    for y, a, v in zip(ys, arms, vals):
        left = 0.0
        segs = [x for x in v[:4]]
        last = max(i for i, x in enumerate(segs) if x > 0)
        for i, (x, (_, c)) in enumerate(zip(segs, parts)):
            if x <= 0:
                continue
            w = x - (gap if i < last else 0)
            hbar(ax, y, left, max(w, 0), 0.56, c, round_end=(i == last))
            left += x
        label(ax, v[-1] + 0.3, y, f"{fmt(v[-1], 1)} s", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[a] for a in arms])
    comma_axis(ax, "x", 0)
    ax.set_xlabel("segundos por paso (media de los 40 pasos; sin el tiempo de crecer)")
    header(fig, "Figura 6. Dónde se va el tiempo de cada paso",
           [Patch(facecolor=c, edgecolor="none", label=n) for n, c in parts])
    return save(fig, out, "fig6_coste_step.png")


def fig_generacion(d, out):
    """Generación: ms por paso de decodificación (48 secuencias) y tiempo de evaluar 850 muestras."""
    gb = next((g for g in reversed(d["genbench"]) if g), None)
    rows_a = []
    if gb:
        m = gb.get("modes", {})
        for key, name in (("eager_base", "sin LoRA (pesos base)"), ("eager_lora", "con LoRA sin fusionar"),
                          ("static_lora", "con LoRA + CUDA graphs")):
            if (m.get(key) or {}).get("ms_per_step"):
                rows_a.append((name, m[key]["ms_per_step"]))
    ev = []
    for size, lab in (("0.5B", "modelo base 0.5B (494 M)"), ("1.5B", "modelo base 1.5B (1,54 B)")):
        w = (d["base"].get(size) or {}).get("wall_s")
        if w:
            ev.append((lab, w))
    a1 = _json(os.path.join(d["runs"], "exp", "main", "A", "eval_val.inference1.json"))
    a2 = _json(os.path.join(d["runs"], "exp", "main", "A", "eval_val.json"))
    if a1 and a2:
        ev += [("A, LoRA fusionado", a1["wall_s"]), ("A, LoRA sin fusionar", a2["wall_s"])]
    if not rows_a and not ev:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 2.9))
    for ax, rows, unit, title, dec in ((axs[0], rows_a, "ms", "ms por paso de decodificación (48 secuencias, 0.5B)", 1),
                                       (axs[1], ev, "s", "segundos para evaluar val (850 muestras)", 0)):
        clean(ax, "x")
        if not rows:
            ax.set_visible(False)
            continue
        ax.set_xlim(0, max(v for _, v in rows) * 1.3)
        ys = list(range(len(rows)))[::-1]
        ax.set_ylim(-0.7, len(rows) - 0.3)
        for y, (name, v) in zip(ys, rows):
            hbar(ax, y, 0, v, 0.56, S1)
            label(ax, v * 1.02 + 0.5, y, f"{fmt(v, dec)} {unit}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([n for n, _ in rows])
        comma_axis(ax, "x", 0)
        ax.set_title(title, fontsize=9)
    header(fig, "Figura 7. La generación la limita la sobrecarga por paso, no el tamaño del modelo")
    return save(fig, out, "fig7_generacion.png")


HE_ORDER = [("base05", "base 0.5B", S1, False), ("A", "A", S1, True), ("A_lr2", "A_lr2", S1, True),
            ("B", "B", S1, True), ("KD_RL", "KD_RL", S1, True), ("base15", "base 1.5B", S2, False),
            ("C", "C (1.5B + GRPO)", S2, True)]


def fig_humaneval(d, out):
    rows = [(lab, d["he"][k], c, t) for k, lab, c, t in HE_ORDER if d["he"].get(k)]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.2))
    clean(ax, "x")
    ys = list(range(len(rows)))[::-1]
    for y, (name, f, c, trained) in zip(ys, rows):
        lo, hi = f.get("ci_pass@1") or (None, None)
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=c, linewidth=2 * PX, solid_capstyle="round", zorder=4)
        dot(ax, f["pass@1"], y, c, filled=trained)
        label(ax, (hi or f["pass@1"]) + 0.012, y, fmt(f["pass@1"], 3), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1)
    comma_axis(ax, "x", 1)
    ax.set_xlabel(f"pass@1 en HumanEval ({rows[0][1].get('problems')} problemas, "
                  f"{rows[0][1].get('n')} muestras por problema)")
    header(fig, "Figura 8. Transferencia a un benchmark externo (HumanEval)", _size_handles())
    return save(fig, out, "fig8_humaneval.png")


def fig_agente(d, out):
    """Éxito del agente (mini-repos con un bug) por modelo y configuración del bucle."""
    order = {(False, 0): (0, "v1: sin vuelta atrás ni paciencia"), (True, 2): (1, "vuelta atrás + paciencia 2"),
             (True, 0): (2, "vuelta atrás, sin paciencia (final)")}
    rows = []
    for b in d["bench"]:
        if not b or b.get("parse_version") != 2:  # tandas anteriores: pytest sin instalar / respuestas sin cabecera
            continue                              # descartadas por el parser (ver INVESTIGACION.md, apéndice A)
        model = "1.5B" if "1.5B" in (b.get("model") or "") else "0.5B"
        key = (bool(b.get("rollback")), int(b.get("patience") or 0))
        if key not in order:
            continue
        k, name = order[key]
        solved = sum(bool(r.get("solved")) for r in b["repos"])
        rows.append(((model == "1.5B", k), f"{model} · {name}", solved, b.get("n"), b.get("secs"), model))
    if not rows:
        return None
    rows.sort()
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.4))
    clean(ax, "x")
    ax.set_xlim(0, 1.2)
    ys = list(range(len(rows)))[::-1]
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for y, (_, name, s_, n, secs, model) in zip(ys, rows):
        hbar(ax, y, 0, max(s_ / n, 0.002), 0.56, S2 if model == "1.5B" else S1)
        label(ax, s_ / n + 0.02, y, f"{s_}/{n}  ·  {fmt(secs / 60, 1)} min", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    comma_axis(ax, "x", 1)
    ax.set_xlabel("fracción de repos arreglados (tests en verde sin tocar el fichero de tests)")
    header(fig, "Figura 9. Agente Generate → Test → Fix sobre 20 mini-repos con un bug",
           [Patch(facecolor=S1, edgecolor="none", label="0.5B"), Patch(facecolor=S2, edgecolor="none", label="1.5B")])
    return save(fig, out, "fig9_agente.png")


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(root, "runs"))
    ap.add_argument("--out", default=os.path.join(root, "docs", "figuras"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d = load(a.runs)
    for f in (fig_calidad, fig_forest, fig_passk, fig_dinamica, fig_coste, fig_generacion, fig_humaneval, fig_agente):
        try:
            f(d, a.out)
        except Exception as e:  # una figura que falla no tumba las demás
            print(f"  [aviso] {f.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
