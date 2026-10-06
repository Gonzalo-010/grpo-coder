"""Auto-mejora encima del bucle GRPO (python train.py improve).

improve.curriculum: v1 = el comportamiento anterior (por defecto: peso p(1-p), los niveles se abren y no se cierran).
                    v2 = currículo medible (experimental, se activa con improve.curriculum: v2):
  * EMA de acierto por problema y por nivel, observaciones por problema y val pass@1 por nivel (observe_eval).
  * Un nivel está DOMINADO si su EMA media >= master_ema, >= 80 % de sus problemas tienen min_obs observaciones y,
    si hay problemas val de ese nivel, su val pass@1 >= master_val. La frontera es el último nivel abierto; se abre
    el siguiente cuando la frontera queda dominada.
  * Repaso: al menos `rehearsal` (15 %) del batch viene de niveles dominados; si el val de un nivel dominado cae más
    de recovery_drop desde su mejor valor, ese repaso sube a `recovery` (25 %) hasta que se recupera.
  * Si la frontera no progresa (acierto medio < stall_rate en las últimas stall_window observaciones), su cuota baja
    a frontier_cap (20 %) y el resto va a niveles abiertos no dominados.
  * Hard-example mining: w = max(0.05, p(1-p)) * (1 + 0.5*casi + 0.5*fix_fail + 0.25*repetido), con un tope de 3x la
    media del pool. casi = EMA de muestras que pasan los visibles pero no resuelven; fix_fail = EMA de grupos de
    reparación sin ninguna corrección; repetido = EMA de fallos con un AST ya visto en ese problema.
Reparación: prompts con un intento fallido y su feedback (improve.feedback: basic | counterexample) del step actual
y de un buffer (incluidos los de `train.py fix`); más improve.synthetic_per_step grupos de reparación sintética
(problems/repair.py, sólo problemas train). Estado versionado en improve.json (migra el formato v1; improve.reset
lo reinicia limpio).
"""
import ast, glob, hashlib, json, math, os, random

from common import ROOT, opt
from rewards.feedback import build as build_feedback
from rollouts.collect import code_key, collect
from rollouts.fix import prompt_for

EMA, MIN_OBS, UNLOCK, FLOOR, BUFFER, VERSION = 0.3, 2, 0.5, 0.05, 256, 2


def _key(pid, code):
    try:
        norm = ast.dump(ast.parse(code))  # mismo programa con otro formato/comentarios = mismo fallo
    except (SyntaxError, ValueError, RecursionError):
        norm = " ".join(code.split())
    return hashlib.sha1(f"{pid}\n{norm}".encode()).hexdigest()[:16]


def _ema(old, x, first):
    return x if first else (1 - EMA) * old + EMA * x


class Improver:
    def __init__(self, run_dir, probs, cfg, fix_root=None):
        self.path, self.cfg = os.path.join(run_dir, "improve.json"), cfg
        self.mode = opt(cfg, "improve.curriculum", "v1")
        self.by_id = {p.id: p for p in probs}
        self.levels = sorted({p.level for p in probs})
        self.rate, self.obs, self.open, self.buffer = {}, {}, 1, []
        self.near, self.fix_fail, self.rep, self.recent = {}, {}, {}, {}
        self.hist, self.val, self.best_val, self.mastered, self.recovering = {}, {}, {}, set(), set()
        self.stalled, self.last = False, {}
        reset = bool(opt(cfg, "improve.reset", False))
        if os.path.exists(self.path) and not reset:
            try:
                with open(self.path) as f:
                    d = json.load(f)
                self._load(d)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                print(f"      [improve] {self.path} ilegible ({e!r}): currículo y buffer empiezan de cero")
                self.__init_empty()
        elif reset and os.path.exists(self.path):
            print("      [improve] improve.reset=true: estado del currículo reiniciado")
        self._remember(self._from_fix_runs(fix_root or os.path.join(ROOT, "runs", "fix")))

    def __init_empty(self):
        self.rate, self.obs, self.open, self.buffer = {}, {}, 1, []
        self.near, self.fix_fail, self.rep, self.recent = {}, {}, {}, {}
        self.hist, self.val, self.best_val, self.mastered, self.recovering = {}, {}, {}, set(), set()

    def _load(self, d):
        """v1 (sin 'version': rate/obs/open_level/buffer) se migra; v2 se carga entero."""
        self.rate = {k: float(v) for k, v in d["rate"].items()}
        self.obs = {k: int(v) for k, v in d["obs"].items() if k in self.rate}  # obs sin tasa: update() fallaría
        self.buffer = [b for b in d["buffer"] if isinstance(b, dict) and {"pid", "code", "feedback", "key"} <= b.keys()]
        # por valor, no por posición: al reanudar con otro --levels la lista de niveles cambia
        self.open = max(1, sum(lv <= d["open_level"] for lv in self.levels))
        if d.get("version", 1) >= 2:
            self.near = {k: float(v) for k, v in d.get("near", {}).items()}
            self.fix_fail = {k: float(v) for k, v in d.get("fix_fail", {}).items()}
            self.rep = {k: float(v) for k, v in d.get("rep", {}).items()}
            self.recent = {k: list(v)[-32:] for k, v in d.get("recent", {}).items()}
            self.hist = {int(k): [float(x) for x in v] for k, v in d.get("hist", {}).items()}
            self.val = {int(k): [float(x) for x in v] for k, v in d.get("val", {}).items()}
            self.best_val = {int(k): float(v) for k, v in d.get("best_val", {}).items()}
            self.mastered = {int(x) for x in d.get("mastered", [])}
            self.recovering = {int(x) for x in d.get("recovering", [])}
            self.stalled = bool(d.get("stalled", False))
        else:
            print("      [improve] improve.json v1 migrado a v2 (EMA/obs/nivel/buffer conservados)")

    # ------------------------------------------------------------------ buffer de fallos
    def _from_fix_runs(self, root):
        out = []
        for path in sorted(glob.glob(os.path.join(root, "*", "attempts.jsonl"))):
            with open(path) as f:
                for line in f:
                    try:
                        a = json.loads(line)
                    except ValueError:
                        continue
                    if (isinstance(a, dict) and not a.get("solved") and a.get("feedback")
                            and isinstance(a.get("code"), str) and a.get("problem") in self.by_id):
                        out.append(dict(pid=a["problem"], code=a["code"], feedback=a["feedback"]))
        return out

    def _remember(self, fails):
        seen = {b["key"] for b in self.buffer}
        for x in fails:
            k = _key(x["pid"], x["code"])
            if k not in seen:
                seen.add(k)
                self.buffer.append(dict(x, key=k))
        self.buffer = self.buffer[-BUFFER:]

    # ------------------------------------------------------------------ pesos y muestreo
    def weight(self, pid):
        p = self.rate.get(pid, 0.5)  # sin observar: máxima prioridad para explorarlo
        base = max(FLOOR, p * (1 - p))
        if self.mode == "v1":
            return base
        return base * (1 + 0.5 * self.near.get(pid, 0.0) + 0.5 * self.fix_fail.get(pid, 0.0)
                       + 0.25 * self.rep.get(pid, 0.0))

    def weights(self, pool):
        """Pesos de hard mining con tope de 3x la media (v2)."""
        w = [self.weight(p.id) for p in pool]
        if self.mode != "v1" and w:
            cap = 3 * sum(w) / len(w)
            w = [min(x, cap) for x in w]
        return w

    def _draw(self, pool, k):
        pool, w, out = list(pool), self.weights(pool), []
        while pool and len(out) < k:
            i = random.choices(range(len(pool)), weights=w)[0]
            out.append(pool.pop(i))
            w.pop(i)
        return out

    def quotas(self, k):
        """(frontera, repaso, consolidación) para un batch de k problemas (v2)."""
        open_lv = self.levels[:self.open]
        front = open_lv[-1]
        reh_lv = [lv for lv in open_lv if lv in self.mastered and lv != front]
        mid_lv = [lv for lv in open_lv if lv not in self.mastered and lv != front]
        reh = 0.0
        if reh_lv:
            reh = float(opt(self.cfg, "improve.rehearsal", 0.15))
            if self.recovering & set(reh_lv):
                reh = max(reh, float(opt(self.cfg, "improve.recovery", 0.25)))
        n_reh = 0
        if reh_lv:  # con k pequeño la frontera nunca se queda sin muestras (k=1: repaso con probabilidad reh)
            n_reh = min(math.ceil(reh * k), k - 1) if k > 1 else int(random.random() < reh)
        n_front = k - n_reh
        if self.stalled and (reh_lv or mid_lv):
            n_front = min(n_front, max(1, math.floor(float(opt(self.cfg, "improve.frontier_cap", 0.2)) * k)))
        return n_front, n_reh, k - n_front - n_reh, front, reh_lv, mid_lv

    def sample(self, probs, k):
        if self.mode == "v1":
            return self._draw([p for p in probs if p.level in self.levels[:self.open]], k)
        n_front, n_reh, n_mid, front, reh_lv, mid_lv = self.quotas(k)
        f_pool = [p for p in probs if p.level == front]
        r_pool = [p for p in probs if p.level in reh_lv]
        m_pool = [p for p in probs if p.level in mid_lv]
        out = self._draw(f_pool, n_front) + self._draw(r_pool, n_reh) + self._draw(m_pool, n_mid)
        rest = [p for p in probs if p.level in self.levels[:self.open] and p not in out]
        out += self._draw(rest, k - len(out))  # rellenar si algún pool se quedó corto
        self.last.update(quota_front=n_front / k, quota_reh=n_reh / k, quota_mid=n_mid / k)
        return out

    # ------------------------------------------------------------------ actualización
    def update(self, batch, results):
        for p, rs in zip(batch, results):
            n = self.obs.get(p.id, 0)
            f = sum(r.solved for _, r in rs) / max(1, len(rs))
            self.rate[p.id] = f if n == 0 else (1 - EMA) * self.rate[p.id] + EMA * f
            self.obs[p.id] = n + 1
            if self.mode == "v1":
                continue
            near = sum(getattr(r, "pv", 0.0) == 1.0 and not r.solved for _, r in rs) / max(1, len(rs))
            self.near[p.id] = _ema(self.near.get(p.id, 0.0), near, n == 0)
            seen = self.recent.setdefault(p.id, [])
            fails = [code_key(s.code) for s, r in rs if not r.solved and isinstance(getattr(s, "code", None), str)]
            reps = sum(1 for kk in fails if kk[:200] in seen)
            self.rep[p.id] = _ema(self.rep.get(p.id, 0.0), reps / len(fails) if fails else 0.0, n == 0)
            seen.extend(kk[:200] for kk in fails)
            del seen[:-32]
        if self.mode == "v1":
            return self._update_v1()
        front = self.levels[self.open - 1]
        fb = [sum(r.solved for _, r in rs) / max(1, len(rs)) for p, rs in zip(batch, results) if p.level == front]
        if fb:
            h = self.hist.setdefault(front, [])
            h.extend(fb)
            del h[:-200]
        self._refresh()

    def _update_v1(self):
        while self.open < len(self.levels):
            opened = [pid for pid, p in self.by_id.items() if p.level in self.levels[:self.open]]
            if any(self.obs.get(pid, 0) < MIN_OBS for pid in opened):
                break
            if sum(self.rate[pid] for pid in opened) / len(opened) < UNLOCK:
                break
            self.open += 1
            print(f"      [improve] se abre el nivel L{self.levels[self.open - 1]}")

    def level_ema(self, lv):
        ids = [pid for pid, p in self.by_id.items() if p.level == lv]
        return sum(self.rate.get(pid, 0.0) for pid in ids) / len(ids) if ids else 0.0

    def _is_mastered(self, lv):
        ids = [pid for pid, p in self.by_id.items() if p.level == lv]
        if not ids:
            return False
        enough = sum(self.obs.get(pid, 0) >= int(opt(self.cfg, "improve.min_obs", 3)) for pid in ids) / len(ids) >= 0.8
        ok_val = not self.val.get(lv) or self.val[lv][-1] >= float(opt(self.cfg, "improve.master_val", 0.5))
        return enough and ok_val and self.level_ema(lv) >= float(opt(self.cfg, "improve.master_ema", 0.7))

    def _refresh(self):
        for lv in self.levels[:self.open]:
            if self._is_mastered(lv):
                self.mastered.add(lv)
        while self.open < len(self.levels) and self.levels[self.open - 1] in self.mastered:
            self.open += 1
            print(f"      [improve] frontera dominada: se abre el nivel L{self.levels[self.open - 1]}")
        front = self.levels[self.open - 1]
        w = int(opt(self.cfg, "improve.stall_window", 20))
        h = self.hist.get(front, [])
        self.stalled = len(h) >= w and sum(h[-w:]) / w < float(opt(self.cfg, "improve.stall_rate", 0.05))

    def observe_eval(self, split, rep):
        """Tras una evaluación por split: val pass@1 por nivel -> dominio y recuperación de niveles."""
        if self.mode == "v1" or split != "val":
            return
        drop = float(opt(self.cfg, "improve.recovery_drop", 0.15))
        for lv_s, v in rep.get("by_level", {}).items():
            lv, x = int(lv_s), v.get("pass@1")
            if x is None:
                continue
            self.val.setdefault(lv, []).append(float(x))
            del self.val[lv][:-5]
            best = self.best_val.get(lv, x)
            if lv in self.mastered and x < best - drop:
                if lv not in self.recovering:
                    print(f"      [improve] val de L{lv} cae {best:.2f} -> {x:.2f}: más repaso de ese nivel")
                self.recovering.add(lv)
            elif lv in self.recovering and x >= best - 0.05:
                self.recovering.discard(lv)
            self.best_val[lv] = max(best, x)
        self._refresh()

    # ------------------------------------------------------------------ reparación
    def repair(self, gen, batch, results, seed):
        """Grupos de reparación para este step: [(clave_grupo, Sample, Result)]."""
        k = int(self.cfg.improve.repair_per_step)
        mode = "basic" if self.mode == "v1" else opt(self.cfg, "improve.feedback", "counterexample")
        fresh = []
        for p, rs in zip(batch, results):
            bad = [(s, r) for s, r in rs if not r.solved]
            if bad:
                s, r = max(bad, key=lambda x: x[1].reward)  # el fallo más cercano a correcto de cada problema
                fb = build_feedback(p, s.code, r, self.cfg, mode, seed, getattr(r, "truncated", False))[0]
                fresh.append(dict(pid=p.id, code=s.code, feedback=fb))
        picks = random.sample(fresh, min(len(fresh), k - k // 2))
        now = {_key(x["pid"], x["code"]) for x in fresh}
        old = [b for b in self.buffer if b["key"] not in now and b["pid"] in self.by_id
               and self.by_id[b["pid"]].level in self.levels[:self.open]]
        picks += random.sample(old, min(len(old), k // 2))
        self._remember(fresh)
        out = []
        if picks:
            probs = [self.by_id[x["pid"]] for x in picks]
            prompts = [prompt_for(gen, self.cfg, p, x["code"], x["feedback"]) for p, x in zip(probs, picks)]
            res = collect(gen, probs, self.cfg.num_rollouts, self.cfg, seed, prompts=prompts)
            out = [(f"{p.id}#fix{j}", s, r) for j, (p, rs) in enumerate(zip(probs, res)) for s, r in rs]
            if self.mode != "v1":
                rep = n = 0
                for p, x, rs in zip(probs, picks, res):
                    failed = not any(r.solved for _, r in rs)
                    self.fix_fail[p.id] = _ema(self.fix_fail.get(p.id, 0.0), float(failed), p.id not in self.fix_fail)
                    rep += sum(code_key(s.code) == code_key(x["code"]) for s, _ in rs)
                    n += len(rs)
                self.last.update(fix_repeat=rep / n if n else None)
        syn = int(opt(self.cfg, "improve.synthetic_per_step", 1)) if self.mode != "v1" else 0
        if syn > 0:
            out += self._synthetic(gen, seed, syn)
        return out

    def _synthetic(self, gen, seed, m):
        from problems import repair
        from rollouts.generate import build_prompt
        base = [p for p in self.by_id.values() if p.level in self.levels[:self.open] and getattr(p, "split", "train") == "train"]
        tasks = repair.tasks(base, per_problem=3, seed=int(self.cfg.seed))
        if not tasks:
            return []
        picks = random.sample(tasks, min(m, len(tasks)))
        prompts = [build_prompt(gen.tok, self.cfg.prompt.system, t.prompt(), self.cfg.prompt.prefill_fence) for t in picks]
        res = collect(gen, picks, self.cfg.num_rollouts, self.cfg, seed, prompts=prompts)
        out = [(f"{t.id}#syn", s, r) for t, rs in zip(picks, res) for s, r in rs]
        self.last.update(syn_n=len(out), syn_solved=sum(r.solved for _, _, r in out))
        return out

    # ------------------------------------------------------------------ estado
    def metrics(self):
        """Estado del currículo para metrics.jsonl (panel)."""
        m = dict(open_level=self.levels[self.open - 1], curriculum=self.mode)
        if self.mode != "v1":
            m.update(mastered=sorted(self.mastered), recovering=sorted(self.recovering), stalled=self.stalled,
                     ema_by_level={str(lv): round(self.level_ema(lv), 4) for lv in self.levels[:self.open]},
                     fix_fail_mean=round(sum(self.fix_fail.values()) / len(self.fix_fail), 4) if self.fix_fail else None,
                     **self.last)
        return m

    def save(self):
        tmp = self.path + ".tmp"
        d = dict(version=VERSION, curriculum=self.mode, rate=self.rate, obs=self.obs,
                 open_level=self.levels[self.open - 1], buffer=self.buffer, near=self.near, fix_fail=self.fix_fail,
                 rep=self.rep, recent=self.recent, hist={str(k): v for k, v in self.hist.items()},
                 val={str(k): v for k, v in self.val.items()}, best_val={str(k): v for k, v in self.best_val.items()},
                 mastered=sorted(self.mastered), recovering=sorted(self.recovering), stalled=self.stalled)
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.path)
