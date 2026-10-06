"""Panel local: python app.py [--port 8501] [--host 127.0.0.1] [--config config.yaml] -> http://localhost:8501

Sólo librería estándar y una página sin dependencias externas. Escucha en 127.0.0.1: el panel lanza entrenamientos
y agentes sobre repos locales. Además exige Host local y JSON en los POST, así que una web cualquiera abierta en el
navegador no puede dispararlo.

GPU: un único trabajo a la vez (cola en un hilo). Mientras entrena un proceso, no se aceptan trabajos de GPU y el
modelo del panel se descarga para dejarle la VRAM.
"""
import argparse, json, math, os, queue, re, signal, subprocess, sys, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import yaml

from common import ROOT, is_training, load_config, read_config, training_pids

UI_DIR = os.path.join(ROOT, "runs", "ui")
# lo que se puede cambiar desde el panel, con su rango válido (fuera de él el entrenamiento fallaría más tarde:
# every_steps=0 divide por cero, top_p>1 rompe la generación, 1 rollout deja todas las ventajas a 0).
# Las marcadas como estado del modelo cambian config_hash -> run nuevo.
EDITABLE = {"num_rollouts": (int, 2, 64), "max_new_tokens": (int, 16, 8192), "generation.temperature": (float, 0.01, 2.0),
            "generation.top_p": (float, 0.01, 1.0), "grpo.problems_per_step": (int, 1, 1024),
            "grpo.max_steps": (int, 1, 10**7), "grpo.kl_coef": (float, 0.0, 10.0), "grpo.updates_per_batch": (int, 1, 16),
            "grpo.lr": (float, 0.0, 1.0), "lora.r": (int, 1, 1024), "lora.alpha": (int, 1, 4096),
            "checkpoint.every_steps": (int, 1, 10**6), "checkpoint.eval_every_steps": (int, 1, 10**6),
            "fix.iters": (int, 0, 20), "improve.repair_per_step": (int, 0, 64), "agent.iters": (int, 1, 20),
            "agent.test_cmd": (str,), "agent.timeout_s": (int, 1, 3600), "agent.max_new_tokens": (int, 16, 8192),
            "fix.engine": (str, ("v1", "v2")), "fix.feedback": (str, ("basic", "counterexample")),
            "fix.candidates": (int, 1, 64), "improve.curriculum": (str, ("v1", "v2")),
            "improve.feedback": (str, ("basic", "counterexample")), "improve.synthetic_per_step": (int, 0, 64),
            "grpo.skip_zero_var": (bool,), "grpo.adv_norm": (str, ("std", "mean", "batch")),
            "eval.select_best": (str, ("legacy", "val")), "rollouts.score_cache": (bool,), "agent.candidates": (int, 1, 8),
            "grpo.dynamic_sampling": (bool,), "eval.patience": (int, 0, 1000), "grpo.minibatches": (int, 1, 16),
            "eval.during_train_n": (int, 1, 64), "agent.patience": (int, 0, 20)}
STATE_KEYS = {"grpo.lr", "lora.r", "lora.alpha"}
MODES = ("phase2", "improve", "fix", "phase1")


class Busy(Exception):
    pass


def _tail(path, n=80):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64_000))
            return "\n".join(f.read().decode(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _jsonl(path, limit=2000, keep=lambda row: False):
    """Filas (objetos) de un .jsonl, submuestreadas a ~limit. Siempre quedan la última y las que cumplan `keep`
    (evaluaciones y eventos: pocas y valiosas, el submuestreo uniforme se comería la mayoría)."""
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):  # una línea `null` o `1` rompería las gráficas en el navegador
                    rows.append(row)
    except OSError:
        pass
    step = max(1, len(rows) // limit)
    if step == 1:
        return rows
    return [r for i, r in enumerate(rows) if i % step == 0 or i == len(rows) - 1 or keep(r)]


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)  # nunca un JSON a medio escribir si dos peticiones coinciden o algo cae


def _check(key, v):
    """Valor de un ajuste del panel convertido y dentro de su rango, o ValueError."""
    if key not in EDITABLE:
        raise ValueError(f"no editable desde el panel: {key}")
    kind, *rng = EDITABLE[key]
    if kind is bool:  # sí/no: sólo true/false (el formulario los envía como booleano o como texto)
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.lower() in ("true", "false"):
            return v.lower() == "true"
        raise ValueError(f"{key}: debe ser true o false")
    if rng and isinstance(rng[0], tuple):  # una de varias opciones
        if v not in rng[0]:
            raise ValueError(f"{key}: debe ser uno de {', '.join(rng[0])}")
        return v
    if isinstance(v, bool) or (kind is int and isinstance(v, float) and not v.is_integer()):
        raise ValueError(f"{key}: valor no válido ({v!r})")
    try:
        v = kind(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: valor no válido ({v!r})") from None
    if rng and not (math.isfinite(v) and rng[0] <= v <= rng[1]):
        raise ValueError(f"{key}: debe estar entre {rng[0]} y {rng[1]}")
    return v


def _finite(x):
    """NaN/Infinity (p. ej. una loss que diverge en meta.json) no son JSON válido para el navegador: -> null."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


class App:
    def __init__(self, config=None, engine_factory=None, ui_dir=UI_DIR):
        self.base, self.ui_dir = config or os.path.join(ROOT, "config.yaml"), ui_dir
        self.engine_factory = engine_factory or self._real_engine
        self.engine, self.engine_ckpt = None, None
        self.jobs, self.sessions, self.proc, self.pending = {}, {}, None, 0
        self.lock = threading.Lock()
        self.q = queue.Queue()
        os.makedirs(ui_dir, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------- configuración ----------
    @staticmethod
    def _real_engine(cfg, checkpoint):
        from model.infer import Engine
        return Engine(cfg, checkpoint)

    def overrides(self):
        return _json(os.path.join(self.ui_dir, "overrides.json")) or {}

    def raw_config(self):
        cfg = read_config(self.base)
        for key, v in self.overrides().items():
            try:
                v = _check(key, v)  # overrides.json editado a mano o de otra versión: lo inválido se ignora
            except ValueError as e:
                print(f"[panel] ajuste ignorado de overrides.json: {e}")
                continue
            *path, last = key.split(".")
            d = cfg
            for p in path:
                d = d.setdefault(p, {})
            d[last] = v
        return cfg

    def cfg(self):
        path = os.path.join(self.ui_dir, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(self.raw_config(), f, sort_keys=False, allow_unicode=True)
        return load_config(path)

    def set_config(self, values):
        ov = self.overrides()
        for key, v in values.items():
            ov[key] = _check(key, v)
        _write_json(os.path.join(self.ui_dir, "overrides.json"), ov)
        return self.config_view()

    def reset_config(self):
        try:
            os.remove(os.path.join(self.ui_dir, "overrides.json"))
        except FileNotFoundError:
            pass
        return self.config_view()

    def config_view(self):
        raw = self.raw_config()

        def get(key):
            d = raw
            try:
                for p in key.split("."):
                    d = d[p]
            except (KeyError, TypeError):  # config.yaml anterior a la opción: el panel la muestra vacía
                return None
            return d
        return dict(values={k: get(k) for k in EDITABLE}, overrides=self.overrides(), state_keys=sorted(STATE_KEYS),
                    choices={k: list(r[1]) for k, r in EDITABLE.items() if len(r) > 1 and isinstance(r[1], tuple)},
                    base=self.base)

    # ---------- entrenamiento (proceso aparte) ----------
    def _train_state(self):
        return _json(os.path.join(self.ui_dir, "train.json")) or {}

    def external_trainings(self):
        return training_pids()

    def training(self):
        """Estado del entrenamiento del panel y de cualquier otro train.py vivo (p. ej. lanzado desde una terminal):
        ambos ocupan la GPU, así que los dos bloquean trabajos de GPU y un segundo entrenamiento."""
        st = self._train_state()
        pid = st.get("pid")
        own = self.proc is not None and self.proc.pid == pid
        alive = bool(pid) and (self.proc.poll() is None if own else is_training(pid))
        if own and not alive:
            st["exit_code"] = self.proc.returncode
        pids = ([pid] if alive else []) + [p for p in self.external_trainings() if p != pid]
        if pids and not alive:  # el que corre no es el del panel: su log viejo confundiría
            st["mode"], st["log"] = "lanzado fuera del panel", None
        return dict(st, running=bool(pids), pids=pids, log=_tail(st["log"]) if st.get("log") else "")

    def train_cmd(self, mode, fresh, cfg_path):
        cmd = [sys.executable, "-u", os.path.join(ROOT, "train.py"), mode, "--config", cfg_path]
        return cmd + (["--fresh"] if fresh and mode in ("phase2", "improve") else [])

    def start_training(self, mode, fresh=False):
        if mode not in MODES:
            raise ValueError(f"modo desconocido: {mode}")
        with self.lock:
            if self.training()["running"]:
                raise Busy("ya hay un entrenamiento en marcha")
            if self.pending:
                raise Busy("hay un trabajo de GPU en curso; espera a que termine")
            self._unload()
            self.cfg()  # escribe runs/ui/config.yaml con los ajustes del panel
            log = os.path.join(self.ui_dir, f"train-{time.strftime('%Y%m%d-%H%M%S')}-{mode}.log")
            with open(log, "w") as out:
                self.proc = subprocess.Popen(self.train_cmd(mode, fresh, os.path.join(self.ui_dir, "config.yaml")),
                                             cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             start_new_session=True)
            _write_json(os.path.join(self.ui_dir, "train.json"),
                        dict(pid=self.proc.pid, mode=mode, fresh=fresh, log=log, started=time.time()))
        return self.training()

    def stop_training(self):
        t = self.training()
        if not t["running"]:
            raise Busy("no hay ningún entrenamiento en marcha")
        for pid in t["pids"]:  # parada segura: train.py guarda checkpoint del último step completo
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:  # terminó entre la consulta y la señal
                pass
        return dict(t, stopping=True)

    # ---------- trabajos de GPU (fix / agente) ----------
    def _unload(self):
        if self.engine is not None:
            self.engine.close()
            self.engine, self.engine_ckpt = None, None

    def _engine(self, cfg, checkpoint):
        if self.engine is None or self.engine_ckpt != checkpoint:
            self._unload()
            self.engine, self.engine_ckpt = self.engine_factory(cfg, checkpoint), checkpoint
        return self.engine

    @staticmethod
    def _validate(kind, p):
        """Errores de formato -> 400 inmediato. Rondas acotadas: un trabajo sin fin bloquearía la cola de GPU."""
        def text(key, required=True):
            v = p.get(key) or ""
            if not isinstance(v, str) or (required and not v.strip()):
                raise ValueError(f"{key}: hace falta un texto")
            return v

        def count(key, lo, hi):
            v = p.get(key)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi):
                raise ValueError(f"{key}: debe ser un entero entre {lo} y {hi}")

        ck = p.get("checkpoint") or None
        if ck is not None:
            runs = os.path.realpath(os.path.join(ROOT, "runs"))
            if not isinstance(ck, str) or os.path.commonpath([os.path.realpath(ck), runs]) != runs or not os.path.isdir(ck):
                raise ValueError("checkpoint: debe ser un directorio de checkpoint dentro de runs/")
        if kind == "fix":
            count("iters", 0, 20)
            count("chains", 1, 16)
            if p.get("problem"):
                from problems.problems import ALL as PROBLEMS  # los 19 originales y el catálogo (train/val/test)
                if not isinstance(p["problem"], str) or p["problem"] not in PROBLEMS:
                    raise ValueError(f"problema desconocido: {p['problem']}")
            else:
                from problems.custom import Task
                p["sig"], p["doc"], p["tests"] = text("sig"), text("doc", False), text("tests")  # null -> ""
                Task(p["sig"], p["doc"], p["tests"])
        else:
            count("iters", 1, 20)
            p["test_cmd"] = text("test_cmd", False)
            if not os.path.isdir(text("repo")) or not text("task").strip():
                raise ValueError("hace falta la ruta de un directorio existente y una tarea")

    def submit(self, kind, params):
        self._validate(kind, params)
        with self.lock:
            if self.training()["running"]:
                raise Busy("la GPU está ocupada por el entrenamiento; páralo o espera")
            job = dict(id=uuid.uuid4().hex[:8], kind=kind, params=params, status="queued", events=[], result=None,
                       error=None, created=time.time())
            self.jobs[job["id"]] = job
            self.pending += 1
        self.q.put(job)
        return job

    def _worker(self):
        while True:
            job = self.q.get()
            job["status"] = "running"
            try:
                job["result"] = getattr(self, "_run_" + job["kind"])(job)
                job["status"] = "done"
            except Exception as e:
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                with self.lock:
                    self.pending -= 1

    def _run_fix(self, job):
        from problems.custom import Task
        from problems.problems import ALL as PROBLEMS
        from rollouts import fix as gtf
        p, cfg = job["params"], self.cfg()
        prob = PROBLEMS[p["problem"]] if p.get("problem") else Task(p["sig"], p["doc"], p["tests"])
        rounds = cfg.fix.iters if p.get("iters") is None else p["iters"]
        gen = self._engine(cfg, p.get("checkpoint") or None).generator()  # antes del run: si no carga, no deja run vacío
        run = os.path.join(ROOT, "runs", "fix", f"ui-{time.strftime('%Y%m%d-%H%M%S')}-{job['id']}")
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "attempts.jsonl"), "a") as f:
            def on_attempt(a, r):
                row = dict(vars(a), detail=r.detail)
                job["events"].append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            chains = gtf.run(gen, [prob], cfg, rounds, p.get("chains") or 1, cfg.seed, on_attempt)
        summ = gtf.summary(chains)
        with open(os.path.join(run, "summary.json"), "w") as f:
            json.dump(dict(summary=summ, task=p, gen=gen.stats), f, indent=1)
        return dict(summary=summ, run=run)

    def _run_agent(self, job):
        from agent.loop import Session
        p, cfg = job["params"], self.cfg()
        eng = self._engine(cfg, p.get("checkpoint") or None)  # antes de copiar el repo: si no carga, no deja copia
        s = Session(p["repo"], p["task"], cfg, p.get("test_cmd") or None)
        self.sessions[job["id"]] = s

        def on_iter(it):
            job["events"].append(dict(baseline=s.baseline) if it is None else dict(it, diff=s.last_diff))
        s.run(eng, int(p["iters"]) if p.get("iters") else None, on_iter)
        return dict(status=s.status, diff=s.last_diff, work=s.work, dir=s.dir)

    def decide(self, job_id, accept):
        s = self.sessions.get(job_id)
        if s is None or self.jobs[job_id]["status"] != "done":
            raise ValueError("sesión de agente no encontrada o sin terminar")
        if s.status in ("accepted", "rejected"):
            raise Busy(f"la sesión ya está {'aceptada' if s.status == 'accepted' else 'rechazada'}")
        if accept:
            s.accept()
        else:
            s.reject()
        self.jobs[job_id]["result"]["status"] = s.status
        return self.job(job_id)

    def job(self, job_id):
        j = self.jobs[job_id]
        return dict(j, events=list(j["events"]))  # copia: el hilo de GPU sigue añadiendo eventos

    # ---------- lectura de runs ----------
    def status(self):
        from rollouts import monitor  # sólo NVML: el fallback con torch crearía un contexto CUDA en el panel
        jobs = sorted(self.jobs.values(), key=lambda j: -j["created"])[:20]
        gpu = monitor.snapshot() if monitor.pynvml else {}
        return dict(training=self.training(), gpu=gpu, engine=dict(loaded=self.engine is not None,
                    checkpoint=self.engine_ckpt), jobs=[{k: j[k] for k in ("id", "kind", "status", "created")} for j in jobs])

    def runs(self):
        out = []
        exp = os.path.join(ROOT, "runs", "exp")  # experimentos: runs/exp/<plan>/<brazo> son runs de phase2
        for plan in sorted(os.listdir(exp), reverse=True) if os.path.isdir(exp) else []:
            for arm in sorted(os.listdir(os.path.join(exp, plan))) if os.path.isdir(os.path.join(exp, plan)) else []:
                d = os.path.join(exp, plan, arm)
                if os.path.isdir(os.path.join(d, "checkpoints")) or os.path.exists(os.path.join(d, "metrics.jsonl")):
                    out.append(dict(kind="exp", name=f"{plan}/{arm}", path=d, checkpoints=self.checkpoints(d),
                                    has_metrics=os.path.exists(os.path.join(d, "metrics.jsonl"))))
        for kind in ("phase2", "fix", "phase1"):
            base = os.path.join(ROOT, "runs", kind)
            for name in sorted(os.listdir(base), reverse=True) if os.path.isdir(base) else []:
                d = os.path.join(base, name)
                if not os.path.isdir(d):
                    continue
                r = dict(kind=kind, name=name, path=d)
                if kind == "phase2":
                    r["checkpoints"] = self.checkpoints(d)
                    r["has_metrics"] = os.path.exists(os.path.join(d, "metrics.jsonl"))
                else:
                    r["summary"] = (_json(os.path.join(d, "summary.json" if kind == "fix" else "metrics.json"))
                                    or {}).get("summary")
                out.append(r)
        ev = os.path.join(ROOT, "runs", "eval")  # train.py eval: una evaluación (eval.json) o una curva (curve.json)
        for name in sorted(os.listdir(ev), reverse=True) if os.path.isdir(ev) else []:
            d = os.path.join(ev, name)
            curve, one = _json(os.path.join(d, "curve.json")), _json(os.path.join(d, "eval.json"))
            items = (curve or {}).get("curve") or ([dict(one, label=one.get("label") or (
                os.path.basename(os.path.normpath(one["checkpoint"])) if one.get("checkpoint") else "base"))] if one else [])
            rows = [dict(label=c.get("label"), step=c.get("step"), split=(curve or one or {}).get("split"),
                         pass1=c.get("pass@1"), ci=c.get("ci_pass@1"), greedy=c.get("greedy")) for c in items]
            if rows:
                out.append(dict(kind="eval", name=name, path=d, evals=rows))
        agent = os.path.join(ROOT, "runs", "agent")
        for name in sorted(os.listdir(agent), reverse=True) if os.path.isdir(agent) else []:
            s = _json(os.path.join(agent, name, "session.json"))
            if s:
                out.append(dict(kind="agent", name=name, path=os.path.join(agent, name), status=s["status"],
                                task=s["task"], repo=s["repo"], iters=len(s["iters"])))
        return out

    @staticmethod
    def checkpoints(run):
        ckdir, out = os.path.join(run, "checkpoints"), []
        for name in sorted(os.listdir(ckdir)) if os.path.isdir(ckdir) else []:
            meta = _json(os.path.join(ckdir, name, "meta.json"))
            if meta:
                out.append(dict(name=name, path=os.path.join(ckdir, name), step=meta.get("step"),
                                metrics=meta.get("metrics", {}), best=name == "best"))
        return out

    @staticmethod
    def metrics(run):
        run, runs = os.path.realpath(run), os.path.realpath(os.path.join(ROOT, "runs"))  # runs/ puede ser un enlace
        if os.path.commonpath([run, runs]) != runs:
            raise ValueError("ruta fuera de runs/")
        return dict(metrics=_jsonl(os.path.join(run, "metrics.jsonl"), keep=lambda r: "step" not in r or "eval_reward" in r or r.get("event") == "eval_split"),
                    gpu=_jsonl(os.path.join(run, "gpu.jsonl"), 500))

    @staticmethod
    def problems():
        from problems.problems import ALL
        order = {"core": 0, "catalog": 1}
        return [dict(id=p.id, level=p.level, prompt=p.prompt(), split=p.split, source=p.source, category=p.category)
                for p in sorted(ALL.values(), key=lambda p: (order.get(p.source, 2), p.level, p.id))]


class Handler(BaseHTTPRequestHandler):
    app, hosts = None, {"localhost", "127.0.0.1", "[::1]"}

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(_finite(body), default=str, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _host_ok(self):
        host = self.headers.get("Host", "")
        return (host if host.endswith("]") else host.rsplit(":", 1)[0]) in self.hosts

    def _handle(self, method):
        if not self._host_ok():
            return self._send(403, {"error": "sólo accesible como localhost"})
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        body = {}
        if method == "POST":
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, {"error": "se espera application/json"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= 1 << 20:
                return self._send(413, {"error": "cuerpo ausente o mayor de 1 MB"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(400, {"error": "JSON inválido"})
            if not isinstance(body, dict):
                return self._send(400, {"error": "se espera un objeto JSON"})
        a, p = self.app, url.path
        routes = {
            ("GET", "/"): lambda: self._page(),
            ("GET", "/api/status"): a.status,
            ("GET", "/api/runs"): a.runs,
            ("GET", "/api/metrics"): lambda: a.metrics(q.get("run", "")),
            ("GET", "/api/problems"): a.problems,
            ("GET", "/api/config"): a.config_view,
            ("POST", "/api/config"): lambda: a.set_config(body),
            ("POST", "/api/config/reset"): a.reset_config,
            ("POST", "/api/train/start"): lambda: a.start_training(body.get("mode", "phase2"), bool(body.get("fresh"))),
            ("POST", "/api/train/stop"): a.stop_training,
            ("POST", "/api/fix"): lambda: a.submit("fix", body),
            ("POST", "/api/agent"): lambda: a.submit("agent", body),
        }
        m = re.fullmatch(r"/api/jobs/(\w+)(?:/(accept|reject))?", p)
        try:
            if (method, p) in routes:
                return routes[(method, p)]() if p == "/" else self._send(200, routes[(method, p)]())
            if m and method == "GET" and not m.group(2) and m.group(1) in a.jobs:
                return self._send(200, a.job(m.group(1)))
            if m and method == "POST" and m.group(2):
                return self._send(200, a.decide(m.group(1), m.group(2) == "accept"))
            return self._send(404, {"error": "no encontrado"})
        except Busy as e:
            return self._send(409, {"error": str(e)})
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            return self._send(400, {"error": str(e) or type(e).__name__})
        except Exception as e:  # nunca cortar la conexión sin respuesta
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _page(self):
        with open(os.path.join(ROOT, "ui", "index.html"), "rb") as f:
            self._send(200, f.read(), "text/html; charset=utf-8")

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


def make_server(app, host="127.0.0.1", port=8501):
    Handler.app = app
    if host not in ("0.0.0.0", "::"):
        Handler.hosts = Handler.hosts | {host}
    return ThreadingHTTPServer((host, port), Handler)


def main():
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # código del modelo con "\d" etc.: sin ruido en consola
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 sólo si Windows no llega a localhost (WSL2)")
    ap.add_argument("--config", help="config base (defecto: config.yaml); el panel guarda sus ajustes aparte")
    a = ap.parse_args()
    srv = make_server(App(a.config), a.host, a.port)
    print(f"panel en http://localhost:{a.port}  (Ctrl+C para salir; un entrenamiento lanzado sigue en marcha)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
