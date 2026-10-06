"""Agente de código local: lee el repo, propone ediciones, ejecuta los tests en el sandbox y corrige con su salida.

Trabaja SIEMPRE sobre una copia (runs/agent/<id>/work). El repo original sólo se toca en accept(), que se niega si
alguno de los ficheros editados cambió en el repo desde que empezó la sesión.

Bucle: tests iniciales -> [propuesta -> escribir -> tests -> análisis] x agent.iters. Cada ronda se puntúa con
(tests verdes, nº de tests que pasan, -nº que fallan). Si una edición EMPEORA la mejor puntuación vista, se revierte
(la copia de trabajo vuelve al mejor estado) y el modelo lo sabe en la ronda siguiente: antes las ediciones malas se
acumulaban y cada ronda partía de un estado peor. Parada: tests verdes, agent.iters agotado o agent.patience rondas
seguidas sin mejorar.
"""
import codecs, hashlib, json, os, re, shlex, shutil, time, uuid
from collections import Counter

from agent.edits import HEAD, diff, is_placeholder, parse, safe
from agent.repo import COPY_SKIP, context, scan
from common import ROOT, opt
from sandbox import command

_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)(?: - (.*))?$", re.M)
_PYTEST_LOC = re.compile(r"^([^\s:]+\.py):(\d+):", re.M)
_PY_FILE = re.compile(r'File "([^"]+\.py)", line (\d+)')
_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)|AssertionError)\b:?\s*(.*)$")
_COUNT = re.compile(r"(\d+) (passed|failed|errors?)\b")


def _src_line(work, path, line):
    """Texto de `path:line` dentro de la copia de trabajo (las rutas del sandbox pueden ser absolutas)."""
    parts = path.replace("\\", "/").lstrip("/").split("/")
    for i in range(len(parts)):
        full = os.path.join(work, *parts[i:])
        if os.path.isfile(full):
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
                return lines[line - 1].strip()[:160] if 0 < line <= len(lines) else None
            except OSError:
                return None
    return None


def summarize_tests(out, work=None):
    """Feedback estructurado de pytest (o de un traceback normal de Python): tests que fallan, excepción, la línea
    del código bajo prueba que falla (prefiere ficheros que no son de tests) y el recuento. dict con 'text'."""
    out = out or ""
    failed = [(m.group(1), (m.group(2) or "").strip()) for m in _FAILED.finditer(out)]
    detail = [ln[1:].strip() for ln in out.splitlines() if ln.startswith("E ")][:3]
    locs = [(f, int(n)) for f, n in _PYTEST_LOC.findall(out)] + [(f, int(n)) for f, n in _PY_FILE.findall(out)]
    src = [x for x in locs if not os.path.basename(x[0]).startswith("test")]
    where = (src or locs)[-1] if (src or locs) else None
    exc = None
    for ln in reversed(out.splitlines()):
        m = _EXC.match(ln.strip())
        if m:
            exc = (m.group(1) + (": " + m.group(2) if m.group(2) else ""))[:200]
            break
    if exc is None and failed and failed[0][1]:
        exc = failed[0][1][:200]
    counts = {}
    for n, k in _COUNT.findall(out):
        counts[k.rstrip("s") if k.startswith("error") else k] = int(n)
    code = _src_line(work, *where) if work and where else None
    lines = []
    if failed:
        lines.append("Failing: " + ", ".join(t for t, _ in failed[:5]))
    if exc:
        lines.append("Error: " + exc)
    lines += [f"Detail: {d}" for d in detail if d not in (exc or "")][:2]
    if where:
        lines.append(f"At {where[0]}:{where[1]}" + (f": `{code}`" if code else ""))
    if counts:
        lines.append("Summary: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    sig = f"{failed[0][0] if failed else '-'}|{(exc or '-').split(':')[0]}|{where[0] + ':' + str(where[1]) if where else '-'}"
    return dict(failed=[t for t, _ in failed], error=exc, where=where, code=code, counts=counts, sig=sig,
                text="\n".join(lines))


def _drop_pyc(full):
    """Borra el bytecode de un .py recién escrito: si la edición conserva el tamaño y cae en el mismo segundo,
    Python reutilizaría el .pyc viejo de __pycache__ y los tests ejecutarían el código anterior."""
    d, base = os.path.split(full)
    stem, pc = os.path.splitext(base)[0] + ".", os.path.join(d, "__pycache__")
    if base.endswith(".py") and os.path.isdir(pc):
        for n in os.listdir(pc):
            if n.startswith(stem):
                try:
                    os.remove(os.path.join(pc, n))
                except OSError:
                    pass


def _passed(out):
    m = re.findall(r"(\d+) passed", out or "")
    return int(m[-1]) if m else 0


def score(res):
    """Puntuación de una ejecución de tests: (verde, pasan, -fallan). Mayor es mejor; comparable entre rondas."""
    c = summarize_tests(res.out)["counts"] if res is not None else {}
    return (int(bool(res and res.ok)), c.get("passed", _passed(res.out if res else "")),
            -(c.get("failed", 0) + c.get("error", 0)))


def _edit_key(edits):
    """Huella de un conjunto de ediciones: el mismo código (AST para .py) en los mismos ficheros = misma edición."""
    from rollouts.collect import code_key
    norm = sorted((rel, code_key(body) if rel.endswith(".py") else body.strip()) for rel, body in edits.items())
    return hashlib.sha1(json.dumps(norm).encode()).hexdigest()[:16]

SYSTEM = ("You are a coding agent working on a local repository. To change a file, reply with a block:\n"
          "FILE: relative/path.py\n```python\n<the complete new content of the file>\n```\n"
          "Write the COMPLETE file, only for the files you change, and keep changes minimal.")


def _style(path):
    """(BOM utf-8, CRLF) del fichero: se conservan al reescribirlo. Si no, en Windows (core.autocrlf) cada edición
    cambiaría todas las líneas en disco, aunque el diff (lectura con saltos universales) no lo mostrara."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return False, False
    return raw.startswith(codecs.BOM_UTF8), b"\r\n" in raw


def _digest(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None
    except IsADirectoryError:
        return "<dir>"


def _read_bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError):
        return None


def _tail(s, n=3000):
    return s if len(s) <= n else "[...]\n" + s[-n:]


class Session:
    def __init__(self, repo, task, cfg, test_cmd=None, root=None):
        if not os.path.isdir(repo):
            raise ValueError(f"no es un directorio: {repo}")
        self.repo, self.task, self.cfg = os.path.realpath(repo), task, cfg
        self.test_cmd = (test_cmd or "").strip() or cfg.agent.test_cmd  # en blanco = el de la config
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.dir = os.path.realpath(os.path.join(root or os.path.join(ROOT, "runs", "agent"), self.id))
        self.work = os.path.join(self.dir, "work")
        shutil.copytree(self.repo, self.work, ignore=self._ignore, symlinks=True)
        self.changed = {}  # ruta relativa -> digest del fichero al empezar la sesión (None si es nuevo)
        self.iters, self.baseline, self.status, self.last_diff = [], None, "running", ""
        self.orig = {}  # ruta relativa -> contenido (bytes) en la copia de trabajo antes de la primera edición
        self.best_score, self.best_files, self.best_res = None, {}, None

    def _ignore(self, d, names):
        skip = set(shutil.ignore_patterns(*COPY_SKIP)(d, names))
        for n in names:  # la carpeta que contiene esta sesión (agente sobre este mismo proyecto): no copiarse a sí misma
            p = os.path.realpath(os.path.join(d, n))
            if self.dir == p or self.dir.startswith(p + os.sep):
                skip.add(n)
        return skip

    def target(self):
        """El fichero de la tarea si es inequívoco: el único fichero no de tests que la tarea nombra y existe en la
        copia de trabajo (p. ej. "Fix solution.py ... Do not modify test_solution.py" -> solution.py). Si no, None
        y una respuesta sin cabecera FILE: no se aplica."""
        names = set(re.findall(r"[\w./-]+\.[A-Za-z0-9]+\b", self.task or ""))
        cands = sorted(n for n in names if not os.path.basename(n).lower().startswith("test")
                       and os.path.isfile(os.path.join(self.work, n)))
        return cands[0] if len(cands) == 1 else None

    def test(self):
        a = self.cfg.agent
        return command.run(shlex.split(self.test_cmd), self.work, self.cfg.sandbox, a.timeout_s, a.mem_mb)

    def prompt(self, engine, last):
        a = self.cfg.agent
        hist = "".join(f"- attempt {it['i']}: changed {', '.join(it['files']) or 'nothing'} -> tests "
                       f"{'passed' if it['passed'] else 'failed'}"
                       f"{' (it repeated an earlier edit, so it was not applied)' if it.get('repeat') else ''}\n"
                       for it in self.iters[-4:])
        fb = summarize_tests(last.out, self.work) if last and not last.ok else None
        tests = (f"Test command: {self.test_cmd}\nLast run (exit code {last.code}):\n"
                 + (f"{fb['text']}\nOutput (tail):\n{_tail(last.out, 1500)}\n" if fb and fb["text"] else
                    f"{_tail(last.out)}\n")) if last else ""
        notes = []
        if self.iters and self.iters[-1].get("repeat"):
            notes.append("Your last reply repeated an edit that was already tried and failed. Propose a DIFFERENT change.")
        if self.iters and self.iters[-1].get("reverted"):
            it = self.iters[-1]
            notes.append(f"Your last edit made the tests WORSE ({it['score_before'][1]} -> {it['score'][1]} passing), so "
                         "it was reverted. The files are back to the best version so far. Try a different change.")
        n_same = self.__dict__.get("sigs", Counter()).get(fb["sig"], 0) if fb else 0
        if n_same >= 2:
            notes.append(f"The same failure has persisted for {n_same} attempts. Try a different approach.")
        ask = f"\nTask: {self.task}\n{tests}" + (f"Previous attempts:\n{hist}" if hist else "") + "".join(
            n + "\n" for n in notes)
        files, limit = scan(self.work), a.context_length - a.max_new_tokens
        budget = 3 * limit  # ~3 caracteres/token en código; se corrige abajo con el tokenizer real
        while True:
            p = engine.chat([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": context(files, self.task + ask, budget) + ask}])
            if engine.n_tokens(p) <= limit or budget < 2000:
                return p
            budget = int(budget * 0.75)

    def _best_of(self, engine, prompt, k):
        """k respuestas; cada edición distinta se prueba en una copia temporal de la copia de trabajo y gana la que
        más tests pasa (luego exit 0; luego la primera). Sólo decide QUÉ texto se aplica: después se aplica por el
        camino normal de step(), así que copiar/aplicar/descartar/conflictos no cambian."""
        texts = [engine.complete(prompt, self.cfg.agent.max_new_tokens) for _ in range(k)]
        a, seen, scores = self.cfg.agent, set(), []
        for i, t in enumerate(texts):
            ed = {r: b for r, b in parse(t, self.target()).items() if not is_placeholder(b)}
            key = _edit_key(ed) if ed else None
            if not ed or key in seen or key in self.__dict__.get("tried", {}):
                scores.append(None)
                continue
            seen.add(key)
            tmp = os.path.join(os.path.dirname(self.work), f"cand_{i}")
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                shutil.copytree(self.work, tmp, symlinks=True)
                for rel, body in ed.items():
                    full = safe(tmp, rel)
                    os.makedirs(os.path.dirname(full), exist_ok=True)
                    with open(full, "w", encoding="utf-8") as f:
                        f.write(body)
                    _drop_pyc(full)
                res = command.run(shlex.split(self.test_cmd), tmp, self.cfg.sandbox, a.timeout_s, a.mem_mb)
                scores.append([int(res.ok), _passed(res.out)])
            except (ValueError, OSError):
                scores.append([0, -1])
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        valid = [i for i, sc in enumerate(scores) if sc is not None]
        best = max(valid, key=lambda i: (scores[i], -i)) if valid else 0
        return texts[best], scores

    def step(self, engine, last):
        t0 = time.time()
        k = max(1, int(opt(self.cfg, "agent.candidates", 1)))
        prompt = self.prompt(engine, last)
        if k == 1:
            text, cand = engine.complete(prompt, self.cfg.agent.max_new_tokens), None
        else:
            text, cand = self._best_of(engine, prompt, k)
        edits = parse(text, self.target())
        inferred = bool(edits) and not HEAD.search(text)  # bloque sin cabecera aplicado al fichero de la tarea
        tried, sigs = self.__dict__.setdefault("tried", {}), self.__dict__.setdefault("sigs", Counter())
        key = _edit_key(edits) if edits else None
        if key is not None and key in tried:  # edición ya probada: ni se escribe ni se vuelve a testear
            it = dict(i=len(self.iters), response=text, files=[], rejected=[], passed=False, code=last.code,
                      output=_tail(last.out), secs=round(time.time() - t0, 1), repeat=True, repeat_of=tried[key],
                      candidates=k, cand_scores=cand)
            self.iters.append(it)
            return it, last
        written, rejected = [], []
        for rel, body in edits.items():
            try:
                if is_placeholder(body):
                    raise ValueError(f"{rel}: es la plantilla del prompt, no código")
                full = safe(self.work, rel)
                if os.path.isdir(full):
                    raise ValueError(f"es un directorio: {rel}")
                rel = os.path.relpath(full, self.work)
                # la copia de trabajo aún tiene el contenido del inicio de la sesión: ese es el original contra el que
                # accept() detecta cambios en el repo (leer el repo aquí daría por bueno lo que cambió mientras tanto)
                orig = self.changed[rel] if rel in self.changed else _digest(full)
                if rel not in self.orig:
                    self.orig[rel] = _read_bytes(full)
                bom, crlf = _style(full)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8-sig" if bom else "utf-8", newline="\r\n" if crlf else "\n") as f:
                    f.write(body.replace("\r\n", "\n").lstrip("\ufeff"))
                _drop_pyc(full)
            except (ValueError, OSError) as e:
                rejected.append(str(e))
                continue
            self.changed[rel] = orig
            written.append(rel)
        res = self.test() if written else last
        fb = summarize_tests(res.out, self.work) if not res.ok else None
        if fb:
            sigs[fb["sig"]] += 1
        if key is not None:
            tried[key] = len(self.iters)
        sc, before = score(res), self.best_score
        # agent.rollback (defecto true): una edición que deja los tests peor que el mejor estado se deshace
        reverted = bool(opt(self.cfg, "agent.rollback", True) and written and before is not None and sc < before)
        if reverted:  # peor que el mejor estado: vuelta atrás (los tests de ese estado siguen siendo el contexto)
            self._restore_best()
        elif written and (before is None or sc > before):
            self._mark_best(sc, res)
        self.last_diff = diff(self.repo, self.work, sorted(self.changed))
        it = dict(i=len(self.iters), response=text, files=written, rejected=rejected, passed=bool(written and res.ok),
                  code=res.code, output=_tail(res.out), secs=round(time.time() - t0, 1), repeat=False,
                  failing=fb["failed"] if fb else [], error_line=fb["where"] if fb else None,
                  sig=fb["sig"] if fb else None, candidates=k, cand_scores=cand, score=list(sc),
                  score_before=list(before) if before else None, reverted=reverted,
                  improved=bool(written and (before is None or sc > before)), inferred_file=inferred)
        self.iters.append(it)
        return it, (self.best_res if reverted else res)

    def _mark_best(self, sc, res):
        self.best_score, self.best_res = sc, res
        self.best_files = {rel: _read_bytes(os.path.join(self.work, rel)) for rel in self.changed}

    def _restore_best(self):
        for rel in self.changed:
            full = os.path.join(self.work, rel)
            want = self.best_files[rel] if rel in self.best_files else self.orig.get(rel)
            if want is None:
                if os.path.isfile(full):
                    os.remove(full)
            else:
                with open(full, "wb") as f:
                    f.write(want)
            _drop_pyc(full)

    def run(self, engine, iters=None, on_iter=None):
        """Tests iniciales (su fallo es el contexto) y hasta `iters` rondas de editar -> testear. Para antes si los
        tests pasan o si agent.patience rondas seguidas no mejoran el mejor resultado."""
        patience = int(opt(self.cfg, "agent.patience", 0))
        try:
            last = self.test()
            self.baseline = dict(code=last.code, passed=last.ok, output=_tail(last.out))
            self.best_score, self.best_res = score(last), last
            if on_iter:
                on_iter(None)
            stall = 0
            for _ in range(iters or self.cfg.agent.iters):
                it, last = self.step(engine, last)
                if on_iter:
                    on_iter(it)
                if it["passed"]:
                    break
                stall = 0 if it.get("improved") else stall + 1
                if patience and stall >= patience:
                    it["stopped"] = "patience"
                    break
        except BaseException:  # error o Ctrl+C: la sesión queda registrada (y su copia localizable), no a medias
            self.status = "error"
            self.save()
            raise
        self.status = "passed" if self.iters and self.iters[-1]["passed"] else "failed"
        self.save()
        return self

    def accept(self):
        conflicts = [rel for rel, d in self.changed.items() if _digest(os.path.join(self.repo, rel)) != d]
        if conflicts:
            raise RuntimeError(f"estos ficheros cambiaron en el repo durante la sesión: {', '.join(conflicts)}")
        dsts = [(rel, safe(self.repo, rel)) for rel in sorted(self.changed)]  # todo validado antes de escribir nada
        for rel, dst in dsts:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".agent-tmp"
            try:
                shutil.copy2(os.path.join(self.work, rel), tmp)
                os.replace(tmp, dst)  # nunca un fichero a medio escribir en el repo
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        self.status = "accepted"
        self.save()

    def reject(self):
        shutil.rmtree(self.work, ignore_errors=True)
        self.status = "rejected"
        self.save()

    def save(self):
        d = dict(id=self.id, repo=self.repo, task=self.task, test_cmd=self.test_cmd, status=self.status,
                 baseline=self.baseline, iters=self.iters, changed=sorted(self.changed), diff=self.last_diff)
        with open(os.path.join(self.dir, "session.json"), "w") as f:
            json.dump(d, f, indent=1)
