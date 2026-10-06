"""Lectura de un repositorio para el agente: ficheros de texto, árbol y selección de lo relevante para la tarea."""
import os, re

# COPY_SKIP: lo que no se copia a la copia de trabajo (control de versiones, cachés, entornos). SKIP_DIRS además
# queda fuera del contexto del modelo pero SÍ se copia: build/, env/ o dist/ pueden ser código que los tests importan.
COPY_SKIP = (".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache",
             ".ruff_cache", ".tox", "*.egg-info")
SKIP_DIRS = COPY_SKIP + ("env", "dist", "build", "runs", ".idea", ".vscode")
TEXT_EXT = {".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".js", ".ts",
            ".sh", ".html", ".css", ".c", ".h", ".cpp", ".rs", ".go", ".java"}
MAX_FILE = 100_000
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _skip(name):
    return name in SKIP_DIRS or name.endswith(".egg-info")


def scan(root):
    """{ruta_relativa: texto} de los ficheros de texto (sin binarios, dependencias ni ficheros enormes)."""
    files = {}
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if not _skip(x))
        for n in sorted(names):
            path = os.path.join(d, n)
            if os.path.splitext(n)[1].lower() not in TEXT_EXT or os.path.islink(path) or os.path.getsize(path) > MAX_FILE:
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    files[os.path.relpath(path, root)] = f.read()
            except (UnicodeDecodeError, OSError):
                continue
    return files


def rank(files, text):
    """Rutas ordenadas por relevancia para `text` (tarea + salida de tests): la ruta citada tal cual (traceback)
    pesa más que coincidir en el nombre, y eso más que compartir identificadores con el contenido."""
    words = {w.lower() for w in WORD.findall(text)}

    def score(path):
        body = {w.lower() for w in WORD.findall(files[path])}
        name = {w.lower() for w in WORD.findall(path)}
        return 10 * (path in text) + 3 * len(words & name) + len(words & body) / (1 + len(body)) ** 0.5

    return sorted(files, key=lambda p: (-score(p), p))


def context(files, text, budget):
    """Árbol del repo + contenido de los ficheros más relevantes que quepan en `budget` caracteres."""
    tree = "\n".join(sorted(files)[:300])
    out, used = [f"Repository files:\n{tree}\n"], len(tree)
    for path in rank(files, text):
        block = f"\nFILE: {path}\n```\n{files[path]}\n```\n"
        if used + len(block) <= budget:  # si no cabe, puede caber otro más pequeño
            out.append(block)
            used += len(block)
    return "".join(out)
