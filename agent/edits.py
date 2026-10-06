"""Formato de edición del agente: `FILE: ruta` seguido de un bloque de código con el contenido COMPLETO del fichero.
Fichero completo en vez de diffs: un modelo pequeño se equivoca mucho menos. Las rutas se validan siempre."""
import difflib, os, re

HEAD = re.compile(r"^FILE:[ \t]*(\S+)[ \t]*$", re.M)
BLOCK = re.compile(r"^```[^\n]*\n(.*?)^```", re.S | re.M)
# La plantilla del prompt de sistema copiada tal cual (los modelos pequeños lo hacen): nunca es contenido de un fichero
PLACEHOLDER = re.compile(r"<the complete new content of the file>", re.I)


def parse(text, default=None):
    """El contenido va de la primera valla tras la cabecera a la ÚLTIMA antes de la siguiente cabecera: así un
    README con bloques de código dentro no se trunca en su primera valla interna.

    default: si la respuesta no trae ninguna cabecera `FILE:` pero sí exactamente UN bloque de código, ese bloque es
    el contenido de `default`. Los modelos pequeños responden así a menudo (en el benchmark, 10 de 23 primeras
    respuestas del 0.5B: el arreglo correcto, sin cabecera, que antes se descartaba). Sólo lo pasa quien sabe sin
    ambigüedad cuál es el fichero (Session.target())."""
    heads, out = list(HEAD.finditer(text)), {}
    for h, nxt in zip(heads, heads[1:] + [None]):
        lines = text[h.end():nxt.start() if nxt else len(text)].split("\n")
        fences = [i for i, line in enumerate(lines) if line.startswith("```")]
        if len(fences) >= 2:
            out[h.group(1).strip("`'\"")] = "\n".join(lines[fences[0] + 1:fences[-1]]) + "\n"
    if not heads and default:
        blocks = BLOCK.findall(text)
        if len(blocks) == 1 and blocks[0].strip():
            out[default] = blocks[0] if blocks[0].endswith("\n") else blocks[0] + "\n"
    return out


def is_placeholder(body):
    return bool(PLACEHOLDER.search(body or ""))


def safe(root, rel):
    """Ruta absoluta de `rel` dentro de root. Error si sale del repo (.., absoluta, symlink) o toca .git."""
    root = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root, rel))
    if not rel or os.path.isabs(rel) or os.path.commonpath([full, root]) != root or full == root:
        raise ValueError(f"ruta fuera del repositorio: {rel}")
    if ".git" in os.path.relpath(full, root).split(os.sep):
        raise ValueError(f"no se edita .git: {rel}")
    return full


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:  # un binario no debe tumbar el diff
            return f.read()
    except FileNotFoundError:
        return ""


def diff(orig_root, work_root, paths):
    out = []
    for rel in paths:
        a, b = _read(os.path.join(orig_root, rel)), _read(os.path.join(work_root, rel))
        out += difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{rel}", f"b/{rel}")
    return "".join(out)
