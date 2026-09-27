"""Turns downloaded pages into short spoken "fact cards" (knowledge/cards.json) using the PC's LLM.

Run from the project root, after knowledge.scrape:
    uv run python -m knowledge.make_cards              # everything
    uv run python -m knowledge.make_cards cine teatro  # rebuild only these (merged into cards.json)

The LLM only summarizes the text it is given (temperature 0 = deterministic, no creativity), and
answers through a JSON schema, so the output is always machine-readable. Every course/lab name
it uses is checked against the source; unknown names mark the card "revisar": true.
"""
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

from knowledge.extract import page_text, page_title, pdf_text
from knowledge.scrape import BASE, CAMPUS, RAW

CARDS = Path(__file__).parent / "cards.json"
OLLAMA = "http://127.0.0.1:11434"
BUILD_MODEL = "gemma3:4b"   # best Spanish cards; llama3.2:3b was faster but copied text verbatim

TITLES = {"perfil": "para quién es", "egresado": "qué hace el egresado",
          "materias": "plan de estudios", "laboratorios": "laboratorios"}

PROGRAM_PROMPT = """Eres redactor de un robot orientador vocacional que habla en voz alta con aspirantes.
Con base ÚNICAMENTE en los textos de abajo sobre el programa "{name}" de la Universidad Autónoma de \
Occidente, escribe fichas breves en español hablado: frases completas, sin listas ni símbolos, de 50 a \
70 palabras cada una. Nunca menciones precios ni valores. Si los textos no traen información para una \
ficha, déjala vacía ("").
perfil: para quién es el programa (intereses y habilidades del aspirante) y de qué se trata.
egresado: qué puede hacer quien se gradúa.
materias: duración, créditos, modalidad y algunas materias representativas de los primeros, intermedios y últimos semestres.
laboratorios: laboratorios y espacios de práctica que mencionan los textos.
En nombres_materias y nombres_laboratorios copia exactamente los nombres de materias y laboratorios que usaste (máximo diez en cada lista).

TEXTO DE LA PÁGINA:
{page}

TEXTO DEL PLAN DE ESTUDIOS (viene de un PDF con tablas, puede estar desordenado):
{plan}"""

PROGRAM_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "string"} for k in TITLES},
        "nombres_materias": {"type": "array", "items": {"type": "string"}},
        "nombres_laboratorios": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*TITLES, "nombres_materias", "nombres_laboratorios"],
}

CAMPUS_PROMPT = """Eres redactor de un robot orientador vocacional que habla en voz alta con aspirantes.
Con base ÚNICAMENTE en el texto de abajo, de la página "{title}" de la Universidad Autónoma de Occidente, \
escribe de una a cuatro fichas breves sobre lo que un aspirante querría saber: actividades, servicios, \
espacios y cómo acceder a ellos. Español hablado, frases completas, sin listas ni símbolos, de 50 a 70 \
palabras cada una, sin precios. Cada ficha lleva un título corto.

TEXTO:
{page}"""

CAMPUS_SCHEMA = {
    "type": "object",
    "properties": {"fichas": {"type": "array", "items": {
        "type": "object",
        "properties": {"titulo": {"type": "string"}, "texto": {"type": "string"}},
        "required": ["titulo", "texto"]}}},
    "required": ["fichas"],
}


MAX_WORDS = 75   # hard cap per card: each extra word costs Pi reading time on every card turn


def plain(text: str) -> str:
    # Lowercase without accents and with single spaces, so "Cálculo" matches "calculo" and a
    # name split across two lines of a PDF table ("Algoritmia y\nprogramación") still matches.
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def missing_names(names: list[str], source: str) -> list[str]:
    src = plain(source)
    return [n for n in names if n.strip() and plain(n) not in src]


def mentions_labs(source: str) -> bool:
    # \b = word boundary. Matches "lab", "labs", "laboratorio", "Fab-Lab"; not "colaboración"
    # (no boundary before "lab") and not "laboral"/"labores" (must be lab, labs or laborator…).
    return re.search(r"\blab(s\b|\b|orator)", plain(source)) is not None


def trim_words(text: str, limit: int = MAX_WORDS) -> str:
    # Keep whole sentences while they fit in `limit` words (always at least the first sentence).
    kept, count = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        n = len(sentence.split())
        if kept and count + n > limit:
            break
        kept.append(sentence)
        count += n
    return " ".join(kept)


def with_retries(fn, tries: int = 3, wait_s: float = 10):
    # Ollama's model runner can crash under GPU-memory pressure (HTTP 500); it restarts on the
    # next request, so waiting a bit and retrying usually works. The last failure is re-raised.
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except OSError as e:   # urllib's HTTPError/URLError are OSError subclasses
            if attempt == tries:
                raise
            print(f"  intento {attempt} falló ({e}); reintentando en {wait_s:.0f} s", flush=True)
            time.sleep(wait_s)


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"{OLLAMA}{path}", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def free_gpu() -> None:
    # Unload every model Ollama has in memory (keep_alive 0 = unload now): the voice agent keeps
    # its LLM loaded forever, and gemma3:4b needs the whole 4 GB GPU to build cards reliably.
    with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=10) as r:
        for m in json.load(r)["models"]:
            _post("/api/generate", {"model": m["name"], "keep_alive": 0})


def ask(prompt: str, schema: dict) -> dict:
    # /api/generate with "format": <JSON schema> makes Ollama constrain the output to that schema.
    body = {"model": BUILD_MODEL, "prompt": prompt, "format": schema, "stream": False,
            "options": {"temperature": 0, "num_ctx": 6144}}   # smaller context = more of the model fits in the GPU
    return json.loads(with_retries(lambda: _post("/api/generate", body))["response"])


def program_cards(slug: str, html: str, plan: str) -> list[dict]:
    name, page = page_title(html), page_text(html)
    out = ask(PROGRAM_PROMPT.format(name=name, page=page[:8000], plan=plan[:5000]), PROGRAM_SCHEMA)
    source = page + "\n" + plan
    cards = []
    for tipo, names_key in (("perfil", None), ("egresado", None),
                            ("materias", "nombres_materias"), ("laboratorios", "nombres_laboratorios")):
        text = trim_words(out[tipo])
        if not text:
            continue   # the source had nothing for this card: omit it, never invent it
        if tipo == "laboratorios" and not mentions_labs(source):
            continue   # the LLM once invented a lab from a course name; no "lab" in source → no card
        missing = missing_names(out[names_key], source) if names_key else []
        if names_key and not out[names_key]:
            missing = ["(no listó nombres: verificar a mano)"]   # can't check what wasn't declared
        card = {"id": f"{slug}-{tipo}", "tipo": tipo, "programa": name, "titulo": f"{name}: {TITLES[tipo]}",
                "texto": text, "fuente": f"{BASE}/programa/{slug}/", "revisar": bool(missing)}
        if missing:
            card["faltan"] = missing   # tells the human reviewer exactly what to check
        cards.append(card)
    return cards


def campus_cards(path: str, html: str) -> list[dict]:
    title = page_title(html) or path
    out = ask(CAMPUS_PROMPT.format(title=title, page=page_text(html)[:8000]), CAMPUS_SCHEMA)
    base_id = path.replace("/", "-")
    return [{"id": f"{base_id}-{i}", "tipo": "campus", "programa": None, "titulo": f["titulo"].strip(),
             "texto": trim_words(f["texto"]), "fuente": f"{BASE}/{path}/", "revisar": False}
            for i, f in enumerate(out["fichas"][:4]) if f["texto"].strip()]


if __name__ == "__main__":
    only = set(sys.argv[1:])   # optional: rebuild just these program slugs / campus ids
    old = json.loads(CARDS.read_text(encoding="utf-8")) if CARDS.exists() and only else []
    cards = [c for c in old if not any(c["id"].startswith(s + "-") for s in only)]
    # One job per source: (id, function that builds its cards).
    jobs = []
    for html_path in sorted((RAW / "programas").glob("*.html")):
        pdf = html_path.with_suffix(".pdf")
        jobs.append((html_path.stem, lambda h=html_path, p=pdf: program_cards(
            h.stem, h.read_text(encoding="utf-8"), pdf_text(p) if p.exists() else "")))
    for path in CAMPUS:
        base_id = path.replace("/", "-")
        jobs.append((base_id, lambda path=path, b=base_id: campus_cards(
            path, (RAW / "campus" / f"{b}.html").read_text(encoding="utf-8"))))

    free_gpu()
    failed = []
    for job_id, build in jobs:
        if only and job_id not in only:
            continue
        try:
            new = build()
        except OSError as e:
            failed.append(job_id)   # skip it and keep going; rerun later with its id as argument
            print(f"{job_id}: FALLÓ ({e})", flush=True)
            continue
        cards += new
        # Save after every source, so a crash never throws away the cards already made.
        CARDS.write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{job_id}: {len(new)} fichas", flush=True)
    review = [c["id"] for c in cards if c["revisar"]]
    print(f"\n{len(cards)} fichas en {CARDS.name}. Para revisar ({len(review)}): {review}")
    if failed:
        print(f"Fallaron ({len(failed)}); reintenta con: uv run python -m knowledge.make_cards {' '.join(failed)}")
