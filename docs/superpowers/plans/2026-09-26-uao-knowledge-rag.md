# UAO Knowledge (RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ground the voice agent in real Universidad Autónoma de Occidente information. Scrape the site once, turn it into short human-reviewable "fact cards", and at runtime inject 1–3 relevant cards per turn, so the agent recommends real programs and answers detail questions without inventing anything.

**Architecture:**
- **Offline build** (`knowledge/`, runs on the PC): `scrape.py` → `knowledge/raw/` → `extract.py` (clean text, prices removed) → `make_cards.py` (the PC LLM writes cards) → `knowledge/cards.json` (committed).
- **Runtime** (`rag.py`): embeds the cards with an Ollama embedding model on the CPU, caches the vectors, and `retrieve()` picks a mode per turn (detail / recommendation / price hand-over).
- `brain.py` puts the cards next to the current user message only, and adds a fixed rules block with the list of official program names to the system prompt.

**Tech Stack:** Python 3.11 (uv), BeautifulSoup4, pypdf, Ollama `/api/generate` with JSON-schema output (build) and `/api/embed` (runtime), numpy.

**Spec:** `docs/superpowers/specs/2026-09-26-uao-knowledge-rag-design.md`

**Ruling carried from planning:**
- The runtime module is named `rag.py`, not `knowledge.py` as in the spec. A module `knowledge.py` next to a package directory `knowledge/` breaks `python -m knowledge.scrape`.
- Cards may carry an extra `faltan` list (names that failed the source check) to help the human reviewer.

## Global Constraints

- Runtime stays offline. Only `knowledge/scrape.py` touches the internet; `make_cards.py` and `eval.py` only call the local Ollama.
- Prices, tuition and scholarships are never stored in cards or given as answers. `strip_prices()` runs on all source text before the LLM sees it.
- Scraping: never fetch URLs containing `?`. Wait 2 s between requests, and skip files already downloaded.
- Pi budget: detail turns get at most `rag_max_cards` (2) cards; recommendation turns get 3 `perfil` cards.
- The system prompt must be byte-identical across turns (Ollama prompt cache). Cards go only in the current user message and are never stored in `Brain.history`.
- The embedding model always runs on the CPU (`"num_gpu": 0`), so the 4 GB GPU stays free for the LLM.
- Comment density stays high: the user is learning and asked for every line to be explained.
- Tests are plain asserts in `test_logic.py`: `uv run python test_logic.py`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push after each task: `git push`.
- Build scripts run as modules from the project root: `uv run python -m knowledge.<script>`.

## Review Focus

1. **Page templates vary** (virtual programs use "¿Qué debes saber del programa?"; some pages put the study plan in HTML and others only in a PDF; some pages have no PDF). The build must not crash, and a card whose source section is missing must be omitted, not invented. → `test_page_text_generic` (Task 2).
2. **Prices in other formats** ("$ 9.500.000" with a space, amounts inside sentences about tuition). They must never reach the LLM. → extra cases in `test_strip_prices` (Task 2).
3. **`cards.json` missing** (fresh clone before building). The agent must still start and converse, just without cards. → the `self.knowledge = None` branch in Task 6, covered by `test_find_cards_without_knowledge`.
4. **Student asks for a program the UAO doesn't have** ("¿Tienen ingeniería aeroespacial?"). The agent must say it isn't offered, not invent one. → replay turn plus the suspicious-name report (Task 8).
5. **Embedding model not pulled, or Ollama down at startup.** Fail fast with `ollama pull <model>`. → `OllamaEmbedder` raises SystemExit (Task 4); manual check in Task 6.

---

### Task 1: Scraper

**Why:** download the program pages, their latest study-plan PDF and the campus pages, once and politely. From then on everything works offline.

**Files:**
- Create: `knowledge/scrape.py`, `knowledge/fixtures/ingenieria-mecatronica.html`
- Modify: `pyproject.toml` (add `beautifulsoup4`, `pypdf`), `.gitignore` (add `knowledge/raw/`), `test_logic.py`

**Interfaces:**
- Produces:
  - `knowledge.scrape.BASE = "https://www.uao.edu.co"`, `RAW` (Path `knowledge/raw`), `CAMPUS` (list of page paths)
  - `program_urls(listing_html: str) -> list[str]`
  - `latest_plan_pdf(program_html: str) -> str | None`
  - Files: `knowledge/raw/programas/<slug>.html` (+ `<slug>.pdf`) and `knowledge/raw/campus/<path-with-dashes>.html`

- [ ] **Step 1: Add the dependencies and ignore raw downloads**

In `pyproject.toml` `dependencies`, after `"kokoro-onnx",` add:
```toml
    "beautifulsoup4",   # HTML parsing for the knowledge build (knowledge/extract.py)
    "pypdf",            # text from the study-plan PDFs (knowledge/extract.py)
```
Append to `.gitignore`:
```
# Raw pages downloaded by knowledge/scrape.py (rebuildable; only cards.json is committed)
knowledge/raw/
```
Run: `uv sync` → Expected: `+ beautifulsoup4`, `+ pypdf`.

- [ ] **Step 2: Save the test fixture** (a real program page, so extraction tests run offline)

```powershell
curl.exe -sL -A "Mozilla/5.0" -o knowledge/fixtures/ingenieria-mecatronica.html https://www.uao.edu.co/programa/ingenieria-mecatronica/
```
Expected: file of about 250 KB.

- [ ] **Step 3: Write the failing tests** (add `from knowledge import scrape` to the imports of `test_logic.py`, and the tests above the runner block)

```python
FIXTURE = Path(__file__).parent / "knowledge" / "fixtures" / "ingenieria-mecatronica.html"


def test_program_urls():
    html = ('<a href="https://www.uao.edu.co/programa/cine/">Cine</a>'
            '<a href="https://www.uao.edu.co/programa/cine/">otra vez</a>'
            '<a href="https://www.uao.edu.co/programa/derecho/">Derecho</a>'
            '<a href="https://www.uao.edu.co/blog/x/">no</a>')
    assert scrape.program_urls(html) == ["https://www.uao.edu.co/programa/cine/",
                                         "https://www.uao.edu.co/programa/derecho/"]


def test_latest_plan_pdf():
    # The page links an old plan (2022), the 2025 plan, a PEP and a tuition PDF: pick the 2025 plan.
    html = FIXTURE.read_text(encoding="utf-8")
    assert scrape.latest_plan_pdf(html).endswith("/2025/01/PLAN-DE-ESTUDIO-ING-METRONICA-2025.pdf")
    assert scrape.latest_plan_pdf("<p>sin pdf</p>") is None
```

- [ ] **Step 4: Run to confirm it fails**

Run: `uv run python test_logic.py` → Expected: `ModuleNotFoundError: No module named 'knowledge.scrape'` (or `ImportError`)

- [ ] **Step 5: Create `knowledge/scrape.py`**

```python
"""Downloads UAO pages and study-plan PDFs into knowledge/raw/. The only step that needs internet.

Run from the project root (rarely: when the university updates its site):
    uv run python -m knowledge.scrape

Etiquette: robots.txt allows /programa/ and /wp-content/uploads/; we never fetch URLs with "?",
wait 2 s between requests, and skip files already downloaded (delete knowledge/raw/ to refresh).
"""
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.uao.edu.co"
RAW = Path(__file__).parent / "raw"
LIST_URL = f"{BASE}/programas-de-pregrado/"
# Campus-life pages (paths under BASE). Prices/tuition pages are deliberately absent.
CAMPUS = [
    "uao-labs/laboratorios-uao",
    "bienestar-universitario-1",
    "arte-y-cultura",
    "arte-y-cultura/danza-contemporanea",
    "arte-y-cultura/expresion-artistica",
    "arte-y-cultura/iniciacion-musical",
    "arte-y-cultura/salsa-cabaret",
    "arte-y-cultura/teatro",
    "deporte-y-recreacion",
    "biblioteca",
    "nuestro-campus",
    "opciones-de-internacionalizacion",
    "grupos-de-investigacion",
]
HEADERS = {"User-Agent": "OrientadorVozBot/0.1 (proyecto academico; descarga unica de paginas publicas)"}
DELAY_S = 2.0


def program_urls(listing_html: str) -> list[str]:
    # Every undergraduate program page is /programa/<slug>/; set() removes duplicate links.
    return sorted(set(re.findall(r'href="(https://www\.uao\.edu\.co/programa/[a-z0-9-]+/)"', listing_html)))


def latest_plan_pdf(program_html: str) -> str | None:
    # WordPress stores uploads as /wp-content/uploads/<year>/<month>/<file>.pdf, so the
    # (year, month) in the URL tells us which study plan is the newest.
    best = None
    for link, year, month in set(re.findall(
            r'href="(https://www\.uao\.edu\.co/wp-content/uploads/(\d{4})/(\d{2})/[^"]+\.pdf)"', program_html, re.I)):
        name = urllib.parse.unquote(link.rsplit("/", 1)[1]).lower()
        if "plan" not in name or "valor" in name or "matricula" in name:
            continue   # not a study plan (PEP, electives, regulations) or a price document
        if best is None or (year, month) > best[0]:
            best = ((year, month), link)
    return best[1] if best else None


def fetch(url: str, dest: Path) -> bytes:
    if dest.exists():
        return dest.read_bytes()   # already downloaded: don't hit the server again
    time.sleep(DELAY_S)
    # Some file names contain accents (INGENIERÍA): percent-encode them for the HTTP request.
    req = urllib.request.Request(urllib.parse.quote(url, safe=":/%"), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"descargado {url}", flush=True)
    return data


if __name__ == "__main__":
    listing = fetch(LIST_URL, RAW / "_lista.html").decode("utf-8", "replace")
    urls = program_urls(listing)
    print(f"{len(urls)} programas", flush=True)
    for url in urls:
        slug = url.rstrip("/").rsplit("/", 1)[1]
        html = fetch(url, RAW / "programas" / f"{slug}.html").decode("utf-8", "replace")
        pdf = latest_plan_pdf(html)
        if pdf:
            fetch(pdf, RAW / "programas" / f"{slug}.pdf")
    for path in CAMPUS:
        fetch(f"{BASE}/{path}/", RAW / "campus" / f"{path.replace('/', '-')}.html")
    print("listo", flush=True)
```

- [ ] **Step 6: Run the tests to confirm they pass**

Run: `uv run python test_logic.py` → Expected: `16 checks passed`

- [ ] **Step 7: Scrape for real**

Run: `uv run python -m knowledge.scrape` (takes ~2–3 min because of the 2 s courtesy delay)
Expected: `30 programas`, then `descargado …` lines, then `listo`. `knowledge/raw/programas/` should hold 30 `.html` files and most programs a `.pdf`; `knowledge/raw/campus/` should hold 13 `.html` files.

- [ ] **Step 8: Commit and push**

```powershell
git add pyproject.toml uv.lock .gitignore knowledge/scrape.py knowledge/fixtures/ingenieria-mecatronica.html test_logic.py
git commit -m "Add polite UAO scraper for program pages, study-plan PDFs and campus pages" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 2: Text extraction (clean text, prices removed)

**Why:** turn noisy WordPress pages and scrambled PDFs into clean text. Prices are stripped deterministically **before** any LLM sees the text, so no card can contain one.

**Files:**
- Create: `knowledge/extract.py`
- Modify: `test_logic.py`

**Interfaces:**
- Produces: `page_text(html: str) -> str`, `page_title(html: str) -> str`, `strip_prices(text: str) -> str`, `pdf_text(path) -> str`

- [ ] **Step 1: Write the failing tests** (add `from knowledge import extract` to the imports)

```python
def test_strip_prices():
    text = ("Duración del programa: 9 periodos académicos - 156 créditos académicos "
            "Valor periodo académico: (16-18 créditos)*: $10.716.000 ** COP Metodología: Presencial "
            "Horario: Día*. ** Valor sujeto a cambios sin previo aviso. Los valores de matrícula no "
            "incluyen la Estampilla Procultura. Título: Ingeniero.")
    out = extract.strip_prices(text)
    assert "$" not in out and "10.716" not in out and "Estampilla" not in out
    assert "Metodología: Presencial" in out and "Título: Ingeniero." in out
    assert "$" not in extract.strip_prices("El semestre vale $ 9.500.000 aprox.")   # space after $


def test_page_text_fixture():
    html = FIXTURE.read_text(encoding="utf-8")
    t = extract.page_text(html)
    assert "Si te gusta encontrar la solución" in t      # perfil del aspirante
    assert "Fab-Lab" in t                                 # labs block (untitled section)
    assert "Énfasis de Automatización" in t              # perfil del egresado
    assert "Xpoilers" not in t                            # news section dropped
    assert "cookies" not in t.lower()                     # faculty/contact section dropped
    assert "$" not in t and "10.716" not in t            # prices stripped
    assert extract.page_title(html) == "Ingeniería Mecatrónica"


def test_page_text_generic():
    # Review Focus #1: an unknown template must still give its text, minus junk sections.
    html = ("<html><body><h1>Programa X</h1><h2>¿Qué debes saber del programa?</h2><p>Forma líderes.</p>"
            "<h2>Noticias</h2><p>Evento del lunes</p></body></html>")
    t = extract.page_text(html)
    assert "Forma líderes." in t and "Evento del lunes" not in t
```

- [ ] **Step 2: Run to confirm it fails**

Run: `uv run python test_logic.py` → Expected: `ImportError` for `knowledge.extract`

- [ ] **Step 3: Create `knowledge/extract.py`**

```python
"""Turns downloaded UAO pages and PDFs into clean text for the card maker. No network here.

Page layouts differ between programs, so we don't depend on exact section names: we keep all
text grouped by <h1>/<h2> headings and only drop the junk sections we know about.
"""
import re

from bs4 import BeautifulSoup, Comment, NavigableString

# Sections that are never useful to a student (matched in lowercase against the heading).
DROP_TITLES = ("noticias", "asesor on-line", "signin", "reset password", "próximos eventos",
               "redes a las que", "facultad de")

# Price removal. The label and the amount are removed wherever they appear...
_PRICE_LABEL = re.compile(r"Valor periodo acad[ée]mico:?\s*(\([^)]*\))?\s*\**:?", re.I)
_AMOUNT = re.compile(r"\$\s?[\d.,]+\s*\**\s*(COP)?")
# ...and whole sentences about tuition are dropped.
_PRICE_WORDS = ("matrícula", "matricula", "estampilla", "valor sujeto", "valores de")


def strip_prices(text: str) -> str:
    text = _AMOUNT.sub("", _PRICE_LABEL.sub("", text))
    paragraphs = []
    for para in text.split("\n\n"):
        sentences = re.split(r"(?<=[.!?])\s+", para)
        paragraphs.append(" ".join(s for s in sentences if not any(w in s.lower() for w in _PRICE_WORDS)))
    return "\n\n".join(p for p in paragraphs if p.strip())


def _soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "svg"]):
        tag.decompose()   # remove code, menus and forms entirely
    return soup


def page_title(html: str) -> str:
    h1 = _soup(html).find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""


def _sections(html: str) -> list[tuple[str | None, str]]:
    # Walk the page in document order; every <h1>/<h2> starts a new (title, text) section.
    # Text before the first heading gets title None (breadcrumbs, widgets): it's dropped later.
    out, title, parts = [], None, []
    for el in _soup(html).body.descendants:
        if getattr(el, "name", None) in ("h1", "h2"):
            out.append((title, " ".join(parts)))
            title, parts = el.get_text(" ", strip=True), []
        elif (isinstance(el, NavigableString) and not isinstance(el, Comment)
              and el.find_parent(["h1", "h2"]) is None):
            text = el.strip()
            if text:
                parts.append(text)
    out.append((title, " ".join(parts)))
    return out


def page_text(html: str) -> str:
    keep = []
    for title, text in _sections(html):
        if title is None or not text:
            continue
        if any(d in title.lower() for d in DROP_TITLES):
            continue
        keep.append(f"## {title}\n{text}" if title else text)   # untitled blocks (e.g. labs) kept as text
    return strip_prices("\n\n".join(keep))


def pdf_text(path) -> str:
    from pypdf import PdfReader   # imported here: only the build needs it
    reader = PdfReader(str(path))
    # Study-plan PDFs are tables: the text comes out in layout order (names split across
    # lines, course codes mixed in). The LLM in make_cards.py is told to expect that.
    return strip_prices("\n".join(page.extract_text() or "" for page in reader.pages))
```

- [ ] **Step 4: Run to confirm it passes**

Run: `uv run python test_logic.py` → Expected: `19 checks passed`.
If `test_page_text_fixture` fails on `Xpoilers` or `cookies`, print `extract._sections(html)` titles and add the offending heading (lowercase substring) to `DROP_TITLES`. Record that as a ruling.

- [ ] **Step 5: Look at the extracted text of 3 different templates**

```powershell
uv run python -c "from pathlib import Path; from knowledge.extract import page_text; [print('=====', s, '\n', page_text(Path(f'knowledge/raw/programas/{s}.html').read_text(encoding='utf-8'))[:1200]) for s in ('psicologia','administracion-virtual','diseno-industrial')]"
```
Expected: readable text for each, including "Perfil" sections and, for Diseño Industrial, "Primer semestre …" courses. No `$` anywhere.

- [ ] **Step 6: Commit and push**

```powershell
git add knowledge/extract.py test_logic.py
git commit -m "Add page/PDF text extraction with deterministic price stripping" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 3: Card maker → `knowledge/cards.json`

**Why:** the PC LLM condenses each source into short spoken cards. A source check flags any course or lab name that doesn't appear in the original text, so a human can review it.

**Files:**
- Create: `knowledge/make_cards.py`, `knowledge/cards.json` (generated, committed)
- Modify: `test_logic.py`

**Interfaces:**
- Consumes: `extract.page_text/page_title/pdf_text`, `scrape.BASE/CAMPUS/RAW`
- Produces:
  - `knowledge.make_cards.plain(text) -> str` (lowercase, no accents)
  - `missing_names(names: list[str], source: str) -> list[str]`
  - Card dicts: `{"id", "tipo", "programa", "titulo", "texto", "fuente", "revisar"[, "faltan"]}`; `tipo` ∈ `perfil|egresado|materias|laboratorios|campus`

- [ ] **Step 1: Write the failing test** (add `from knowledge import make_cards` to the imports)

```python
def test_missing_names():
    source = "Primer semestre: Cálculo 1, Álgebra lineal. Laboratorio Fab-Lab."
    # Accents and case don't matter; an invented name is reported.
    assert make_cards.missing_names(["calculo 1", "ÁLGEBRA LINEAL", "Robótica Cuántica"], source) == ["Robótica Cuántica"]
    assert make_cards.missing_names(["", "Fab-Lab"], source) == []
```

- [ ] **Step 2: Run to confirm it fails**

Run: `uv run python test_logic.py` → Expected: `ImportError` for `knowledge.make_cards`

- [ ] **Step 3: Create `knowledge/make_cards.py`**

```python
"""Turns downloaded pages into short spoken "fact cards" (knowledge/cards.json) using the PC's LLM.

Run from the project root, after knowledge.scrape:
    uv run python -m knowledge.make_cards              # everything
    uv run python -m knowledge.make_cards cine teatro  # rebuild only these (merged into cards.json)

The LLM only summarizes the text it is given (temperature 0 = deterministic, no creativity), and
answers through a JSON schema, so the output is always machine-readable. Every course/lab name
it uses is checked against the source; unknown names mark the card "revisar": true.
"""
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

from knowledge.extract import page_text, page_title, pdf_text
from knowledge.scrape import BASE, CAMPUS, RAW

CARDS = Path(__file__).parent / "cards.json"
OLLAMA = "http://127.0.0.1:11434"
BUILD_MODEL = "gemma3:4b"   # the build runs once on the PC's GPU; gemma3 writes good Spanish JSON

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
En nombres_materias y nombres_laboratorios copia exactamente los nombres de materias y laboratorios que usaste.

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


def plain(text: str) -> str:
    # Lowercase without accents, so "Cálculo" matches "calculo".
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def missing_names(names: list[str], source: str) -> list[str]:
    src = plain(source)
    return [n for n in names if n.strip() and plain(n).strip() not in src]


def ask(prompt: str, schema: dict) -> dict:
    # /api/generate with "format": <JSON schema> makes Ollama constrain the output to that schema.
    body = {"model": BUILD_MODEL, "prompt": prompt, "format": schema, "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192}}
    req = urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.load(r)["response"])


def program_cards(slug: str, html: str, plan: str) -> list[dict]:
    name, page = page_title(html), page_text(html)
    out = ask(PROGRAM_PROMPT.format(name=name, page=page[:12000], plan=plan[:8000]), PROGRAM_SCHEMA)
    source = page + "\n" + plan
    cards = []
    for tipo, names_key in (("perfil", None), ("egresado", None),
                            ("materias", "nombres_materias"), ("laboratorios", "nombres_laboratorios")):
        text = out[tipo].strip()
        if not text:
            continue   # the source had nothing for this card: omit it, never invent it
        missing = missing_names(out[names_key], source) if names_key else []
        card = {"id": f"{slug}-{tipo}", "tipo": tipo, "programa": name, "titulo": f"{name}: {TITLES[tipo]}",
                "texto": text, "fuente": f"{BASE}/programa/{slug}/", "revisar": bool(missing)}
        if missing:
            card["faltan"] = missing   # tells the human reviewer exactly what to check
        cards.append(card)
    return cards


def campus_cards(path: str, html: str) -> list[dict]:
    title = page_title(html) or path
    out = ask(CAMPUS_PROMPT.format(title=title, page=page_text(html)[:12000]), CAMPUS_SCHEMA)
    base_id = path.replace("/", "-")
    return [{"id": f"{base_id}-{i}", "tipo": "campus", "programa": None, "titulo": f["titulo"].strip(),
             "texto": f["texto"].strip(), "fuente": f"{BASE}/{path}/", "revisar": False}
            for i, f in enumerate(out["fichas"][:4]) if f["texto"].strip()]


if __name__ == "__main__":
    only = set(sys.argv[1:])   # optional: rebuild just these program slugs / campus ids
    old = json.loads(CARDS.read_text(encoding="utf-8")) if CARDS.exists() and only else []
    cards = [c for c in old if not any(c["id"].startswith(s + "-") for s in only)]
    for html_path in sorted((RAW / "programas").glob("*.html")):
        slug = html_path.stem
        if only and slug not in only:
            continue
        pdf = html_path.with_suffix(".pdf")
        new = program_cards(slug, html_path.read_text(encoding="utf-8"), pdf_text(pdf) if pdf.exists() else "")
        cards += new
        print(f"{slug}: {len(new)} fichas", flush=True)
    for path in CAMPUS:
        base_id = path.replace("/", "-")
        if only and base_id not in only:
            continue
        new = campus_cards(path, (RAW / "campus" / f"{base_id}.html").read_text(encoding="utf-8"))
        cards += new
        print(f"{base_id}: {len(new)} fichas", flush=True)
    CARDS.write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
    review = [c["id"] for c in cards if c["revisar"]]
    print(f"\n{len(cards)} fichas en {CARDS.name}. Para revisar ({len(review)}): {review}")
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run python test_logic.py` → Expected: `20 checks passed`

- [ ] **Step 5: Try 2 programs and 1 campus page, then read the result**

```powershell
uv run python -m knowledge.make_cards ingenieria-mecatronica psicologia deporte-y-recreacion
uv run python -c "import json; [print(c['id'], c['revisar'], c.get('faltan',''), '\n ', c['texto'], '\n') for c in json.load(open('knowledge/cards.json', encoding='utf-8'))]"
```
Expected: ~4 cards for each program and 1–4 for deportes, each 50–70 words of natural Spanish, with no prices. Mecatrónica's labs card mentions the Fab-Lab. If the cards are poor (lists, English, invented facts), adjust `PROGRAM_PROMPT` wording and rerun before continuing. Record any prompt change as a ruling.

- [ ] **Step 6: Build everything**

Run: `uv run python -m knowledge.make_cards` (≈ 5–10 min on the GPU)
Expected: one line per program and campus page, then a total of about 130–170 cards and a `revisar` list. Open `knowledge/cards.json` and read every card in the `revisar` list. Fix or delete wrong names by hand, and set `"revisar": false` once a card is checked.

- [ ] **Step 7: Commit and push**

```powershell
git add knowledge/make_cards.py knowledge/cards.json test_logic.py
git commit -m "Add LLM card maker with source check; generate UAO knowledge cards" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 4: Runtime search (`rag.py`)

**Why:** find the 1–3 cards relevant to what the student just said, in ~100 ms on the CPU, and decide the turn's mode (detail, recommendation or price hand-over).

**Files:**
- Create: `rag.py`
- Modify: `test_logic.py`

**Interfaces:**
- Produces:
  - `rag.CARDS`, `rag.VECTORS`, `rag.PRICE_CARD` (dict with `titulo`, `texto`)
  - `OllamaEmbedder(host: str, model: str)`, callable as `(texts: list[str], query: bool = False) -> np.ndarray` (unit-length rows)
  - `Knowledge(embed, model_name: str, cards_path=CARDS, cache_path=VECTORS)`: `.cards`, `.ranked(query) -> list[tuple[float, dict]]`, `.search(query, k, threshold, kinds=None) -> list[dict]`, `.program_names() -> list[str]`
  - `choose_mode(text: str, turn: int, recommend_after: int) -> "precio"|"recomendacion"|"detalle"`
  - `retrieve(kb, text, user_turns: list[str], last_reply: str, cfg) -> tuple[str, list[dict]]` (`user_turns` already includes `text`)

- [ ] **Step 1: Write the failing tests** (add `import json` and `import rag` to the imports)

```python
_VEC = {"A. perfil a": [1, 0, 0], "B. labs b": [0, 1, 0], "C. campus c": [0, 0, 1],
        "q-ab": [0.8, 0.6, 0], "nada": [-1, 0, 0]}


def _fake_embed(calls):
    def embed(texts, query=False):
        calls.append(len(texts))
        v = np.array([_VEC[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return embed


def _kb(calls, tmp):
    cards = [{"id": "a", "tipo": "perfil", "programa": "A", "titulo": "A", "texto": "perfil a"},
             {"id": "b", "tipo": "laboratorios", "programa": "A", "titulo": "B", "texto": "labs b"},
             {"id": "c", "tipo": "campus", "programa": None, "titulo": "C", "texto": "campus c"}]
    (tmp / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    return rag.Knowledge(_fake_embed(calls), "fake", tmp / "cards.json", tmp / "v.npz")


def test_knowledge_search():
    tmp = Path(tempfile.mkdtemp())
    kb = _kb([], tmp)
    ids = lambda cards: [c["id"] for c in cards]
    assert ids(kb.search("q-ab", k=2, threshold=0.3)) == ["a", "b"]     # ordered by similarity
    assert ids(kb.search("q-ab", k=2, threshold=0.7)) == ["a"]          # threshold cuts b (0.6)
    assert ids(kb.search("q-ab", k=1, threshold=0.3)) == ["a"]          # k limit
    assert ids(kb.search("q-ab", k=2, threshold=0.3, kinds={"laboratorios"})) == ["b"]
    assert kb.search("nada", k=2, threshold=0.3) == []                  # nothing similar → nothing
    assert kb.program_names() == ["A"]


def test_vector_cache():
    tmp, calls = Path(tempfile.mkdtemp()), []
    _kb(calls, tmp)
    _kb(calls, tmp)                          # same cards.json → vectors come from the cache
    assert calls == [3]
    cards = json.loads((tmp / "cards.json").read_text(encoding="utf-8"))
    cards[0]["texto"] = "labs b"; cards[0]["titulo"] = "B"
    (tmp / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    rag.Knowledge(_fake_embed(calls), "fake", tmp / "cards.json", tmp / "v.npz")
    assert calls == [3, 3]                   # file changed → recomputed


def test_choose_mode():
    assert rag.choose_mode("¿Cuánto cuesta el semestre?", 1, 6) == "precio"
    assert rag.choose_mode("¿Hay becas?", 1, 6) == "precio"
    assert rag.choose_mode("¿Qué carrera me recomiendas?", 2, 6) == "recomendacion"
    assert rag.choose_mode("Me gusta el fútbol", 6, 6) == "recomendacion"   # automatic, once
    assert rag.choose_mode("Me gusta el fútbol", 7, 6) == "detalle"
    assert rag.choose_mode("¿Qué laboratorios hay?", 2, 6) == "detalle"


def test_retrieve():
    calls = []

    class KB:
        def search(self, query, k, threshold, kinds=None):
            calls.append((query, k, threshold, kinds))
            return [{"id": "x"}]

    cfg = SimpleNamespace(recommend_after_turns=6, rag_max_cards=2, rag_threshold=0.45)
    assert rag.retrieve(KB(), "¿cuánto vale?", ["¿cuánto vale?"], "", cfg) == ("precio", [rag.PRICE_CARD])
    assert calls == []                                                   # no search for prices
    mode, _ = rag.retrieve(KB(), "sí, esa", ["hola", "sí, esa"], "¿Te gusta la robótica?", cfg)
    assert mode == "detalle" and calls[-1] == ("¿Te gusta la robótica? sí, esa", 2, 0.45, None)
    mode, _ = rag.retrieve(KB(), "¿qué me recomiendas?", ["me gusta dibujar", "¿qué me recomiendas?"], "", cfg)
    assert mode == "recomendacion" and calls[-1] == ("me gusta dibujar ¿qué me recomiendas?", 3, -1.0, {"perfil"})
```

- [ ] **Step 2: Run to confirm it fails**

Run: `uv run python test_logic.py` → Expected: `ModuleNotFoundError: No module named 'rag'`

- [ ] **Step 3: Create `rag.py`**

```python
"""Runtime search over the UAO fact cards (knowledge/cards.json). Offline; PC and Pi.

How semantic search works: an embedding model turns a text into a vector (a list of numbers)
such that texts with similar MEANING get vectors pointing in similar directions. We embed every
card once (cached), embed what the student said, and rank cards by cosine similarity (the dot
product of unit-length vectors: 1 = same meaning, 0 = unrelated).
"""
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
CARDS = ROOT / "knowledge" / "cards.json"
VECTORS = ROOT / "models" / "knowledge_vectors.npz"   # cache, rebuilt when cards.json changes

# Price questions get this instead of a search: the robot hands over to humans.
PRICE_CARD = {"id": "precios", "titulo": "Precios y becas",
              "texto": "El estudiante pregunta por precios, matrícula o becas. No des cifras: dile con "
                       "amabilidad que eso lo explica mejor un asesor de admisiones de la UAO."}
PRICE_WORDS = ("precio", "cuánto cuesta", "cuanto cuesta", "cuánto vale", "cuanto vale", "matrícula",
               "matricula", "beca", "financiación", "financiacion", "valor del semestre")
RECOMMEND_WORDS = ("recomiendas", "recomendarías", "recomienda", "qué carrera", "que carrera",
                   "qué puedo estudiar", "que puedo estudiar", "qué debería estudiar", "que deberia estudiar")
# Some embedding models are trained with instruction prefixes; using them improves retrieval.
EMBED_PREFIXES = {"embeddinggemma": ("task: search result | query: ", "title: none | text: ")}


class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.url, self.model = host.rstrip("/"), model

    def __call__(self, texts: list[str], query: bool = False) -> np.ndarray:
        q_prefix, d_prefix = EMBED_PREFIXES.get(self.model.split(":")[0], ("", ""))
        prefix = q_prefix if query else d_prefix
        body = {"model": self.model, "input": [prefix + t for t in texts], "keep_alive": -1,
                "options": {"num_gpu": 0}}   # CPU only: the GPU's 4 GB belong to the LLM
        req = urllib.request.Request(f"{self.url}/api/embed", json.dumps(body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                vectors = np.array(json.load(r)["embeddings"], dtype=np.float32)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Falla el modelo de embeddings '{self.model}' ({e}). Ejecuta: ollama pull {self.model}")
        except urllib.error.URLError:
            raise SystemExit(f"Ollama no responde en {self.url}. Abre la app de Ollama (o ejecuta: ollama serve)")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # unit length → dot = cosine


class Knowledge:
    def __init__(self, embed, model_name: str, cards_path=CARDS, cache_path=VECTORS):
        raw = Path(cards_path).read_bytes()
        self.cards, self.embed = json.loads(raw), embed
        # The cache key changes if the cards OR the embedding model change.
        key = hashlib.sha256(raw + model_name.encode()).hexdigest()
        cache = Path(cache_path)
        if cache.exists():
            data = np.load(cache)
            if str(data["key"]) == key:
                self.vectors = data["vectors"]
                return
        self.vectors = embed([f"{c['titulo']}. {c['texto']}" for c in self.cards])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, key=key, vectors=self.vectors)

    def ranked(self, query: str) -> list[tuple[float, dict]]:
        scores = self.vectors @ self.embed([query], query=True)[0]   # one dot product per card
        return [(float(scores[i]), self.cards[i]) for i in np.argsort(-scores)]

    def search(self, query: str, k: int, threshold: float, kinds=None) -> list[dict]:
        hits = []
        for score, card in self.ranked(query):
            if score < threshold:
                break   # sorted: everything after this is even less similar
            if kinds and card["tipo"] not in kinds:
                continue
            hits.append(card)
            if len(hits) == k:
                break
        return hits

    def program_names(self) -> list[str]:
        return sorted({c["programa"] for c in self.cards if c["programa"]})


def choose_mode(text: str, turn: int, recommend_after: int) -> str:
    low = text.lower()
    if any(w in low for w in PRICE_WORDS):
        return "precio"
    if any(w in low for w in RECOMMEND_WORDS) or turn == recommend_after:
        return "recomendacion"
    return "detalle"


def retrieve(kb, text: str, user_turns: list[str], last_reply: str, cfg) -> tuple[str, list[dict]]:
    mode = choose_mode(text, len(user_turns), cfg.recommend_after_turns)
    if mode == "precio":
        return mode, [PRICE_CARD]
    if mode == "recomendacion":
        # Everything the student has said vs. the "who is this program for" cards; always top 3.
        return mode, kb.search(" ".join(user_turns), 3, -1.0, kinds={"perfil"})
    # Detail: the agent's last question gives context to short answers like "sí, esa".
    return mode, kb.search(f"{last_reply} {text}".strip(), cfg.rag_max_cards, cfg.rag_threshold)
```

- [ ] **Step 4: Run to confirm it passes**

Run: `uv run python test_logic.py` → Expected: `24 checks passed`

- [ ] **Step 5: Commit and push**

```powershell
git add rag.py test_logic.py
git commit -m "Add runtime card search with cached embeddings and per-turn retrieval modes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 5: Grounded brain

**Why:** the LLM must see the official program list on every turn and the relevant cards on the turns that need them, without breaking Ollama's prompt cache.

**Files:**
- Modify: `brain.py`, `test_logic.py`

**Interfaces:**
- Produces:
  - `Brain(host, model, threads, num_gpu, max_turns=10, program_names=())`
  - `.system: str`
  - `.stream_reply(user_text, stop=None, cards=())`
  - `llm_worker(brain, text, out_q, stop, cards=())`

- [ ] **Step 1: Write the failing test and update the old fake**

In `test_llm_worker_always_ends`, change the fake to accept cards:
```python
        def stream_reply(self, text, stop, cards=()):
```
Add:
```python
class _FakeResp:
    # Stands in for the HTTP response: a context manager that yields NDJSON lines.
    def __init__(self, lines): self.lines = lines
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def __iter__(self): return iter(self.lines)


def test_brain_injects_cards_only_in_current_turn():
    b = brain.Brain("http://x", "m", 4, 0, program_names=["Ingeniería Mecatrónica"])
    sent = []

    def fake_post(path, body):
        sent.append(body)
        return _FakeResp([json.dumps({"message": {"content": "Hola."}, "done": True}).encode()])

    b._post = fake_post
    card = {"titulo": "Mecatrónica: laboratorios", "texto": "Tiene Fab-Lab."}
    list(b.stream_reply("¿Qué labs hay?", cards=[card]))
    list(b.stream_reply("Gracias"))
    first, second = sent
    assert "Tiene Fab-Lab." in first["messages"][-1]["content"]           # card in the current turn
    assert first["messages"][0] == second["messages"][0]                  # system prompt identical
    assert "Ingeniería Mecatrónica" in first["messages"][0]["content"]    # official list in system
    assert all("Fab-Lab" not in m["content"] for m in second["messages"])  # card not kept in history
    assert b.history[0] == {"role": "user", "content": "¿Qué labs hay?"}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `uv run python test_logic.py` → Expected: `TypeError: Brain.__init__() got an unexpected keyword argument 'program_names'`

- [ ] **Step 3: Change `brain.py`**

After `SYSTEM_PROMPT = """…"""`, add:
```python
# Added to the system prompt when knowledge cards exist. Built ONCE per run so the system prompt
# never changes between turns: Ollama can then reuse its cached reading of it (prompt cache).
RAG_RULES = """

Reglas sobre la universidad:
Solo menciona programas, materias, laboratorios y servicios que aparezcan en la "Información \
verificada de la UAO" o en la lista oficial de programas de abajo.
Si no tienes la información, dilo con honestidad y sugiere hablar con un asesor de la UAO.
Nunca des precios, valores de matrícula ni montos de becas.
Programas de pregrado de la UAO: {names}."""
```
Change the constructor signature and add `self.system`:
```python
    def __init__(self, host: str, model: str, threads: int, num_gpu: int, max_turns: int = 10,
                 program_names=()):
        self.url = host.rstrip("/")
        self.model = model
        self.system = SYSTEM_PROMPT + (RAG_RULES.format(names=", ".join(program_names)) if program_names else "")
```
(keep the rest of `__init__` as is). In `warmup()`, replace `"content": SYSTEM_PROMPT` with `"content": self.system`.
Replace the start of `stream_reply` up to `reply = ""` with:
```python
    def stream_reply(self, user_text: str, stop=None, cards=()):
        content = user_text
        if cards:
            # Verified facts travel ONLY with this turn's message; the history keeps the plain
            # text, so later turns don't carry (and re-read) old cards.
            facts = "\n".join(f"- {c['titulo']}: {c['texto']}" for c in cards)
            content = f"Información verificada de la UAO:\n{facts}\n\nEstudiante: {user_text}"
        self.history.append({"role": "user", "content": user_text})
        # Keep an odd number of messages so the history always starts with a user message.
        self.history = self.history[-(2 * self.max_turns - 1):]
        body = {
            "model": self.model, "stream": True, "keep_alive": -1, "options": self.options,
            "messages": [{"role": "system", "content": self.system}] + self.history[:-1]
                        + [{"role": "user", "content": content}],
        }
        reply = ""
```
Change `llm_worker`:
```python
def llm_worker(brain, text: str, out_q, stop, cards=()) -> None:
```
and inside it `for token in brain.stream_reply(text, stop, cards):`.

- [ ] **Step 4: Run to confirm it passes**

Run: `uv run python test_logic.py` → Expected: `25 checks passed`

- [ ] **Step 5: Commit and push**

```powershell
git add brain.py test_logic.py
git commit -m "Ground the brain: official program list in system prompt, cards only in the current turn" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 6: Wire retrieval into the voice loop

**Why:** connect `rag.retrieve()` to `main.py`, so every spoken turn gets the right cards, and add the settings.

**Files:**
- Modify: `main.py`, `config.toml`, `test_logic.py`

**Interfaces:**
- Consumes: `rag.CARDS/Knowledge/OllamaEmbedder/retrieve`, `Brain(program_names=…)`, `llm_worker(…, cards)`
- Produces: `Agent.find_cards(text) -> list[dict]`, `Agent.knowledge`, `Agent.user_turns`, `Agent.last_reply`

- [ ] **Step 1: Write the failing test and update the test helpers**

In `_agent()`, after the `a.mic, a.vad, …` line, add:
```python
    a.knowledge, a.user_turns, a.last_reply = None, [], ""
```
In `_filler_run`'s `SlowBrain`, change the signature to `def stream_reply(self, text, stop, cards=()):`.
Add:
```python
def test_find_cards_without_knowledge():
    # Review Focus #3: no cards.json → the agent still works, just without cards.
    a = _agent([], barge_in=False)
    assert a.find_cards("¿Qué laboratorios hay?") == []
    assert a.user_turns == ["¿Qué laboratorios hay?"]
```

- [ ] **Step 2: Run to confirm it fails**

Run: `uv run python test_logic.py` → Expected: `AttributeError: 'Agent' object has no attribute 'find_cards'`

- [ ] **Step 3: Add the settings to `config.toml`** (in `[common]`, after `ollama_host`)

```toml
# Knowledge (RAG) — see docs/superpowers/specs/2026-09-26-uao-knowledge-rag-design.md
embed_model = "embeddinggemma"   # Ollama embedding model (CPU); chosen by knowledge/eval.py
rag_threshold = 0.45             # minimum similarity for a card to be used in a detail turn
rag_max_cards = 2                # cards per detail turn (each ~60 words ≈ +1.5 s on a Pi)
recommend_after_turns = 6        # on this turn the agent recommends programs even if not asked
```

- [ ] **Step 4: Change `main.py`**

Add the import:
```python
from rag import CARDS, Knowledge, OllamaEmbedder, retrieve
```
In `Agent.__init__`, replace the `self.brain = Brain(...)` line with:
```python
        # Knowledge cards are optional: without knowledge/cards.json the agent still converses.
        self.knowledge = None
        if CARDS.exists():
            self.knowledge = Knowledge(OllamaEmbedder(cfg.ollama_host, cfg.embed_model), cfg.embed_model)
        names = self.knowledge.program_names() if self.knowledge else ()
        self.brain = Brain(cfg.ollama_host, cfg.llm_model, cfg.threads, cfg.llm_num_gpu, program_names=names)
```
and next to `self.ready_turn = None` add:
```python
        self.user_turns = []               # everything the student said (recommendation query)
        self.last_reply = ""               # the agent's last answer (context for detail queries)
```
Add the method to `Agent` (after `user_interrupting`):
```python
    def find_cards(self, text: str) -> list[dict]:
        self.user_turns.append(text)
        if self.knowledge is None:
            return []
        mode, cards = retrieve(self.knowledge, text, self.user_turns, self.last_reply, self.cfg)
        print(f"[rag] {mode}: {', '.join(c['id'] for c in cards) or '-'}", flush=True)
        return cards
```
In `respond()`, right after `print(f"\nTú: {text}", flush=True)`, add `cards = self.find_cards(text)` and change the LLM thread line to:
```python
        threading.Thread(target=llm_worker, args=(self.brain, text, tokens, stop, cards), daemon=True).start()
```
After `print(f"Orienta: {' '.join(spoken)}", flush=True)`, add:
```python
        self.last_reply = " ".join(spoken)
```

- [ ] **Step 5: Run to confirm it passes**

Run: `uv run python test_logic.py` → Expected: `26 checks passed`

- [ ] **Step 6: Pull the embedding model and do a startup check**

```powershell
ollama pull embeddinggemma
uv run python -c "import main; from config import load_config; a = main.Agent(load_config('pc')); print(len(a.knowledge.cards), 'fichas'); print(a.find_cards('¿Qué laboratorios tiene mecatrónica?'))"
```
Expected: the card count, then `[rag] detalle: ingenieria-mecatronica-laboratorios, …` and the card list.
Then run `ollama rm embeddinggemma`, rerun the same command, and expect `Falla el modelo de embeddings … ollama pull embeddinggemma`. Pull it again afterwards.

- [ ] **Step 7: Commit and push**

```powershell
git add main.py config.toml test_logic.py
git commit -m "Wire UAO card retrieval into the voice loop" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 7: Retrieval evaluation → choose embedding model and threshold

**Why:** pick the embedding model and `rag_threshold` with numbers, not by eye.

**Files:**
- Create: `knowledge/eval.py`
- Modify: `config.toml` (only if the numbers say so)

**Interfaces:**
- Consumes: `rag.Knowledge`, `rag.OllamaEmbedder`, `config.load_config`

- [ ] **Step 1: Create `knowledge/eval.py`**

```python
"""Retrieval evaluation: does the search find the right card for real student questions?

Run: uv run python -m knowledge.eval embeddinggemma granite-embedding:278m
For each model and threshold it prints how many questions got an acceptable card in the top 2
(or, for off-topic questions, correctly got nothing).
"""
import sys

from config import load_config
from rag import ROOT, Knowledge, OllamaEmbedder

# (question, acceptable card-id prefixes; None = nothing should be retrieved)
QUESTIONS = [
    ("¿Qué laboratorios tiene mecatrónica?", ("ingenieria-mecatronica-laboratorios",)),
    ("¿Qué materias se ven en ingeniería mecatrónica?", ("ingenieria-mecatronica-materias",)),
    ("Me gusta la robótica y armar cosas", ("ingenieria-mecatronica", "ingenieria-electronica")),
    ("Quiero ayudar a las personas con sus problemas emocionales", ("psicologia",)),
    ("Me interesan las leyes y la justicia", ("derecho",)),
    ("Me gusta dibujar y diseñar productos", ("diseno-industrial", "diseno-de-la-comunicacion-grafica")),
    ("Quiero hacer películas", ("cine", "narrativas-y-entretenimiento-digital")),
    ("Me gusta la inteligencia artificial y los datos", ("ingenieria-de-datos",)),
    ("Quiero trabajar cuidando el medio ambiente", ("ingenieria-ambiental",)),
    ("¿Qué hace un ingeniero biomédico?", ("ingenieria-biomedica",)),
    ("Me gustan los negocios y el comercio internacional", ("mercadeo", "administracion")),
    ("¿Hay carreras virtuales?", ("administracion-virtual", "contaduria-publica1-virtual",
                                  "ingenieria-informatica-virtual", "mercadeo-global-virtual",
                                  "publicidad-virtual", "virtual-ingenieria-industrial",
                                  "ingenieria-electronica-y-telecomunicaciones-virtual")),
    ("¿Dónde puedo hacer deporte en la universidad?", ("deporte-y-recreacion",)),
    ("¿Tienen equipo de fútbol?", ("deporte-y-recreacion",)),
    ("¿Hay clases de teatro?", ("arte-y-cultura",)),
    ("Me gusta bailar salsa", ("arte-y-cultura",)),
    ("¿Cómo es la biblioteca?", ("biblioteca",)),
    ("¿Puedo hacer un intercambio en otro país?", ("opciones-de-internacionalizacion",)),
    ("¿Los laboratorios abren de noche?", ("uao-labs",)),
    ("¿Hay apoyo psicológico para los estudiantes?", ("bienestar-universitario",)),
    ("¿Cómo es el campus?", ("nuestro-campus",)),
    ("¿Puedo investigar siendo estudiante?", ("grupos-de-investigacion",)),
    ("¿Quién ganó el partido de ayer?", None),
    ("¿Qué hora es?", None),
    ("Me gusta comer pizza", None),
]
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def correct(ranked, expected, threshold) -> bool:
    top2 = [card for score, card in ranked if score >= threshold][:2]
    if expected is None:
        return not top2
    return any(card["id"].startswith(p) for card in top2 for p in expected)


if __name__ == "__main__":
    host = load_config("pc").ollama_host
    for model in sys.argv[1:] or ["embeddinggemma"]:
        cache = ROOT / "models" / f"eval_{model.replace(':', '-')}.npz"
        kb = Knowledge(OllamaEmbedder(host, model), model, cache_path=cache)
        ranked = [kb.ranked(q) for q, _ in QUESTIONS]   # embed each question once
        print(f"\n=== {model}")
        for t in THRESHOLDS:
            ok = sum(correct(r, exp, t) for r, (_, exp) in zip(ranked, QUESTIONS))
            print(f"umbral {t:.2f}: {ok}/{len(QUESTIONS)} correctas")
        for (q, exp), r in zip(QUESTIONS, ranked):
            score, card = r[0]
            print(f"  {score:.2f} {card['id']:45} ← {q}")
```

- [ ] **Step 2: Pull the second candidate and run the evaluation**

```powershell
ollama pull granite-embedding:278m
uv run python -m knowledge.eval embeddinggemma granite-embedding:278m
```
Expected: a table per model. Pick the model and threshold with the most correct answers. On a tie, prefer the higher threshold, since it injects fewer irrelevant cards and so is faster on the Pi. If an expected card id doesn't exist (the card maker named it differently), fix the prefix in `QUESTIONS`, not the result.

- [ ] **Step 3: Set the winners in `config.toml`**

Update `embed_model` and `rag_threshold` in `[common]` with the chosen values and write the score in the comment (e.g. `# 22/25 in knowledge/eval.py`).

- [ ] **Step 4: Commit and push**

```powershell
git add knowledge/eval.py config.toml
git commit -m "Add retrieval evaluation; choose embedding model and threshold" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 8: End-to-end replay and README

**Why:** verify the whole chain (retrieval → grounded LLM) on both profiles, catch invented program names, and measure what cards cost on the `pi` profile.

**Files:**
- Create: `knowledge/replay.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `rag.Knowledge/OllamaEmbedder/retrieve`, `brain.Brain`, `sentences.SentenceSplitter`, `knowledge.make_cards.plain`

- [ ] **Step 1: Create `knowledge/replay.py`**

```python
"""End-to-end check without audio: replays student turns through retrieval + the grounded LLM.

Run: uv run python -m knowledge.replay pc|pi
Prints each reply with its mode, cards and time to first sentence, and flags any
"Ingeniería X"-style program name that isn't in the official list (possible invention).
"""
import re
import sys
import time

from brain import Brain
from config import load_config
from knowledge.make_cards import plain
from rag import Knowledge, OllamaEmbedder, retrieve
from sentences import SentenceSplitter

TURNS = [
    "Hola, ¿cómo estás?",
    "Estoy viendo la universidad y me gustaría ver qué carrera es la mejor para mí.",
    "Me gusta jugar fútbol, comer y arreglar carros.",
    "Creo que crear algo y resolver problemas.",
    "Me gusta mucho la tecnología pero también lo físico, construir cosas.",
    "¿Qué me recomiendas estudiar?",
    "¿Qué laboratorios tiene mecatrónica?",
    "¿Qué materias se ven en los primeros semestres?",
    "¿Hay equipo de fútbol en la universidad?",
    "¿Cuánto cuesta el semestre?",
    "¿Tienen ingeniería aeroespacial?",
]
NAME_RE = re.compile(r"(?:Ingenier[ií]a|Licenciatura|Tecnolog[ií]a)(?: (?:de|en|del))? [A-ZÁÉÍÓÚ][a-záéíóúñ]+")

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    kb = Knowledge(OllamaEmbedder(cfg.ollama_host, cfg.embed_model), cfg.embed_model)
    official = [plain(n) for n in kb.program_names()]
    brain = Brain(cfg.ollama_host, cfg.llm_model, cfg.threads, cfg.llm_num_gpu, program_names=kb.program_names())
    brain.check()
    brain.warmup()
    user_turns, last = [], ""
    for text in TURNS:
        user_turns.append(text)
        t0 = time.perf_counter()
        mode, cards = retrieve(kb, text, user_turns, last, cfg)
        splitter, first, reply = SentenceSplitter(), None, ""
        for token in brain.stream_reply(text, cards=cards):
            reply += token
            if first is None and splitter.feed(token):
                first = time.perf_counter() - t0
        first = first or time.perf_counter() - t0
        last = reply
        suspicious = [m for m in NAME_RE.findall(reply) if not any(plain(m) in o for o in official)]
        print(f"\nTú: {text}\n[{mode}: {', '.join(c['id'] for c in cards) or '-'} | 1ª frase {first:.2f}s]")
        print(f"Orienta: {reply.strip()}")
        if suspicious:
            print(f"  ⚠ revisar nombres: {suspicious}")
```

- [ ] **Step 2: Replay on both profiles**

```powershell
uv run python -m knowledge.replay pc
uv run python -m knowledge.replay pi
```
Expected on both:
- **Recommendation turn:** names real UAO programs, e.g. Mecatrónica or Mecánica.
- **Labs and courses turns:** use the Mecatrónica cards.
- **Fútbol turn:** uses a deportes card.
- **Price turn:** hands over to an advisor, with no figures.
- **Aeroespacial turn:** says it isn't offered. A `⚠ revisar` flag on "Ingeniería Aeroespacial" is acceptable only if the reply says it doesn't exist.

On `pi`, compare the `1ª frase` times of turns with cards against turns without, and write down the difference.
If a reply invents a program or a fact, record it and adjust `RAG_RULES` or the card text. Do not change the tests.

- [ ] **Step 3: Add a section to `README.md`**

````markdown
## UAO knowledge (RAG)

The agent answers from `knowledge/cards.json`: short fact cards about the Universidad Autónoma de
Occidente's programs and campus life, generated from the public website and reviewed by humans.

```
uv run python -m knowledge.scrape        # download pages (internet; ~3 min, polite 2 s delay)
uv run python -m knowledge.make_cards    # write cards with the PC LLM (~5–10 min); review "revisar" ones
uv run python -m knowledge.eval embeddinggemma   # retrieval quality
uv run python -m knowledge.replay pc     # end-to-end check without audio
```

To update: rerun scrape (delete `knowledge/raw/` first) and make_cards, review `git diff knowledge/cards.json`, commit.
Prices, tuition and scholarships are never answered; the agent hands those over to admissions staff.
````
Add below it the measured numbers from Step 2 (card turn vs no-card turn on `pi`).

- [ ] **Step 4: Run all the checks and commit**

Run: `uv run python test_logic.py` → Expected: `26 checks passed`
```powershell
git add knowledge/replay.py README.md
git commit -m "Add end-to-end knowledge replay and document the RAG workflow" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
