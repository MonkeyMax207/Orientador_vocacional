"""Turns downloaded UAO pages and PDFs into clean text for the card maker. No network here.

Page layouts differ between programs, so we don't depend on exact section names: we keep all
text grouped by <h1>/<h2> headings and only drop the junk sections we know about.
"""
import re

from bs4 import BeautifulSoup, Comment, NavigableString

# Sections that are never useful to a student (matched in lowercase against the heading).
DROP_TITLES = ("noticias", "asesor on-line", "signin", "reset password", "próximos eventos",
               "redes a las que", "facultad de")

# A real heading is short; the longest useful one seen ("Código Snies: … vigencia 7 años") is ~20 words.
MAX_HEADING_WORDS = 25

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
    soup = _soup(html)
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    # Some templates (virtual programs) have no <h1>. Their <title> holds the name plus SEO text,
    # e.g. "¿Administración de Empresas Virtual – UAO | Gestión y Liderazgo": keep what comes before
    # the first separator, then drop a trailing "UAO" / "en la UAO" and stray punctuation.
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    name = re.split(r"\s*[|–]\s*|\s-\s", title)[0]
    return re.sub(r"\s+(en la\s+)?UAO\b.*$", "", name).strip(" ¿?¡!.")


def _sections(html: str) -> list[tuple[str | None, str]]:
    # Walk the page in document order; every <h1>/<h2> starts a new (title, text) section.
    # Text before the first heading gets title None (breadcrumbs, widgets): it's dropped later.
    out, title, parts = [], None, []
    for el in _soup(html).body.descendants:
        if getattr(el, "name", None) in ("h1", "h2"):
            out.append((title, " ".join(parts)))
            heading = el.get_text(" ", strip=True)
            if len(heading.split()) > MAX_HEADING_WORDS:
                # Some pages wrap a whole table (e.g. the study plan) in an <h2>: that's content.
                title, parts = "", [heading]
            else:
                title, parts = heading, []
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
