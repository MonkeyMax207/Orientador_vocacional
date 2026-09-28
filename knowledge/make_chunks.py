"""Splits the downloaded UAO pages and study-plan PDFs into overlapping text chunks for search
(knowledge/chunks.json). No LLM involved: the raw text itself is the knowledge, so nothing is
summarized or invented. On the PC the LLM reads 3 chunks (~450 words) in ~0.3 s.

Run after knowledge.scrape:  uv run python -m knowledge.make_chunks
"""
import json
from pathlib import Path

from knowledge.extract import page_text, page_title, pdf_text
from knowledge.scrape import BASE, CAMPUS, RAW

CHUNKS = Path(__file__).parent / "chunks.json"
SIZE, STRIDE = 150, 120   # 150-word windows; 30 words of overlap so no fact is cut in half


def chunk_words(text: str, size: int = SIZE, stride: int = STRIDE) -> list[str]:
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break          # this window already reached the end of the text
        i += stride
    return chunks


def chunks_of(base_id: str, title: str, programa, text: str, url: str) -> list[dict]:
    # Same fields as the old cards, so rag.Knowledge can search them unchanged.
    return [{"id": f"{base_id}-{i}", "tipo": "texto", "programa": programa, "titulo": title,
             "texto": chunk, "fuente": url} for i, chunk in enumerate(chunk_words(text))]


if __name__ == "__main__":
    chunks = []
    for html_path in sorted((RAW / "programas").glob("*.html")):
        html, slug = html_path.read_text(encoding="utf-8"), html_path.stem
        name, url = page_title(html), f"{BASE}/programa/{slug}/"
        chunks += chunks_of(slug, name, name, page_text(html), url)
        pdf = html_path.with_suffix(".pdf")
        if pdf.exists():   # course lists: title says so, so "materias de X" finds them
            chunks += chunks_of(f"{slug}-plan", f"{name}: plan de estudios y materias", name, pdf_text(pdf), url)
    for path in CAMPUS:
        base_id = path.replace("/", "-")
        html = (RAW / "campus" / f"{base_id}.html").read_text(encoding="utf-8")
        chunks += chunks_of(base_id, page_title(html) or path, None, page_text(html), f"{BASE}/{path}/")
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(chunks)} fragmentos en {CHUNKS.name}")
