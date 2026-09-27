"""Downloads UAO pages and study-plan PDFs into knowledge/raw/. The only step that needs internet.

Run from the project root (rarely: when the university updates its site):
    uv run python -m knowledge.scrape

Etiquette: robots.txt allows /programa/ and /wp-content/uploads/; we never fetch URLs with "?",
wait 2 s between requests, and skip files already downloaded (delete knowledge/raw/ to refresh).
"""
import re
import time
import urllib.error
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
    "investigaciones-uao",   # was /grupos-de-investigacion/ (404 since 2026-09)
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


def fetch(url: str, dest: Path) -> bytes | None:
    if dest.exists():
        return dest.read_bytes()   # already downloaded: don't hit the server again
    time.sleep(DELAY_S)
    # Some file names contain accents (INGENIERÍA): percent-encode them for the HTTP request.
    req = urllib.request.Request(urllib.parse.quote(url, safe=":/%"), headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        # A page the university removed (404) must not stop the whole download: warn and move on.
        print(f"AVISO: {url} → {e.code}, se omite", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"descargado {url}", flush=True)
    return data


if __name__ == "__main__":
    listing = fetch(LIST_URL, RAW / "_lista.html")
    if listing is None:
        raise SystemExit(f"No se pudo descargar la lista de programas ({LIST_URL})")
    listing = listing.decode("utf-8", "replace")
    urls = program_urls(listing)
    print(f"{len(urls)} programas", flush=True)
    for url in urls:
        slug = url.rstrip("/").rsplit("/", 1)[1]
        html = fetch(url, RAW / "programas" / f"{slug}.html")
        if html is None:
            continue
        pdf = latest_plan_pdf(html.decode("utf-8", "replace"))
        if pdf:
            fetch(pdf, RAW / "programas" / f"{slug}.pdf")
    for path in CAMPUS:
        fetch(f"{BASE}/{path}/", RAW / "campus" / f"{path.replace('/', '-')}.html")
    print("listo", flush=True)
