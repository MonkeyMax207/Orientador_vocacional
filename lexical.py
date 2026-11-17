"""BM25 retrieval over the UAO knowledge, for the Pi — no embeddings, no GPU, no model.

Why lexical and not the embedding retrieval in `rag.py`: the 2026-11-16 judging put the
fine-tuned 1.7B at a 46.5% error rate, with the dominant failure being confabulated specifics
— course names almost right, assigned to the wrong program. Facts that a 1.7B cannot hold have
to be retrieved. But the reason RAG was dropped was latency on the Pi, and that reason was
sound: `rag.py` needs an embedding model loaded beside the LLM and an HTTP round trip per turn.

BM25 needs neither. It is a word-count ranking over 400 short documents: an inverted index
built in milliseconds, queried in microseconds, in pure Python with no dependency the Pi does
not already have. It keeps "local and economical" intact and gives back nearly all of the
latency that dropping RAG was meant to win.

What it cannot do is match meaning without shared words — "¿cuánto cuesta?" will not find a
chunk that says "matrícula" unless the word appears. That is acceptable here because the
questions name programs and subjects, which is exactly what lexical matching is good at, and
because the alternative is a model that invents those same names.

Formula: Okapi BM25 with the standard k1=1.5, b=0.75.

    uv run python -m lexical "materias de arquitectura"
"""
import json
import math
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
K1 = 1.5          # term-frequency saturation: beyond ~2 occurrences, more adds little
# Cuantas veces se repite el titulo y el programa en el indice. Ver la nota en BM25.__init__.
PESO_PROGRAMA = 3
B = 0.75          # how much to punish long documents
# Words too common in this corpus to discriminate; keeping them lets every query match
# every program's boilerplate. "programa" appears in nearly all 400 documents.
VACIAS = {"de", "la", "el", "los", "las", "un", "una", "y", "o", "en", "a", "que", "con", "por",
          "para", "del", "al", "se", "su", "sus", "es", "son", "como", "lo", "mas", "muy", "este",
          "esta", "estos", "estas", "ese", "esa", "eso", "si", "no", "me", "te", "le", "yo", "tu",
          "hay", "tiene", "tienen", "puede", "pueden", "ser", "estar", "sobre", "programa", "uao",
          "universidad", "autonoma", "occidente", "estudiante", "estudiantes"}


def normalize(text: str) -> str:
    """Lowercase, accent-free: 'Diseño' and 'diseno' must be the same token, because students
    type without accents and the STT output is inconsistent about them."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", (text or "").lower())
                         if unicodedata.category(c) != "Mn")
    return sin_tildes


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize(text))
            if len(t) > 2 and t not in VACIAS]


class BM25:
    """An inverted index over documents, built once at startup.

    `docs` is a list of dicts carrying at least "texto"; whatever else they hold is returned
    untouched, so the caller decides what a document is.
    """

    def __init__(self, docs: list[dict], campos: tuple[str, ...] = ("titulo", "programa", "texto"),
                 peso_programa: int = PESO_PROGRAMA):
        self.docs = docs
        # El titulo y el nombre del programa se repiten `peso_programa` veces en el indice. Es el
        # truco clasico de ponderacion por campo en BM25, y aqui corrige algo medido: en
        # "cuenteme de arquitectura, que materias hay en el sexto semestre", la palabra "sexto"
        # pesaba idf 3.98 y "arquitectura" solo 2.93, asi que el documento correcto caia al
        # puesto 5 — fuera de un top-3. El nombre del programa es el desambiguador, y confundir
        # programas era el modo de falla dominante del modelo (46.5% de error sin recuperar).
        self.tokens = []
        for d in docs:
            cuerpo = tokenize(str(d.get("texto", "") or ""))
            clave = tokenize(" ".join(str(d.get(c, "") or "") for c in campos if c != "texto"))
            self.tokens.append(cuerpo + clave * peso_programa)
        self.largos = [len(t) for t in self.tokens]
        self.largo_medio = (sum(self.largos) / len(self.largos)) if self.largos else 0.0
        self.frecuencias: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = {}
        for toks in self.tokens:
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.frecuencias.append(tf)
            for t in tf:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        self.n = len(docs)

    def idf(self, term: str) -> float:
        """Rare terms weigh more. The +0.5s keep it finite for a term in every document."""
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = [0.0] * self.n
        for term in terms:
            if term not in self.doc_freq:
                continue
            idf = self.idf(term)
            for i, tf in enumerate(self.frecuencias):
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = f * (K1 + 1) / (f + K1 * (1 - B + B * self.largos[i] / (self.largo_medio or 1)))
                out[i] += idf * norm
        return out

    def search(self, query: str, k: int = 3, minimo: float = 0.0) -> list[tuple[dict, float]]:
        """Top `k` documents above `minimo`, best first. Empty when nothing matches, which is
        the honest answer and lets the caller say it does not know."""
        puntajes = self.scores(query)
        orden = sorted(range(self.n), key=lambda i: puntajes[i], reverse=True)
        return [(self.docs[i], puntajes[i]) for i in orden[:k] if puntajes[i] > minimo]


def load(cards: Path | None = None, chunks: Path | None = None) -> BM25:
    """The index over both knowledge files, the same sources quiz.py judges against."""
    docs = []
    for path in (cards or ROOT / "knowledge" / "cards.json",
                 chunks or ROOT / "knowledge" / "chunks.json"):
        docs.extend(json.loads(Path(path).read_text()))
    return BM25(docs)


if __name__ == "__main__":
    import sys
    import time

    indice = load()
    t = time.perf_counter()
    indice2 = load()
    construccion = (time.perf_counter() - t) * 1000
    consulta = " ".join(sys.argv[1:]) or "materias de arquitectura"
    t = time.perf_counter()
    hits = indice.search(consulta, k=3)
    ms = (time.perf_counter() - t) * 1000
    print(f"{indice.n} documentos indexados en {construccion:.0f} ms; "
          f"consulta en {ms:.2f} ms\n")
    for doc, s in hits:
        print(f"  [{s:5.1f}] {doc.get('programa') or '?'} — {doc.get('titulo') or ''}")
        print(f"          {str(doc.get('texto'))[:110]}")
