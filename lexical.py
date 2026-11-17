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

    def program_names(self) -> list[str]:
        """Los nombres oficiales de los programas, para que el cerebro no invente carreras.

        Mismo contrato que rag.Knowledge.program_names(): los dos motores de recuperacion son
        intercambiables desde main.py, y el 2026-11-17 este metodo faltaba aqui, asi que la Pi
        se cayo al arrancar con BM25 — la clase servia para buscar pero no cumplia la interfaz.
        """
        return sorted({d["programa"] for d in self.docs if d.get("programa")})

    def _vocabulario(self) -> list[str]:
        if not hasattr(self, "_vocab"):
            self._vocab = sorted(self.doc_freq)
        return self._vocab

    def expandir(self, term: str, maximo: int = 6) -> list[str]:
        """El termino, o los del indice que empiezan por el si no aparece tal cual.

        Sin esto, "cuanto DURA" no alcanza "DURACION del programa: 8 periodos academicos" —
        comprobado el 2026-11-17 con el robot hablando: el dato estaba en el corpus y al modelo
        le llegaron tablas de creditos, asi que respondio que la carrera "dura 139 creditos".
        Es la debilidad conocida de la busqueda lexica, y un prefijo la cubre casi entera sin
        traer un modelo de por medio.

        Se expande SIEMPRE, no solo cuando el termino falta: "dura" existe por su cuenta en el
        corpus, asi que una condicion "si no aparece" dejaba el caso real sin arreglar.

        Solo para terminos de 4 letras o mas: con menos, el prefijo arrastra medio diccionario.
        """
        if len(term) < 4:
            return [term] if term in self.doc_freq else []
        import bisect
        vocab = self._vocabulario()
        i = bisect.bisect_left(vocab, term)
        fuera = []
        # El termino exacto entra primero si existe; bisect ya lo deja en esta posicion.
        while i < len(vocab) and vocab[i].startswith(term) and len(fuera) < maximo:
            fuera.append(vocab[i])
            i += 1
        return fuera

    def idf(self, term: str) -> float:
        """Rare terms weigh more. The +0.5s keep it finite for a term in every document."""
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        # Cada termino se expande a sus derivados por prefijo (ver expandir), y los derivados
        # se reparten el peso para que una palabra con muchas variantes no pese mas que una
        # exacta: "dura" no debe valer seis veces por alcanzar duracion, duraderos, durante...
        terms = []
        for t in tokenize(query):
            variantes = self.expandir(t)
            terms += [(v, 1.0 / len(variantes)) for v in variantes]
        out = [0.0] * self.n
        for term, peso in terms:
            idf = self.idf(term) * peso
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


def _ventanas(texto: str, ancho: int, paso: int):
    """Trozos solapados de `ancho` caracteres. Deliberadamente NO se corta por frases: el
    corpus es texto de folleto con tablas, y al separar por puntos "Titulo que otorga:" quedaba
    en un trozo y "Psicologo" en otro, que es justo perder el dato. Una ventana mantiene juntos
    la etiqueta y su valor."""
    texto = " ".join((texto or "").split())
    if len(texto) <= ancho:
        return [texto] if texto else []
    return [texto[i:i + ancho] for i in range(0, max(1, len(texto) - ancho + paso), paso)]


def fragmentos(indice: "BM25", consulta: str, docs: int = 3, n: int = 2,
               ancho: int = 260, paso: int = 90) -> list[dict]:
    """Los `n` trozos mas pertinentes de los `docs` mejores documentos.

    Por que existe: medido en la Pi el 2026-11-17, el robot tarda de 7 a 11 segundos en decir su
    primera palabra, y casi todo ese tiempo es leer el prompt a ~43 fichas/s. Tres documentos
    recortados a 300 caracteres son ~250 fichas, pero la respuesta suele vivir en una franja
    pequena de uno de ellos; el resto se paga en silencio frente al visitante.

    Se conserva el programa y el titulo de cada trozo: sin ellos el modelo pierde a que carrera
    pertenece el dato, que es precisamente su falla dominante.
    """
    mejores = indice.search(consulta, k=docs)
    candidatos = [(v, doc) for doc, _ in mejores
                  for v in _ventanas(str(doc.get("texto", "")), ancho, paso)]
    if not candidatos:
        return []
    # El nombre del programa sale de la consulta aqui: ya hizo su trabajo eligiendo el
    # documento, y dentro de el solo es ruido. Sin esto, a "y cuanto dura" le ganaban las
    # ventanas que repetian "psicologia" en vez de la que dice "8 periodos academicos".
    del_programa = set()
    for _, doc in candidatos:
        del_programa |= set(tokenize(str(doc.get("programa", ""))))
    afinada = " ".join(t for t in tokenize(consulta) if t not in del_programa) or consulta

    sub = BM25([{"titulo": "", "programa": "", "texto": v} for v, _ in candidatos],
               peso_programa=1)
    puntajes = sub.scores(afinada)
    orden = sorted(range(len(candidatos)), key=lambda i: puntajes[i], reverse=True)
    salida, tomados = [], []
    for i in orden:
        if len(salida) >= n:
            break
        ventana, doc = candidatos[i]
        # Dos ventanas contiguas del mismo documento dicen casi lo mismo y se pagan dos veces.
        if any(d is doc and (v in ventana or ventana in v or
                             len(set(v.split()) & set(ventana.split())) > len(ventana.split()) * 0.5)
               for v, d in tomados):
            continue
        tomados.append((ventana, doc))
        salida.append({"id": doc.get("id", ""), "programa": doc.get("programa", ""),
                        "titulo": doc.get("titulo", ""), "texto": ventana})
    return salida


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
