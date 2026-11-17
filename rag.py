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
CHUNKS = ROOT / "knowledge" / "chunks.json"            # full-text chunks (knowledge/make_chunks.py)
CHUNK_VECTORS = ROOT / "models" / "chunk_vectors.npz"

# Price questions get this instead of a search: the robot hands over to humans.
PRICE_CARD = {"id": "precios", "titulo": "Precios y becas",
              "texto": "El estudiante pregunta por precios, matrícula o becas. No des cifras: dile con "
                       "amabilidad que eso lo explica mejor un asesor de admisiones de la UAO."}
PRICE_WORDS = ("precio", "cuánto cuesta", "cuanto cuesta", "cuánto vale", "cuanto vale", "matrícula",
               "matricula", "beca", "financiación", "financiacion", "valor del semestre")
RECOMMEND_WORDS = ("recomiendas", "recomendarías", "recomienda", "qué carrera", "que carrera",
                   "qué puedo estudiar", "que puedo estudiar", "qué debería estudiar", "que deberia estudiar")
# Card types used in normal (detail) turns: facts the student asks about. Program "perfil" and
# "egresado" cards are only for recommendation turns; offered on every turn they turned the agent
# into a salesman that pushed a career at each vague sentence.
DETAIL_KINDS = {"materias", "laboratorios", "campus"}
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
            # 900s: embedding every chunk is one batched request; on the DGX's CPU-only Ollama
            # (embeddinggemma) under load (e.g. a NIM container loading concurrently) 253 chunks
            # measured ~1.4s each, close to the old 300s ceiling (2026-09-29).
            with urllib.request.urlopen(req, timeout=900) as r:
                vectors = np.array(json.load(r)["embeddings"], dtype=np.float32)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Falla el modelo de embeddings '{self.model}' ({e}). Ejecuta: ollama pull {self.model}")
        except urllib.error.URLError:
            raise SystemExit(f"Ollama no responde en {self.url}. Abre la app de Ollama (o ejecuta: ollama serve)")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # unit length → dot = cosine


class Knowledge:
    def __init__(self, embed, model_name: str, cards_path=CARDS, cache_path=VECTORS):
        raw = Path(cards_path).read_bytes()
        # Cards flagged "revisar" failed the source check (possibly invented names): skip them
        # until a human reviews the card and sets "revisar": false in cards.json.
        self.cards = [c for c in json.loads(raw) if not c.get("revisar")]
        self.embed = embed
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
        return self._ranked_by(self.embed([query], query=True)[0])

    def _ranked_by(self, query_vector) -> list[tuple[float, dict]]:
        scores = self.vectors @ query_vector   # one dot product per card
        return [(float(scores[i]), self.cards[i]) for i in np.argsort(-scores)]

    @staticmethod
    def _pick(ranked, k: int, threshold: float, kinds) -> list[dict]:
        hits = []
        for score, card in ranked:
            if score < threshold:
                break   # sorted: everything after this is even less similar
            if kinds and card["tipo"] not in kinds:
                continue
            hits.append(card)
            if len(hits) == k:
                break
        return hits

    def search(self, query: str, k: int, threshold: float, kinds=None) -> list[dict]:
        return self._pick(self.ranked(query), k, threshold, kinds)

    def search_best(self, queries: list[str], k: int, threshold: float, kinds=None) -> list[dict]:
        # Several phrasings of the same need, embedded in ONE request (cheap on the Pi); the one
        # whose best card matches most strongly decides which cards are returned.
        rankings = [self._ranked_by(v) for v in self.embed(queries, query=True)]
        # Compare queries by their best card among the allowed kinds (-1 if none: no cards at all).
        best = max(rankings, key=lambda ranked: next(
            (score for score, card in ranked if not kinds or card["tipo"] in kinds), -1.0))
        return self._pick(best, k, threshold, kinds)

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
    # Detail: the student's words alone, and together with their previous turn. The previous
    # turn keeps an implicit topic ("¿qué materias hay?" right after asking about mecatrónica);
    # search_best keeps whichever matches more strongly, so a change of topic still wins.
    # The agent's own (longer) reply is used only as a last resort: it would steer the search.
    previous = user_turns[-2] if len(user_turns) > 1 else ""
    queries = [text] + ([f"{previous} {text}"] if previous else [])
    cards = kb.search_best(queries, cfg.rag_max_cards, cfg.rag_threshold, DETAIL_KINDS)
    if not cards and last_reply:
        cards = kb.search(f"{last_reply} {text}", cfg.rag_max_cards, cfg.rag_threshold, DETAIL_KINDS)
    return mode, cards


def make_search(cfg):
    """Returns (knowledge, search, gate) over the full-text chunks, or (None, None, None) if none.
    search(query) -> text the LLM receives as the result of its buscar_uao tool call.
    gate(text) -> True if the student's words resemble some UAO text (offer the tool)."""
    if not CHUNKS.exists():
        return None, None, None
    kb = Knowledge(OllamaEmbedder(cfg.ollama_host, cfg.embed_model), cfg.embed_model, CHUNKS, CHUNK_VECTORS)

    def search(query: str) -> str:
        hits = kb.search(query, cfg.rag_max_cards, cfg.rag_threshold)
        return "\n\n".join(f"[{h['titulo']}] {h['texto']}" for h in hits) or "No encontré información sobre eso."

    def gate(text: str) -> bool:
        return kb.ranked(text)[0][0] >= cfg.rag_gate   # chat scores ~0.15–0.28, UAO questions ~0.36+
    return kb, search, gate


def make_finder_bm25(cfg):
    """El mismo contrato que make_finder, con BM25 lexico en vez de embeddings.

    Por que existe: el 2026-11-16 el modelo afinado solo (sin recuperar nada) quedo en 46.5% de
    error juzgado contra la fuente, con el modo de falla dominante en inventar especificos —
    materias casi correctas, atribuidas al programa equivocado. Recuperar bajo ese error a 19.7%
    en fp16. Pero la razon para quitar el RAG era la latencia en la Pi, y era buena: embeddings
    exigen un segundo modelo cargado y una ida y vuelta HTTP por turno.

    BM25 no necesita nada de eso: 400 documentos indexados en 27 ms, consulta en 0.11 ms, Python
    puro. Conserva "local y economico" y devuelve casi toda la latencia que se buscaba ganar.

    El turno anterior se concatena a la consulta por la misma razon que en la version con
    embeddings, y ahi esta medido cuanto importa: con el programa en la consulta, la
    recuperacion acierta el programa el 100% de las veces contra el 40% sin el.
    """
    import lexical
    if not CHUNKS.exists():
        return None, lambda text, previous="": []
    kb = lexical.load()

    # rag_fragmentos > 0 entrega franjas del documento en vez del documento entero. En la Pi
    # cada ficha de prompt son ~23 ms, y la respuesta casi siempre vive en una franja pequena:
    # mandar el parrafo completo se paga en silencio antes de que el robot hable.
    trozos = getattr(cfg, "rag_fragmentos", 0)

    def find(text: str, previous: str = "") -> list[dict]:
        consulta = f"{previous} {text}".strip() if previous else text
        hits = kb.search(consulta, k=cfg.rag_max_cards, minimo=cfg.rag_gate_bm25)
        if not hits:
            return []                      # la compuerta decide: sin nada relevante, no se inventa
        if trozos:
            return lexical.fragmentos(kb, consulta, docs=cfg.rag_max_cards, n=trozos)
        return [doc for doc, _ in hits]
    return kb, find


def make_finder_grafo(cfg):
    """Hechos estructurados primero; la busqueda por texto solo como ultimo recurso.

    Nace de la primera conversacion real del robot, el 2026-11-17, que mostro dos cosas que
    ningun quiz habia revelado:

      * "Hola, como va todo?" trajo documentos del CAFS y el robot hablo de espacios
        deportivos. 79 de 400 documentos no pertenecen a ninguna carrera y secuestraban la
        charla desde el primer turno.
      * El programa en foco se perdia entre turnos: "y es dificil?" trajo Ingenieria Mecanica
        mientras se hablaba de Mercadeo.

    Por eso aqui el programa es **pegajoso**: una vez nombrado, se arrastra hasta que la
    persona nombre otro. Y si no hay ninguno en foco, la compuerta sube a `rag_gate_charla`,
    de modo que un saludo no arrastre folletos. El umbral salio de medir: la charla puntua
    0-4.8 en BM25 y una pregunta sobre la UAO 7.6 o mas.
    """
    import graph
    import lexical
    grafo = graph.build()
    kb = lexical.load()
    trozos = getattr(cfg, "rag_fragmentos", 0)
    gate_charla = getattr(cfg, "rag_gate_charla", 6.0)
    estado = {"programa": ""}

    def find(text: str, previous: str = "") -> list[dict]:
        # Primero el turno actual solo: si la persona nombra otra carrera, el foco cambia ahi
        # mismo. Solo si no nombra ninguna se mira el contexto (su turno anterior y la ultima
        # respuesta del robot), y por ultimo se conserva el foco anterior. Sin este orden el
        # foco quedaba pegado: el 2026-11-17 llevaba nueve turnos en Ingenieria Mecatronica
        # mientras la conversacion ya iba por Cine.
        nombrado = graph.programa_de(text, grafo, "")
        programa = nombrado or graph.programa_de(previous, grafo, "") or estado["programa"]
        estado["programa"] = programa

        # 0. Pregunta por el catalogo: se entregan los nombres oficiales y nada mas.
        catalogo = graph.listar(grafo, text)
        if catalogo:
            return [{"id": "grafo:catalogo", "programa": "", "titulo": "Programas",
                      "texto": catalogo}]

        # 1. Un hecho exacto de la ficha del programa: ~20 fichas de prompt en vez de ~140.
        # El programa ya se decidio arriba; consultar() lo recalculaba por su cuenta con el
        # contexto y podia discrepar. El 2026-11-17, a "hablame de la carrera de cine" el foco
        # dijo Cine y el dato entregado fue el de Ingenieria Mecatronica, que seguia en el
        # contexto. Dos sitios decidiendo lo mismo es un sitio de mas.
        # Sin campo reconocido pero con una carrera en foco, se entrega su perfil: "hablame de
        # la carrera de cine" no pide un dato concreto, y caer a texto libre ahi fue lo que
        # devolvio parrafos de Mecatronica cuando se preguntaba por Cine.
        campo = graph.campo_de(text) or graph.campo_de(previous) or ("perfil" if nombrado else "")
        dato = graph.hecho(grafo, programa, campo) if programa else ""
        if dato:
            return [{"id": f"grafo:{programa}", "programa": programa,
                      "titulo": programa, "texto": dato}]

        # 2. Sin programa en foco, solo se busca si la pregunta de verdad habla de la UAO.
        # El programa entra en la consulta de respaldo: medido el 2026-11-16, con el en la
        # consulta el acierto de programa es 100% y sin el 40%. Sin esta linea, "¿es dificil?"
        # mientras se hablaba de Mercadeo traia Ingenieria Ambiental y Arquitectura.
        # El turno anterior solo entra si el actual NO nombro carrera. Si la nombro, arrastrar
        # el turno viejo la contradice: el 2026-11-17, "hablame de la carrera de cine" despues
        # de hablar de mecatronica devolvio parrafos de Mecatronica, porque esa palabra seguia
        # en la consulta y pesa mas que "cine", que tiene cuatro letras.
        partes = (programa, text) if nombrado else (programa, previous, text)
        consulta = " ".join(x for x in partes if x).strip()
        minimo = cfg.rag_gate_bm25 if programa else gate_charla
        hits = kb.search(consulta, k=cfg.rag_max_cards, minimo=minimo)
        if not hits:
            return []
        if trozos:
            return lexical.fragmentos(kb, consulta, docs=cfg.rag_max_cards, n=trozos)
        return [doc for doc, _ in hits]

    return kb, find


def make_finder(cfg):
    """Returns (knowledge, find). find(text) -> the chunks to put next to the student's message:
    none for plain chat (below rag_gate), the top rag_max_cards for questions about the UAO.
    (Measured: llama3.2:3b uses facts placed in the message, but ignored the same facts when they
    came back from a tool call; the gate keeps chat free of forced information.)"""
    if not CHUNKS.exists():
        return None, lambda text: []
    kb = Knowledge(OllamaEmbedder(cfg.ollama_host, cfg.embed_model), cfg.embed_model, CHUNKS, CHUNK_VECTORS)

    def find(text: str, previous: str = "") -> list[dict]:
        queries = [text] + ([f"{previous} {text}"] if previous else [])
        rankings = [kb._ranked_by(v) for v in kb.embed(queries, query=True)]   # one embed request
        if rankings[0][0][0] < cfg.rag_gate:
            return []   # the gate looks at the student's CURRENT words only: chat stays chat
        # The previous question may carry the topic ("¿y qué materias…?" after asking about
        # mecatrónica): keep whichever phrasing matches more strongly.
        best = max(rankings, key=lambda ranked: ranked[0][0])
        return kb._pick(best, cfg.rag_max_cards, cfg.rag_threshold, None)
    return kb, find


def make_retriever(cfg):
    """La unica puerta que usan server.py y main.py. El perfil decide el motor."""
    backend = getattr(cfg, "rag_backend", "embed")
    if backend == "grafo":
        return make_finder_grafo(cfg)
    if backend == "bm25":
        return make_finder_bm25(cfg)
    if backend == "ninguno":
        return None, lambda text, previous="": []
    return make_finder(cfg)
