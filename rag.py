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
    # Detail: search with the student's own words. Only if that finds nothing (short answers
    # like "sí, esa") retry with the agent's last reply as context. Mixing them always would let
    # the previous (longer) reply steer the search back to the previous topic.
    cards = kb.search(text, cfg.rag_max_cards, cfg.rag_threshold)
    if not cards and last_reply:
        cards = kb.search(f"{last_reply} {text}", cfg.rag_max_cards, cfg.rag_threshold)
    return mode, cards
