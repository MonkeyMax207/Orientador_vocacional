"""End-to-end check without audio: replays student turns through retrieval + the grounded LLM.

Run: uv run python -m knowledge.replay pc|pi
Prints each reply with its mode, cards and time to first sentence, and flags any
"Ingeniería X"-style program name that isn't in the official list (possible invention).
"""
import re
import sys
import time

from brain import make_brain
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
    brain = make_brain(cfg, program_names=kb.program_names())
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
