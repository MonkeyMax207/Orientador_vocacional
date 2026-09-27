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
    ("¿Puedo investigar siendo estudiante?", ("investigaciones-uao",)),   # page moved from grupos-de-investigacion
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
