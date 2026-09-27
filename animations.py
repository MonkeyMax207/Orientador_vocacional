"""Keyword → animation lookup. No AI involved: a dictionary is instant and predictable.

The Player (audio_io.py) calls trigger() at the moment a sentence STARTS PLAYING,
so the movement is synchronized with the voice, not with when the text was generated.
"""
import re
import unicodedata

# Edit freely: key = word in the agent's speech, value = animation name.
KEYWORDS = {
    "hola": "saludar", "bienvenido": "saludar", "bienvenida": "saludar",
    "adiós": "despedir", "chao": "despedir",
    "claro": "asentir", "exacto": "asentir", "genial": "asentir", "excelente": "asentir",
    "interesante": "inclinar_cabeza", "cuéntame": "inclinar_cabeza", "pregunta": "inclinar_cabeza",
}


def _plain(text: str) -> str:
    # Lowercase and strip accents so "Adiós", "ADIOS" and "adios" all match.
    # NFD splits "ó" into "o" + a combining accent mark (category "Mn"), which we drop.
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


_TABLE = {_plain(k): v for k, v in KEYWORDS.items()}   # normalized once at import time


def match(sentence: str) -> list[str]:
    found = []
    for word in re.findall(r"\w+", _plain(sentence)):   # \w+ = whole words, so "holanda" ≠ "hola"
        anim = _TABLE.get(word)
        if anim and anim not in found:
            found.append(anim)
    return found


def trigger(name: str) -> None:
    # Sub-project 4 replaces this print with the servo command.
    print(f"[ANIM] {name}", flush=True)
