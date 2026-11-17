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


# El animatronico se conecta aqui una sola vez, al arrancar. Queda como variable de modulo y
# no como parametro porque trigger() lo llama el Player desde el hilo del audio, que no tiene
# —ni deberia tener— una referencia al hardware. Sin conectar, todo sigue siendo un print.
_OSO = None


def conectar(oso) -> None:
    """Enchufa el puente con el ESP32. `oso` puede ser None: entonces solo se imprime."""
    global _OSO
    _OSO = oso


def trigger(name: str) -> None:
    print(f"[ANIM] {name}", flush=True)
    if _OSO is not None:
        _OSO.animacion(name)


def boca(nivel: int) -> None:
    """Abertura de la boca, 0-100. La llama el Player por cada trozo de audio que suena, unas
    21 veces por segundo, asi que tiene que ser barata y no puede lanzar excepciones: una
    mandibula no vale una voz cortada."""
    if _OSO is not None:
        try:
            _OSO.boca(nivel)
        except Exception:
            pass
