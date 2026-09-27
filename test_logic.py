"""Plain-assert checks for the pure-logic parts of the agent (no audio, no models).

Run: uv run python test_logic.py
Each test_* function raises AssertionError if the logic is broken.
"""
import numpy as np

import animations
import config
import download_models
import sentences
import vad


def test_config_profiles_have_same_keys():
    # If a key exists in [pc] but not in [pi] the Pi would crash at startup with
    # AttributeError. Loading both and comparing their keys catches that on the PC.
    pc, pi = config.load_config("pc"), config.load_config("pi")
    assert set(vars(pc)) == set(vars(pi)), set(vars(pc)) ^ set(vars(pi))
    assert pc.profile == "pc" and pi.profile == "pi"


def _split(tokens):
    # Simulates the LLM stream: feed token by token, then flush at the end.
    s = sentences.SentenceSplitter()
    out = []
    for t in tokens:
        out += s.feed(t)
    return out + s.flush()


def test_sentence_splitter():
    # Tokens cut words in half; the splitter must still find sentence ends.
    assert _split(["¡Cla", "ro! Me ", "gusta."]) == ["¡Claro!", "Me gusta."]
    # Abbreviation: "Dr." is not a sentence end.
    assert _split(["El Dr. Pérez dice hola. ", "Adiós"]) == ["El Dr. Pérez dice hola.", "Adiós"]
    # Decimal numbers: no whitespace after the dot, so no split.
    assert _split(["Cuesta 3.5 millones. Sí"]) == ["Cuesta 3.5 millones.", "Sí"]
    assert _split(["¿Te gusta? ", "¡Genial!"]) == ["¿Te gusta?", "¡Genial!"]
    assert _split(["Mmm... ", "déjame ver."]) == ["Mmm...", "déjame ver."]
    assert _split([]) == []


def test_clean_for_tts():
    # Review Focus #2: markdown and emojis must never reach the voice.
    assert sentences.clean_for_tts("**Hola** 😊 #1") == "Hola 1"
    assert sentences.clean_for_tts("- Ingeniería de sistemas") == "Ingeniería de sistemas"
    assert sentences.clean_for_tts("✨🎉") == ""


def test_animation_keywords():
    assert animations.match("¡Hola! Qué interesante.") == ["saludar", "inclinar_cabeza"]
    assert animations.match("ADIOS, amigo") == ["despedir"]        # case and accents don't matter
    assert animations.match("Me gusta Holanda") == []               # whole words only
    assert animations.match("Hola, hola") == ["saludar"]            # no duplicates


def test_piper_urls():
    onnx, cfg = download_models.piper_urls("es_MX-ald-medium")
    base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium"
    assert onnx == base + ".onnx" and cfg == base + ".onnx.json"


def test_turn_detector():
    # 96 ms silence = 3 chunks ends a turn; 64 ms = 2 chunks of speech minimum; keep 2 chunks before speech.
    td = vad.TurnDetector(silence_ms=96, min_speech_ms=64, pre_roll_chunks=2)
    s, q = np.ones(512, np.float32), np.zeros(512, np.float32)   # s = speech chunk, q = quiet chunk
    feed = lambda seq: [td.feed(c, c is s) for c in seq]

    assert all(r is None for r in feed([q, q, q, q]))            # silence only → no turn
    res = feed([s, s, q, q, q])                                    # speech then 3 silences → turn ends
    assert all(r is None for r in res[:-1])
    assert len(res[-1]) == 7 * 512                                 # 2 pre-roll + 2 speech + 3 silence
    assert all(r is None for r in feed([s, q, q, q]))            # 1-chunk click → dropped as noise
    # Review Focus #4: a pause shorter than silence_ms does NOT split the turn.
    res = feed([s, s, q, q, s, s, q, q, q])
    assert all(r is None for r in res[:-1]) and res[-1] is not None
    assert len(res[-1]) == 9 * 512                                 # the whole thing is ONE turn


if __name__ == "__main__":
    # Collect every function whose name starts with test_ and run it.
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} checks passed")
