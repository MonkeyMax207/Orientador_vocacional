"""Plain-assert checks for the pure-logic parts of the agent (no audio, no models).

Run: uv run python test_logic.py
Each test_* function raises AssertionError if the logic is broken.
"""
import animations
import config
import sentences


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


if __name__ == "__main__":
    # Collect every function whose name starts with test_ and run it.
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} checks passed")
