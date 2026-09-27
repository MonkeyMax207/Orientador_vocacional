"""Plain-assert checks for the pure-logic parts of the agent (no audio, no models).

Run: uv run python test_logic.py
Each test_* function raises AssertionError if the logic is broken.
"""
import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import animations
import brain
import config
import download_models
import main
import sentences
import stt
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
    assert sentences.clean_for_tts("⭐ Arte & diseño") == "Arte y diseño"     # ⭐ is outside the old ranges
    assert sentences.clean_for_tts("👨‍💻 Sistemas") == "Sistemas"              # compound emoji: no ZWJ left
    assert sentences.clean_for_tts("2. Derecho") == "Derecho"                # numbered-list marker
    assert sentences.clean_for_tts("3.5 millones") == "3.5 millones"         # ...but not a decimal


def test_numbered_list_is_split_by_item():
    # Small LLMs answer the final summary with a list. Each item must be its own sentence,
    # not "Medicina 2." (the number glued to the wrong item).
    assert _split(["Te sugiero:\n1. Medicina\n2. Derecho\n", "3. Arte\n¿Cuál te gusta más?"]) == [
        "Te sugiero:", "1. Medicina", "2. Derecho", "3. Arte", "¿Cuál te gusta más?"]


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


def test_stt_clean():
    # Review Focus #1: noise and Whisper's known hallucinations must become "".
    assert stt.clean("  Me gusta la biología. ") == "Me gusta la biología."
    assert stt.clean("Subtítulos realizados por la comunidad de Amara.org") == ""
    assert stt.clean("¡Gracias por ver el video!") == ""
    assert stt.clean("Eh... mmm") == ""
    assert stt.clean("...") == ""
    assert stt.clean("Sí") == "Sí"                     # short but real answers survive
    for noise in ("Mmmm.", "Ehh...", "Hmmm", "[Música]", "(risas)", "Música"):
        assert stt.clean(noise) == "", noise           # stretched fillers and sound tags are noise
    assert stt.clean("Ajá") == "Ajá"                   # a real Spanish "yes"
    assert stt.clean("Me gustan las películas con subtítulos") != ""   # the word alone is fine


def test_stt_gpu_failure_gives_fix_it_message():
    # CUDA libraries load lazily on the FIRST transcription, not when the model is created.
    # A failure there must become the "use cpu/int8" message, not a traceback mid-startup.
    real = stt.STT.transcribe
    stt.STT.transcribe = lambda self, audio: (_ for _ in ()).throw(RuntimeError("cublas64_12.dll not found"))
    try:
        stt.STT("base", "cpu", "int8", 4)
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "whisper_device" in str(e), e
    finally:
        stt.STT.transcribe = real


def test_llm_worker_always_ends():
    # Review Focus #3: if Ollama dies mid-reply, the consumer must still get the None end marker.
    class Broken:
        def stream_reply(self, text, stop):
            yield "Hola"
            raise ConnectionError("Ollama se cayó")

    q = queue.Queue()
    brain.llm_worker(Broken(), "hola", q, threading.Event())
    assert q.get_nowait() == "Hola"
    assert q.get_nowait() is None


def test_slow_filler_waits_for_first_filler():
    # The 2nd filler must never be queued while the 1st still plays: it would delay an answer
    # that is about to arrive (pi profile answers land right around slow_llm_s).
    assert not main.slow_filler_due(elapsed=3.5, limit=3.0, answer_started=False, already_played=False, player_busy=True)
    assert main.slow_filler_due(elapsed=3.5, limit=3.0, answer_started=False, already_played=False, player_busy=False)
    assert not main.slow_filler_due(elapsed=2.0, limit=3.0, answer_started=False, already_played=False, player_busy=False)
    assert not main.slow_filler_due(elapsed=3.5, limit=3.0, answer_started=True, already_played=False, player_busy=False)
    assert not main.slow_filler_due(elapsed=3.5, limit=3.0, answer_started=False, already_played=True, player_busy=False)


class _FakeMic:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def pending(self):
        out, self.chunks = self.chunks, []
        return out

    def read(self, timeout=None):
        return self.chunks.pop(0)


class _FakeVAD:
    def is_speech(self, chunk):
        return bool(chunk[0])   # our fake speech chunks are ones, silence is zeros

    def reset(self):
        pass


def _agent(chunks, barge_in):
    # A real Agent without __init__: no models or audio devices, only the logic under test.
    a = main.Agent.__new__(main.Agent)
    a.cfg = SimpleNamespace(barge_in=barge_in, min_speech_ms=250, silence_ms=600, slow_llm_s=3.0)
    a.mic, a.vad, a.turns, a.ready_turn = _FakeMic(chunks), _FakeVAD(), vad.TurnDetector(600, 250), None
    return a


def test_barge_in_keeps_turn_from_backlog():
    # While STT/TTS block the main thread, a whole short utterance (speech + 600 ms silence)
    # can pile up in the mic queue. It must interrupt AND come back as the next turn.
    s, q = np.ones(512, np.float32), np.zeros(512, np.float32)
    a = _agent([s] * 10 + [q] * 19, barge_in=True)
    assert a.user_interrupting()
    assert len(a.listen()) == 28 * 512   # 10 speech + 18 silence chunks (the turn's end)


def test_noise_turn_cuts_filler():
    # A cough starts a filler; once STT says "just noise", the filler must stop.
    stopped = []

    class Player:
        def put(self, audio, anims=()): pass
        def stop(self): stopped.append(True)
        def busy(self): return False

    a = _agent([], barge_in=False)
    a.player = Player()
    a.fillers = SimpleNamespace(pick=lambda slow=False: np.zeros(10, np.int16))
    a.stt = SimpleNamespace(transcribe=lambda audio: "Mmm")
    main.RECORDINGS = Path(tempfile.mkdtemp())   # keep the test's wav out of recordings/
    a.respond(np.zeros(512, np.float32))
    assert stopped


if __name__ == "__main__":
    # Collect every function whose name starts with test_ and run it.
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} checks passed")
