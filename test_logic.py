"""Plain-assert checks for the pure-logic parts of the agent (no audio, no models).

Run: uv run python test_logic.py
Each test_* function raises AssertionError if the logic is broken.
"""
import json
import queue
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import animations
import brain
import config
import download_models
import main
import rag
import sentences
import stt
import vad
from knowledge import extract, make_cards, scrape


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
        def stream_reply(self, text, stop, cards=()):
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
    a.cfg = SimpleNamespace(barge_in=barge_in, min_speech_ms=250, silence_ms=600, slow_llm_s=3.0,
                            filler_after_s=0.8)
    a.mic, a.vad, a.turns, a.ready_turn = _FakeMic(chunks), _FakeVAD(), vad.TurnDetector(600, 250), None
    a.knowledge, a.user_turns, a.last_reply = None, [], ""
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


def _filler_run(llm_delay_s):
    # Runs one real respond() with fakes; returns True if a filler clip was queued.
    FILLER = np.zeros(3, np.int16)
    puts = []

    class Player:
        def put(self, audio, anims=()): puts.append(audio)
        def stop(self): pass
        def busy(self): return False

    class SlowBrain:
        def stream_reply(self, text, stop, cards=()):
            time.sleep(llm_delay_s)
            yield "Hola, qué bien."

    a = _agent([], barge_in=False)
    a.cfg.silence_ms, a.cfg.filler_after_s = 0, 0.3   # filler due 0.3 s after the turn ends
    a.player, a.brain = Player(), SlowBrain()
    a.fillers = SimpleNamespace(pick=lambda slow=False: FILLER)
    a.stt = SimpleNamespace(transcribe=lambda audio: "Hola")
    a.tts = SimpleNamespace(synthesize=lambda text: np.zeros(5, np.int16))
    main.RECORDINGS = Path(tempfile.mkdtemp())
    a.respond(np.zeros(512, np.float32))
    time.sleep(0.4)   # give a wrongly-uncancelled timer the chance to fire
    return any(p is FILLER for p in puts)


def test_filler_only_when_answer_is_late():
    # A person doesn't say "dame un segundo" when they already know the answer.
    assert not _filler_run(llm_delay_s=0.0)   # answer ready at once → no filler
    assert _filler_run(llm_delay_s=0.8)       # answer late → filler covers the wait


FIXTURE = Path(__file__).parent / "knowledge" / "fixtures" / "ingenieria-mecatronica.html"


def test_program_urls():
    html = ('<a href="https://www.uao.edu.co/programa/cine/">Cine</a>'
            '<a href="https://www.uao.edu.co/programa/cine/">otra vez</a>'
            '<a href="https://www.uao.edu.co/programa/derecho/">Derecho</a>'
            '<a href="https://www.uao.edu.co/blog/x/">no</a>')
    assert scrape.program_urls(html) == ["https://www.uao.edu.co/programa/cine/",
                                         "https://www.uao.edu.co/programa/derecho/"]


def test_latest_plan_pdf():
    # The page links an old plan (2022), the 2025 plan, a PEP and a tuition PDF: pick the 2025 plan.
    html = FIXTURE.read_text(encoding="utf-8")
    assert scrape.latest_plan_pdf(html).endswith("/2025/01/PLAN-DE-ESTUDIO-ING-METRONICA-2025.pdf")
    assert scrape.latest_plan_pdf("<p>sin pdf</p>") is None


def test_strip_prices():
    text = ("Duración del programa: 9 periodos académicos - 156 créditos académicos "
            "Valor periodo académico: (16-18 créditos)*: $10.716.000 ** COP Metodología: Presencial "
            "Horario: Día*. ** Valor sujeto a cambios sin previo aviso. Los valores de matrícula no "
            "incluyen la Estampilla Procultura. Título: Ingeniero.")
    out = extract.strip_prices(text)
    assert "$" not in out and "10.716" not in out and "Estampilla" not in out
    assert "Metodología: Presencial" in out and "Título: Ingeniero." in out
    assert "$" not in extract.strip_prices("El semestre vale $ 9.500.000 aprox.")   # space after $


def test_page_text_fixture():
    html = FIXTURE.read_text(encoding="utf-8")
    t = extract.page_text(html)
    assert "Si te gusta encontrar la solución" in t      # perfil del aspirante
    assert "Fab-Lab" in t                                 # labs block (untitled section)
    assert "Énfasis de Automatización" in t              # perfil del egresado
    assert "Xpoilers" not in t                            # news section dropped
    assert "cookies" not in t.lower()                     # faculty/contact section dropped
    assert "$" not in t and "10.716" not in t            # prices stripped
    assert extract.page_title(html) == "Ingeniería Mecatrónica"


def test_page_title_without_h1():
    # Virtual-program pages have no <h1>: the name must come from <title> (SEO text removed),
    # otherwise the program vanishes from the official list and the agent says it doesn't exist.
    cases = {
        "¿Administración de Empresas Virtual – UAO | Gestión y Liderazgo": "Administración de Empresas Virtual",
        "Contaduría Pública y Finanzas Virtual UAO | Gestión y Planeación": "Contaduría Pública y Finanzas Virtual",
        "Ingeniería Informática Virtual en la UAO – Carrera en Tecnología.": "Ingeniería Informática Virtual",
        "Ingeniería Industrial Virtual UAO | Gestión, Producción y Logística": "Ingeniería Industrial Virtual",
    }
    for title, name in cases.items():
        assert extract.page_title(f"<html><head><title>{title}</title></head><body><h2>Archivos</h2></body></html>") == name
    assert extract.page_title("<html><body><h1>Cine</h1></body></html>") == "Cine"   # <h1> still wins


def test_page_text_generic():
    # Review Focus #1: an unknown template must still give its text, minus junk sections.
    html = ("<html><body><h1>Programa X</h1><h2>¿Qué debes saber del programa?</h2><p>Forma líderes.</p>"
            "<h2>Noticias</h2><p>Evento del lunes</p></body></html>")
    t = extract.page_text(html)
    assert "Forma líderes." in t and "Evento del lunes" not in t


def test_page_text_long_heading_is_content():
    # Some pages wrap a whole study-plan table inside an <h2>: a "heading" that long is content.
    html = ("<html><body><h1>Diseño</h1><h2>Plan de estudios <span>Primer semestre Fundamentos "
            "matemáticos Dibujo I Inglés I Desarrollo personal Ética Segundo semestre Física Materiales "
            "Inglés II Tercer semestre Ergonomía Procesos de manufactura Modelado 3D Historia del "
            "diseño Dibujo II</span></h2>"
            "</body></html>")
    assert "Dibujo II" in extract.page_text(html)


def test_missing_names():
    source = "Primer semestre: Cálculo 1, Álgebra lineal. Laboratorio Fab-Lab."
    # Accents and case don't matter; an invented name is reported.
    assert make_cards.missing_names(["calculo 1", "ÁLGEBRA LINEAL", "Robótica Cuántica"], source) == ["Robótica Cuántica"]
    assert make_cards.missing_names(["", "Fab-Lab"], source) == []
    # PDF tables split names across lines: whitespace differences must not count as missing.
    assert make_cards.missing_names(["Algoritmia y programación"], "Algoritmia y\nprogramación") == []
    # PDFs use typographic ligatures ("ﬁ" is ONE character) and hyphenate words at line ends.
    assert make_cards.missing_names(["Matemáticas Financieras"], "Matemáticas ﬁnancieras") == []
    assert make_cards.missing_names(["Mecánica de Fluidos"], "Mecánica de ﬂuidos") == []
    assert make_cards.missing_names(["actos y procedimientos"], "actos y procedimien-\ntos") == []


def test_labs_only_when_source_mentions_them():
    # The LLM once turned the course "Neurociencias" into a lab for a page that names no labs.
    assert make_cards.mentions_labs("Usarás el Fab-Lab y el Laboratorio de Automática")
    assert not make_cards.mentions_labs("Materias: Neurociencias, Procesos psicológicos, Colaboración")
    assert not make_cards.mentions_labs("Prepárate para el mundo laboral y sus labores")   # "labor…" ≠ lab
    assert make_cards.mentions_labs("Contamos con labs de cómputo")


def test_with_retries():
    # Ollama's runner can crash mid-build (HTTP 500); it restarts itself, so retrying works.
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("500")
        return "ok"

    assert make_cards.with_retries(flaky, tries=3, wait_s=0) == "ok" and len(attempts) == 3
    try:
        make_cards.with_retries(lambda: (_ for _ in ()).throw(OSError("500")), tries=2, wait_s=0)
        raise AssertionError("expected OSError")
    except OSError:
        pass


def test_ask_retries_cut_json():
    # A runner restart can cut the model's answer mid-JSON ("Unterminated string"): that must be
    # retried like a network error, not crash the whole build.
    replies = iter([{"response": '{"fichas": [{"titulo": "Acceso'},          # cut off
                    {"response": '{"fichas": []}'}])
    real_post, real_wait = make_cards._post, make_cards.RETRY_WAIT_S
    make_cards._post, make_cards.RETRY_WAIT_S = (lambda path, body: next(replies)), 0
    try:
        assert make_cards.ask("prompt", make_cards.CAMPUS_SCHEMA) == {"fichas": []}
    finally:
        make_cards._post, make_cards.RETRY_WAIT_S = real_post, real_wait


def test_trim_words():
    text = "Uno dos tres cuatro. Cinco seis siete. Ocho nueve diez once doce."
    assert make_cards.trim_words(text, 7) == "Uno dos tres cuatro. Cinco seis siete."   # whole sentences
    assert make_cards.trim_words(text, 3) == "Uno dos tres cuatro."                     # keeps at least one
    assert make_cards.trim_words(text, 99) == text


_VEC = {"A. perfil a": [1, 0, 0], "B. labs b": [0, 1, 0], "C. campus c": [0, 0, 1],
        "q-ab": [0.8, 0.6, 0], "nada": [-1, 0, 0]}


def _fake_embed(calls):
    def embed(texts, query=False):
        calls.append(len(texts))
        v = np.array([_VEC[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return embed


def _kb(calls, tmp):
    cards = [{"id": "a", "tipo": "perfil", "programa": "A", "titulo": "A", "texto": "perfil a"},
             {"id": "b", "tipo": "laboratorios", "programa": "A", "titulo": "B", "texto": "labs b"},
             {"id": "c", "tipo": "campus", "programa": None, "titulo": "C", "texto": "campus c"}]
    (tmp / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    return rag.Knowledge(_fake_embed(calls), "fake", tmp / "cards.json", tmp / "v.npz")


def test_knowledge_search():
    tmp = Path(tempfile.mkdtemp())
    kb = _kb([], tmp)
    ids = lambda cards: [c["id"] for c in cards]
    assert ids(kb.search("q-ab", k=2, threshold=0.3)) == ["a", "b"]     # ordered by similarity
    assert ids(kb.search("q-ab", k=2, threshold=0.7)) == ["a"]          # threshold cuts b (0.6)
    assert ids(kb.search("q-ab", k=1, threshold=0.3)) == ["a"]          # k limit
    assert ids(kb.search("q-ab", k=2, threshold=0.3, kinds={"laboratorios"})) == ["b"]
    assert kb.search("nada", k=2, threshold=0.3) == []                  # nothing similar → nothing
    assert kb.program_names() == ["A"]


def test_knowledge_skips_cards_under_review():
    # A card flagged "revisar" may contain invented courses/labs: never use it until a human clears it.
    tmp = Path(tempfile.mkdtemp())
    cards = [{"id": "a", "tipo": "perfil", "programa": "A", "titulo": "A", "texto": "perfil a"},
             {"id": "b", "tipo": "materias", "programa": "A", "titulo": "B", "texto": "labs b", "revisar": True}]
    (tmp / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    kb = rag.Knowledge(_fake_embed([]), "fake", tmp / "cards.json", tmp / "v.npz")
    assert [c["id"] for c in kb.cards] == ["a"]
    assert kb.search("q-ab", k=2, threshold=0.3) == [cards[0]]


def test_vector_cache():
    tmp, calls = Path(tempfile.mkdtemp()), []
    _kb(calls, tmp)
    _kb(calls, tmp)                          # same cards.json → vectors come from the cache
    assert calls == [3]
    cards = json.loads((tmp / "cards.json").read_text(encoding="utf-8"))
    cards[0]["texto"] = "labs b"; cards[0]["titulo"] = "B"
    (tmp / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    rag.Knowledge(_fake_embed(calls), "fake", tmp / "cards.json", tmp / "v.npz")
    assert calls == [3, 3]                   # file changed → recomputed


def test_choose_mode():
    assert rag.choose_mode("¿Cuánto cuesta el semestre?", 1, 6) == "precio"
    assert rag.choose_mode("¿Hay becas?", 1, 6) == "precio"
    assert rag.choose_mode("¿Qué carrera me recomiendas?", 2, 6) == "recomendacion"
    assert rag.choose_mode("Me gusta el fútbol", 6, 6) == "recomendacion"   # automatic, once
    assert rag.choose_mode("Me gusta el fútbol", 7, 6) == "detalle"
    assert rag.choose_mode("¿Qué laboratorios hay?", 2, 6) == "detalle"


def test_retrieve():
    calls = []

    class KB:
        def search(self, query, k, threshold, kinds=None):
            calls.append((query, k, threshold, kinds))
            return [{"id": "x"}]

    cfg = SimpleNamespace(recommend_after_turns=6, rag_max_cards=2, rag_threshold=0.45)
    assert rag.retrieve(KB(), "¿cuánto vale?", ["¿cuánto vale?"], "", cfg) == ("precio", [rag.PRICE_CARD])
    assert calls == []                                                   # no search for prices
    # Detail: the student's own words first. The agent's previous (long) reply must NOT steer the
    # search: in the replay it made "¿Qué laboratorios…?" fetch the cards of the previous topic.
    mode, _ = rag.retrieve(KB(), "¿Qué laboratorios hay?", ["hola", "¿Qué laboratorios hay?"],
                           "La mecatrónica tiene muchas materias interesantes…", cfg)
    assert mode == "detalle" and calls[-1] == ("¿Qué laboratorios hay?", 2, 0.45, None)
    mode, _ = rag.retrieve(KB(), "¿qué me recomiendas?", ["me gusta dibujar", "¿qué me recomiendas?"], "", cfg)
    assert mode == "recomendacion" and calls[-1] == ("me gusta dibujar ¿qué me recomiendas?", 3, -1.0, {"perfil"})


def test_retrieve_falls_back_to_context():
    # A short answer like "sí, esa" finds nothing alone: then retry with the agent's last question.
    calls = []

    class KB:
        def search(self, query, k, threshold, kinds=None):
            calls.append(query)
            return [{"id": "robotica"}] if "robótica" in query else []

    cfg = SimpleNamespace(recommend_after_turns=6, rag_max_cards=2, rag_threshold=0.45)
    mode, cards = rag.retrieve(KB(), "sí, esa", ["hola", "sí, esa"], "¿Te gusta la robótica?", cfg)
    assert (mode, cards) == ("detalle", [{"id": "robotica"}])
    assert calls == ["sí, esa", "¿Te gusta la robótica? sí, esa"]


def test_find_cards_without_knowledge():
    # Review Focus #3: no cards.json → the agent still works, just without cards.
    a = _agent([], barge_in=False)
    assert a.find_cards("¿Qué laboratorios hay?") == []
    assert a.user_turns == ["¿Qué laboratorios hay?"]


class _FakeResp:
    # Stands in for the HTTP response: a context manager that yields NDJSON lines.
    def __init__(self, lines): self.lines = lines
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def __iter__(self): return iter(self.lines)


def test_brain_injects_cards_only_in_current_turn():
    b = brain.Brain("http://x", "m", 4, 0, program_names=["Ingeniería Mecatrónica"])
    sent = []

    def fake_post(path, body):
        sent.append(body)
        return _FakeResp([json.dumps({"message": {"content": "Hola."}, "done": True}).encode()])

    b._post = fake_post
    card = {"titulo": "Mecatrónica: laboratorios", "texto": "Tiene Fab-Lab."}
    list(b.stream_reply("¿Qué labs hay?", cards=[card]))
    list(b.stream_reply("Gracias"))
    first, second = sent
    assert "Tiene Fab-Lab." in first["messages"][-1]["content"]           # card in the current turn
    # The model must know the facts are a hidden note, not something the student wrote
    # (in the replay it answered "gracias por recordarme la información verificada").
    assert "el estudiante no" in first["messages"][-1]["content"]
    assert first["messages"][0] == second["messages"][0]                  # system prompt identical
    assert "Ingeniería Mecatrónica" in first["messages"][0]["content"]    # official list in system
    assert all("Fab-Lab" not in m["content"] for m in second["messages"])  # card not kept in history
    assert b.history[0] == {"role": "user", "content": "¿Qué labs hay?"}


if __name__ == "__main__":
    # Collect every function whose name starts with test_ and run it.
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} checks passed")
