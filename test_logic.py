"""Plain-assert checks for the pure-logic parts of the agent (no audio, no models).

Run: uv run python test_logic.py
Each test_* function raises AssertionError if the logic is broken.
"""
import json
import os
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
import fillers
import esp32
import graph
import respuestas
import lexical
from finetune import judge, make_dataset, masking, memory, quiz, schema, teacher, thermal, validate
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
    dgx = config.load_config("dgx")
    assert set(vars(dgx)) == set(vars(pc)), set(vars(dgx)) ^ set(vars(pc))
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
    main.RECORDINGS = main.CONVERSACIONES = Path(tempfile.mkdtemp())   # keep the test's wav out of recordings/
    a.respuestas = SimpleNamespace(buscar=lambda texto: ("", ""))
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
    # Sin respuesta inmediata a proposito: esta prueba es sobre las muletillas, y "Hola" ahora
    # toma el atajo de respuestas.py, asi que la dejaria sin ejercitar nada.
    a.respuestas = SimpleNamespace(buscar=lambda texto: ("", ""))
    a.stt = SimpleNamespace(transcribe=lambda audio: "que materias tiene mecatronica")
    a.tts = SimpleNamespace(synthesize=lambda text: np.zeros(5, np.int16))
    main.RECORDINGS = main.CONVERSACIONES = Path(tempfile.mkdtemp())
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

        def search_best(self, queries, k, threshold, kinds=None):
            calls.append((queries, k, threshold, kinds))
            return [{"id": "x"}]

    cfg = SimpleNamespace(recommend_after_turns=6, rag_max_cards=2, rag_threshold=0.45)
    assert rag.retrieve(KB(), "¿cuánto vale?", ["¿cuánto vale?"], "", cfg) == ("precio", [rag.PRICE_CARD])
    assert calls == []                                                   # no search for prices
    # Detail: the student's words alone AND with their previous turn (keeps an implicit topic:
    # "¿qué materias…?" right after asking about mecatrónica). The agent's long reply is not used.
    mode, _ = rag.retrieve(KB(), "¿Qué materias hay?", ["¿Labs de mecatrónica?", "¿Qué materias hay?"],
                           "La mecatrónica tiene muchos laboratorios…", cfg)
    assert mode == "detalle"
    # Program-profile cards only in recommendation mode: in normal turns they made the agent push
    # careers at every vague sentence ("estoy pensando qué estudiar" → Ingeniería Industrial).
    assert calls[-1] == (["¿Qué materias hay?", "¿Labs de mecatrónica? ¿Qué materias hay?"], 2, 0.45,
                         rag.DETAIL_KINDS)
    assert "perfil" not in rag.DETAIL_KINDS and "egresado" not in rag.DETAIL_KINDS
    mode, _ = rag.retrieve(KB(), "¿qué me recomiendas?", ["me gusta dibujar", "¿qué me recomiendas?"], "", cfg)
    assert mode == "recomendacion" and calls[-1] == ("me gusta dibujar ¿qué me recomiendas?", 3, -1.0, {"perfil"})


def test_retrieve_falls_back_to_context():
    # A short answer like "sí, esa" finds nothing: then retry with the agent's last question.
    calls = []

    class KB:
        def search(self, query, k, threshold, kinds=None):
            calls.append(query)
            return [{"id": "robotica"}] if "robótica" in query else []

        def search_best(self, queries, k, threshold, kinds=None):
            calls.append(queries)
            return []

    cfg = SimpleNamespace(recommend_after_turns=6, rag_max_cards=2, rag_threshold=0.45)
    mode, cards = rag.retrieve(KB(), "sí, esa", ["hola", "sí, esa"], "¿Te gusta la robótica?", cfg)
    assert (mode, cards) == ("detalle", [{"id": "robotica"}])
    assert calls == [["sí, esa", "hola sí, esa"], "¿Te gusta la robótica? sí, esa"]


def test_search_best_picks_strongest_query():
    # Both queries embedded in ONE call; the one with the strongest match decides the cards.
    calls = []
    kb = _kb(calls, Path(tempfile.mkdtemp()))
    calls.clear()
    assert [c["id"] for c in kb.search_best(["nada", "q-ab"], k=1, threshold=0.3)] == ["a"]
    assert kb.search_best(["nada"], k=1, threshold=0.3) == []
    assert calls == [2, 1]                        # one embedding request per search_best call


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
    # The student's words close the message, unlabeled: a "Lo que dijo el estudiante:" label made
    # the model talk ABOUT the student in third person instead of TO them.
    assert first["messages"][-1]["content"].endswith("]\n\n¿Qué labs hay?")
    assert first["messages"][0] == second["messages"][0]                  # system prompt identical
    assert "Ingeniería Mecatrónica" in first["messages"][0]["content"]    # official list in system
    assert all("Fab-Lab" not in m["content"] for m in second["messages"])  # card not kept in history
    assert b.history[0] == {"role": "user", "content": "¿Qué labs hay?"}



class _UrlopenCtx:
    def __init__(self, body): self.body = body
    def __enter__(self): return self.body
    def __exit__(self, *exc): return False


def _sse(lines):
    # One SSE "data: ..." line per list element; mirrors _FakeResp but for NimBrain's format.
    return _FakeResp([f"data: {json.dumps(x)}".encode() if x != "[DONE]" else b"data: [DONE]" for x in lines])


def test_nimbrain_streams_content_and_skips_ollama_fields():
    b = brain.NimBrain("http://x", "m")
    sent = []
    b._post = lambda path, body: sent.append((path, body)) or _sse(
        [{"choices": [{"delta": {"content": "Hola"}}]},
         {"choices": [{"delta": {"content": "."}}]},
         "[DONE]"])
    assert "".join(b.stream_reply("Hola")) == "Hola."
    path, body = sent[0]
    assert path == "/v1/chat/completions"
    assert "options" not in body and "keep_alive" not in body


def test_nimbrain_assembles_tool_call_split_across_chunks():
    searched = []
    b = brain.NimBrain("http://x", "m", search=lambda q: searched.append(q) or "Fab-Lab.")
    replies = iter([
        _sse([{"choices": [{"delta": {"tool_calls": [
                  {"index": 0, "function": {"name": "buscar_uao", "arguments": ""}}]}}]},
              {"choices": [{"delta": {"tool_calls": [
                  {"index": 0, "function": {"arguments": "{\"consulta\""}}]}}]},
              {"choices": [{"delta": {"tool_calls": [
                  {"index": 0, "function": {"arguments": ": \"laboratorios\"}"}}]}}]},
              {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
              "[DONE]"]),
        _sse([{"choices": [{"delta": {"content": "Tiene el Fab-Lab."}}]}, "[DONE]"]),
    ])
    b._post = lambda path, body: next(replies)
    assert "".join(b.stream_reply("¿Qué labs hay?")) == "Tiene el Fab-Lab."
    assert searched == ["laboratorios"]


def test_nimbrain_empty_stream_yields_empty_reply():
    b = brain.NimBrain("http://x", "m")
    b._post = lambda path, body: _sse([{"choices": [{"delta": {}, "finish_reason": "stop"}]}, "[DONE]"])
    assert "".join(b.stream_reply("Hola")) == ""
    assert b.history[-1] == {"role": "assistant", "content": ""}


def test_nimbrain_check_reports_missing_model():
    import unittest.mock as mock
    import io
    b = brain.NimBrain("http://x", "otro-modelo")
    resp = io.BytesIO(json.dumps({"data": [{"id": "nvidia/nemotron-nano-9b-v2"}]}).encode())
    with mock.patch("urllib.request.urlopen", return_value=_UrlopenCtx(resp)):
        try:
            b.check()
            assert False, "expected SystemExit"
        except SystemExit as e:
            assert "otro-modelo" in str(e) and "nemotron-nano-9b-v2" in str(e)


def test_make_brain_selects_backend_from_config():
    ollama_cfg = SimpleNamespace(llm_backend="ollama", ollama_host="http://x", llm_model="m",
                                  threads=4, llm_num_gpu=0)
    nim_cfg = SimpleNamespace(llm_backend="nim", nim_host="http://y", llm_model="m")
    assert type(brain.make_brain(ollama_cfg)) is brain.Brain
    b = brain.make_brain(nim_cfg)
    assert type(b) is brain.NimBrain and b.url == "http://y"


def test_chunk_words():
    from knowledge import make_chunks
    words = " ".join(f"w{i}" for i in range(300))
    chunks = make_chunks.chunk_words(words, size=150, stride=120)
    assert [len(c.split()) for c in chunks] == [150, 150, 60]   # windows overlap by 30 words
    assert chunks[1].split()[0] == "w120"
    assert make_chunks.chunk_words("", 150, 120) == []


def test_brain_searches_only_when_the_model_asks():
    # The model decides: a tool call triggers ONE search, then the answer streams normally.
    searched = []
    b = brain.Brain("http://x", "m", 4, 0, search=lambda q: searched.append(q) or "Fab-Lab y Automática.")
    replies = iter([
        [json.dumps({"message": {"content": "", "tool_calls": [
            {"function": {"name": "buscar_uao", "arguments": {"consulta": "laboratorios mecatrónica"}}}]}}).encode()],
        [json.dumps({"message": {"content": "Tiene el Fab-Lab."}, "done": True}).encode()],
    ])
    sent = []
    b._post = lambda path, body: sent.append(body) or _FakeResp(next(replies))
    assert "".join(b.stream_reply("¿Qué labs tiene mecatrónica?")) == "Tiene el Fab-Lab."
    assert searched == ["laboratorios mecatrónica"]
    assert "tools" in sent[0] and "tools" not in sent[1]                 # no search loops
    assert sent[1]["messages"][-1] == {"role": "tool", "content": "Fab-Lab y Automática."}
    assert b.history == [{"role": "user", "content": "¿Qué labs tiene mecatrónica?"},
                         {"role": "assistant", "content": "Tiene el Fab-Lab."}]   # no tool noise
    # A normal chat turn: the model answers directly, no search.
    replies = iter([[json.dumps({"message": {"content": "¡Qué bien!"}, "done": True}).encode()]])
    assert "".join(b.stream_reply("Me gusta el fútbol")) == "¡Qué bien!" and len(searched) == 1


def test_brain_offers_search_only_past_the_gate():
    # Small models call a tool whenever it is offered, so it is only offered when the student's
    # words resemble some UAO text (gate); plain chat never gets the tool.
    b = brain.Brain("http://x", "m", 4, 0, search=lambda q: "x", gate=lambda text: "laboratorio" in text)
    sent = []
    b._post = lambda path, body: sent.append(body) or _FakeResp(
        [json.dumps({"message": {"content": "Ok."}, "done": True}).encode()])
    list(b.stream_reply("Hola, ¿cómo estás?"))
    list(b.stream_reply("¿Qué laboratorio hay?"))
    assert "tools" not in sent[0] and "tools" in sent[1]

OFICIALES = ["Ingeniería Mecatrónica", "Ingeniería Mecánica",
             "Ingeniería Electrónica y Telecomunicaciones", "Psicología"]


def test_validate_rejects_invented_program():
    ej = {"p": "¿Tienen aeroespacial?", "r": "Sí, tenemos Ingeniería Aeroespacial."}
    assert "programa_inventado" in validate.check(ej, source="", official=OFICIALES)


def test_validate_accepts_official_program_despite_accents_and_case():
    ej = {"p": "¿Qué hay?", "r": "Tenemos ingenieria mecatronica."}
    assert "programa_inventado" not in validate.check(
        ej, source="ingeniería mecatrónica", official=OFICIALES)


def test_validate_accepts_multiword_official_program_from_one_word_match():
    ej = {"p": "¿Qué hay?", "r": "Tenemos Ingeniería Electrónica y Telecomunicaciones."}
    assert "programa_inventado" not in validate.check(
        ej, source="Ingeniería Electrónica y Telecomunicaciones", official=OFICIALES)


def test_validate_allows_absent_program_only_when_denied():
    # El bucket de negativas TIENE que nombrar carreras que no existen, para negarlas.
    negada = {"p": "¿Tienen aeroespacial?", "r": "No, no tenemos Ingeniería Aeroespacial."}
    afirmada = {"p": "¿Tienen aeroespacial?", "r": "Claro, Ingeniería Aeroespacial es buenísima."}
    assert validate.check(negada, source="", official=OFICIALES, allow_absent=True) == []
    assert "sin_negacion" in validate.check(afirmada, source="", official=OFICIALES,
                                             allow_absent=True)


def test_validate_rejects_invented_non_engineering_program():
    # 16 de los 30 programas reales no empiezan por Ingeniería/Licenciatura/Tecnología, así que
    # el regex heredado de replay.py era ciego a ellos. "Diseño Gráfico" no existe en la UAO:
    # el programa real es "Diseño de la Comunicación Gráfica".
    ej = {"p": "¿Qué más hay?", "r": "También tenemos Diseño Gráfico y Medicina."}
    assert "programa_inventado" in validate.check(ej, source="", official=OFICIALES)


def test_absent_list_never_contains_a_real_program():
    # "Derecho" y "Arquitectura" SÍ los ofrece la UAO: negarlos sería un error factual.
    reales = ["Derecho", "Arquitectura", "Psicología", "Diseño de la Comunicación Gráfica"]
    ausentes = validate.absent_from(reales)
    assert "Derecho" not in ausentes and "Arquitectura" not in ausentes
    assert "Medicina" in ausentes


def test_validate_keeps_honest_decline_despite_no_overlap():
    # Una respuesta honesta de "el texto no lo dice" no comparte vocabulario con la fuente por
    # construcción; descartarla nos quitaría justo los ejemplos que enseñan a admitir ignorancia.
    ej = {"p": "¿Hay laboratorio?", "r": "No se menciona ninguno en particular."}
    assert "sin_anclaje" not in validate.check(
        ej, source="Fab-Lab para impresión y Laboratorio de Automática", official=OFICIALES)


def test_validate_measures_dialogue_length_per_turn():
    # El bug que mato los 400 dialogos de la primera corrida: ocho turnos de diecisiete
    # palabras son ocho respuestas correctas, no una de ciento treinta y seis.
    dialogo = {"turnos": [{"u": f"pregunta {i}", "a": "Eso suena muy interesante, cuéntame un poco "
                                                      "más de lo que te gusta hacer."}
                           for i in range(8)]}
    assert "longitud" not in validate.check(dialogo, source="", official=OFICIALES)


def test_validate_still_rejects_a_single_overlong_dialogue_turn():
    dialogo = {"turnos": [{"u": "hola", "a": "palabra " * 80}]}
    assert "longitud" in validate.check(dialogo, source="", official=OFICIALES)


def test_validate_skips_anchoring_for_dialogues():
    # "¡Qué bien!" no comparte vocabulario con ninguna ficha y es una respuesta correcta.
    dialogo = {"turnos": [{"u": "Me gusta el fútbol", "a": "¡Qué bien! ¿Lo juegas seguido?"}]}
    assert validate.check(dialogo, source="Fab-Lab y Laboratorio de Automática",
                           official=OFICIALES) == []


def test_validate_rejects_price_written_in_words():
    ej = {"p": "¿Cuánto cuesta?", "r": "El semestre vale tres millones de pesos."}
    assert "precio" in validate.check(ej, source="", official=OFICIALES)


def test_validate_accepts_price_refusal_without_a_figure():
    # Decir "precios" no es dar un precio: el rechazo honesto debe pasar la compuerta.
    ej = {"p": "¿Cuánto cuesta el semestre?",
          "r": "Eso no te lo puedo decir; lo explica mejor un asesor de admisiones."}
    assert validate.check(ej, source="", official=OFICIALES) == []


def test_validate_rejects_markdown_and_bullets():
    ej = {"p": "¿Qué labs?", "r": "Tiene:\n- **Fab-Lab**\n- Automática"}
    assert "formato" in validate.check(ej, source="Fab-Lab y Automática", official=OFICIALES)


def test_validate_rejects_invented_contact():
    ej = {"p": "¿A quién llamo?", "r": "Escribe a info@uao.edu.co."}
    assert "contacto" in validate.check(ej, source="", official=OFICIALES)


def test_validate_rejects_answer_unmoored_from_source():
    ej = {"p": "¿Qué labs tiene?", "r": "Tiene un reactor nuclear y un túnel de viento."}
    assert "sin_anclaje" in validate.check(
        ej, source="Fab-Lab para impresión y Laboratorio de Automática", official=OFICIALES)


def test_validate_accepts_answer_anchored_in_source():
    ej = {"p": "¿Qué labs tiene?", "r": "Tiene el Fab-Lab y el laboratorio de Automática."}
    assert validate.check(
        ej, source="Fab-Lab para impresión y Laboratorio de Automática",
        official=OFICIALES) == []


def test_dedup_crosses_buckets():
    ejs = [{"p": "¿Qué laboratorios tiene Mecatrónica?", "r": "a", "bucket": "fichas"},
           {"p": "¿qué laboratorios tiene mecatronica?", "r": "b", "bucket": "chunks"},
           {"p": "¿Hay equipo de fútbol?", "r": "c", "bucket": "chunks"}]
    assert len(validate.dedup(ejs)) == 2


class _FakeTermica:
    """Guardia térmica con temperatura y señales simuladas."""

    def __init__(self, temps):
        self.serie, self.enviadas, self.avisos = list(temps), [], []
        self.guard = thermal.ThermalGuard(read=self._read, pids=self._pids,
                                           send=self._send, log=self.avisos.append)

    def _pids(self, patterns=thermal.HEAVY):
        """Como heavy_pids: el patron decide. Solo el runner devuelve memoria al morir."""
        return [111] if patterns == thermal.RELEASABLE else [111, 222]

    def _read(self):
        return [self.serie.pop(0)] if self.serie else [0.0]

    def _send(self, pids, sig):
        import signal as s
        self.enviadas.append(({s.SIGSTOP: "STOP", s.SIGCONT: "CONT", s.SIGKILL: "KILL",
                                s.SIGTERM: "TERM"}[sig],
                               list(pids)))
        return list(pids)


def test_quiz_distinctive_tokens_identify_one_program_each():
    tokens = quiz.distinctive_tokens(OFICIALES)
    assert tokens.get("mecatronica") == "Ingeniería Mecatrónica"
    assert tokens.get("mecanica") == "Ingeniería Mecánica"
    assert "ingenieria" not in tokens          # no identifica a ninguno


def test_quiz_scores_expected_terms():
    r = quiz.score("Tiene el Fab-Lab y el laboratorio de Automática.", ["fab", "automatica"],
                    program="Ingeniería Mecatrónica", official=OFICIALES)
    assert r["aciertos"] == 1.0 and not r["contaminacion_cruzada"]


def test_quiz_detects_cross_contamination():
    # La métrica que decide el sub-proyecto: el hecho atribuido al programa equivocado.
    r = quiz.score("Eso es del laboratorio de Mecánica.", ["fab"],
                    program="Ingeniería Mecatrónica", official=OFICIALES)
    assert r["contaminacion_cruzada"] is True
    assert r["aciertos"] == 0.0


def test_quiz_flags_invented_program_bad_format_and_price():
    r = quiz.score("- **Ingeniería Aeroespacial** cuesta tres millones de pesos.", ["fab"],
                    program="Ingeniería Mecatrónica", official=OFICIALES)
    assert r["inventado"] and not r["formato_ok"] and r["precio_dado"]


def test_quiz_expected_terms_are_facts_not_style():
    # Una palabra que solo está en la redacción del maestro no es un hecho que exigir.
    terms = quiz.expected_terms("Tiene el Fab-Lab, genial para prototipar.",
                                 "Cuenta con Fab-Lab para impresión tridimensional.")
    assert "genial" not in terms


def test_quiz_expected_terms_exclude_the_questions_own_words():
    # Exigirlas premia al modelo que repite la pregunta y castiga al que contesta directo.
    terms = quiz.expected_terms("Pensamiento creativo son cinco creditos.",
                                 "Pensamiento creativo: cinco creditos del plan.",
                                 question="cuantos creditos tiene pensamiento creativo")
    assert "pensamiento" not in terms and "creativo" not in terms
    assert "cinco" in terms


def _meminfo_file(total_kb, avail_kb):
    p = Path(tempfile.mkdtemp()) / "meminfo"
    p.write_text(f"MemTotal:       {total_kb} kB\nMemFree:         1000 kB\n"
                  f"MemAvailable:   {avail_kb} kB\nBuffers:          500 kB\n")
    return str(p)


def test_memory_check_passes_with_headroom():
    # 116 GB disponibles: lo que había tras el reinicio.
    ok, msg = memory.check(needed_gb=25, reserve_gb=8,
                            path=_meminfo_file(126 * 1024**2, 116 * 1024**2))
    assert ok, msg


def test_memory_check_fails_when_ollama_holds_the_pool():
    # El escenario del 2026-11-15: los modelos cargados dejaron el pozo casi vacío.
    ok, msg = memory.check(needed_gb=25, reserve_gb=8,
                            path=_meminfo_file(126 * 1024**2, 20 * 1024**2))
    assert not ok
    assert "ollama" in msg.lower()   # el mensaje tiene que decir qué hacer


def test_memory_require_aborts_instead_of_oomming_the_machine():
    try:
        memory.require(needed_gb=25, reserve_gb=8, path=_meminfo_file(126 * 1024**2, 5 * 1024**2))
        assert False, "debió abortar"
    except SystemExit as e:
        assert "MEMORIA INSUFICIENTE" in str(e)


def test_thermal_memory_pressure_never_freezes_the_main_job():
    """La regresion del 2026-11-16: congelar por memoria dejo el entrenamiento trabado 22 min.

    SIGSTOP no devuelve memoria, asi que la presion que lo congelo no podia ceder. Ahora la
    presion solo TERMINA lo que libera al salir, y jamas congela.
    """
    t = _FakeTermica([40, 40])
    t.guard.tick(pressure=True)
    assert "STOP" not in [a for a, _ in t.enviadas]
    assert t.enviadas == [("TERM", [111])]      # el runner, no el entrenamiento (222)


def test_thermal_memory_pressure_does_not_latch():
    """Y aunque la presion siga, el trabajo principal sigue corriendo: nada queda trabado."""
    t = _FakeTermica([40, 40, 40])
    for _ in range(3):
        t.guard.tick(pressure=True)
    assert t.guard.frozen == []


def test_thermal_memory_pressure_warns_when_there_is_nothing_to_release():
    t = _FakeTermica([40])
    t.guard.pids = lambda patterns=thermal.HEAVY: []      # ya no queda runner que soltar
    t.guard.tick(pressure=True)
    assert all(not pids for _, pids in t.enviadas)     # no se señaló a nadie
    assert any("NO se congela" in a for a in t.avisos)


def test_thermal_still_freezes_for_heat_while_memory_is_tight():
    """Calor y presion a la vez: el calor manda, porque ahi congelar si sirve."""
    t = _FakeTermica([95])
    t.guard.tick(pressure=True)
    assert ("STOP", [111, 222]) in t.enviadas


def test_thermal_freezes_above_threshold_and_thaws_when_cool():
    t = _FakeTermica([80, 95, 92, 84])
    for _ in range(4):
        t.guard.tick()
    # 80: nada. 95: congela. 92: sigue congelado (no reanuda hasta <= 85). 84: reanuda.
    assert [a for a, _ in t.enviadas] == ["STOP", "CONT"]


def test_thermal_kills_as_last_resort():
    t = _FakeTermica([101])
    t.guard.tick()
    assert t.enviadas == [("KILL", [111, 222])]


def test_thermal_does_not_refreeze_what_is_already_frozen():
    t = _FakeTermica([95, 96, 97])
    for _ in range(3):
        t.guard.tick()
    assert [a for a, _ in t.enviadas] == ["STOP"]


def test_thermal_wait_until_cool_holds_then_returns():
    serie = iter([91, 90, 84])
    esperas = []
    assert thermal.wait_until_cool(read=lambda: [next(serie)], sleep=esperas.append,
                                    log=lambda m: None) is True
    assert esperas == [30, 30]


def test_thermal_wait_until_cool_passes_through_when_already_cool():
    assert thermal.wait_until_cool(read=lambda: [70.0], sleep=lambda s: None,
                                    log=lambda m: None) is False


def test_thermal_heavy_pids_never_includes_this_process():
    # Leer /proc en vez de pgrep: un patrón en la línea de comandos coincide con ese comando.
    assert os.getpid() not in thermal.heavy_pids(patterns=("python",))


def _fake_render(messages, add_generation_prompt=False):
    # Imita una plantilla de chat real: append-only, con cabecera de turno.
    out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
    return out + "<|assistant|>" if add_generation_prompt else out


def _fake_encode(text):
    # Un "token" por caracter: suficiente para verificar los tramos.
    return [ord(c) for c in text]


def test_masking_labels_every_assistant_turn_not_only_the_last():
    # El bug que tuvo este archivo: de un diálogo de ocho turnos se entrenaba uno.
    messages = [{"role": "system", "content": "S"},
                {"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "U2"}, {"role": "assistant", "content": "A2"}]
    ids, labels = masking.build(messages, _fake_render, _fake_encode)
    entrenado = "".join(chr(l) for l in labels if l != masking.IGNORE)
    assert "A1" in entrenado and "A2" in entrenado


def test_masking_never_labels_user_or_system_text():
    messages = [{"role": "system", "content": "SECRETO"},
                {"role": "user", "content": "PREGUNTA"}, {"role": "assistant", "content": "R"}]
    _, labels = masking.build(messages, _fake_render, _fake_encode)
    entrenado = "".join(chr(l) for l in labels if l != masking.IGNORE)
    assert "SECRETO" not in entrenado and "PREGUNTA" not in entrenado


def test_masking_trained_fraction_is_meaningful():
    messages = [{"role": "user", "content": "U"}, {"role": "assistant", "content": "RESPUESTA"}]
    _, labels = masking.build(messages, _fake_render, _fake_encode)
    assert 0.1 < masking.trained_fraction(labels) < 0.9


def test_schema_to_chat_single_turn_uses_brains_system_prompt():
    d = schema.to_chat({"p": "¿Qué labs?", "r": "El Fab-Lab."})
    assert [m["role"] for m in d["messages"]] == ["system", "user", "assistant"]
    # Entrenar con un prompt distinto al que sirve brain.py es la falla silenciosa clásica.
    assert d["messages"][0]["content"] == brain.SYSTEM_PROMPT
    assert d["messages"][-1]["content"] == "El Fab-Lab."


def test_schema_to_chat_multiturn_preserves_order():
    d = schema.to_chat({"turnos": [{"u": "Hola", "a": "¡Hola!"},
                                    {"u": "¿Y labs?", "a": "El Fab-Lab."}]})
    assert [m["role"] for m in d["messages"]] == ["system", "user", "assistant",
                                                   "user", "assistant"]


def test_split_holds_out_phrasings_not_facts():
    ejs = [{"p": f"pregunta {fid} {i}", "r": "x", "fact_id": fid}
           for fid in ("a", "b") for i in range(10)]
    train, quiz = make_dataset.split_by_phrasing(ejs, quiz_frac=0.2)
    assert {e["fact_id"] for e in train} == {"a", "b"}   # todos los hechos se entrenan
    assert {e["fact_id"] for e in quiz} == {"a", "b"}
    assert not ({e["p"] for e in train} & {e["p"] for e in quiz})


def test_as_items_handles_list_and_dialogue_shapes():
    assert make_dataset._as_items([{"p": "a", "r": "b"}, "basura"]) == [{"p": "a", "r": "b"}]
    dialogo = {"turnos": [{"u": "a", "a": "b"}]}
    assert make_dataset._as_items(dialogo) == [dialogo]
    assert make_dataset._as_items("nada") == []


def test_usable_rejects_dialogue_with_a_half_turn():
    # El maestro parte un turno en dos objetos: {"u":...} y luego {"a":...}. Pasaba el chequeo
    # de arriba y reventaba schema.to_chat al final de una corrida de horas (9 de 258 dialogos).
    partido = {"turnos": [{"u": "hola", "a": "¡Hola!"}, {"u": "solo pregunta"}, {"a": "solo respuesta"}]}
    assert make_dataset._as_items(partido) == []
    assert validate.check(partido, source="", official=OFICIALES) == ["forma_invalida"]
    entero = {"turnos": [{"u": "hola", "a": "¡Hola!"}]}
    assert make_dataset._as_items(entero) == [entero]


def test_as_items_drops_other_key_names():
    # El maestro a veces responde {"pregunta":..,"respuesta":..}: dejarlo pasar reventaba
    # schema.to_chat al final de una corrida de horas.
    assert make_dataset._as_items([{"pregunta": "a", "respuesta": "b"}]) == []


def test_validate_rejects_empty_or_shapeless_example():
    # Pasaba todas las demás compuertas y entrenaba al modelo a no decir nada.
    assert validate.check({"p": "¿Y?", "r": "  "}, source="", official=OFICIALES) == ["forma_invalida"]
    assert validate.check({"pregunta": "a", "respuesta": "b"}, source="",
                           official=OFICIALES) == ["forma_invalida"]


def test_teacher_extracts_json_from_prose():
    # El maestro envuelve el JSON en texto por mucho que el prompt lo prohíba.
    raw = 'Claro, aquí tienes:\n[{"p":"¿Qué labs?","r":"El Fab-Lab."}]\nEspero que sirva.'
    assert teacher._extract_json(raw) == [{"p": "¿Qué labs?", "r": "El Fab-Lab."}]


def test_teacher_extracts_json_ignoring_brackets_inside_strings():
    raw = '[{"p":"¿Qué dice el [anexo]?","r":"Dice }{ cosas."}]'
    assert teacher._extract_json(raw) == [{"p": "¿Qué dice el [anexo]?", "r": "Dice }{ cosas."}]


def test_teacher_retries_malformed_json_then_succeeds():
    respuestas = iter(['{"roto": ', '[{"p":"a","r":"b"}]'])
    real_post = teacher._post
    teacher._post = lambda host, body: {"message": {"content": next(respuestas)}}
    try:
        assert teacher.ask_json([{"role": "user", "content": "x"}], host="http://x",
                                 model="m", retries=2) == [{"p": "a", "r": "b"}]
    finally:
        teacher._post = real_post


def test_teacher_ask_many_keeps_order_and_marks_failures():
    real_post = teacher._post
    teacher._post = lambda host, body: (
        {"message": {"content": "[]"}} if "ok" in body["messages"][0]["content"]
        else {"message": {"content": "basura"}})
    try:
        jobs = [[{"role": "user", "content": "ok1"}], [{"role": "user", "content": "malo"}],
                [{"role": "user", "content": "ok2"}]]
        assert teacher.ask_many(jobs, host="http://x", model="m", workers=2) == [[], None, []]
    finally:
        teacher._post = real_post


def test_entrypoints_default_to_the_detected_profile():
    """El 2026-11-17 main.py caia en el perfil "pc" sin argumentos y la Pi arranco con la ruta
    de embeddings: tres minutos vectorizando en su CPU, con el BM25 ya instalado al lado. La
    deteccion existia en config.py pero nadie la usaba."""
    import re
    for archivo in ("main.py", "server.py", "fillers.py"):
        texto = Path(archivo).read_text()
        linea = next(l for l in texto.splitlines() if "load_config(sys.argv" in l)
        assert "DEFAULT_PROFILE" in linea, f"{archivo} no usa el perfil detectado: {linea.strip()}"
        assert '"pc"' not in linea, f"{archivo} todavia cae en pc"


def test_detected_profile_tells_a_pi_from_a_dgx():
    """Las dos son aarch64: mirar solo la arquitectura mandaria la DGX al perfil de la Pi."""
    assert config._perfil_por_defecto() in ("pi", "dgx", "pc")


def test_both_retrievers_satisfy_the_same_interface():
    """main.py usa el objeto de conocimiento sin saber que motor es. Si uno no tiene lo que el
    otro si, la Pi se cae al arrancar — paso el 2026-11-17 con program_names()."""
    import re
    usados = set()
    for archivo in ("main.py", "server.py"):
        usados |= set(re.findall(r"self\.knowledge\.(\w+)", Path(archivo).read_text()))
    assert usados, "la prueba no encontro ningun uso: revisa el analisis"
    for nombre in usados:
        assert hasattr(rag.Knowledge, nombre), f"Knowledge no tiene {nombre}"
        assert hasattr(lexical.BM25, nombre), f"BM25 no tiene {nombre}"


class _PuertoFalso:
    """Un puerto serie de mentira: guarda lo que se le escribe."""
    def __init__(self): self.escrito = []
    def write(self, datos): self.escrito.append(datos.decode("ascii")); return len(datos)
    def close(self): pass
    def reset_input_buffer(self): pass

    def ordenes(self):
        return [l for l in "".join(self.escrito).split("\n") if l]


def _oso(puerto):
    return esp32.Animatronico("/dev/falso", abrir=lambda p, b: puerto, log=lambda m: None)


def test_audio_device_accepts_a_list_of_candidate_names():
    """El mismo modulo USB se llama distinto en Windows y en Linux, asi que config.toml da
    varios nombres. Yo borre esta funcion el 2026-11-17 al sobreescribir audio_io.py, y la Pi
    quedo con una lista en la configuracion y un lector que no entendia listas; no exploto
    solo porque nadie arranco el robot en ese rato."""
    import audio_io
    assert audio_io.pick_device("default", "input") == "default"   # cadena suelta, tal cual

    encontrados = []
    def consulta(nombre, tipo):
        encontrados.append(nombre)
        if nombre != "El Bueno":
            raise ValueError("no existe")
    real = audio_io.sd.query_devices
    audio_io.sd.query_devices = consulta
    try:
        elegido = audio_io.pick_device(["No Existe", "El Bueno", "Tampoco"], "input")
        assert elegido == "El Bueno", elegido
        assert encontrados == ["No Existe", "El Bueno"]            # para en cuanto acierta
        try:
            audio_io.pick_device(["Ninguno"], "input")
            assert False, "deberia avisar en vez de seguir sin audio"
        except SystemExit as e:
            assert "list_devices" in str(e)
    finally:
        audio_io.sd.query_devices = real


def test_config_device_names_match_what_the_code_accepts():
    """El perfil de la Pi trae una lista; el del PC una cadena. Los dos tienen que funcionar."""
    import tomllib
    d = tomllib.load(open("config.toml", "rb"))
    for perfil in ("pc", "pi"):
        valor = d[perfil].get("input_device", d["common"].get("input_device"))
        assert isinstance(valor, (str, list)), f"{perfil}: {valor!r}"


def test_animatronic_absent_is_not_an_error():
    """Sin ESP32 el robot tiene que hablar igual. Hoy no hay ninguno enchufado a la Pi, y
    nadie deberia quedarse sin voz porque falte un servo."""
    mudo = esp32.Animatronico("", log=lambda m: None)
    assert not mudo.conectado
    mudo.boca(50); mudo.parpadear(); mudo.animacion("saludar"); mudo.reposo()   # no revienta


def test_animatronic_survives_a_port_that_fails_to_open():
    def explota(puerto, baud): raise OSError("no existe")
    oso = esp32.Animatronico("/dev/noexiste", abrir=explota, log=lambda m: None)
    assert not oso.conectado
    oso.boca(80)              # y sigue sin quejarse


def test_mouth_level_follows_loudness():
    import numpy as np
    assert esp32.nivel_de_boca(np.zeros(256, dtype="int16")) == 0
    suave = esp32.nivel_de_boca(np.full(256, 900, dtype="int16"))
    fuerte = esp32.nivel_de_boca(np.full(256, 9000, dtype="int16"))
    assert 0 < suave < fuerte == 100
    assert esp32.nivel_de_boca(np.full(256, 30000, dtype="int16")) == 100   # sin pasarse
    assert esp32.nivel_de_boca([]) == 0


def test_mouth_skips_repeats_so_the_servo_does_not_buzz():
    """Mandarle veinte veces por segundo la misma posicion a un servo lo hace zumbar y
    calentarse, y ademas llena el puerto de ruido inutil."""
    puerto = _PuertoFalso()
    oso = _oso(puerto)
    oso._vivo = False                      # sin escritor: se mira el buzon directamente
    oso.boca(50)
    assert oso._pendiente == "M50"
    oso._pendiente = None
    for _ in range(5):
        oso.boca(50)                       # el mismo valor: no debe volver a enviarse
        oso.boca(52)                       # dentro del umbral: tampoco
    assert oso._pendiente is None, oso._pendiente
    oso.boca(90)                           # un cambio de verdad si pasa
    assert oso._pendiente == "M90"
    # 0 y 100 siempre pasan aunque esten dentro del umbral: cerrar del todo importa.
    oso._pendiente = None
    oso.boca(100); assert oso._pendiente == "M100"
    oso._pendiente = None
    oso.boca(0); assert oso._pendiente == "M0"


def test_mouth_updates_overwrite_instead_of_piling_up():
    """Llegar tarde con la posicion de una mandibula no sirve de nada: solo importa la
    ultima. Si se encolaran, la boca iria retrasada respecto a la voz y nunca alcanzaria."""
    import time
    puerto = _PuertoFalso()
    oso = _oso(puerto)
    oso._vivo = False                      # el escritor no corre: se inspecciona el buzon
    for nivel in (10, 30, 60, 95):
        oso.boca(nivel)
    assert oso._pendiente == "M95", oso._pendiente
    assert oso._cola == []                 # la boca no ocupa la cola de ordenes


def test_named_animations_are_queued_not_overwritten():
    """Un parpadeo o un saludo si tienen que llegar todos: pisarlos seria perder gestos."""
    puerto = _PuertoFalso()
    oso = _oso(puerto)
    oso._vivo = False
    oso.animacion("saludar"); oso.parpadear(); oso.mirar(10, 90)
    assert oso._cola == ["Asaludar", "B", "L10,90"]


def test_animations_module_works_without_hardware():
    """animations.trigger() lo llama el hilo del audio; si el oso falta o falla, la voz sigue."""
    animations.conectar(None)
    animations.trigger("saludar")          # solo imprime
    animations.boca(50)

    class Roto:
        def animacion(self, n): raise RuntimeError("cable suelto")
        def boca(self, n): raise RuntimeError("cable suelto")
    animations.conectar(Roto())
    animations.boca(50)                    # tragado: una mandibula no vale una voz cortada
    animations.conectar(None)


def test_canned_reply_skips_the_model_for_greetings_and_prices():
    """El modelo es el 57% de un turno en la Pi y no aporta nada a un saludo. Y los precios
    dejan de depender de que se porte bien: una frase escrita a mano no alucina."""
    r = respuestas.Respuestas()
    for entrada, esperado in (("hola", "saludo"), ("Hola, ¿cómo estás?", "como_estas"),
                               ("cuanto cuesta la carrera", "precio"),
                               ("cual es la mas economica", "precio"),
                               ("hay beca para psicologia", "precio"),
                               ("muchas gracias", "gracias")):
        texto, regla = r.buscar(entrada)
        assert regla == esperado, f"{entrada} -> {regla}"
        assert texto


def test_canned_reply_does_not_hijack_real_questions():
    """Lo que necesita datos tiene que llegar al grafo: si esta tabla se los traga, el robot
    responde generalidades donde tenia el dato exacto."""
    r = respuestas.Respuestas()
    for entrada in ("que materias tiene psicologia", "cuanto dura derecho",
                     "hola, me interesa mecatronica", "que laboratorios hay",
                     "me gusta dibujar que me recomiendas"):
        assert r.buscar(entrada) == ("", ""), entrada


def test_canned_price_reply_never_quotes_a_number():
    """El requisito duro: ni cifras ni promesas de beca."""
    import re
    for _, patron, opciones in respuestas.TABLA:
        for frase in opciones:
            assert not re.search(r"\d", frase), frase


def test_fillers_cycle_before_repeating():
    """random.choice con seis clips repetia alguna en tres de cada cinco turnos, y eso es lo
    que delata a una maquina. La bolsa barajada las agota todas antes de volver a empezar."""
    import types
    f = types.SimpleNamespace(normal=list("abcdef"), slow=list("wxyz"),
                               _bolsas={False: [], True: []})
    f.pick = types.MethodType(fillers.Fillers.pick, f)
    salidas = [f.pick() for _ in range(6)]
    assert sorted(salidas) == list("abcdef"), salidas
    assert sorted(f.pick(slow=True) for _ in range(4)) == list("wxyz")
    # Las dos bolsas son independientes: gastar las lentas no toca las normales.
    assert sorted(f.pick() for _ in range(6)) == list("abcdef")


def test_filler_phrases_are_speakable():
    """Sin digitos ni markup: el robot las dice en voz alta. Y cortas, porque su trabajo es
    tapar un hueco, no llenarlo."""
    import re
    for frase in fillers.FILLERS:
        assert len(frase.split()) <= 6, frase
    for frase in fillers.FILLERS + fillers.SLOW_FILLERS:
        assert not re.search(r"\d|\*|#|_", frase), frase
        assert frase[0].isupper() and frase.endswith((".", "?", "!")), frase
    assert len(set(fillers.FILLERS)) == len(fillers.FILLERS)


def test_config_limits_actually_reach_the_brain():
    """max_turns, max_tokens y max_sentences llevaban en config.toml sin que nadie los leyera:
    el tope real salia del Modelfile y el historial crecia hasta 19 mensajes. En una charla de
    20 turnos del 2026-11-17 eso llevo la primera palabra de 4,7 s a 30,3 s en la Pi."""
    import types
    cfg = types.SimpleNamespace(llm_backend="ollama", ollama_host="http://x", llm_model="m",
                                 threads=4, llm_num_gpu=0, max_turns=3, max_tokens=60,
                                 rag_card_chars=300)
    b = brain.make_brain(cfg)
    assert b.max_turns == 3, "max_turns no llego al cerebro"
    assert b.options.get("num_predict") == 60, "max_tokens no llego al cerebro"
    assert b.card_chars == 300


def test_pi_remembers_fewer_turns_than_the_dgx():
    """La Pi paga cada ficha del historial en espera; la DGX no."""
    import tomllib
    d = tomllib.load(open("config.toml", "rb"))
    assert d["pi"]["max_turns"] < d["common"]["max_turns"]


def test_graph_covers_every_program_with_the_hard_facts():
    g = graph.build()
    assert len(g) == 30
    for campo, minimo in (("creditos", 30), ("titulo", 30), ("duracion", 28),
                           ("perfil", 30), ("egresado", 30)):
        n = sum(1 for nodo in g.values() if nodo.get(campo))
        assert n >= minimo, f"{campo}: solo {n}/30"


def test_graph_trims_the_brochure_tail_off_a_value():
    """La extraccion por regex no sabe donde termina un dato: el titulo de Psicologia salia
    "psicologo becas y financiacion directorio de" el 2026-11-17."""
    assert graph.recortar("psicologo becas y financiacion directorio") == "psicologo"
    assert graph.recortar("arquitecto descripcion del programa") == "arquitecto"
    for nodo in graph.build().values():
        assert len(nodo.get("titulo", "").split()) <= 6, nodo.get("titulo")


def test_graph_field_comes_from_the_current_question():
    """"y cuanto dura" despues de "hablame del plan de estudios" devolvia las materias, porque
    la coincidencia mas larga ganaba y venia del turno anterior."""
    g = graph.build()
    hecho, _ = graph.consultar(g, "y cuanto dura", "Ingeniería Mecatrónica",
                                contexto="me puedes hablar del plan de estudios")
    assert "periodos" in hecho, hecho


def test_graph_detects_short_and_ambiguous_program_names():
    """"Cine" tiene cuatro letras y con el umbral anterior era indetectable: el foco saltaba a
    Comunicacion Social, que respondia que Cine es parte de ese programa. Y "contaduria" esta
    en dos nombres, asi que ninguna palabra suya es distintiva."""
    g = graph.build()
    assert graph.programa_de("hay algo relacionado con el cine", g, "") == "Cine"
    assert graph.programa_de("cuentame de contaduria", g, "") == "Contaduría Pública"
    assert "Datos" in graph.programa_de("que tal la ingenieria de datos", g, "")
    # Y nada de falsos positivos con charla.
    for q in ("hola como estas", "me gusta dibujar", "que carreras tienen"):
        assert graph.programa_de(q, g, "") == "", q


def test_graph_generic_words_come_from_the_catalog():
    """Escrita a mano, esa lista dejo fuera "datos" y rompio un programa entero."""
    g = graph.build()
    gen = graph._genericas(g)
    assert "ingenieria" in gen and "virtual" in gen     # en varios nombres
    assert "datos" not in gen and "cine" not in gen     # en uno solo


def test_graph_keeps_the_program_across_turns():
    g = graph.build()
    assert graph.programa_de("y cuanto dura", g, "Psicología") == "Psicología"
    assert "Mercadeo" in graph.programa_de("hablame de mercadeo", g, "Psicología")


def test_graph_answers_catalog_questions_with_official_names():
    """En la conversacion real del 2026-11-17 el robot dijo que Mercadeo y Negocios
    Internacionales es una carrera "relacionada con el arte". Los nombres oficiales son el
    unico antidoto contra inventar carreras."""
    g = graph.build()
    todos = graph.listar(g, "que carreras tienen")
    assert "Arquitectura" in todos
    # Se entregan pocos nombres a proposito: con los 30 delante, el modelo los recitaba todos
    # (43 palabras, 29 segundos en la Pi). El resto se anuncia, no se lista.
    assert todos.count(",") < graph.MAX_LISTA + 1 and "más" in todos
    solo_ing = graph.listar(g, "que carreras de ingenieria hay")
    assert "Ingeniería Ambiental" in solo_ing and "Arquitectura" not in solo_ing
    assert graph.listar(g, "hola como va todo") == ""
    assert graph.listar(g, "cuanto dura psicologia") == ""


def test_graph_says_nothing_rather_than_guessing():
    """Un dato ausente tiene que poder decirse "no lo se": rellenarlo con lo que haya cerca es
    exactamente como el modelo respondia que la carrera "dura 139 creditos"."""
    g = graph.build()
    assert graph.hecho(g, "Psicología", "inventado") == ""
    assert graph.hecho(g, "Carrera Que No Existe", "duracion") == ""
    assert graph.consultar(g, "hola como va todo") == ("", "")


def test_graph_facts_are_far_smaller_than_prose():
    """La razon de ser del grafo: en la Pi cada ficha de prompt son ~23 ms de silencio."""
    g = graph.build()
    hecho, _ = graph.consultar(g, "cuanto dura", "Ingeniería Mecatrónica")
    assert 0 < len(hecho) < 120, hecho


def test_bm25_prefix_expansion_reaches_word_forms():
    """"cuanto DURA" tiene que alcanzar "DURACION": el 2026-11-17 el robot contesto que la
    carrera "dura 139 creditos" teniendo "8 periodos academicos" en el corpus, porque la
    busqueda lexica comparaba palabras exactas."""
    kb = lexical.load()
    assert "duracion" in kb.expandir("dura")
    assert "laboratorios" in kb.expandir("labor")
    # Se expande aunque el termino exacto exista: "dura" esta en el corpus por su cuenta.
    assert len(kb.expandir("dura")) > 1


def test_bm25_short_terms_are_not_expanded():
    """Con menos de cuatro letras el prefijo arrastra medio diccionario."""
    kb = lexical.load()
    assert len(kb.expandir("de")) <= 1 and len(kb.expandir("psi")) <= 1


def test_bm25_finds_the_duration_fact():
    kb = lexical.load()
    texto = " ".join(f["texto"] for f in lexical.fragmentos(kb, "estudio psicologia y cuanto dura"))
    assert "periodos" in texto.lower(), texto[:160]


def test_fragments_are_much_smaller_than_whole_documents():
    """La razon de ser de los fragmentos: en la Pi cada ficha de prompt son ~23 ms."""
    kb = lexical.load()
    consulta = "cuenteme de arquitectura que materias hay en el sexto semestre"
    enteros = sum(len(str(d.get("texto", ""))) for d, _ in kb.search(consulta, k=3))
    trozos = sum(len(f["texto"]) for f in lexical.fragmentos(kb, consulta))
    assert trozos < enteros / 2, f"{trozos} contra {enteros}"


def test_bm25_lists_the_official_programs():
    kb = lexical.load()
    nombres = kb.program_names()
    assert "Arquitectura" in nombres and "Psicología" in nombres
    assert all(n for n in nombres) and len(nombres) == len(set(nombres))


def test_focus_and_fact_never_disagree_across_a_conversation():
    """El dato entregado tiene que ser del programa que el foco dice. El 2026-11-17, a
    "hablame de la carrera de cine" el foco decia Cine y el texto era de Ingenieria
    Mecatronica: find() decidia el programa y consultar() lo recalculaba por su cuenta."""
    class Cfg:
        rag_backend = "grafo"; rag_max_cards = 3; rag_gate_bm25 = 1.0
        rag_gate_charla = 6.0; rag_fragmentos = 2
    _, find = rag.make_retriever(Cfg())
    guion = [("", "me puedes hablar del programa de ingenieria mecatronica", "Mecatrónica"),
             ("me puedes hablar del programa de ingenieria mecatronica",
              "me puedes hablar de la carrera de cine", "Cine"),
             ("me puedes hablar de la carrera de cine", "y cuanto dura", "Cine"),
             ("y cuanto dura", "que materias tiene", "Cine")]
    for previo, pregunta, esperado in guion:
        docs = find(pregunta, previo)
        assert docs, pregunta
        prog = docs[0].get("programa") or ""
        assert esperado in prog, f"{pregunta}: foco {prog}, esperaba {esperado}"
        # Y el texto tiene que hablar de ese programa, no de otro.
        assert prog.split()[-1].lower()[:5] in str(docs[0]["texto"]).lower(), docs[0]["texto"][:90]


def test_retriever_dispatches_by_profile():
    """server.py y main.py llaman una sola puerta; el perfil elige el motor. Sin esto habria
    que editar codigo para cambiar de la DGX a la Pi, que es como se cuelan las diferencias
    entre lo que se mide y lo que se despliega."""
    class Cfg:
        rag_backend = "ninguno"
        rag_max_cards = 3
        rag_gate_bm25 = 1.0
    kb, find = rag.make_retriever(Cfg())
    assert kb is None and find("hola de arquitectura") == []

    Cfg.rag_backend = "bm25"
    kb, find = rag.make_retriever(Cfg())
    hits = find("como se llama el titulo", previous="estudio psicologia")
    assert hits and any("Psicolog" in (h.get("programa") or "") for h in hits)


def test_retriever_bm25_uses_the_previous_turn_as_topic():
    """Medido el 2026-11-16: con el programa en la consulta la recuperacion acierta el programa
    el 100% de las veces, contra el 40% sin el. El turno anterior es ese contexto."""
    class Cfg:
        rag_backend = "bm25"
        rag_max_cards = 3
        rag_gate_bm25 = 1.0
    _, find = rag.make_retriever(Cfg())
    sin = find("que materias hay en el sexto semestre")
    con = find("que materias hay en el sexto semestre", previous="cuenteme de arquitectura")
    assert any("Arquitectura" in (h.get("programa") or "") for h in con)
    assert con != sin


def test_quiz_does_not_punish_an_honest_refusal_as_an_invention():
    """Nombrar una carrera ausente PARA rechazarla es el comportamiento correcto. El
    2026-11-16 esta medicion lo contaba como invencion y restaba puntos por acertar."""
    oficiales = ["Ingeniería Ambiental", "Arquitectura", "Mercadeo Global – Virtual"]
    bien = quiz.score("No contamos con Ingeniería Química, pero sí con Ingeniería Ambiental.",
                      [], program="", official=oficiales)
    assert not bien["inventado"]
    mal = quiz.score("Claro, puedes estudiar Ingeniería Química en la UAO.",
                     [], program="", official=oficiales)
    assert mal["inventado"]


def test_lexical_normalizes_accents_because_students_type_without_them():
    assert lexical.normalize("Diseño Gráfico") == "diseno grafico"
    assert lexical.tokenize("de la Arquitectura") == ["arquitectura"]   # vacias fuera


def test_lexical_drops_corpus_wide_words():
    """'programa' esta en casi los 400 documentos: dejarlo haria que toda consulta matchee todo."""
    assert "programa" not in lexical.tokenize("el programa de psicologia")
    assert "psicologia" in lexical.tokenize("el programa de psicologia")


def test_lexical_ranks_the_matching_program_first():
    docs = [{"titulo": "Psicología", "programa": "Psicología",
             "texto": "El titulo otorgado es Psicologo, nueve periodos academicos."},
            {"titulo": "Arquitectura", "programa": "Arquitectura",
             "texto": "Taller de proyecto y urbanismo, 150 creditos."},
            {"titulo": "Mecatrónica", "programa": "Ingeniería Mecatrónica",
             "texto": "Automatizacion y control, Fab-Lab."}]
    idx = lexical.BM25(docs)
    mejor, puntaje = idx.search("como se llama el titulo del programa de psicologia", k=1)[0]
    assert mejor["programa"] == "Psicología" and puntaje > 0


def test_lexical_searches_title_and_program_not_only_body():
    """Una pregunta que nombra la carrera debe alcanzar sus documentos aunque el cuerpo no la
    repita: el nombre del programa es parte del texto buscable."""
    docs = [{"titulo": "Arquitectura: laboratorios", "programa": "Arquitectura",
             "texto": "Los talleres de proyecto se llaman laboratorios."},
            {"titulo": "Derecho: consultorio", "programa": "Derecho",
             "texto": "Brigadas juridicas desde el quinto semestre."}]
    idx = lexical.BM25(docs)
    assert idx.search("arquitectura", k=1)[0][0]["programa"] == "Arquitectura"


def test_lexical_returns_nothing_when_nothing_matches():
    """Sin coincidencia, lista vacia: deja que quien llama diga que no sabe, en vez de
    entregarle al modelo un documento irrelevante que invite a inventar."""
    idx = lexical.BM25([{"titulo": "Psicología", "programa": "Psicología", "texto": "Psicologo"}])
    assert idx.search("cuanto cuesta el parqueadero de bicicletas") == []


def test_judge_prompt_withholds_the_reference_answer():
    """El juez debe ver la FUENTE, nunca la respuesta de referencia: mostrarsela reproduciria
    el defecto que este modulo existe para arreglar (comparar redacciones, no hechos)."""
    msgs = judge.build_prompt("¿hay Neurociencias?", "Si, hay Neurociencias",
                              "El programa incluye Neurociencias en el tercer ciclo.", "Psicologia")
    texto = " ".join(m["content"] for m in msgs)
    assert "tercer ciclo" in texto                 # la fuente si
    assert "Psicologia" in texto
    assert "referencia" not in texto.lower()       # la respuesta de referencia no


def test_judge_parse_rejects_unknown_verdicts():
    assert judge.parse({"veredicto": "correcta", "por_que": "ok"})[0] == "correcta"
    assert judge.parse({"veredicto": "CORRECTA"})[0] == "correcta"
    assert judge.parse({"veredicto": "mas o menos"})[0] == ""    # no se cuela como aprobada
    assert judge.parse(None)[0] == ""
    assert judge.parse("correcta")[0] == ""


def test_judge_summary_excludes_the_unanswerable_from_the_denominator():
    """Contar una pregunta imposible como fallo del modelo lo subestimaria igual que la
    metrica vieja; y una respuesta sin juicio tampoco puede contar como buena."""
    s = judge.summarize(["correcta", "correcta", "incorrecta", "incompleta", "imposible", ""])
    assert s["juzgables"] == 4                      # 2 correctas + 1 incorrecta + 1 incompleta
    assert s["exactitud"] == 0.5
    assert s["exactitud_laxa"] == 0.75
    assert s["tasa_error"] == 0.25
    assert s["imposible"] == 1 and s["sin_juicio"] == 1


def test_judge_summary_of_nothing_is_zero_not_a_crash():
    s = judge.summarize([])
    assert s["exactitud"] == 0.0 and s["juzgables"] == 0


if __name__ == "__main__":
    # Collect every function whose name starts with test_ and run it.
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} checks passed")
