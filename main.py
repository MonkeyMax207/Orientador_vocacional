"""The voice agent: LISTENING → THINKING → SPEAKING, forever.

Run: uv run python main.py [pc|pi]

Threads involved:
  - PortAudio's mic thread  → pushes 32 ms chunks into Mic.q
  - Player thread           → plays queued clips, fires animations
  - LLM worker thread       → pushes Ollama tokens into a queue (one per turn)
  - this main thread        → VAD, STT, sentence splitting, Piper, orchestration
"""
import queue
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import animations
from audio_io import SAMPLE_RATE, Mic, Player, save_wav, to_int16
from brain import llm_worker, make_brain
from config import DEFAULT_PROFILE, load_config
from fillers import Fillers
from respuestas import Respuestas
from rag import make_retriever
from sentences import SentenceSplitter, clean_for_tts
from stt import STT, clean
from tts import TTS
from vad import CHUNK_MS, VAD, TurnDetector

ROOT = Path(__file__).parent
RECORDINGS = ROOT / "recordings"
# Las pruebas lo redirigen, igual que RECORDINGS. Sin esto, test_logic.py escribia turnos
# falsos en el registro real: el 2026-11-17 aparecieron como "Hola" repetido y "que materias
# tiene mecatronica" con stt 0.0s, y yo los diagnostique como un bucle del microfono. El
# registro es la evidencia de como se comporta el robot; contaminarlo cuesta horas.
CONVERSACIONES = ROOT / "logs"


def slow_filler_due(elapsed, limit, answer_started, already_played, player_busy) -> bool:
    # Play a 2nd filler only if: the answer is late, none has started, we haven't already,
    # AND the 1st filler has finished. Queued behind the 1st one it would delay an answer
    # that is probably about to arrive.
    return elapsed > limit and not answer_started and not already_played and not player_busy


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        print("Cargando modelos...", flush=True)
        # UAO knowledge is optional (knowledge/chunks.json); the LLM decides when to search it.
        self.knowledge, self.find = make_retriever(cfg)
        names = self.knowledge.program_names() if self.knowledge else ()
        self.brain = make_brain(cfg, program_names=names)
        self.brain.check()     # fail fast if Ollama is down or the model is missing
        self.brain.warmup()    # load the LLM + cache the system prompt
        self.stt = STT(cfg.whisper_model, cfg.whisper_device,
                       cfg.whisper_compute, cfg.threads)
        # pace y pause_s vienen del trabajo hecho en la Pi: el ritmo se alarga sin bajar el
        # tono, y la pausa evita que las frases se atropellen (cada una se sintetiza apenas el
        # LLM la escribe). Opcionales, asi que la DGX y el PC siguen funcionando sin ellos.
        self.tts = TTS(cfg.tts_engine, cfg.voice, cfg.tts_device,
                       getattr(cfg, "voice_pace", 1.0), getattr(cfg, "sentence_pause_s", 0.0))
        # warm-up
        self.tts.synthesize("Hola.")
        self.fillers = Fillers(cfg.voice)
        self.respuestas = Respuestas()
        self.vad = VAD(cfg.vad_threshold)
        self.turns = TurnDetector(cfg.silence_ms, cfg.min_speech_ms)
        # voice_rate < 1 plays the same samples slower: a deeper "character" voice at zero CPU cost.
        self.player = Player(cfg.output_device, int(
            self.tts.sample_rate * cfg.voice_rate))
        # opened last so no audio piles up while loading
        self.mic = Mic(cfg.input_device)
        self.ready_turn = None             # a full user turn captured while we were speaking
        # everything the student said (recommendation query)
        self.user_turns = []
        # the agent's last answer (context for detail queries)
        self.last_reply = ""
        RECORDINGS.mkdir(exist_ok=True)

    def feed_mic(self, chunk):
        return self.turns.feed(chunk, self.vad.is_speech(chunk))

    def listen(self) -> np.ndarray:
        # A complete turn may already be waiting (captured during barge-in, see below).
        if self.ready_turn is not None:
            audio, self.ready_turn = self.ready_turn, None
            return audio
        # LISTENING: run every 32 ms chunk through the VAD until a full turn comes back.
        while True:
            audio = self.feed_mic(self.mic.read())
            if audio is not None:
                return audio

    def user_interrupting(self) -> bool:
        # Called repeatedly while the agent is thinking/speaking.
        if not self.cfg.barge_in:
            self.mic.pending()   # discard what the mic heard (mostly our own voice)
            return False
        for chunk in self.mic.pending():
            # Keep building the user's turn, so their words aren't lost. While STT or Piper
            # blocked this thread, a WHOLE short turn may have piled up: keep it for listen().
            audio = self.feed_mic(chunk)
            if audio is not None:
                self.ready_turn = audio
                return True
        # Interrupt once the user has said as much as a valid turn needs (min_speech_ms).
        return self.turns.speech_chunks * CHUNK_MS >= self.cfg.min_speech_ms

    def _anotar(self, pregunta, respuesta, cards, t_end, t_stt, t_first) -> None:
        """Una linea JSON por turno en logs/conversacion-<fecha>.jsonl. Nunca interrumpe: si
        escribir falla, el robot sigue hablando."""
        try:
            registro = {
                "hora": datetime.now().isoformat(timespec="seconds"),
                "pregunta": pregunta,
                "respuesta": respuesta,
                "docs": [{"id": c.get("id", ""), "programa": c.get("programa", ""),
                           "texto": str(c.get("texto", ""))[:300]} for c in cards],
                "stt_s": round(t_stt - t_end, 2),
                "primera_palabra_s": round(t_first - t_stt, 2) if t_first else None,
            }
            destino = CONVERSACIONES / f"conversacion-{datetime.now():%Y%m%d}.jsonl"
            destino.parent.mkdir(exist_ok=True)
            with open(destino, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[log] no se pudo anotar el turno: {e}", flush=True)

    def find_cards(self, text: str) -> list[dict]:
        # UAO text only for questions about the UAO (rag_gate); plain chat gets nothing.
        self.user_turns.append(text)
        previous = self.user_turns[-2] if len(self.user_turns) > 1 else ""
        # La respuesta anterior del robot tambien es contexto, y hacia falta: el 2026-11-17,
        # tras contestar "si, hay la carrera de Cine", la pregunta siguiente ("¿y que materias
        # se ven en esa carrera?") seguia trayendo Ingenieria Mecatronica, porque el foco solo
        # miraba lo que decia el estudiante. "Esa carrera" era la que el robot acababa de
        # nombrar, y el grafo no se enteraba.
        contexto = f"{self.last_reply} {previous}".strip()
        cards = self.find(text, contexto) if self.knowledge else []
        if cards:
            print(f"[rag] {', '.join(c['id'] for c in cards)}", flush=True)
        return cards

    def respond(self, audio: np.ndarray) -> None:
        # The user actually stopped talking silence_ms ago (that's how we detected the end).
        t_detect = time.perf_counter()
        t_end = t_detect - self.cfg.silence_ms / 1000
        # THINKING: the filler is SCHEDULED, not played. threading.Timer runs play_filler on its
        # own thread after a delay, unless we cancel() it first. So it only sounds if no answer
        # is ready filler_after_s after the user stopped: that still keeps first sound < 2 s,
        # but a quick answer comes without "Mmm..." in front of it.
        t_filler = []   # the timer thread records here when the filler really started

        def play_filler():
            t_filler.append(time.perf_counter())
            self.player.put(self.fillers.pick(), ["pensar"])

        filler = threading.Timer(
            max(0.0, t_end + self.cfg.filler_after_s - t_detect), play_filler)
        filler.start()
        save_wav(
            RECORDINGS / f"turn_{datetime.now():%Y%m%d_%H%M%S}.wav", to_int16(audio), SAMPLE_RATE)

        text = clean(self.stt.transcribe(audio))
        t_stt = time.perf_counter()
        if not text:
            print("(ruido ignorado)", flush=True)
            filler.cancel()
            self.player.stop()   # it was a cough or a bang: cut the filler short if it started
            self.finish_speaking()
            return
        print(f"\nTú: {text}", flush=True)
        # verified UAO facts for this turn (maybe none)
        # Respuesta inmediata para lo que no necesita al modelo: saludos, agradecimientos y
        # —sobre todo— precios. Idea de Juan Diego el 2026-11-17, tras oir al robot tardar
        # varios segundos en decir "Hola, que bien". Ahorra el 57% del turno que se va en el
        # modelo, y de paso saca el requisito de "nunca dar precios" de las manos del modelo:
        # una frase escrita a mano no alucina.
        lista, regla = self.respuestas.buscar(text)
        if lista:
            filler.cancel()
            self.player.stop()        # la muletilla no tiene sentido si ya hay respuesta
            print(f"[inmediata:{regla}] {lista}", flush=True)
            self.player.put(self.tts.synthesize(lista), ["hablar"])
            print(f"Orienta: {lista}", flush=True)
            self.last_reply = lista
            self._anotar(text, lista, [], t_end, t_stt, None)
            self.finish_speaking()
            return

        cards = self.find_cards(text)

        # Start the LLM on its own thread; tokens arrive in `tokens` while we keep working.
        tokens, stop = queue.Queue(), threading.Event()
        threading.Thread(target=llm_worker, args=(
            self.brain, text, tokens, stop, cards), daemon=True).start()

        splitter, spoken = SentenceSplitter(), []
        t_sentence = t_first = None
        slow_played = False
        while True:
            if self.user_interrupting():
                stop.set()
                filler.cancel()
                self.player.stop()
                print("[interrumpido]", flush=True)
                return
            try:
                # wait max 50 ms so we can check other things
                token = tokens.get(timeout=0.066)
            except queue.Empty:
                # Answer still not ready long after the user stopped? Say a second filler.
                if slow_filler_due(time.perf_counter() - t_end, self.cfg.slow_llm_s,
                                   t_first is not None, slow_played, self.player.busy()):
                    self.player.put(self.fillers.pick(slow=True), ["pensar"])
                    slow_played = True
                continue
            # None = the LLM finished: flush the last partial sentence.
            for sentence in splitter.flush() if token is None else splitter.feed(token):
                sentence = clean_for_tts(sentence)
                if not sentence:
                    continue
                t_sentence = t_sentence or time.perf_counter()
                # SPEAKING: synthesize and queue. It plays right after the filler/previous sentence.
                clip = self.tts.synthesize(sentence)
                filler.cancel()   # the answer is here: a filler not yet started is no longer needed
                # Un respiro si justo antes sono una muletilla: encadenadas sin aire suenan
                # atropelladas, aunque el reloj diga que el tiempo esta bien. Solo antes de la
                # primera frase, y solo si hubo muletilla.
                pausa = getattr(self.cfg, "pausa_tras_filler_s", 0.0)
                if t_first is None and pausa and (t_filler or slow_played):
                    import numpy as _np
                    self.player.put(_np.zeros(int(self.tts.sample_rate * pausa), dtype=clip.dtype))
                self.player.put(clip, animations.match(sentence))
                t_first = t_first or time.perf_counter()
                spoken.append(sentence)
            if token is None:
                break

        print(f"Orienta: {' '.join(spoken)}", flush=True)
        self.last_reply = " ".join(spoken)
        # El 2026-11-17 la primera conversacion real del robot se perdio al cerrar la terminal, y
        # con ella la unica evidencia de como se comporta fuera de un quiz. Cada turno queda
        # escrito con lo que recupero y cuanto tardo: sin esto se diagnostica a ciegas y se
        # termina midiendo lo que es comodo medir en vez de lo que falla.
        self._anotar(text, " ".join(spoken), cards, t_end, t_stt, t_first)
        if t_first:
            first_sound = min(t_filler + [t_first]) - t_end
            source = "filler" if t_filler and t_filler[0] < t_first else "respuesta"
            print(f"[tiempos] primer sonido {first_sound:.2f}s ({source}) | stt {t_stt - t_detect:.2f}s | "
                  f"llm 1ª frase {t_sentence - t_stt:.2f}s | tts 1ª frase {t_first - t_sentence:.2f}s | "
                  f"respuesta lista {t_first - t_end:.2f}s", flush=True)
        self.finish_speaking()

    def finish_speaking(self) -> None:
        # Wait for the queued audio to finish (still watching for barge-in), then reset.
        while self.player.busy():
            if self.user_interrupting():
                self.player.stop()
                print("[interrumpido]", flush=True)
                return
            time.sleep(0.03)
        if not self.cfg.barge_in:
            self.mic.pending()   # throw away the echo of our own voice
            self.vad.reset()
            self.turns.reset()


def main():
    # Sin argumento, la maquina decide: la Pi no debe caer en el perfil del PC, que usa
    # embeddings y CUDA. El 2026-11-17 ese defecto dejo a la Pi tres minutos vectorizando 253
    # fragmentos en su CPU antes de que nadie pudiera hablarle.
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROFILE)
    agent = Agent(cfg)
    print(
        f"Listo (perfil {cfg.profile}). Habla cuando quieras. Ctrl+C para salir.", flush=True)
    try:
        while True:
            agent.respond(agent.listen())
    except KeyboardInterrupt:
        print("\nAdiós")


if __name__ == "__main__":
    main()
