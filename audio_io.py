"""Microphone and speaker. The only module that talks to the sound card.

Connection: PortAudio (C library, bundled inside the sounddevice wheel) opens the
USB module. It calls our callback from its own thread every 32 ms with fresh
samples; we push them into a queue.Queue, a thread-safe FIFO the main loop reads from.
"""
import queue
import threading
import wave

import numpy as np
import sounddevice as sd

import animations

SAMPLE_RATE = 16000   # Hz. Whisper and Silero VAD both expect 16 kHz.
CHUNK = 512           # samples per block = 32 ms at 16 kHz (the size Silero VAD requires)


def to_int16(audio: np.ndarray) -> np.ndarray:
    # float32 in [-1, 1] → int16 in [-32767, 32767]; clip first so loud peaks don't wrap around.
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def save_wav(path, audio: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as w:   # stdlib wave: plain PCM .wav files
        w.setnchannels(1)                     # mono
        w.setsampwidth(2)                     # 2 bytes per sample = int16
        w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())


def load_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate()


class Mic:
    def __init__(self, device: str):
        self.q = queue.Queue()
        try:
            self.stream = sd.InputStream(
                device=device, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=CHUNK,              # PortAudio calls us with exactly 512 samples
                callback=self._callback,
            )
        except ValueError as e:               # device not found, or the name matches several
            raise SystemExit(f"Micrófono '{device}': {e}\nEjecuta: uv run python list_devices.py")
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        # Runs on PortAudio's thread: do the minimum and return fast, or audio drops out.
        if status:
            print(f"[mic] {status}", flush=True)   # e.g. "input overflow" = we read too slowly
        self.q.put(indata[:, 0].copy())            # indata is (frames, channels); keep channel 0.
                                                   # copy() because PortAudio reuses the buffer

    def read(self, timeout=None) -> np.ndarray:
        return self.q.get(timeout=timeout)         # blocks until the next 32 ms chunk arrives

    def pending(self) -> list[np.ndarray]:
        # Everything recorded so far, without waiting. Used while the agent is speaking.
        chunks = []
        while True:
            try:
                chunks.append(self.q.get_nowait())
            except queue.Empty:
                return chunks


class Player:
    """Plays queued clips one after another on a background thread."""

    def __init__(self, device: str, sample_rate: int):
        self.q = queue.Queue()                 # items: (int16 audio, animation names)
        self._stop = threading.Event()         # thread-safe flag used to cut playback short
        try:
            self.stream = sd.OutputStream(
                device=device, samplerate=sample_rate, channels=1, dtype="int16",
                latency="low",                 # small device buffer = sound starts sooner
            )
        except ValueError as e:
            raise SystemExit(f"Altavoz '{device}': {e}\nEjecuta: uv run python list_devices.py")
        self.stream.start()
        # daemon=True: this thread dies automatically when the program exits.
        threading.Thread(target=self._run, daemon=True).start()

    def put(self, audio: np.ndarray, anims=()) -> None:
        self.q.put((audio, list(anims)))

    def busy(self) -> bool:
        # unfinished_tasks counts items queued OR currently playing (task_done not called yet).
        return self.q.unfinished_tasks > 0

    def stop(self) -> None:
        # Barge-in: cut the current clip and throw away everything queued.
        # ponytail: a stop() landing exactly between get() and clear() lets one clip play; add a generation counter if that shows up.
        self._stop.set()
        while True:
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                break

    def _run(self):
        while True:
            audio, anims = self.q.get()        # waits here until something is queued
            self._stop.clear()
            for name in anims:                 # fire the animations as this clip STARTS
                animations.trigger(name)
            for i in range(0, len(audio), 1024):   # write in small pieces so stop() reacts fast
                if self._stop.is_set():
                    break
                self.stream.write(audio[i:i + 1024].reshape(-1, 1))   # (frames, channels)
            self.q.task_done()


if __name__ == "__main__":
    # Self-check: record 3 s from the module's mic and play it back through its speaker.
    # Run: uv run python audio_io.py [pc|pi]
    import sys
    import time
    from pathlib import Path

    from config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    mic = Mic(cfg.input_device)
    print("Habla durante 3 segundos...")
    chunks = [mic.read() for _ in range(3 * SAMPLE_RATE // CHUNK)]
    audio = np.concatenate(chunks)
    # Peak level: ~0.0 = mic dead or muted, 0.1–0.8 = good, 1.0 = clipping (too loud)
    print(f"nivel pico: {np.abs(audio).max():.2f}")
    Path("recordings").mkdir(exist_ok=True)
    save_wav("recordings/mic_test.wav", to_int16(audio), SAMPLE_RATE)
    player = Player(cfg.output_device, SAMPLE_RATE)
    player.put(to_int16(audio))
    while player.busy():
        time.sleep(0.1)
    print("Guardado en recordings/mic_test.wav")
