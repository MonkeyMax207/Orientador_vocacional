"""Voice activity detection (is this 32 ms chunk speech?) + turn detection (has the user finished?).

Silero VAD is a tiny neural network (~2 MB) exported to ONNX, a portable model format.
onnxruntime executes it on the CPU in about 1 ms per chunk, with no PyTorch needed.
"""
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).parent / "models" / "silero_vad.onnx"
CHUNK_MS = 32        # 512 samples at 16 kHz
CONTEXT = 64         # Silero v5 expects the last 64 samples of the previous chunk in front of the new one


class VAD:
    def __init__(self, threshold: float = 0.5):
        if not MODEL.exists():
            raise SystemExit(f"Falta {MODEL}. Ejecuta: uv run python download_models.py")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1      # the model is tiny; 1 thread leaves the cores for Whisper/LLM
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        # The model is recurrent: `state` carries memory of the audio heard so far.
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self.context, chunk])[np.newaxis, :].astype(np.float32)  # shape (1, 576)
        out, self.state = self.session.run(
            None,   # None = return all outputs: [speech probability, new state]
            {"input": x, "state": self.state, "sr": np.array(16000, dtype=np.int64)},
        )
        self.context = chunk[-CONTEXT:]
        return float(out[0][0])

    def is_speech(self, chunk: np.ndarray) -> bool:
        return self.prob(chunk) > self.threshold


class TurnDetector:
    """Collects chunks into one user turn; returns the audio when the turn ends."""

    def __init__(self, silence_ms: int, min_speech_ms: int, pre_roll_chunks: int = 10):
        self.silence_limit = silence_ms // CHUNK_MS    # e.g. 600 ms → 18 chunks
        self.min_speech = min_speech_ms // CHUNK_MS    # e.g. 250 ms → 7 chunks
        # Pre-roll: the VAD reacts a few ms late, so we keep the last ~320 ms before speech
        # started. Otherwise the first syllable gets cut ("ola" instead of "hola").
        self.pre = deque(maxlen=pre_roll_chunks)
        self.reset()

    def reset(self) -> None:
        self.buf = []            # chunks of the current turn (empty = not in a turn)
        self.speech_chunks = 0   # speech chunks in this turn (main.py reads this for barge-in)
        self.silence = 0         # consecutive silent chunks
        self.pre.clear()

    def feed(self, chunk: np.ndarray, is_speech: bool):
        if not self.buf:                    # waiting for someone to start talking
            if is_speech:
                self.buf = list(self.pre) + [chunk]
                self.speech_chunks, self.silence = 1, 0
            else:
                self.pre.append(chunk)
            return None
        self.buf.append(chunk)              # inside a turn: keep everything, pauses included
        if is_speech:
            self.speech_chunks += 1
            self.silence = 0                # a short pause followed by speech resets the count
        else:
            self.silence += 1
        if self.silence >= self.silence_limit:
            audio, enough = np.concatenate(self.buf), self.speech_chunks >= self.min_speech
            self.reset()
            return audio if enough else None   # too little speech = a click or cough, ignore it
        return None


if __name__ == "__main__":
    # Live meter to tune vad_threshold: talk, stay quiet, and watch the bar.
    # Run: uv run python vad.py [pc|pi]
    import sys

    from audio_io import Mic
    from config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    vad, mic = VAD(cfg.vad_threshold), Mic(cfg.input_device)
    print("Habla y mira la barra (Ctrl+C para salir)")
    while True:
        p = vad.prob(mic.read())
        print(f"\r{'#' * int(p * 40):<40} {p:.2f} {'VOZ' if p > cfg.vad_threshold else '   '}", end="", flush=True)
