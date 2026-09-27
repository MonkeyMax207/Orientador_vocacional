"""Text-to-speech with Piper: a small neural voice (VITS model, exported to ONNX).

Piper first converts letters to phonemes with espeak-ng (bundled in the wheel), then the
ONNX model turns phonemes into a waveform. It runs faster than real time even on a Pi.
"""
from pathlib import Path

import numpy as np
from piper import PiperVoice

VOICES = Path(__file__).parent / "models" / "piper"


class TTS:
    def __init__(self, voice: str):
        path = VOICES / f"{voice}.onnx"
        if not path.exists():
            raise SystemExit(f"Falta la voz {path}. Ejecuta: uv run python download_models.py")
        self.voice = PiperVoice.load(str(path))          # also reads <voice>.onnx.json next to it
        self.sample_rate = self.voice.config.sample_rate  # 22050 Hz for "medium" voices

    def synthesize(self, text: str) -> np.ndarray:
        # Piper yields one AudioChunk per sentence it finds; join them into a single clip.
        chunks = [c.audio_int16_array for c in self.voice.synthesize(text)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


if __name__ == "__main__":
    # Listen to the voice and measure its speed. Run: uv run python tts.py [pc|pi] [text...]
    import sys
    import time

    from audio_io import Player
    from config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    text = " ".join(sys.argv[2:]) or "¡Hola! Soy Orienta. ¿Qué materias disfrutas más en el colegio?"
    tts = TTS(cfg.voice)
    tts.synthesize("Hola.")   # warm-up
    t = time.perf_counter()
    audio = tts.synthesize(text)
    dt, dur = time.perf_counter() - t, len(audio) / tts.sample_rate
    # RTF (real-time factor) = synthesis time / audio length. Below 1 = faster than speaking it.
    print(f"síntesis {dt:.2f}s para {dur:.2f}s de audio (RTF {dt / dur:.2f})")
    player = Player(cfg.output_device, tts.sample_rate)
    player.put(audio)
    while player.busy():
        time.sleep(0.1)
