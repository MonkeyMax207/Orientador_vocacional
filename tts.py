"""Text-to-speech with two interchangeable engines, chosen per profile in config.toml:

- Piper:  small VITS voices exported to ONNX. Fast (RTF ~0.1 on PC, ~0.25 on a Pi 5) but more robotic.
- Kokoro: an 82M-parameter model (StyleTTS2-based), also ONNX. Much more natural, ~7x slower:
          fine on the PC, too slow for real-time conversation on the Pi's CPU.

Both first turn letters into phonemes with espeak-ng (bundled in their wheels), then the neural
model turns phonemes into a waveform. Both return int16 audio, so the rest of the code doesn't care.
"""
from pathlib import Path

import numpy as np

MODELS = Path(__file__).parent / "models"


class TTS:
    def __init__(self, engine: str, voice: str, device: str = "cpu"):
        self.engine, self.voice_name = engine, voice
        if engine == "piper":
            from piper import PiperVoice               # imported here: load only the engine in use
            path = MODELS / "piper" / f"{voice}.onnx"
            if not path.exists():
                raise SystemExit(f"Falta la voz {path}. Ejecuta: uv run python download_models.py")
            self.voice = PiperVoice.load(str(path))    # also reads <voice>.onnx.json next to it
            self.sample_rate = self.voice.config.sample_rate   # 22050 Hz for "medium" voices
        elif engine == "kokoro":
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
            model, voices = MODELS / "kokoro" / "kokoro-v1.0.onnx", MODELS / "kokoro" / "voices-v1.0.bin"
            if not model.exists() or not voices.exists():
                raise SystemExit(f"Falta Kokoro en {model.parent}. Ejecuta: uv run python download_models.py")
            # An ONNX Runtime "execution provider" is the backend that runs the model. We list them
            # in order of preference: CUDA (GPU) first, CPU as fallback if the GPU can't be used.
            providers = ["CPUExecutionProvider"]
            ort.set_default_logger_severity(3)   # errors only: hide harmless CUDA optimization warnings
            if device == "cuda":
                ort.preload_dlls()   # load CUDA/cuDNN from the nvidia pip packages before the session
                providers.insert(0, "CUDAExecutionProvider")
            session = ort.InferenceSession(str(model), providers=providers)
            # voices-v1.0.bin holds every voice as a "style vector"; `voice` picks one (e.g. em_santa).
            self.kokoro = Kokoro.from_session(session, str(voices))
            self.sample_rate = 24000                   # Kokoro always outputs 24 kHz
        else:
            raise SystemExit(f"tts_engine desconocido '{engine}'. Usa 'piper' o 'kokoro'")

    def synthesize(self, text: str) -> np.ndarray:
        if self.engine == "piper":
            # Piper yields one AudioChunk per sentence it finds; join them into a single clip.
            chunks = [c.audio_int16_array for c in self.voice.synthesize(text)]
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        # Kokoro returns float32 in [-1, 1] (and its sample rate, which we already know).
        samples, _sr = self.kokoro.create(text, voice=self.voice_name, speed=1.0, lang="es")
        return (np.clip(samples, -1, 1) * 32767).astype(np.int16)


if __name__ == "__main__":
    # Listen to the voice and measure its speed. Run: uv run python tts.py [pc|pi] [text...]
    import sys
    import time

    from audio_io import Player
    from config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    text = " ".join(sys.argv[2:]) or "¡Hola! Soy Orienta. ¿Qué materias disfrutas más en el colegio?"
    tts = TTS(cfg.tts_engine, cfg.voice, cfg.tts_device)
    tts.synthesize("Hola.")   # warm-up
    t = time.perf_counter()
    audio = tts.synthesize(text)
    dt, dur = time.perf_counter() - t, len(audio) / tts.sample_rate
    # RTF (real-time factor) = synthesis time / audio length. Below 1 = faster than speaking it.
    print(f"síntesis {dt:.2f}s para {dur:.2f}s de audio (RTF {dt / dur:.2f})")
    # voice_rate < 1 plays the same samples slower: deeper, calmer "character" voice, zero CPU cost.
    player = Player(cfg.output_device, int(tts.sample_rate * cfg.voice_rate))
    player.put(audio)
    while player.busy():
        time.sleep(0.1)
