"""Filler phrases: pre-generated clips that play the instant the user stops talking.

They hide the STT + LLM delay (the "Mmm, déjame pensar..." a person would say).
Generate once per voice:  uv run python fillers.py [pc|pi]
Clips go to models/fillers/<voice>/ (ignored by git, rebuilt by this script).
"""
import random
from pathlib import Path

from audio_io import load_wav, save_wav

FILLER_DIR = Path(__file__).parent / "models" / "fillers"

# Normal fillers: 1.5–3 s long, neutral enough to follow any question or answer.
FILLERS = [
    "Mmm... déjame pensarlo un momento.",
    "Entiendo, dame un segundo.",
    "Muy bien... a ver, déjame pensar.",
    "Mmm, qué interesante, un momento.",
    "Vale, déjame organizar las ideas.",
]
# Slow fillers: played if the answer still isn't ready after slow_llm_s.
SLOW_FILLERS = [
    "Sigo pensando, ya casi.",
    "Dame un momentito más.",
]


class Fillers:
    def __init__(self, voice: str):
        d = FILLER_DIR / voice
        self.normal = [load_wav(p)[0] for p in sorted(d.glob("filler_*.wav"))]
        self.slow = [load_wav(p)[0] for p in sorted(d.glob("slow_*.wav"))]
        if not self.normal or not self.slow:
            raise SystemExit(f"Faltan los fillers de '{voice}'. Ejecuta: uv run python fillers.py")
        self._last = None

    def pick(self, slow: bool = False):
        options = self.slow if slow else self.normal
        # Never repeat the previous clip back-to-back (it sounds robotic).
        choices = [a for a in options if a is not self._last] or options
        self._last = random.choice(choices)
        return self._last


if __name__ == "__main__":
    import sys

    from config import load_config
    from tts import TTS

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    tts = TTS(cfg.voice)
    out = FILLER_DIR / cfg.voice
    out.mkdir(parents=True, exist_ok=True)
    for prefix, texts in (("filler", FILLERS), ("slow", SLOW_FILLERS)):
        for i, text in enumerate(texts):
            audio = tts.synthesize(text)
            save_wav(out / f"{prefix}_{i}.wav", audio, tts.sample_rate)
            print(f"{prefix}_{i}.wav  {len(audio) / tts.sample_rate:.1f}s  {text}")
