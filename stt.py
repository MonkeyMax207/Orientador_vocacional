"""Speech-to-text with faster-whisper (OpenAI's Whisper model on the CTranslate2 engine).

CTranslate2 is a C++ inference engine: int8 on CPU and float16 on GPU, several times
faster than the original PyTorch Whisper and much lighter.
"""
import importlib.util
import os
import re
from pathlib import Path

WHISPER_DIR = Path(__file__).parent / "models" / "whisper"

# Phrases Whisper invents on silence or noise (it was trained on subtitled YouTube videos).
HALLUCINATIONS = ("amara.org", "subtítulos", "suscríbete", "gracias por ver")
NOISE_WORDS = {"eh", "em", "mm", "mmm", "hmm", "ah", "uh", "ajá"}


def clean(text: str) -> str:
    t = text.strip()
    low = t.lower()
    if any(h in low for h in HALLUCINATIONS):
        return ""
    words = re.findall(r"\w+", low)
    if not words or all(w in NOISE_WORDS for w in words):
        return ""
    return t


def _add_cuda_dlls() -> None:
    # Windows only: pip put cuBLAS/cuDNN DLLs in .venv/Lib/site-packages/nvidia/*/bin,
    # but Windows doesn't search there. Add those folders to the DLL search path.
    # On Linux the libraries are found through their own mechanism, so do nothing.
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None:
        return
    for root in spec.submodule_search_locations:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]   # for lazily loaded DLLs


class STT:
    def __init__(self, model: str, device: str, compute_type: str, threads: int):
        if device == "cuda":
            _add_cuda_dlls()
        from faster_whisper import WhisperModel   # imported here so test_logic.py stays light
        try:
            self.model = WhisperModel(
                model, device=device, compute_type=compute_type, cpu_threads=threads,
                download_root=str(WHISPER_DIR),
                local_files_only=True,   # never touch the internet at runtime (offline requirement)
            )
        except Exception as e:
            raise SystemExit(f"No pude cargar Whisper '{model}' ({e}). Ejecuta: uv run python download_models.py")

    def transcribe(self, audio) -> str:
        segments, _info = self.model.transcribe(
            audio,                              # float32 numpy array, 16 kHz mono
            language="es",                      # fixed: skips language detection (faster, no mistakes)
            beam_size=1,                        # greedy decoding: fastest, good enough for short turns
            condition_on_previous_text=False,   # each turn is independent; avoids repetition loops
            without_timestamps=True,            # we don't need word timings
        )
        # `segments` is a generator: the actual decoding happens while we iterate it.
        return " ".join(s.text.strip() for s in segments).strip()


if __name__ == "__main__":
    # Run: uv run python stt.py [pc|pi] [file.wav]   (defaults to the recording from audio_io.py)
    import sys
    import time

    import numpy as np

    from audio_io import load_wav
    from config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    path = sys.argv[2] if len(sys.argv) > 2 else "recordings/mic_test.wav"
    stt = STT(cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute, cfg.threads)
    audio = load_wav(path)[0].astype(np.float32) / 32768   # int16 → float32 in [-1, 1]
    stt.transcribe(audio[:16000])   # warm-up: the first call pays one-time setup (CUDA kernels, memory)
    t = time.perf_counter()
    text = stt.transcribe(audio)
    print(f"{time.perf_counter() - t:.2f}s  crudo: {text!r}  limpio: {clean(text)!r}")
