"""Speech-to-text with faster-whisper (OpenAI's Whisper model on the CTranslate2 engine).

CTranslate2 is a C++ inference engine: int8 on CPU and float16 on GPU, several times
faster than the original PyTorch Whisper and much lighter.
"""
import ctypes
import importlib.util
import os
import re
from pathlib import Path

import numpy as np

WHISPER_DIR = Path(__file__).parent / "models" / "whisper"

# Phrases Whisper invents on silence or noise (it was trained on subtitled YouTube videos).
HALLUCINATIONS = ("amara.org", "subtítulos realizados", "subtítulos por", "suscríbete", "gracias por ver")
# Hesitation sounds, written with repeated letters collapsed ("mmmm" → "m", "ehh" → "eh").
# "ajá" is NOT here: it's a real Spanish "yes".
NOISE_WORDS = {"eh", "em", "m", "hm", "ah", "uh"}


def clean(text: str) -> str:
    t = text.strip()
    low = t.lower()
    if any(h in low for h in HALLUCINATIONS):
        return ""
    if re.fullmatch(r"[\[(].*[\])]", t) or low.strip(".!¡ ") == "música":
        return ""   # sound tags Whisper writes for non-speech: "[Música]", "(risas)"
    words = [re.sub(r"(\w)\1+", r"\1", w) for w in re.findall(r"\w+", low)]   # collapse repeats
    if not words or all(w in NOISE_WORDS for w in words):
        return ""
    return t


def _add_cuda_libs() -> None:
    # pip put the cuBLAS/cuDNN libraries in .venv/.../site-packages/nvidia/*/, where neither
    # Windows nor Linux looks by default. CTranslate2 loads them lazily, by name, on the first
    # GPU transcription, so we make them findable before that happens.
    spec = importlib.util.find_spec("nvidia")
    if spec is None:
        return   # no NVIDIA packages (e.g. the Raspberry Pi): nothing to do
    for root in spec.submodule_search_locations:
        if os.name == "nt":
            # Windows: DLLs live in nvidia/*/bin → add those folders to the DLL search path.
            for bin_dir in Path(root).glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]   # for lazily loaded DLLs
        else:
            # Linux: .so files live in nvidia/*/lib. Changing LD_LIBRARY_PATH from inside the
            # process has no effect, so load each library now; later lookups by name
            # ("libcublas.so.12") then find the already-loaded copy.
            for lib in sorted(Path(root).glob("*/lib/lib*.so*")):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass   # a library we don't need (or whose deps load later); skip it


class STT:
    def __init__(self, model: str, device: str, compute_type: str, threads: int):
        if device == "cuda":
            _add_cuda_libs()
        from faster_whisper import WhisperModel   # imported here so test_logic.py stays light
        try:
            self.model = WhisperModel(
                model, device=device, compute_type=compute_type, cpu_threads=threads,
                download_root=str(WHISPER_DIR),
                local_files_only=True,   # never touch the internet at runtime (offline requirement)
            )
        except Exception as e:
            raise SystemExit(f"No pude cargar Whisper '{model}' ({e}). Ejecuta: uv run python download_models.py")
        try:
            # Warm-up: the first transcription pays one-time setup, and it's also when the CUDA
            # libraries actually load. Doing it here turns a GPU problem into a clear message.
            self.transcribe(np.zeros(16000, dtype=np.float32))
        except Exception as e:
            raise SystemExit(f"Whisper falló en '{device}' ({e}). "
                             "En config.toml usa whisper_device = 'cpu' y whisper_compute = 'int8'")

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
    t = time.perf_counter()
    text = stt.transcribe(audio)
    print(f"{time.perf_counter() - t:.2f}s  crudo: {text!r}  limpio: {clean(text)!r}")
