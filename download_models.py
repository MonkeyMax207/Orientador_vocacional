"""One-time download of every model file into models/. The ONLY step that needs internet.

Run: uv run python download_models.py [extra_voice ...]
(Extra Piper voice names let you download alternatives to compare by ear.)
The LLMs are managed by Ollama instead: `ollama pull <model>` (printed at the end).
"""
import sys
import urllib.request
from pathlib import Path

from config import load_config

MODELS = Path(__file__).parent / "models"
# Silero VAD, pinned to release v5.1.2 so the model's inputs never change under us.
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def piper_urls(voice: str) -> tuple[str, str]:
    # Voice names look like "es_MX-ald-medium" = <lang>_<REGION>-<speaker>-<quality>.
    # On Hugging Face they live at es/es_MX/ald/medium/es_MX-ald-medium.onnx (+ .onnx.json config).
    lang_region, name, quality = voice.split("-")
    lang = lang_region.split("_")[0]
    base = f"{PIPER_BASE}/{lang}/{lang_region}/{name}/{quality}/{voice}"
    return base + ".onnx", base + ".onnx.json"


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"ya existe  {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"descargando {url}")
    tmp = dest.with_name(dest.name + ".part")   # a cut download never looks like a complete file
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


if __name__ == "__main__":
    from faster_whisper.utils import download_model   # imported here: only needed when downloading

    fetch(SILERO_URL, MODELS / "silero_vad.onnx")
    voices = {load_config(p).voice for p in ("pc", "pi")} | set(sys.argv[1:])
    for voice in voices:
        onnx_url, json_url = piper_urls(voice)
        fetch(onnx_url, MODELS / "piper" / f"{voice}.onnx")
        fetch(json_url, MODELS / "piper" / f"{voice}.onnx.json")
    for profile in ("pc", "pi"):
        cfg = load_config(profile)
        # Same cache_dir that stt.py passes as download_root, so STT finds it offline.
        download_model(cfg.whisper_model, cache_dir=str(MODELS / "whisper"))
        print(f"whisper '{cfg.whisper_model}' listo. LLM: ejecuta  ollama pull {cfg.llm_model}")
