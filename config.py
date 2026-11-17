"""Loads config.toml and returns the settings for one profile ("pc" or "pi")."""
import platform
import tomllib                      # TOML parser built into Python 3.11+ (no extra dependency)
from pathlib import Path
from types import SimpleNamespace   # lets us write cfg.silence_ms instead of cfg["silence_ms"]


# La Pi elige su perfil sola: sin esto, `main.py` sin argumentos intentaba la GPU CUDA del PC y
# fallaba. Viene del trabajo hecho en la Pi. Se mira el modelo de hardware y no solo la
# arquitectura, porque la DGX tambien es aarch64 y no debe tomarse por una Raspberry.
def _perfil_por_defecto() -> str:
    try:
        modelo = open("/proc/device-tree/model", "rb").read().decode(errors="ignore")
    except OSError:
        modelo = ""
    if "Raspberry Pi" in modelo:
        return "pi"
    return "pc" if platform.machine() != "aarch64" else "dgx"


DEFAULT_PROFILE = _perfil_por_defecto()

ROOT = Path(__file__).parent        # project folder, so paths work from any working directory


def load_config(profile: str) -> SimpleNamespace:
    with open(ROOT / "config.toml", "rb") as f:   # tomllib requires the file opened in binary mode
        data = tomllib.load(f)
    if profile == "common" or profile not in data:
        options = [k for k in data if k != "common"]
        raise SystemExit(f"Perfil desconocido '{profile}'. Opciones: {options}")
    # Profile keys override common ones: {**a, **b} keeps b's value when both have a key.
    return SimpleNamespace(**{**data["common"], **data[profile]}, profile=profile)
