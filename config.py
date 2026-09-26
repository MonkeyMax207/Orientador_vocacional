"""Loads config.toml and returns the settings for one profile ("pc" or "pi")."""
import tomllib                      # TOML parser built into Python 3.11+ (no extra dependency)
from pathlib import Path
from types import SimpleNamespace   # lets us write cfg.silence_ms instead of cfg["silence_ms"]

ROOT = Path(__file__).parent        # project folder, so paths work from any working directory


def load_config(profile: str) -> SimpleNamespace:
    with open(ROOT / "config.toml", "rb") as f:   # tomllib requires the file opened in binary mode
        data = tomllib.load(f)
    if profile == "common" or profile not in data:
        options = [k for k in data if k != "common"]
        raise SystemExit(f"Perfil desconocido '{profile}'. Opciones: {options}")
    # Profile keys override common ones: {**a, **b} keeps b's value when both have a key.
    return SimpleNamespace(**{**data["common"], **data[profile]}, profile=profile)
