"""Confirms both GPU users (Whisper and Ollama) really run on the RTX 3050, from native Windows.

Run: uv run python check_gpu.py
"""
import json
import time
import urllib.request

import numpy as np

from brain import Brain
from config import load_config

cfg = load_config("pc")

print("1) faster-whisper en GPU...")
try:
    from stt import STT
    stt = STT(cfg.whisper_model, "cuda", "float16", cfg.threads)
    t = time.perf_counter()
    stt.transcribe(np.zeros(2 * 16000, dtype=np.float32))   # 2 s of silence exercises cuBLAS + cuDNN
    print(f"   OK ({time.perf_counter() - t:.2f}s)")
except BaseException as e:   # BaseException also catches the SystemExit raised by STT
    print(f"   FALLÓ: {e}\n   → en [pc] usa whisper_device='cpu' y whisper_compute='int8'")

print("2) Ollama en GPU...")
b = Brain(cfg.ollama_host, cfg.llm_model, cfg.threads, cfg.llm_num_gpu)
b.check()
b.warmup()
with urllib.request.urlopen(cfg.ollama_host + "/api/ps") as r:   # /api/ps = models currently loaded
    for m in json.load(r)["models"]:
        print(f"   {m['name']}: {m['size_vram'] / 1e9:.1f} GB en VRAM de {m['size'] / 1e9:.1f} GB")
