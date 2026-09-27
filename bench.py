"""Latency benchmark: runs recorded turns through STT → LLM (first sentence) → TTS.

Uses the recordings/turn_*.wav files that main.py saves.
Run: uv run python bench.py [pc|pi]
"""
import sys
import time
from pathlib import Path

import numpy as np

from audio_io import load_wav
from brain import Brain
from config import load_config
from sentences import SentenceSplitter, clean_for_tts
from stt import STT, clean
from tts import TTS

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
files = sorted((Path(__file__).parent / "recordings").glob("turn_*.wav"))[:10]
if not files:
    raise SystemExit("No hay grabaciones. Conversa un poco con main.py primero.")

brain = Brain(cfg.ollama_host, cfg.llm_model, cfg.threads, cfg.llm_num_gpu)
brain.check()
brain.warmup()
stt = STT(cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute, cfg.threads)
# (STT warms itself up inside its constructor)
tts = TTS(cfg.voice)
tts.synthesize("Hola.")

rows = []
for f in files:
    audio = load_wav(f)[0].astype(np.float32) / 32768
    t0 = time.perf_counter()
    text = clean(stt.transcribe(audio))
    t1 = time.perf_counter()
    if not text:
        continue
    brain.history.clear()   # each question is measured on its own, like a first turn
    splitter, first = SentenceSplitter(), ""
    for token in brain.stream_reply(text):
        done = splitter.feed(token)
        if done:
            first = clean_for_tts(done[0])
            break           # we only time the FIRST sentence: that's what the user waits for
    else:                   # (for/else: runs only if the loop finished without break)
        first = clean_for_tts(" ".join(splitter.flush()))
    t2 = time.perf_counter()
    tts.synthesize(first)
    t3 = time.perf_counter()
    rows.append((t1 - t0, t2 - t1, t3 - t2))
    print(f"stt {t1 - t0:.2f}s | llm 1ª frase {t2 - t1:.2f}s | tts {t3 - t2:.2f}s | {text[:50]!r}")

stt_avg, llm_avg, tts_avg = np.mean(rows, axis=0)
ready = cfg.silence_ms / 1000 + stt_avg + llm_avg + tts_avg
print(f"\nPerfil {cfg.profile} ({len(rows)} turnos): stt {stt_avg:.2f}s | llm {llm_avg:.2f}s | "
      f"tts {tts_avg:.2f}s | respuesta lista ≈ {ready:.2f}s tras dejar de hablar")
