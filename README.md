# Orientador vocacional por voz (offline)

Voice agent that talks with students in Spanish and helps them discover their vocational interests.
Fully offline. Runs on Windows/Ubuntu for development and on a Raspberry Pi 5 inside an animatronic.

Design: `docs/superpowers/specs/2026-09-26-offline-voice-loop-design.md`

## Setup (once, needs internet)

```
uv sync                                  # Python 3.11 + dependencies into .venv/
uv run python download_models.py         # VAD, Whisper and Piper files into models/
ollama pull qwen2.5:3b                   # LLM for the pc profile
ollama pull qwen2.5:1.5b                 # LLM for the pi profile
uv run python fillers.py pc              # generate the filler clips with the Piper voice
```

## Run (offline)

```
uv run python list_devices.py            # check the USB audio module is found
uv run python main.py pc                 # fast profile (GPU)
uv run python main.py pi                 # Pi-like profile (CPU, 4 threads, small models)
uv run python test_logic.py              # logic checks
uv run python bench.py pi                # latency benchmark over recorded turns
```

## Measured latency (PC, 2026-09-26)

Six synthetic Spanish turns (two Piper voices), `uv run python bench.py <profile>`:

| Profile | STT | LLM 1st sentence | TTS 1st sentence | Answer ready | First sound |
|---|---|---|---|---|---|
| pc (Whisper small GPU, qwen2.5:3b GPU) | 0.41 s | 0.23 s | 0.40 s | 1.65 s | ~0.60 s |
| pi (on PC, 4 threads: Whisper base int8, qwen2.5:1.5b CPU) | 1.50 s | 0.81 s | 0.21 s | 3.12 s | ~0.60 s |
| pi with Whisper small int8 (rejected) | 3.64 s | 0.73 s | 0.21 s | 5.18 s | ~0.60 s |

"Answer ready" includes the 0.6 s end-of-turn silence. Expect the real Pi 5 to be ~1.5–2.5x slower than the `pi` row.
Known trade-off: Whisper `base` is fast enough but mishears some short/unclear Spanish ("Sí" → "¡Tienes!"); `small` is accurate but too slow on CPU. To revisit in the Pi port.
