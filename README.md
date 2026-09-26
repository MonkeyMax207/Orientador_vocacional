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
