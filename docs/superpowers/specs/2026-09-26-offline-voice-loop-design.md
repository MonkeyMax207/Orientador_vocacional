# Offline Spanish Voice Loop — Design (Sub-project 1 of 4)

## Goal

A fully offline, Spanish-speaking voice agent that acts as a vocational guide.
It runs first on the Windows PC and then, unchanged, on a Raspberry Pi 5 (8 GB)
inside an animatronic.

**Success criteria**
- **Time to first sound (TFS) < 2 s.** TFS is measured from the moment the user stops speaking to the first audio from the speaker.
- On the `pi` profile, the real answer starts within about 3–4 s, and filler audio covers the gap.
- No internet connection is used at runtime.
- The same code runs on Windows 11, Ubuntu 22.04 and Raspberry Pi OS; only the config profile changes.

**Project roadmap** (each sub-project gets its own spec)
1. **Voice loop on the PC** ← this spec
2. RAG over the university's careers data
3. Port to the Raspberry Pi 5
4. Animatronic control. In this spec, keyword → animation events are only printed.

## Decisions (agreed)

| Topic | Decision | Why |
|---|---|---|
| Connectivity | Fully offline | Requirement |
| Language | Spanish only | Lets us pick models that are strong in Spanish |
| Latency hiding | Pre-recorded filler phrases + sentence-by-sentence streaming | TFS is met by the filler; the LLM delay is hidden |
| Animations | Keyword → animation lookup table, no inference | Zero CPU cost, predictable |
| Turn-taking | Always listening with VAD; `barge_in` config flag | Muting the mic while the agent speaks is the reliable default; interrupting can be enabled |
| Architecture | Plain Python + threads and `queue.Queue`, calling each library directly | Every library blocks anyway; threads are simpler to follow than asyncio; minimal overhead on the Pi |
| LLM runtime | Ollama (local server) | Native GPU on Windows, one-click install on all 3 platforms, easy CPU/GPU switch |
| Python | 3.11 in a `uv` virtual environment | Matches Raspberry Pi OS Bookworm; speech libraries support it |
| GPU access | Native Windows, no WSL | Ollama and faster-whisper both support CUDA on Windows. WSL is only the last fallback. |
| Profiles | `pc` (GPU, larger models) and `pi` (CPU, 4 threads, small models) in one config | Tune quality and measure Pi-like latency with the same code |
| Audio hardware | Waveshare USB audio module ("USB PnP Audio Device"), used on the PC too | Same mic and speaker as the robot; standard USB audio, no driver needed |

## Components

| File | What it does | Depends on |
|---|---|---|
| `config.toml` + `config.py` | Profiles `pc` and `pi`: model names, device (cuda/cpu), threads, `barge_in`, audio device names, `silence_ms`, `slow_llm_s` | — |
| `audio_io.py` | Mic stream at 16 kHz mono in 32 ms (512-sample) chunks; speaker playback queue | `sounddevice` |
| `vad.py` | `is_speech(chunk)`; detects the end of a turn after `silence_ms` (default 600) | Silero VAD ONNX model via `onnxruntime` (no PyTorch) |
| `stt.py` | `transcribe(audio) -> str`, with language fixed to `es` | `faster-whisper` (`pc`: cuda float16; `pi`: cpu int8) |
| `brain.py` | Vocational-guide persona (system prompt) + conversation history; `stream_reply(text)` yields tokens | Ollama HTTP API (`/api/chat`, `stream: true`, `keep_alive`) |
| `tts.py` | `synthesize(sentence) -> audio` | `piper-tts` with a Spanish voice |
| `sentences.py` | Splits the token stream into sentences, without breaking on "Dr.", "3.5"; `clean_for_tts()` strips markdown and emojis | — |
| `animations.py` | `KEYWORDS = {"hola": "saludar", ...}`; `check(sentence)` prints `[ANIM] name` | — |
| `fillers.py` (generates clips into `models/fillers/<voice>/`) | 1.5–3 s filler phrases, generated once with the same Piper voice; `Fillers.pick()` | `tts.py` |
| `main.py` | State machine LISTENING → THINKING → SPEAKING; logs per-stage timings; saves every user turn to `recordings/` | everything above |
| `download_models.py` | One-time download of the Silero, Piper and Whisper files into `models/` (the only step that needs internet) | `urllib`, `faster-whisper` |
| `check_gpu.py` | Confirms that Ollama and faster-whisper can see the GPU from Windows | — |
| `list_devices.py` | Prints the audio devices so their names can go in `config.toml` | `sounddevice` |
| `bench.py` | Runs recorded Spanish questions through STT → LLM → TTS per profile and prints stage latencies | everything above |

Starting model candidates (`bench.py` picks the final ones):
- **Whisper:** `pc` = `small`; `pi` = `base` or `small` int8.
- **LLM:** `pc` = ~3B (e.g. Qwen2.5-3B, fully on the 4 GB GPU); `pi` = 1–1.5B, 4-bit quantized.
- **Piper:** a Spanish `medium` voice (e.g. `es_MX-*` or `es_ES-*`), chosen by ear.

## Data flow (one turn)

```
mic ─► VAD ─(end of turn)─► STT ─► text ─► LLM (streaming)
                  │                           │ tokens
                  ▼                           ▼
          play filler clip          sentence splitter ─► keyword check ─► [ANIM]
          + [ANIM] pensar                     │
                                              ▼
                                   Piper ─► speaker queue ─► speaker
```

1. **LISTENING:** VAD consumes the mic chunks. When speech starts, the chunks are buffered. After `silence_ms` of silence, the turn ends.
2. **THINKING:** a random filler plays immediately and `[ANIM] pensar` fires. STT runs, the text goes to the LLM, and the tokens stream in.
3. Each completed sentence is checked for keywords, synthesized by Piper and queued for playback. It starts playing as soon as the filler ends.
4. **SPEAKING:** when the queue drains, return to LISTENING.
   - `barge_in: false`: mic input is discarded while the agent is speaking.
   - `barge_in: true`: speech detected while the agent is speaking stops playback and the LLM stream, and the agent listens.

## Latency budget (estimates; `main.py` logs the real numbers)

| Stage | `pc` | `pi` |
|---|---|---|
| End-of-turn silence | 0.6 s | 0.6 s |
| **Filler starts (TFS)** | **~0.65 s** | **~0.65 s** |
| STT | 0.2 s | 0.6–1.0 s |
| LLM first sentence | 0.3 s | 1.0–1.5 s |
| Piper, first sentence | 0.1 s | 0.5–0.7 s |
| **Real answer starts** | **~1.3 s** | **~3–4 s** |

Ways to keep the gap covered on the Pi:
- filler phrases of 1.5–3 s;
- `keep_alive` so the model stays loaded, and a short system prompt that Ollama keeps cached;
- the persona prompt asks for a short opening sentence.

If that's still not enough, these can be tuned in config: Whisper size, Piper voice, `silence_ms`.

## Error handling

- **Ollama isn't running or the model is missing:** fail at startup with the exact `ollama pull` command to run.
- **The audio device isn't found:** fail at startup and print the available devices.
- **STT returns empty text, or a single filler word:** ignore it and go back to LISTENING.
- **No first sentence from the LLM within `slow_llm_s`:** play a second filler.

## Testing

- `vad.py`, `stt.py`, `tts.py` and `brain.py` each have a `__main__` self-check (e.g. `python stt.py sample.wav` prints the text and the time taken).
- `test_logic.py` holds plain `assert` checks (no framework) for the sentence splitter, TTS text cleaning, keyword matching, turn detection, STT noise filtering and config profiles. Run: `uv run python test_logic.py`.
- `bench.py` validates the latency budget on both profiles.
- Audio is recorded through the Waveshare module before any filter is added. Filters are added only if the recordings show a need. The ASUS "AI Noise-cancelling" virtual devices are not used.

## Out of scope (later sub-projects)

- RAG / university programs (sub-project 2)
- Pi installation and tuning (sub-project 3)
- Servo control, lip-sync (sub-project 4)
- Hardware or software echo cancellation (only if barge-in on the Pi turns out to be needed)
- Wake word
