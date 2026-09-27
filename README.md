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

## After the first live test (2026-09-26)

Feedback: fillers sounded pointless (the answer was already ready), the voice sounded robotic, and the LLM repeated questions and ignored questions asked to it. Changes, each measured first:

| Change | Why (measured) |
|---|---|
| New prompt: answer the student first, react to what they said, never repeat a question, cope with STT errors | Replaying the real conversation: same `qwen2.5:3b` stopped repeating questions |
| pc LLM `llama3.2:3b` (was `qwen2.5:3b`) | Best of 4 on the replay: answered questions about itself, read "Pugar fútbol" as "jugar fútbol"; 0.23 s to first sentence |
| pi LLM stays `qwen2.5:1.5b` with the new prompt | `gemma3:1b` worse; `llama3.2:3b` on 4 CPU threads 2.9 s/sentence (~6 s on a Pi) |
| pc voice Kokoro `em_santa` on the GPU (onnxruntime-gpu 1.22 + cuDNN 9.10) | More natural; 0.3–0.5 s per sentence on GPU vs 3–4 s on CPU |
| pi voice Piper `es_ES-davefx-medium` | Kokoro is slower than real time on a Pi 5 CPU (RTF ~1.3–1.6 est.); Piper ~0.25 |
| Short fillers ("Mmm...", "A ver...") only if no answer after `filler_after_s` (1.0 s) | A quick answer no longer waits behind "dame un segundo" |
| `voice_rate` knob: plays audio slower = deeper "turtle" voice | Zero CPU cost; tune by ear (e.g. 0.9) |
| Rejected: Whisper `initial_prompt` vocabulary hint | Made `small` recite the hint over noise and turn "Sí" into "Sin" |

pc end to end with everything on the 4 GB GPU (3.8 GB used): first sound 1.0 s (filler), answer ready 1.4–1.8 s.

## UAO knowledge (RAG)

The agent answers from `knowledge/cards.json`: 147 short fact cards about the Universidad Autónoma de
Occidente's 30 undergraduate programs and campus life, generated from the public website and reviewed
(cards flagged `revisar` are never used). Prices, tuition and scholarships are never answered; the agent
hands those over to admissions staff.

```
uv run python -m knowledge.scrape        # download pages (internet; ~3 min, polite 2 s delay)
uv run python -m knowledge.make_cards    # write cards with the PC LLM; review the "revisar" ones
uv run python -m knowledge.eval embeddinggemma   # retrieval quality (21/25 at threshold 0.30)
uv run python -m knowledge.replay pc     # end-to-end check without audio
```

To update: delete `knowledge/raw/`, rerun scrape and make_cards, review `git diff knowledge/cards.json`, commit.
Before `replay`/`main`, unload models from other profiles (`ollama stop <model>`): Ollama keeps at most 3
loaded, and evicting the chat model to load the embedder costs ~5 s per turn.

Measured with `knowledge/replay.py` (PC, 2026-09-27), time to first sentence:

| Profile | Turns without cards | Turns with cards | Notes |
|---|---|---|---|
| pc (llama3.2:3b GPU, 2 cards) | 0.2–0.4 s | 0.3–1.4 s | Every program named exists; labs/courses/sports from the right cards |
| pi (qwen2.5:1.5b CPU, 1 card) | 0.6 s | 2.6–6 s (16–17 s late in the chat) | Right cards, but the 1.5B model drifts (fixates on unrelated programs, embellishes cards) |
