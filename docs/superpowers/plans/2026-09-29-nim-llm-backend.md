# LLM backend on the DGX via NIM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the `dgx` profile serve chat through the `nvidia-nemotron-nano-9b-v2-dgx-spark` NIM instead of Ollama, with a one-line config revert if it doesn't measure up.

**Architecture:** `Brain` (in `brain.py`) already separates persona/RAG logic (system prompt, history, `buscar_uao` tool) from HTTP transport. Extract three transport-specific hook points (`CHAT_PATH`, `_request_extra()`, `_warmup_extra()`) with no behavior change for the existing Ollama path, then add `NimBrain(Brain)` overriding transport only: OpenAI-style `/v1/chat/completions` SSE streaming instead of Ollama's `/api/chat` NDJSON, and `/v1/models` for the health check instead of `/api/tags`. A `make_brain(cfg, ...)` factory picks the class from `cfg.llm_backend`.

**Tech Stack:** Python 3.11 stdlib only (`urllib`, `json`) — matches the existing project constraint (see `brain.py`'s module docstring: "Only the Python standard library is used, with no HTTP dependency"). No new pip dependencies.

**Spec:** `docs/superpowers/specs/2026-09-29-nim-llm-backend-design.md`

## Global Constraints

- stdlib-only HTTP (`urllib`), no `requests`/`httpx`/`sseclient` — matches the rest of the project.
- RAG embeddings stay on Ollama (`OllamaEmbedder`, `rag.py:148,166`) — this plan never touches `rag.py`.
- Existing `Brain`/Ollama behavior and its tests (`test_logic.py:518-585`) must keep passing unchanged — no signature or body-shape change for the Ollama path.
- `[pc]`/`[pi]` profiles keep working exactly as today; only `[dgx]` opts into the new backend.
- Reversible by config alone: `llm_backend = "ollama"` in `config.toml` must restore today's behavior with no code change.

## Review Focus

- NIM unreachable at `nim_host` (container not started yet) → a clear Spanish `SystemExit` message, not a raw `URLError` traceback (mirrors `Brain.check()`'s existing Ollama message). Covered in Task 1.
- Configured `llm_model` not in the NIM's `/v1/models` response (typo, or container still loading) → a message listing the models that *are* available, not a silent wrong-model request. Covered in Task 1.
- A tool call's `arguments` arrive split across several small SSE chunks (OpenAI streaming sends function-call arguments token-by-token) → must reassemble into valid JSON before `stream_reply` calls `.get("consulta", ...)` on it. A test with only one fragment wouldn't catch a concatenation-order bug. Covered in Task 1 (3+ fragments).
- A turn whose stream has no content and no tool call (e.g., only a `finish_reason` chunk, then `[DONE]`) must yield an empty string, not raise. Covered in Task 1.
- Switching `llm_backend` between `"ollama"` and `"nim"` must be a pure config change — verified by exercising `make_brain` with both values pointing at the same fake config object. Covered in Task 2.

---

### Task 1: `NimBrain` transport, unit-tested with fakes (no real NIM needed)

**Files:**
- Modify: `brain.py` (add `CHAT_PATH`, `_request_extra()`, `_warmup_extra()` hooks to `Brain`; add `NimBrain` class)
- Test: `test_logic.py` (append near the existing `Brain` tests, `test_logic.py:518-585`)

**Interfaces:**
- Produces: `brain.NimBrain(host: str, model: str, max_turns: int = 10, program_names=(), search=None, gate=None)` — same public surface as `Brain` (`.check()`, `.warmup()`, `.stream_reply(user_text, stop=None, cards=())`, `.history`), so callers don't need to know which class they hold.
- Produces on `Brain`: `Brain.CHAT_PATH` (class attr, default `"/api/chat"`), `Brain._request_extra(self) -> dict`, `Brain._warmup_extra(self) -> dict`. `NimBrain` overrides all three plus `check()` and `_stream()`.

- [ ] **Step 1: Refactor `Brain` to expose the three hook points, with identical resulting behavior**

  In `brain.py`:
  - Add class attribute `CHAT_PATH = "/api/chat"` on `Brain`.
  - In `_stream`, change `self._post("/api/chat", body)` to `self._post(self.CHAT_PATH, body)`.
  - In `stream_reply`, replace the body dict's `"keep_alive": -1, "options": self.options` with `**self._request_extra()`, and add:
    ```python
    def _request_extra(self) -> dict:
        return {"keep_alive": -1, "options": self.options}
    ```
  - In `warmup`, replace the body dict's `"keep_alive": -1, ... "options": {**self.options, "num_predict": 1}` with `**self._warmup_extra()`, change `self._post("/api/chat", body)` to `self._post(self.CHAT_PATH, body)`, and add:
    ```python
    def _warmup_extra(self) -> dict:
        return {"keep_alive": -1, "options": {**self.options, "num_predict": 1}}
    ```

- [ ] **Step 2: Run the existing Brain tests to confirm no behavior changed**

  Run: `uv run python test_logic.py`
  Expected: `35 checks passed` (same count as before this task — the refactor must be invisible to these tests).

- [ ] **Step 3: Write the failing tests for `NimBrain`**

  Append to `test_logic.py`:
  ```python
  def _sse(lines):
      # One SSE "data: ..." line per list element; mirrors _FakeResp but for NimBrain's format.
      return _FakeResp([f"data: {json.dumps(x)}".encode() if x != "[DONE]" else b"data: [DONE]" for x in lines])

  def test_nimbrain_streams_content_and_skips_ollama_fields():
      b = brain.NimBrain("http://x", "m")
      sent = []
      b._post = lambda path, body: sent.append((path, body)) or _sse(
          [{"choices": [{"delta": {"content": "Hola"}}]},
           {"choices": [{"delta": {"content": "."}}]},
           "[DONE]"])
      assert "".join(b.stream_reply("Hola")) == "Hola."
      path, body = sent[0]
      assert path == "/v1/chat/completions"
      assert "options" not in body and "keep_alive" not in body

  def test_nimbrain_assembles_tool_call_split_across_chunks():
      searched = []
      b = brain.NimBrain("http://x", "m", search=lambda q: searched.append(q) or "Fab-Lab.")
      replies = iter([
          _sse([{"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"name": "buscar_uao", "arguments": ""}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "{\"consulta\""}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": ": \"laboratorios\"}"}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                "[DONE]"]),
          _sse([{"choices": [{"delta": {"content": "Tiene el Fab-Lab."}}]}, "[DONE]"]),
      ])
      b._post = lambda path, body: next(replies)
      assert "".join(b.stream_reply("¿Qué labs hay?")) == "Tiene el Fab-Lab."
      assert searched == ["laboratorios"]

  def test_nimbrain_empty_stream_yields_empty_reply():
      b = brain.NimBrain("http://x", "m")
      b._post = lambda path, body: _sse([{"choices": [{"delta": {}, "finish_reason": "stop"}]}, "[DONE]"])
      assert "".join(b.stream_reply("Hola")) == ""
      assert b.history[-1] == {"role": "assistant", "content": ""}

  def test_nimbrain_check_reports_missing_model():
      b = brain.NimBrain("http://x", "otro-modelo")
      import io
      import unittest.mock as mock
      resp = io.BytesIO(json.dumps({"data": [{"id": "nvidia/nemotron-nano-9b-v2"}]}).encode())
      with mock.patch("urllib.request.urlopen", return_value=_UrlopenCtx(resp)):
          try:
              b.check()
              assert False, "expected SystemExit"
          except SystemExit as e:
              assert "otro-modelo" in str(e) and "nemotron-nano-9b-v2" in str(e)
  ```
  Add the small `_UrlopenCtx` helper next to `_FakeResp` (a context manager wrapping a `BytesIO` so `json.load(r)` works inside `with urllib.request.urlopen(...) as r:`):
  ```python
  class _UrlopenCtx:
      def __init__(self, body): self.body = body
      def __enter__(self): return self.body
      def __exit__(self, *exc): return False
  ```

- [ ] **Step 4: Run to verify the new tests fail**

  Run: `uv run python test_logic.py`
  Expected: `AttributeError: module 'brain' has no attribute 'NimBrain'` (or similar — the class doesn't exist yet).

- [ ] **Step 5: Implement `NimBrain` in `brain.py`**

  ```python
  class NimBrain(Brain):
      """Same persona/RAG logic as Brain (inherited stream_reply); only the transport differs:
      an OpenAI-compatible NIM instead of Ollama."""
      CHAT_PATH = "/v1/chat/completions"

      def __init__(self, host, model, max_turns=10, program_names=(), search=None, gate=None):
          super().__init__(host, model, threads=0, num_gpu=0, max_turns=max_turns,
                            program_names=program_names, search=search, gate=gate)

      def _request_extra(self) -> dict:
          return {}   # the NIM's own engine config decides sampling; nothing to override per request

      def _warmup_extra(self) -> dict:
          return {"max_tokens": 1}

      def check(self) -> None:
          try:
              with urllib.request.urlopen(self.url + "/v1/models", timeout=3) as r:
                  ids = [m["id"] for m in json.load(r)["data"]]
          except urllib.error.URLError:
              raise SystemExit(f"El NIM no responde en {self.url}. ¿Está corriendo el contenedor?")
          if self.model not in ids:
              raise SystemExit(f"El NIM no sirve el modelo '{self.model}'. Disponibles: {ids}")

      def _stream(self, body: dict, stop, calls: list):
          pending = {}   # tool-call index -> accumulating {"name": str, "arguments": str}
          with self._post(self.CHAT_PATH, body) as resp:
              for raw in resp:
                  if stop is not None and stop.is_set():
                      break
                  line = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip()
                  if not line.startswith("data:"):
                      continue
                  data = line[len("data:"):].strip()
                  if data == "[DONE]":
                      break
                  delta = json.loads(data)["choices"][0].get("delta", {})
                  for tc in delta.get("tool_calls") or []:
                      slot = pending.setdefault(tc["index"], {"name": "", "arguments": ""})
                      fn = tc.get("function", {})
                      slot["name"] += fn.get("name") or ""
                      slot["arguments"] += fn.get("arguments") or ""
                  if delta.get("content"):
                      yield delta["content"]
          for slot in pending.values():
              calls.append({"function": {"name": slot["name"], "arguments": json.loads(slot["arguments"])}})
  ```
  Note: `warmup()`'s request body still includes `"stream": False` (inherited unchanged from `Brain.warmup`), so `NimBrain` never needs `_stream` during warmup — only `check()` and `stream_reply()`'s streaming path use the new pieces.

- [ ] **Step 6: Run to verify all tests pass**

  Run: `uv run python test_logic.py`
  Expected: `40 checks passed` (35 existing + 5 new).

- [ ] **Step 7: Commit**

  ```bash
  git add brain.py test_logic.py
  git commit -m "feat: add NimBrain, an OpenAI-compatible NIM transport for Brain"
  ```

---

### Task 2: Config-driven backend selection (`make_brain`)

**Files:**
- Modify: `brain.py` (add `make_brain` factory)
- Modify: `config.toml` (`[common]` default + `[dgx]` override)
- Modify: `server.py:22,39`, `main.py:22,48`, `knowledge/replay.py:36` (use the factory)
- Test: `test_logic.py`

**Interfaces:**
- Consumes: `brain.NimBrain`, `brain.Brain` (Task 1).
- Produces: `brain.make_brain(cfg, program_names=()) -> Brain` (returns a `NimBrain` when `cfg.llm_backend == "nim"`, else a `Brain`). Every later caller uses this instead of constructing `Brain`/`NimBrain` directly.

- [ ] **Step 1: Write the failing test**

  Append to `test_logic.py`:
  ```python
  def test_make_brain_selects_backend_from_config():
      from types import SimpleNamespace
      ollama_cfg = SimpleNamespace(llm_backend="ollama", ollama_host="http://x", llm_model="m",
                                    threads=4, llm_num_gpu=0)
      nim_cfg = SimpleNamespace(llm_backend="nim", nim_host="http://y", llm_model="m")
      assert type(brain.make_brain(ollama_cfg)) is brain.Brain
      b = brain.make_brain(nim_cfg)
      assert type(b) is brain.NimBrain and b.url == "http://y"
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run python test_logic.py`
  Expected: `AttributeError: module 'brain' has no attribute 'make_brain'`

- [ ] **Step 3: Implement `make_brain(cfg, program_names=()) -> Brain` in `brain.py`**

  ```python
  def make_brain(cfg, program_names=()):
      if getattr(cfg, "llm_backend", "ollama") == "nim":
          return NimBrain(cfg.nim_host, cfg.llm_model, program_names=program_names)
      return Brain(cfg.ollama_host, cfg.llm_model, cfg.threads, cfg.llm_num_gpu, program_names=program_names)
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run python test_logic.py`
  Expected: `41 checks passed`

- [ ] **Step 5: Add the config keys**

  In `config.toml`, under `[common]` add (near `ollama_host`):
  ```toml
  llm_backend = "ollama"       # "ollama" or "nim"; only [dgx] overrides this today
  ```
  Under `[dgx]` add, and update the comment on `llm_model`:
  ```toml
  llm_backend = "nim"
  nim_host = "http://127.0.0.1:8001"   # NIM container; see docs/GUIA-DE-PRUEBAS.md §10
  llm_model = "qwen2.5:32b"    # TODO after first docker run: replace with the id GET /v1/models
                                # on nim_host returns for nvidia-nemotron-nano-9b-v2-dgx-spark
  ```

- [ ] **Step 6: Switch the three call sites to the factory**

  - `brain.py` already defines `make_brain`; no self-reference needed.
  - `server.py:22`: `from brain import Brain` → `from brain import make_brain`. `server.py:39`: `self.brain = Brain(cfg.ollama_host, cfg.llm_model, cfg.threads, cfg.llm_num_gpu, program_names=names)` → `self.brain = make_brain(cfg, program_names=names)`.
  - `main.py:22`: `from brain import Brain, llm_worker` → `from brain import llm_worker, make_brain`. `main.py:48`: same replacement as above.
  - `knowledge/replay.py:11`: `from brain import Brain` → `from brain import make_brain`. `knowledge/replay.py:36`: `brain = Brain(cfg.ollama_host, cfg.llm_model, cfg.threads, cfg.llm_num_gpu, program_names=kb.program_names())` → `brain = make_brain(cfg, program_names=kb.program_names())`.

- [ ] **Step 7: Run the full test suite and a config sanity check**

  Run: `uv run python test_logic.py`
  Expected: `41 checks passed`
  Run: `uv run python -c "from config import load_config; c=load_config('dgx'); print(c.llm_backend, c.nim_host); c2=load_config('pc'); print(c2.llm_backend)"`
  Expected: `nim http://127.0.0.1:8001` then `ollama`

- [ ] **Step 8: Commit**

  ```bash
  git add brain.py config.toml server.py main.py knowledge/replay.py test_logic.py
  git commit -m "feat: select LLM backend (Ollama/NIM) from config via make_brain"
  ```

---

### Task 3: Run the NIM on the DGX and measure against the Ollama baseline

This task is operational (needs an NGC account and a running container), not code — it closes the
spec's "Prerrequisitos" and "Medición" sections. Do it in this session if NGC access is available;
otherwise leave it as the next session's starting point (`docs/GUIA-DE-PRUEBAS.md` §10 will carry it).

**Files:**
- Modify: `docs/GUIA-DE-PRUEBAS.md` (§10, replace the "próximos pasos... por evaluar" LLM bullet with concrete steps)
- Modify: `README.md` (append a measured row once numbers exist, same table style as the existing latency table)
- Modify: `config.toml` (`[dgx].llm_model`, once Step 2 below reveals the real id)

- [ ] **Step 1: One-time NGC access**

  - Create a free account at ngc.nvidia.com and generate an API key.
  - `docker login nvcr.io` (username `$oauthtoken`, password = the API key).

- [ ] **Step 2: Start the NIM and discover its model id**

  Run: `docker run --rm --gpus all -p 8001:8000 nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant`
  Wait for its ready log line, then in another shell:
  Run: `curl -s http://127.0.0.1:8001/v1/models | python3 -m json.tool`
  Expected: JSON with a `data` array containing one object whose `id` is the model name to put in `config.toml`.

- [ ] **Step 3: Point config at the real model id and verify `check()`/`warmup()`**

  Update `[dgx].llm_model` in `config.toml` to the id from Step 2.
  Run: `uv run python brain.py dgx`
  Expected: no `SystemExit`; a `Tú:` prompt appears. Type a line, confirm a Spanish reply streams back and `[primer token ...s]` prints.

- [ ] **Step 4: Measure against the baseline**

  Run: `uv run python -m knowledge.replay dgx`
  Compare against the existing `qwen2.5:32b`/Ollama numbers in `README.md`'s "UAO knowledge (RAG)" table: check program names are all real, the right cards get cited, and note the time-to-first-sentence per turn.

- [ ] **Step 5: Record the result and update the guide**

  - Add a row to the `README.md` measurement table (same columns as the existing one) with the NIM numbers, following the file's existing pattern of noting *why* a result was kept or rejected.
  - In `docs/GUIA-DE-PRUEBAS.md` §10, replace the LLM bullet under "Próximos pasos" with the concrete commands from Steps 1-3 above (account, `docker run`, `curl /v1/models`), so this doesn't need to be rediscovered.
  - If the NIM measures worse than Ollama on quality (hallucinated programs, wrong cards) or is impractically slow, set `[dgx].llm_backend = "ollama"` back and note the rejection reason in `README.md`, same as the "Rejected: Whisper `initial_prompt`" row already there.

- [ ] **Step 6: Commit**

  ```bash
  git add config.toml README.md docs/GUIA-DE-PRUEBAS.md
  git commit -m "docs: measure NIM LLM backend against the Ollama baseline on the DGX"
  ```

---

## Self-Review Notes

- **Spec coverage:** Architecture (backend split) → Task 1. `config.toml` keys + wiring → Task 2. Prerequisites + Medición + rollback note → Task 3. Out-of-scope items (STT/TTS, `pc`/`pi` profiles) are untouched by every task, as required.
- **Type/signature consistency:** `make_brain` (Task 2) matches the exact `NimBrain`/`Brain` constructor signatures fixed in Task 1. `CHAT_PATH`/`_request_extra`/`_warmup_extra` names are used identically in both tasks.
- **Review Focus:** all five items map to a concrete test in Task 1 or Task 2 (listed above).
- **Proportion:** Tasks 1-2 are fully test-driven and independent of the container being up; Task 3 is intentionally light on code (it's the measurement/rollback step the spec calls for) and heavy on exact commands, so a future session can pick it up without re-deriving them.
