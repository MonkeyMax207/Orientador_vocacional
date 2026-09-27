"""The LLM: vocational-guide persona + conversation memory, streamed from Ollama.

Connection: Ollama is a local server (http://localhost:11434). We POST the whole
conversation to /api/chat with "stream": true, and it answers with NDJSON: one JSON
object per line, each carrying the next few characters of the reply.
Only the Python standard library is used (urllib), with no HTTP dependency.
"""
import json
import urllib.error
import urllib.request

# Everything the agent writes is SPOKEN, hence the rules about format and length.
SYSTEM_PROMPT = """Eres Orienta, una orientadora vocacional amable y cercana de la universidad. \
Conversas en español con estudiantes que están eligiendo qué carrera estudiar.

Todo lo que escribes se convierte en voz, así que sigue estas reglas:
Responde con una a tres frases cortas.
Empieza siempre con una frase muy corta, como "¡Claro!", "Entiendo." o "¡Qué bien!".
No uses listas, viñetas, asteriscos, emojis ni símbolos. Escribe los números con palabras.
Haz una sola pregunta a la vez.

Tu objetivo es conocer a la persona: qué materias disfruta, qué actividades le hacen perder \
la noción del tiempo, en qué es buena, si prefiere trabajar con personas, con datos, con las \
manos o creando, y qué le importa para su futuro. Después de unas cinco o seis preguntas, \
resume lo que aprendiste y sugiere áreas de estudio que encajen con sus intereses. \
Si no conoces los programas específicos de la universidad, dilo con honestidad."""


class Brain:
    def __init__(self, host: str, model: str, threads: int, num_gpu: int, max_turns: int = 10):
        self.url = host.rstrip("/")
        self.model = model
        self.max_turns = max_turns   # remembered exchanges; bounds prompt size (and Pi latency)
        self.history = []            # [{"role": "user"|"assistant", "content": str}, ...]
        # Ollama "options" = llama.cpp runtime settings. Keep them identical in every
        # request: changing num_ctx/num_gpu/num_thread makes Ollama RELOAD the model (slow).
        self.options = {
            "num_thread": threads,   # CPU threads for generation
            "num_gpu": num_gpu,      # layers on the GPU (0 = CPU only, like the Pi)
            "num_ctx": 2048,         # context window in tokens: enough for ~10 exchanges
            "temperature": 0.7,      # some variety, but still coherent
        }

    def _post(self, path: str, body: dict):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=120)

    def check(self) -> None:
        # Fail at startup with a fix-it message, not in the middle of a conversation.
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.load(r)["models"]]
        except urllib.error.URLError:
            raise SystemExit(f"Ollama no responde en {self.url}. Abre la app de Ollama (o ejecuta: ollama serve)")
        if self.model not in names and f"{self.model}:latest" not in names:
            raise SystemExit(f"Falta el modelo '{self.model}'. Ejecuta: ollama pull {self.model}")

    def warmup(self) -> None:
        # Loads the model into RAM/VRAM and processes the system prompt once. Ollama then
        # reuses that work (prompt cache) on every turn, so only the NEW words cost time.
        body = {
            "model": self.model, "stream": False,
            "keep_alive": -1,   # -1 = never unload the model (a kiosk must answer instantly)
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "options": {**self.options, "num_predict": 1},   # generate just 1 token
        }
        with self._post("/api/chat", body) as r:
            r.read()

    def stream_reply(self, user_text: str, stop=None):
        self.history.append({"role": "user", "content": user_text})
        # Keep an odd number of messages so the history always starts with a user message.
        self.history = self.history[-(2 * self.max_turns - 1):]
        body = {
            "model": self.model, "stream": True, "keep_alive": -1, "options": self.options,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
        }
        reply = ""
        try:
            with self._post("/api/chat", body) as resp:
                for line in resp:                       # one JSON object per line (NDJSON)
                    if stop is not None and stop.is_set():
                        break                           # leaving the `with` closes the connection → Ollama stops
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        reply += token
                        yield token                     # hand the piece to the caller right away
                    if chunk.get("done"):
                        break
        finally:
            # Remember what was said, even if interrupted, so the next turn has context.
            self.history.append({"role": "assistant", "content": reply})


def llm_worker(brain, text: str, out_q, stop) -> None:
    """Runs on its own thread: moves tokens from the LLM into a queue for main.py.

    The final None is guaranteed (finally), so the reader never waits forever,
    even if Ollama crashes mid-reply.
    """
    try:
        for token in brain.stream_reply(text, stop):
            out_q.put(token)
    except Exception as e:
        print(f"[llm] error: {e}", flush=True)
    finally:
        out_q.put(None)


if __name__ == "__main__":
    # Text chat to tune the persona without audio. Run: uv run python brain.py [pc|pi]
    import sys
    import time

    from config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    b = Brain(cfg.ollama_host, cfg.llm_model, cfg.threads, cfg.llm_num_gpu)
    b.check()
    b.warmup()
    print("Escribe como si fueras el estudiante (Ctrl+C para salir).")
    while True:
        text = input("\nTú: ")
        t, first = time.perf_counter(), None
        print("Orienta: ", end="")
        for tok in b.stream_reply(text):
            first = first or time.perf_counter() - t
            print(tok, end="", flush=True)
        print(f"\n[primer token {first or 0:.2f}s, total {time.perf_counter() - t:.2f}s]")
