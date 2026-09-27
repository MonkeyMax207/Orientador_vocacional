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
# Revised after replaying a real conversation against several models: the old prompt forced an
# opener ("¡Claro!") and listed questions the model then repeated. This one asks it to answer
# the student's questions first, never repeat a question, and cope with transcription errors.
SYSTEM_PROMPT = """Eres Orienta, el orientador vocacional de la universidad. Conversas en voz alta, \
en español, con un estudiante que está decidiendo qué estudiar.

Cómo hablar:
Usa frases cortas y naturales, como en una conversación real: una o dos frases por turno.
Si el estudiante te pregunta algo, respóndelo primero con sinceridad y en pocas palabras, y luego sigue.
Antes de preguntar, reacciona a algo concreto de lo que acaba de decir.
Nunca repitas una pregunta que ya hiciste. Si ya sabes algo, pasa a otro tema.
La transcripción de su voz puede tener errores: si algo no tiene sentido, pide amablemente que lo repita.
No uses listas, símbolos ni emojis. Escribe los números con palabras.

Qué averiguar, un tema a la vez y en el orden en que fluya la charla: materias favoritas, actividades \
que disfruta, en qué es bueno, si prefiere trabajar con personas, con datos, con las manos o creando, \
y qué le importa para su futuro. Cuando tengas cuatro o cinco temas, resume lo que entendiste y \
sugiere dos o tres áreas de estudio que encajen. Si no conoces los programas específicos de la \
universidad, dilo con honestidad."""

# Added to the system prompt when knowledge cards exist. Built ONCE per run so the system prompt
# never changes between turns: Ollama can then reuse its cached reading of it (prompt cache).
RAG_RULES = """

Reglas sobre la universidad:
Solo menciona programas, materias, laboratorios y servicios que aparezcan en la "Información \
verificada de la UAO" o en la lista oficial de programas de abajo.
Si no tienes la información, dilo con honestidad y sugiere hablar con un asesor de la UAO.
Nunca des precios, valores de matrícula ni montos de becas.
Programas de pregrado de la UAO: {names}."""


class Brain:
    def __init__(self, host: str, model: str, threads: int, num_gpu: int, max_turns: int = 10,
                 program_names=()):
        self.url = host.rstrip("/")
        self.model = model
        self.system = SYSTEM_PROMPT + (RAG_RULES.format(names=", ".join(program_names)) if program_names else "")
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
            "messages": [{"role": "system", "content": self.system}],
            "options": {**self.options, "num_predict": 1},   # generate just 1 token
        }
        with self._post("/api/chat", body) as r:
            r.read()

    def stream_reply(self, user_text: str, stop=None, cards=()):
        content = user_text
        if cards:
            # Verified facts travel ONLY with this turn's message; the history keeps the plain
            # text, so later turns don't carry (and re-read) old cards.
            facts = "\n".join(f"- {c['titulo']}: {c['texto']}" for c in cards)
            content = f"Información verificada de la UAO:\n{facts}\n\nEstudiante: {user_text}"
        self.history.append({"role": "user", "content": user_text})
        # Keep an odd number of messages so the history always starts with a user message.
        self.history = self.history[-(2 * self.max_turns - 1):]
        body = {
            "model": self.model, "stream": True, "keep_alive": -1, "options": self.options,
            "messages": [{"role": "system", "content": self.system}] + self.history[:-1]
                        + [{"role": "user", "content": content}],
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


def llm_worker(brain, text: str, out_q, stop, cards=()) -> None:
    """Runs on its own thread: moves tokens from the LLM into a queue for main.py.

    The final None is guaranteed (finally), so the reader never waits forever,
    even if Ollama crashes mid-reply.
    """
    try:
        for token in brain.stream_reply(text, stop, cards):
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
