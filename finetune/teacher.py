"""The teacher: a local Ollama model that writes the training examples.

Connection: Ollama's /api/chat with "stream": false. Only the standard library is
used (urllib), like brain.py — nothing here runs on the Pi, but keeping the same
shape means the Pi's dependency set never grows.

The teacher wraps its JSON in prose ("Claro, aquí tienes: [...]") no matter how
firmly the prompt forbids it, so _extract_json recovers the first balanced JSON
value from the reply instead of trusting json.loads on the whole thing.
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = "http://127.0.0.1:11435"   # the parallel instance; the system one queues serially


def _post(host: str, body: dict) -> dict:
    req = urllib.request.Request(host.rstrip("/") + "/api/chat",
                                 json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def _extract_json(raw: str):
    """The first balanced JSON array or object in `raw`, or None if there is none."""
    starts = [i for i in (raw.find("["), raw.find("{")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    opening = raw[start]
    closing = "]" if opening == "[" else "}"
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_string:                      # brackets inside strings must not count
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == opening:
            depth += 1
        elif c == closing:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def ask_json(messages: list[dict], *, host: str = HOST, model: str, retries: int = 2,
             num_predict: int = 900, temperature: float = 0.9, seed: int | None = None):
    """Asks the teacher and returns the parsed JSON, or None after `retries` bad replies."""
    body = {"model": model, "stream": False, "keep_alive": -1, "messages": messages,
            "options": {"num_gpu": 99, "temperature": temperature, "num_predict": num_predict}}
    if seed is not None:
        body["options"]["seed"] = seed
    for _ in range(retries + 1):
        data = _extract_json(_post(host, body).get("message", {}).get("content", ""))
        if data is not None:
            return data
    return None


def ask_many(jobs: list[list[dict]], *, host: str = HOST, model: str, workers: int = 8,
             num_predict: int = 900, seed_base: int = 0):
    """Runs `jobs` concurrently. Results keep the input order; a failed job is None.

    One bad job must not sink a run of hundreds, so every exception becomes None.
    """
    def one(indexed):
        i, messages = indexed
        try:
            return ask_json(messages, host=host, model=model, num_predict=num_predict,
                            seed=seed_base + i)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, enumerate(jobs)))


if __name__ == "__main__":
    # Throughput probe, to pick the teacher by measurement:
    #   uv run python -m finetune.teacher --probe qwen2.5:14b 8
    import sys
    import time

    if "--probe" in sys.argv:
        model = sys.argv[sys.argv.index("--probe") + 1]
        workers = int(sys.argv[sys.argv.index("--probe") + 2])
        prompt = [{"role": "user", "content":
                   "Genera 5 pares pregunta-respuesta en espanol hablado sobre este texto, en JSON "
                   '[{"p":...,"r":...}]. Texto: Ingenieria Mecatronica tiene Fab-Lab para impresion '
                   "3D, Laboratorio de Automatica para control y robotica, y Laboratorio de Electronica."}]
        t = time.perf_counter()
        out = ask_many([prompt] * workers, model=model, workers=workers, num_predict=200)
        wall = time.perf_counter() - t
        ok = sum(1 for o in out if o)
        print(f"{model} concurrencia={workers} pared={wall:.1f}s validos={ok}/{workers} "
              f"agregado≈{workers * 200 / wall:.1f} tok/s")
