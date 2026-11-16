"""Factual quiz: does the fine-tuned student actually know the UAO?

Deterministic scoring by term matching, no LLM judge: cheap, reproducible, and it will not
flatter us. The questions are the held-out phrasings from the dataset split — the same facts the
model trained on, asked in wordings it never saw.

The metric that decides this sub-project is cross-contamination: a 0.6-1B model learns style well
and blurs facts, and the way that shows up is attributing Mecatrónica's labs to Mecánica.

    uv run python -m finetune.quiz --model orienta:0.6b-q8        # el que corre en la Pi
    PYTHONPATH=. ~/.venvs/orienta-train/bin/python -m finetune.quiz --hf ~/orienta-ft/final

--model mide un tag de Ollama, que es lo que de verdad se sirve. --hf mide un checkpoint local en
fp16: sirve como linea base para comparar entrenamientos, NO como cifra de aceptacion, porque
cuantizar daña los hechos memorizados mas que el estilo.
"""
import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

from finetune import validate

ROOT = Path(__file__).parent.parent
GENERIC = frozenset("""ingenieria licenciatura tecnologia virtual modelo dual aleman publica
finanzas global digitales social medios empresas negocios internacionales periodismo""".split())


def distinctive_tokens(official: list[str]) -> dict[str, str]:
    """Token -> program, for tokens that identify exactly one program.

    'mecatronica' identifies Ingeniería Mecatrónica; 'ingenieria' identifies nothing, and
    'comunicacion' appears in two programs, so both drop out by construction.
    """
    counts: dict[str, list[str]] = {}
    for program in official:
        for token in re.findall(r"[a-zñ]+", validate.normalize(program)):
            if len(token) > 4 and token not in GENERIC:
                counts.setdefault(token, []).append(program)
    return {t: progs[0] for t, progs in counts.items() if len(progs) == 1}


def expected_terms(reference: str, source: str, question: str = "") -> list[str]:
    """The specific facts an answer should carry.

    Content words in both the teacher's answer and the source text — a term only in the
    teacher's wording is style, not fact — minus anything already in the question. Demanding
    the question's own words rewards a model that parrots it and penalises one that answers
    straight: "¿cuántos créditos de pensamiento creativo?" is answered by "cinco créditos",
    and requiring "pensamiento" there measures echoing, not knowledge.
    """
    shared = validate._content_tokens(reference) & validate._content_tokens(source)
    return sorted(shared - validate._content_tokens(question))


def score(answer: str, terms: list[str], *, program: str, official: list[str]) -> dict:
    plain = validate.normalize(answer)
    words = set(re.findall(r"[a-zñ]+", plain))
    hits = [t for t in terms if t in plain]

    tokens = distinctive_tokens(official)
    asked = {t for t, p in tokens.items() if p == program}
    others = {t for t in tokens if t not in asked}
    # Contaminacion: nombra otro programa y no el preguntado. Conservador a proposito: tambien
    # marca un "es parecida a X" legitimo, y preferimos sobrecontar este fallo que perderlo.
    contaminated = bool(words & others) and not bool(words & asked)

    return {
        "aciertos": len(hits) / len(terms) if terms else None,
        "contaminacion_cruzada": contaminated,
        "inventado": bool(validate._mentions_absent_program(answer, official)),
        "formato_ok": not (validate.MARKDOWN.search(answer) or validate.DIGITS.search(answer)
                            or validate.EMOJI.search(answer)),
        "precio_dado": validate._quotes_an_amount(answer),
    }


def _sources() -> dict[str, dict]:
    """fact_id -> source record, for both cards and chunks."""
    out = {}
    for name, path in (("ficha", ROOT / "knowledge" / "cards.json"),
                        ("chunk", ROOT / "knowledge" / "chunks.json")):
        for row in json.loads(path.read_text()):
            out[f"{name}:{row['id']}"] = row
    return out


def ask_ollama(question: str, *, host: str, model: str, system: str) -> str:
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": question}],
            "options": {"temperature": 0.0, "num_predict": 120}}
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"].strip()


def _hf_asker(path: str, system: str):
    """Local checkpoint. Imported lazily so test_logic.py can import this module without torch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetune.train import render
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()

    def ask(question: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        ids = tok(render(tok, messages, add_generation_prompt=True), return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=120, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ask


def main() -> None:
    from brain import SYSTEM_PROMPT

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="tag de Ollama (la cifra que cuenta)")
    ap.add_argument("--hf", help="checkpoint local fp16 (solo linea base)")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--quiz", default="finetune/data/quiz.jsonl")
    ap.add_argument("--n", type=int, default=200, help="preguntas a muestrear")
    args = ap.parse_args()
    if not (args.model or args.hf):
        raise SystemExit("hace falta --model <tag de ollama> o --hf <ruta al checkpoint>")

    rows = [json.loads(l) for l in (ROOT / args.quiz).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "p" in r and "r" in r]
    sources = _sources()
    cards = json.loads((ROOT / "knowledge" / "cards.json").read_text())
    official = sorted({c["programa"] for c in cards if c.get("programa")})

    random.Random(11).shuffle(rows)
    rows = rows[:args.n]
    if args.hf:
        print("AVISO: midiendo fp16, linea base para comparar entrenamientos. La cifra de "
              "aceptacion se mide sobre el GGUF cuantizado.", flush=True)
        ask = _hf_asker(args.hf, SYSTEM_PROMPT)
    else:
        ask = lambda q: ask_ollama(q, host=args.host, model=args.model, system=SYSTEM_PROMPT)

    agg = {"aciertos": [], "contaminacion": 0, "inventado": 0, "formato_malo": 0, "precio": 0,
           "con_terminos": 0}
    precio_rows, fallos = [], []
    for i, row in enumerate(rows, 1):
        source = sources.get(row.get("fact_id", ""), {})
        terms = expected_terms(row["r"], source.get("texto", ""), row["p"])
        answer = ask(row["p"])
        s = score(answer, terms, program=source.get("programa", ""), official=official)
        if s["aciertos"] is not None:
            agg["aciertos"].append(s["aciertos"])
            agg["con_terminos"] += 1
        agg["contaminacion"] += s["contaminacion_cruzada"]
        agg["inventado"] += s["inventado"]
        agg["formato_malo"] += not s["formato_ok"]
        agg["precio"] += s["precio_dado"]
        if s["precio_dado"]:
            precio_rows.append((row["p"], answer))
        elif s["inventado"] or (s["aciertos"] is not None and s["aciertos"] == 0 and terms):
            fallos.append((row["p"], answer, terms))
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    n = len(rows)
    precision = sum(agg["aciertos"]) / len(agg["aciertos"]) if agg["aciertos"] else 0.0
    print(f"\n=== {args.model or args.hf} sobre {n} preguntas ===")
    print(f"precision factual       {precision:6.1%}   (umbral >= 80%)")
    print(f"contaminacion cruzada   {agg['contaminacion'] / n:6.1%}   (umbral < 5%, invalida > 15%)")
    print(f"programas inventados    {agg['inventado']:6d}     (umbral 0)")
    print(f"precios dados           {agg['precio']:6d}     (umbral 0)")
    print(f"formato no hablable     {agg['formato_malo']:6d}     (umbral 0)")
    print(f"preguntas con terminos esperados: {agg['con_terminos']}/{n}")
    for q, a in precio_rows[:3]:
        print(f"\n[PRECIO] P: {q[:90]}\n         R: {a[:140]}")
    for q, a, t in fallos[:4]:
        print(f"\n[FALLO] P: {q[:90]}\n        R: {a[:140]}\n        esperaba: {t[:6]}")


if __name__ == "__main__":
    main()
