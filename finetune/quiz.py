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

import lexical
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


# Formas de decir "eso no lo ofrecemos". Separadas de validate.DECLINES, que son formas de
# decir "la fuente no lo dice": aqui se niega la OFERTA, no el conocimiento.
NIEGA_OFRECER = ("no ofrecemos", "no contamos con", "no tenemos", "no existe", "no hay",
                 "no la ofrece", "no lo ofrece", "no esta disponible")


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
        # Una negativa honesta nombra la carrera ausente PARA rechazarla, que es justo lo que
        # queremos: "no contamos con Ingenieria Quimica, pero si con Ambiental" se marcaba como
        # invencion el 2026-11-16 y contaba contra el modelo. validate.py ya tenia la exencion;
        # aqui faltaba, asi que la medicion castigaba el comportamiento correcto.
        "inventado": bool(validate._mentions_absent_program(answer, official))
                     and not any(d in plain for d in validate.DECLINES)
                     and not any(d in plain for d in NIEGA_OFRECER),
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
    ap.add_argument("--guardar", help="JSONL donde dejar cada respuesta, para re-puntuar sin "
                                      "volver a generar (una corrida son ~12 min de GPU)")
    ap.add_argument("--recuperar", type=int, default=0,
                    help="documentos BM25 a poner en el prompt (0 = sin recuperacion)")
    ap.add_argument("--fragmentos", type=int, default=0,
                    help="franjas del documento en vez del documento entero (lo que corre la Pi)")
    ap.add_argument("--con-programa", action="store_true",
                    help="incluir el programa en la consulta, como lo haria el robot a mitad "
                          "de conversacion, que sabe de que carrera se esta hablando")
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
    # Recuperacion lexica opcional. El 2026-11-16 el modelo afinado quedo en 46.5% de error con
    # el modo de falla dominante en inventar especificos; los hechos que un 1.7B no retiene hay
    # que recuperarlos. BM25 y no embeddings: 0.13 ms por consulta en la Pi, sin modelo cargado.
    indice = lexical.load() if args.recuperar else None

    def con_contexto(pregunta: str, programa: str) -> str:
        if not indice:
            return pregunta
        consulta = f"{programa} {pregunta}" if (args.con_programa and programa) else pregunta
        if args.fragmentos:
            trozos = lexical.fragmentos(indice, consulta, docs=args.recuperar,
                                         n=args.fragmentos)
            if not trozos:
                return pregunta
            bloque = "\n".join(f"- {d.get('titulo') or d.get('programa') or ''}: {d['texto']}"
                                for d in trozos)
            return (f"Informacion de la UAO:\n{bloque}\n\n"
                    f"Responde esta pregunta usando solo esa informacion, en UNA frase corta y "
                    f"hablada.\n\nPregunta: {pregunta}")
        hits = indice.search(consulta, k=args.recuperar)
        if not hits:
            return pregunta
        bloque = "\n".join(f"- {d.get('titulo') or d.get('programa') or ''}: "
                            f"{str(d.get('texto'))[:600]}" for d, _ in hits)
        # Instrucciones duras porque la primera version produjo respuestas de 77 palabras
        # listando el documento, y frases como "el texto no especifica", que un robot que habla
        # en voz alta nunca puede decir: para el visitante no existe ningun texto.
        return (f"Informacion de la UAO:\n{bloque}\n\n"
                f"Responde esta pregunta usando solo esa informacion, en UNA frase corta y "
                f"hablada, como se la dirias a alguien de frente. No leas la lista completa: "
                f"di lo esencial. Nunca menciones \"el texto\", \"la informacion\" ni que te "
                f"dieron datos. Escribe los numeros en palabras (\"nueve semestres\", no \"9\"), "
                f"porque esto se va a leer en voz alta. Si la informacion no alcanza, dilo en "
                f"una frase.\n\n"
                f"Pregunta: {pregunta}")

    if args.hf:
        print("AVISO: midiendo fp16, linea base para comparar entrenamientos. La cifra de "
              "aceptacion se mide sobre el GGUF cuantizado.", flush=True)
        ask = _hf_asker(args.hf, SYSTEM_PROMPT)
    else:
        ask = lambda q: ask_ollama(q, host=args.host, model=args.model, system=SYSTEM_PROMPT)

    agg = {"aciertos": [], "contaminacion": 0, "inventado": 0, "formato_malo": 0, "precio": 0,
           "con_terminos": 0}
    precio_rows, fallos = [], []
    # Las respuestas se tiraban a la basura al terminar. Una corrida son ~12 min de GPU y la
    # metrica puede cambiar — ya cambio dos veces — asi que re-puntuar sin re-generar vale oro.
    guardado = open(ROOT / args.guardar, "w", encoding="utf-8") if args.guardar else None
    for i, row in enumerate(rows, 1):
        source = sources.get(row.get("fact_id", ""), {})
        terms = expected_terms(row["r"], source.get("texto", ""), row["p"])
        answer = ask(con_contexto(row["p"], source.get("programa") or ""))
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
        if guardado:
            guardado.write(json.dumps({"p": row["p"], "esperada": row["r"], "modelo": answer,
                                        "terminos": terms, "aciertos": s["aciertos"],
                                        "fact_id": row.get("fact_id", ""),
                                        "programa": source.get("programa", "")},
                                       ensure_ascii=False) + "\n")
            guardado.flush()      # por si la maquina se apaga de nuevo a mitad de corrida
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    if guardado:
        guardado.close()
        print(f"respuestas guardadas en {args.guardar}", flush=True)
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
