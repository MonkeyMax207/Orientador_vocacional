"""Judge the student's saved answers against the source text, with the local teacher.

Why this exists: `quiz.py`'s "precision factual" is the fraction of the *reference answer's*
content words that appear in the student's answer. That is recall of wording. On 2026-11-16 it
scored a correct answer ("Sí, hay Seminario Taller del Diseño Industrial II") at zero because
the reference happened to add "en el quinto semestre", while missing real errors like offering
electronics courses as Arquitectura's curriculum. A number that does both cannot decide
anything, so this module measures correctness instead.

Two deliberate choices:

  * The judge never sees the reference answer. Showing it would reproduce the very failure
    being fixed — the judge would compare phrasings instead of checking facts. It sees the
    source text, which is the ground truth the reference was itself written from.
  * A third of the quiz is follow-ups severed from their conversation ("¿en qué consiste *ese*
    plan de estudios?") and some are STT noise turned into a question. Those are not failures
    of the model, so the judge gets to say "imposible" and they are reported apart instead of
    being counted as wrong.

    uv run python -m finetune.judge --respuestas logs/respuestas_ft17b.jsonl
"""
import argparse
import json
from pathlib import Path

from finetune import teacher

ROOT = Path(__file__).parent.parent
VEREDICTOS = ("correcta", "incompleta", "incorrecta", "imposible")


def build_prompt(question: str, answer: str, source: str, program: str) -> list[dict]:
    """The judging turn. `source` is the ground truth; the reference answer is withheld."""
    return [{"role": "system", "content":
             "Eres un evaluador estricto. Juzgas si la respuesta de un asistente es correcta "
             "SEGUN EL TEXTO FUENTE que se te da, no segun tu conocimiento general ni segun "
             "como esta redactada. Respondes solo JSON."},
            {"role": "user", "content":
             f"TEXTO FUENTE{f' (programa: {program})' if program else ''}:\n{source[:2500]}\n\n"
             f"PREGUNTA DEL USUARIO:\n{question}\n\n"
             f"RESPUESTA DEL ASISTENTE:\n{answer}\n\n"
             "Clasifica la respuesta en una de estas cuatro categorias:\n"
             '- "correcta": lo que afirma concuerda con el texto fuente y responde la pregunta.\n'
             '  La redaccion puede ser totalmente distinta a la del texto; eso no importa.\n'
             '- "incompleta": todo lo que dice es cierto, pero omite lo que se le pregunto.\n'
             '- "incorrecta": contradice el texto fuente, habla de otro programa, o inventa\n'
             '  datos (nombres de programas, materias, contactos, cifras).\n'
             '- "imposible": la pregunta no se puede responder porque depende de una\n'
             '  conversacion previa ("ese plan", "esa parte") o esta demasiado truncada.\n\n'
             'Devuelve exactamente: {"veredicto": "...", "por_que": "una frase corta"}'}]


def parse(reply) -> tuple[str, str]:
    """(veredicto, por_que). Anything unrecognised becomes ('', ...) so it is not silently
    counted as a pass — an unparseable judgement is a missing judgement, not a good one."""
    if not isinstance(reply, dict):
        return "", "respuesta del juez ilegible"
    v = str(reply.get("veredicto", "")).strip().lower()
    return (v if v in VEREDICTOS else ""), str(reply.get("por_que", ""))[:200]


def summarize(veredictos: list[str]) -> dict:
    """Counts plus the two rates that matter, both computed over *judgeable* items only.

    "imposible" and unparsed verdicts leave the denominator: counting an unanswerable question
    as a failure of the model would understate it exactly the way the old metric did.
    """
    conteo = {v: veredictos.count(v) for v in VEREDICTOS}
    conteo["sin_juicio"] = sum(1 for v in veredictos if not v)
    base = conteo["correcta"] + conteo["incompleta"] + conteo["incorrecta"]
    conteo["juzgables"] = base
    conteo["exactitud"] = (conteo["correcta"] / base) if base else 0.0
    conteo["exactitud_laxa"] = ((conteo["correcta"] + conteo["incompleta"]) / base) if base else 0.0
    conteo["tasa_error"] = (conteo["incorrecta"] / base) if base else 0.0
    return conteo


def main() -> None:
    from finetune.quiz import _sources

    ap = argparse.ArgumentParser()
    ap.add_argument("--respuestas", default="logs/respuestas_ft17b.jsonl")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--host", default=teacher.HOST)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--salida", default="logs/juicio.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.respuestas).read_text().splitlines() if l.strip()]
    sources = _sources()
    print(f"{len(rows)} respuestas a juzgar con {args.model}", flush=True)

    jobs, usables = [], []
    for r in rows:
        src = sources.get(r.get("fact_id", ""), {}).get("texto", "")
        if not src:
            continue                      # sin fuente no hay verdad contra la que juzgar
        usables.append(r)
        jobs.append(build_prompt(r["p"], r["modelo"], src, r.get("programa") or ""))
    print(f"{len(jobs)} tienen texto fuente", flush=True)

    replies = teacher.ask_many(jobs, host=args.host, model=args.model, workers=args.workers,
                               num_predict=200)
    veredictos = []
    with open(ROOT / args.salida, "w", encoding="utf-8") as out:
        for r, reply in zip(usables, replies):
            v, why = parse(reply)
            veredictos.append(v)
            out.write(json.dumps({**r, "veredicto": v, "por_que": why},
                                  ensure_ascii=False) + "\n")

    s = summarize(veredictos)
    print(f"\n=== juicio de {args.respuestas} ===")
    print(f"correctas      {s['correcta']:4d}")
    print(f"incompletas    {s['incompleta']:4d}   (ciertas pero no responden lo preguntado)")
    print(f"incorrectas    {s['incorrecta']:4d}   (contradicen la fuente o inventan)")
    print(f"imposibles     {s['imposible']:4d}   (dependen de contexto que no existe)")
    print(f"sin juicio     {s['sin_juicio']:4d}   (el juez no devolvio JSON legible)")
    print(f"\nsobre {s['juzgables']} juzgables:")
    print(f"  exactitud estricta  {s['exactitud']:6.1%}   (solo 'correcta')")
    print(f"  exactitud laxa      {s['exactitud_laxa']:6.1%}   (correcta + incompleta)")
    print(f"  tasa de error       {s['tasa_error']:6.1%}   (inventa o contradice)")
    print(f"\ndetalle por respuesta en {args.salida}", flush=True)


if __name__ == "__main__":
    main()
