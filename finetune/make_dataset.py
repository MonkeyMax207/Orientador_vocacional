"""Generates the training dataset by distilling a local teacher over the UAO source text.

Run (pilot, ~40 examples, to eyeball before committing hours):
    uv run python -m finetune.make_dataset --limit 40 --out finetune/data/piloto --model qwen2.5:14b
Run (full):
    uv run python -m finetune.make_dataset --out finetune/data --model qwen2.5:14b

Writes as it goes and skips jobs already present on restart, so an interrupted overnight run
resumes instead of starting over.
"""
import argparse
import json
import random
import re
from pathlib import Path

from finetune import schema, teacher, validate

ROOT = Path(__file__).parent.parent
CARDS = ROOT / "knowledge" / "cards.json"
CHUNKS = ROOT / "knowledge" / "chunks.json"

# Phrasings the teacher is told to vary across, so the student sees each fact many ways.
REGISTROS = ["directa y corta", "coloquial, como un amigo", "dudosa, como quien no sabe el nombre exacto",
             "indirecta, preguntando por lo que le interesa", "con una palabra mal transcrita",
             "larga y enredada", "muy breve, dos o tres palabras"]

FORMA = ("Reglas de forma, obligatorias: espanol hablado, sin markdown, sin negritas, sin vinetas, "
         "sin emoji, sin cifras (los numeros en palabras), una o dos frases por respuesta. "
         "Nunca des precios, valores de matricula ni montos de becas. Nunca inventes telefonos "
         "ni correos. Solo menciona programas, materias o laboratorios que aparezcan en el texto.")

# No hay lista fija: se calcula filtrando contra los programas oficiales, porque "Derecho" y
# "Arquitectura" SI los ofrece la UAO y entrenar su negacion seria un error factual.

FUERA_DE_TEMA = ["¿Quien gano el mundial?", "¿Me cuentas un chiste?", "¿Que hora es?",
                 "¿Cual es la capital de Francia?", "¿Me ayudas con una tarea de calculo?",
                 "¿Que piensas de la politica?", "¿Sabes cocinar?"]


def _cards() -> list[dict]:
    return [c for c in json.loads(CARDS.read_text()) if not c.get("revisar")]


def _chunks() -> list[dict]:
    return json.loads(CHUNKS.read_text())


def prompt_factual(source: dict, n: int, estricto: bool, especifico: bool = False) -> list[dict]:
    """`especifico` exige nombrar los datos textuales de la fuente.

    El 0.6B entrenado con la version anterior saco 16,5% de precision factual: aprendio a
    responder glosas genericas ("tiene laboratorios para experimentar con sistemas mecanicos")
    en vez de nombrar el Fab-Lab. La causa esta en el propio prompt: pedir "una o dos frases"
    sin exigir los datos concretos entrena vaguedad. Esta variante pide lo contrario.
    """
    extra = ("Responde SOLO con lo que dice el texto; si algo no esta en el texto, no lo menciones. "
             if estricto else "")
    registros = ", ".join(random.sample(REGISTROS, k=min(4, len(REGISTROS))))
    if especifico:
        exigencia = (
            "REGLA PRINCIPAL: cada respuesta tiene que NOMBRAR los datos textuales del texto "
            "—nombres propios de laboratorios, materias, programas, espacios o servicios— "
            "escritos igual que en el texto. Prohibido resumir con generalidades: 'tiene "
            "laboratorios de su area' esta MAL; 'tiene el Fab-Lab y el laboratorio de Automatica' "
            "esta BIEN. Si la pregunta pide una lista, enumera los elementos reales. "
            "Prioriza preguntas cuya respuesta sea un dato concreto y verificable en el texto. "
            "Puedes usar hasta tres frases cuando haga falta para nombrarlos todos. "
            "PERO dilo con tus palabras, como un amigo hablando: copia los NOMBRES PROPIOS "
            "textuales y nada mas. Prohibido pegar frases de folleto del texto: 'seras un "
            "profesional formado con vision y liderazgo capaz de ejecutar procesos de "
            "direccionamiento estrategico' esta MAL por sonar a publicidad; 'vas a aprender a "
            "dirigir empresas, y haces practica en empresas alemanas' esta BIEN. ")
    else:
        exigencia = ""
    return [{"role": "system", "content":
             "Eres un generador de datos de entrenamiento para un orientador vocacional que habla "
             "en voz alta con estudiantes. Respondes SIEMPRE con JSON valido y nada mas."},
            {"role": "user", "content":
             f"Texto oficial de la Universidad Autonoma de Occidente:\n\n{source['texto']}\n\n"
             f"Escribe {n} pares pregunta-respuesta en espanol sobre ESTE texto. Las preguntas las "
             f"hace un joven que esta pensando que estudiar. Varia mucho el fraseo entre estos "
             # La exigencia va al final, pegada a la instruccion de salida: es lo ultimo que lee
             # el maestro y tiene que pesar mas que el "una o dos frases" de FORMA.
             f"registros: {registros}. {extra}{FORMA}\n\n{exigencia}\n"
             'Devuelve solo JSON: [{"p":"pregunta","r":"respuesta"}, ...]'}]


def prompt_dialogo(cards: list[dict], n_turnos: int, programas: list[str]) -> list[dict]:
    contexto = "\n".join(f"- {c['titulo']}: {c['texto']}" for c in cards)
    nombres = ", ".join(programas)
    return [{"role": "system", "content":
             "Eres un generador de datos de entrenamiento. Respondes SIEMPRE con JSON valido."},
            {"role": "user", "content":
             f"Datos oficiales de la Universidad Autonoma de Occidente:\n\n{contexto}\n\n"
             f"Escribe una conversacion de {n_turnos} turnos entre un joven que no sabe que estudiar "
             f"y Orienta, un amigo cercano que conoce la universidad. Orienta escucha primero, "
             f"reacciona a algo concreto de lo que el joven dijo, hace UNA sola pregunta por turno, "
             f"nunca repite una pregunta, y solo recomienda carreras cuando ya conoce a la persona o "
             f"cuando se lo piden. No es un vendedor. Cada respuesta de Orienta: una o dos frases, "
             f"nunca mas. {FORMA}\n\n"
             f"Si Orienta nombra una carrera, usa EXACTAMENTE uno de estos nombres, sin variarlo: "
             f"{nombres}.\n\n"
             'Devuelve solo JSON: {"turnos":[{"u":"lo que dice el joven","a":"lo que responde Orienta"}, ...]}'}]


def prompt_negativas(clase: str, n: int, programas: list[str]) -> list[dict]:
    casos = {
        "precio": ("El joven pregunta por precios, matricula, cuotas o becas. Orienta NO da cifras "
                   "ni rangos: dice con amabilidad que eso lo explica mejor un asesor de admisiones "
                   "de la universidad. Varia mucho el fraseo de la pregunta: 'cuanto vale', 'me "
                   "alcanza con lo que tengo', 'es caro', 'hay descuentos', 'se puede pagar por "
                   "cuotas', 'cuanto pagaria al semestre', 'que tan costoso es'."),
        "precio_insiste": ("El joven YA recibio la respuesta de que los precios los explica "
                           "admisiones, y insiste: pide un aproximado, un rango, 'solo dime mas o "
                           "menos', 'un numero redondo', 'entre cuanto y cuanto'. Orienta se niega "
                           "otra vez, con calidez y sin ceder NUNCA una cifra, ni aproximada ni de "
                           "rango, y vuelve a derivar a admisiones. Escribe el par con la "
                           "insistencia como pregunta."),
        "ausente": (f"El joven pregunta por una carrera que la universidad NO ofrece "
                    f"(por ejemplo: {', '.join(random.sample(validate.absent_from(programas), 4))}). "
                    f"Orienta lo dice con honestidad, sin inventar, y ofrece contar lo que si hay. "
                    f"Usa SOLO estos nombres de programas que SI existen, escritos exactamente asi: "
                    f"{', '.join(random.sample(programas, min(10, len(programas))))}."),
        "contacto": ("El joven pide un telefono, un correo o un link. Orienta no tiene datos de "
                     "contacto propios y no los inventa: sugiere preguntar en la universidad."),
        "fuera": (f"El joven pregunta algo ajeno a la universidad y a su futuro "
                  f"(por ejemplo: {', '.join(random.sample(FUERA_DE_TEMA, 3))}). Orienta responde "
                  f"breve y con calidez, y vuelve con naturalidad a la conversacion."),
        "ruido": ("La transcripcion de la voz del joven llego sin sentido o cortada (por ejemplo: "
                  "'esto que el tiene gusta por', 'mmm no se eh'). Orienta no adivina: pide que se "
                  "lo repita con CALIDEZ y cercania, como un amigo que no escucho bien en un lugar "
                  "ruidoso, nunca con sequedad ni dando ordenes."),
    }
    return [{"role": "system", "content":
             "Eres un generador de datos de entrenamiento. Respondes SIEMPRE con JSON valido."},
            {"role": "user", "content":
             f"{casos[clase]}\n\nEscribe {n} pares pregunta-respuesta distintos de esa situacion, "
             f"variando mucho el fraseo de la pregunta. {FORMA}\n\n"
             'Devuelve solo JSON: [{"p":"pregunta","r":"respuesta"}, ...]'}]


def stt_noise(text: str, rng: random.Random) -> str:
    """Light Whisper-style damage: the student will meet this input in production."""
    swaps = [("v", "b"), ("ll", "y"), ("z", "s"), ("ción", "cion"), ("qu", "c"), ("h", "")]
    out = text
    for _ in range(rng.randint(1, 2)):
        a, b = rng.choice(swaps)
        out = out.replace(a, b, 1)
    words = out.split()
    if len(words) > 4 and rng.random() < 0.4:
        del words[rng.randrange(len(words))]          # Whisper drops short words
    return " ".join(words)


def _load_done(path: Path) -> tuple[list[dict], set[str]]:
    """Rows from a previous run, re-checked for shape.

    Resumed rows must not be trusted blindly: anything written before a gate existed would
    sail straight into the dataset. The job ids still count as done, so a job whose rows are
    discarded here is not generated again — the gates are what decide, not the history.
    """
    if not path.exists():
        return [], set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if _usable(r)], {r["job_id"] for r in rows if "job_id" in r}


def _usable(d) -> bool:
    """Shape check, including inside each dialogue turn.

    The teacher occasionally splits one turn into two objects ({"u":...} then {"a":...}), which
    passed a top-level "turnos in d" check and then raised KeyError in schema.to_chat at the end
    of a multi-hour run. Those dialogues are dropped rather than repaired: pairing them up is
    guesswork, and they were 9 of 258.
    """
    if not isinstance(d, dict):
        return False
    if "turnos" in d:
        turnos = d["turnos"]
        return (isinstance(turnos, list) and bool(turnos)
                and all(isinstance(t, dict) and str(t.get("u", "")).strip()
                        and str(t.get("a", "")).strip() for t in turnos))
    return "p" in d and "r" in d


def _as_items(data) -> list[dict]:
    """The teacher answers with a list of pairs, or with one object holding "turnos".

    Anything with other key names ("pregunta"/"respuesta") is dropped here rather than
    reaching schema.to_chat, which would raise on the missing key.
    """
    if isinstance(data, list):
        return [d for d in data if _usable(d)]
    return [data] if _usable(data) else []


def generate(jobs: list[tuple[str, list[dict], str, bool]], *, model: str, host: str,
             official: list[str], out_path: Path, rejects_path: Path, workers: int,
             batch: int, num_predict: int = 1300) -> list[dict]:
    """Runs jobs in batches, validating and appending results as they arrive.

    Each job is (job_id, messages, source_text, allow_absent). Resumes by job_id.
    """
    kept, done = _load_done(out_path)
    pending = [j for j in jobs if j[0] not in done]
    print(f"{len(done)} ya hechos, {len(pending)} pendientes", flush=True)
    with out_path.open("a") as out, rejects_path.open("a") as bad:
        for start in range(0, len(pending), batch):
            lote = pending[start:start + batch]
            respuestas = teacher.ask_many([m for _, m, _, _ in lote], host=host, model=model,
                                           workers=workers, seed_base=start,
                                           num_predict=num_predict)
            for (job_id, _, source, allow_absent), data in zip(lote, respuestas):
                if data is None:
                    bad.write(json.dumps({"job_id": job_id, "motivos": ["sin_json"]}) + "\n")
                    continue
                for item in _as_items(data):
                    motivos = validate.check(item, source=source, official=official,
                                              allow_absent=allow_absent)
                    if motivos:
                        bad.write(json.dumps({"job_id": job_id, "motivos": motivos,
                                               "ejemplo": item}, ensure_ascii=False) + "\n")
                        continue
                    item["job_id"], item["fact_id"] = job_id, job_id.rsplit("#", 1)[0]
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    kept.append(item)
            out.flush()
            bad.flush()
            print(f"  lote {start // batch + 1}: {len(kept)} validos acumulados", flush=True)
    return kept


def split_by_phrasing(examples: list[dict], quiz_frac: float = 0.2):
    """Every fact stays in training; what is held out are unseen question phrasings."""
    por_hecho: dict[str, list[dict]] = {}
    for e in examples:
        por_hecho.setdefault(e.get("fact_id", "?"), []).append(e)
    train, quiz = [], []
    for _, grupo in sorted(por_hecho.items()):
        if len(grupo) < 3:                      # too few phrasings to spare one
            train.extend(grupo)
            continue
        corte = max(1, int(len(grupo) * quiz_frac))
        quiz.extend(grupo[:corte])
        train.extend(grupo[corte:])
    return train, quiz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--host", default=teacher.HOST)
    ap.add_argument("--out", default="finetune/data")
    ap.add_argument("--limit", type=int, default=0, help="solo las primeras N fuentes (piloto)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--por-ficha", type=int, default=10)
    ap.add_argument("--por-chunk", type=int, default=8)
    ap.add_argument("--pasadas", type=int, default=2, help="pasadas por fuente, con fraseo distinto")
    ap.add_argument("--pasadas-especificas", type=int, default=0,
                    help="pasadas que exigen nombrar los datos textuales de la fuente")
    ap.add_argument("--dialogos", type=int, default=400)
    ap.add_argument("--negativas", type=int, default=30, help="llamadas por clase de negativa")
    ap.add_argument("--num-predict", type=int, default=1300,
                    help="techo de tokens; los dialogos de 8 turnos no caben en 900")
    ap.add_argument("--solo-escribir", action="store_true",
                    help="no generar nada: rehacer train/quiz desde crudo.jsonl")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    cards, chunks = _cards(), _chunks()
    programas = sorted({c["programa"] for c in cards if c.get("programa")})
    print(f"{len(cards)} fichas, {len(chunks)} chunks, {len(programas)} programas", flush=True)

    if args.limit:
        cards, chunks = cards[:args.limit], chunks[:args.limit]

    # Varias pasadas por fuente con distinto fraseo: es la palanca principal para que un modelo
    # pequeno memorice un hecho. Subir --pasadas y relanzar solo genera las pasadas nuevas.
    jobs: list[tuple[str, list[dict], str, bool]] = []
    pases = [chr(ord("a") + i) for i in range(args.pasadas)]
    # Las pasadas 'esp*' exigen nombrar datos textuales: son las que atacan el 16,5% de precision
    # factual de la primera ronda. Ids distintos, asi que la reanudacion solo genera estas.
    esp = [f"esp{i}" for i in range(args.pasadas_especificas)]
    for c in cards:
        for p in pases:
            jobs.append((f"ficha:{c['id']}#{p}", prompt_factual(c, args.por_ficha, False),
                          c["texto"], False))
        for p in esp:
            jobs.append((f"ficha:{c['id']}#{p}",
                          prompt_factual(c, args.por_ficha, False, especifico=True),
                          c["texto"], False))
    for ch in chunks:
        for p in pases:
            jobs.append((f"chunk:{ch['id']}#{p}", prompt_factual(ch, args.por_chunk, True),
                          ch["texto"], False))
        for p in esp:
            jobs.append((f"chunk:{ch['id']}#{p}",
                          prompt_factual(ch, args.por_chunk, True, especifico=True),
                          ch["texto"], False))
    n_dialogos = max(1, args.dialogos if not args.limit else 3)
    for i in range(n_dialogos):
        muestra = rng.sample(cards, k=min(3, len(cards)))
        fuente = " ".join(c["texto"] for c in muestra)
        jobs.append((f"dialogo:{i}", prompt_dialogo(muestra, 8, programas), fuente, False))
    n_neg = args.negativas if not args.limit else 1
    # precio y precio_insiste van al doble (dos tandas con ids distintos, si no la reanudacion
    # las saltaria): es requisito duro, y el modelo fabrico una cifra en la prueba de cordura
    # aunque dio 0 precios en las 200 preguntas del quiz.
    for clase, tanda in (("precio", "a"), ("precio", "b"),
                          ("precio_insiste", "a"), ("precio_insiste", "b"),
                          ("ausente", "a"), ("contacto", "a"), ("fuera", "a"), ("ruido", "a")):
        for i in range(n_neg):
            # Sin anclaje a un texto fuente: estas respuestas son politica, no datos.
            jobs.append((f"neg:{clase}#{tanda}{i}", prompt_negativas(clase, 8, programas), "",
                          clase == "ausente"))
    rng.shuffle(jobs)
    print(f"{len(jobs)} llamadas al maestro", flush=True)

    if args.solo_escribir:
        kept, _ = _load_done(out_dir / "crudo.jsonl")
        print(f"{len(kept)} ejemplos leidos de crudo.jsonl (sin generar nada)", flush=True)
    else:
        kept = generate(jobs, model=args.model, host=args.host, official=programas,
                        out_path=out_dir / "crudo.jsonl", rejects_path=out_dir / "rechazos.jsonl",
                        workers=args.workers, batch=args.batch, num_predict=args.num_predict)
    kept = validate.dedup(kept)
    print(f"{len(kept)} tras deduplicar", flush=True)

    for e in kept:                               # ruido de STT en ~20% de las entradas
        if "p" in e and rng.random() < 0.2:
            e["p"] = stt_noise(e["p"], rng)

    train, quiz = split_by_phrasing(kept)
    # El bucket viaja junto a los mensajes (el entrenador ignora las claves extra) para poder
    # excluir un bucket flojo mas tarde sin volver a generar nada.
    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps({**schema.to_chat(e), "bucket": e.get("job_id", "?").split("#")[0]},
                              ensure_ascii=False) for e in train) + "\n")
    (out_dir / "quiz.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in quiz) + "\n")
    (out_dir / "muestra.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in kept[:40]) + "\n")
    print(f"listo: {len(train)} entrenamiento, {len(quiz)} quiz -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
