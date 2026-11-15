"""Deterministic gates every generated example must pass. No LLM judge: cheap and reproducible.

A dirty dataset teaches the student to hallucinate, so rejection is the default and the
reasons are explicit — a whole bucket failing shows up as a pattern in rechazos.jsonl
instead of quietly poisoning the weights.

Reasons returned: "programa_inventado", "precio", "formato", "contacto", "sin_anclaje",
"longitud", "sin_negacion".
"""
import re
import unicodedata

# Spoken form: every answer is read aloud by Piper, so markdown and digits are defects.
MARKDOWN = re.compile(r"\*\*|##|__|^\s*[-*•]\s|\n\s*[-*•]\s|\|")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
DIGITS = re.compile(r"\d")
EMAIL_OR_URL = re.compile(r"[\w.+-]+@[\w-]+\.\w+|https?://|www\.|\.edu\.co|\.com\b")
PROGRAM = re.compile(r"(?:ingenieria|licenciatura|tecnologia)\s+(?:de\s+|en\s+|del\s+)?([a-zñ]+)")
# Money: the refusal examples say the word "precios" without quoting a figure, so the gate
# looks for an AMOUNT, not for the topic.
NUMBER_WORDS = ("un", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
                "veinte", "treinta", "cuarenta", "cincuenta", "cien", "ciento", "quinientos", "mil")
MONEY_WORDS = ("peso", "pesos", "dolar", "dolares", "euro", "euros", "smmlv", "salario minimo")
BIG_AMOUNTS = ("millon", "millones", "smmlv")
DENIAL = ("no ", "ninguno", "ninguna", "tampoco", "no tenemos", "no ofrece", "no existe", "no hay")
# Honest "the source doesn't say" answers have almost no lexical overlap with the source by
# construction, so the anchoring gate must not throw them away: they are the examples that
# teach the student to admit ignorance instead of inventing.
DECLINES = ("no se menciona", "no menciona", "no dice", "no especifica", "no aparece",
            "no tengo ese dato", "no sabria", "no estoy seguro", "no lo precisa", "no detalla")
# Careers a model would plausibly invent for a Colombian university, including near-misses of
# real UAO names: "Diseño Gráfico" for "Diseño de la Comunicación Gráfica", "Ingeniería de
# Sistemas" for "Ingeniería Informática". Single words that double as subjects (economía,
# música, biología) are deliberately absent: they appear legitimately inside answers.
ABSENT_CANDIDATES = ("Medicina", "Odontologia", "Enfermeria", "Veterinaria", "Fisioterapia",
                     "Trabajo Social", "Biologia Marina", "Diseno Grafico", "Diseno Gráfico",
                     "Comunicacion Audiovisual", "Administracion de Negocios",
                     "Ingenieria Aeroespacial", "Ingenieria Nuclear", "Ingenieria Petrolera",
                     "Ingenieria Civil", "Ingenieria Quimica", "Ingenieria de Sistemas",
                     "Ingenieria Agronomica", "Ingenieria Naval", "Arquitectura Naval")


def absent_from(official: list[str]) -> list[str]:
    """Candidates that the official list does not contain — computed, never hand-maintained.

    'Derecho' and 'Arquitectura' ARE UAO programs, so a hardcoded absent list drifts into
    teaching the student to deny real ones.
    """
    oficiales = [normalize(o) for o in official]
    return [c for c in ABSENT_CANDIDATES if not any(normalize(c) in o for o in oficiales)]
STOPWORDS = frozenset("""el la los las un una unos unas de del al a en y o que como para por con sin
sobre tiene tienen hay es son esta estan muy mas pero si no te se lo le su sus mi tu eso esa ese
esto esta aqui alli tambien puedes puede pueden donde cuando cual cuales quien porque""".split())
MAX_WORDS = 60


def normalize(text: str) -> str:
    """Lowercase, accent-free: so 'Mecatrónica' and 'mecatronica' compare equal."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zñ0-9]+", normalize(text))
    return {w for w in words if len(w) > 3 and w not in STOPWORDS}


def _mentions_absent_program(text: str, official: list[str]) -> list[str]:
    """Program names in `text` that the UAO does not offer.

    Two detectors, because only 14 of the 30 official programs start with
    Ingeniería/Licenciatura/Tecnología — the pattern inherited from replay.py is blind to
    Arquitectura, Cine, Psicología, Diseño Industrial and the rest of the catalog.
    """
    plain = normalize(text)
    oficiales = [normalize(o) for o in official]
    absent = []
    for match in PROGRAM.finditer(plain):                    # "Ingeniería X" style
        candidate = re.sub(r"\s+", " ", match.group(0)).strip()
        if not any(candidate in o for o in oficiales):
            absent.append(candidate)
    for candidate in absent_from(official):                  # everything else, by name
        if normalize(candidate) in plain:
            absent.append(normalize(candidate))
    return absent


def _quotes_an_amount(text: str) -> bool:
    plain = normalize(text)
    if "$" in text or any(b in plain for b in BIG_AMOUNTS):
        return True
    has_money = any(m in plain for m in MONEY_WORDS)
    words = plain.split()
    return has_money and any(n in words for n in NUMBER_WORDS)


def check(example: dict, *, source: str, official: list[str],
          allow_absent: bool = False) -> list[str]:
    """Rejection reasons for one example; an empty list means it is usable.

    `allow_absent` is for the refusals bucket, where naming a program the UAO does not
    offer is the whole point — there the gate flips and demands an explicit denial.
    """
    answer = example.get("r") or " ".join(t.get("a", "") for t in example.get("turnos", []))
    question = example.get("p") or " ".join(t.get("u", "") for t in example.get("turnos", []))
    reasons = []

    # Shape first: the teacher sometimes answers with other key names ("pregunta"/"respuesta"),
    # and an example with an empty answer would otherwise pass every other gate and train the
    # student to say nothing.
    if not answer.strip() or not question.strip():
        return ["forma_invalida"]

    if MARKDOWN.search(answer) or EMOJI.search(answer) or DIGITS.search(answer):
        reasons.append("formato")
    if len(answer.split()) > MAX_WORDS:
        reasons.append("longitud")
    if _quotes_an_amount(answer):
        reasons.append("precio")
    if EMAIL_OR_URL.search(answer) or re.search(r"\b\d[\d\s-]{5,}", answer):
        reasons.append("contacto")

    absent = _mentions_absent_program(answer, official)
    if allow_absent:
        if absent and not any(d in normalize(answer) for d in DENIAL):
            reasons.append("sin_negacion")   # named a program we don't have, without denying it
    elif absent:
        reasons.append("programa_inventado")

    declina = any(d in normalize(answer) for d in DECLINES)
    if source and not declina:
        tokens = _content_tokens(answer)
        shared = tokens & _content_tokens(source)
        if tokens and len(shared) / len(tokens) < 0.25:
            reasons.append("sin_anclaje")
    return reasons


def dedup(examples: list[dict]) -> list[dict]:
    """Drops questions that repeat an earlier one, comparing across every bucket.

    The key is the normalized content-token set, so punctuation, case and accents do not
    hide a duplicate.
    """
    seen, kept = set(), []
    for example in examples:
        question = example.get("p") or " ".join(t.get("u", "") for t in example.get("turnos", []))
        key = frozenset(_content_tokens(question))
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(example)
    return kept
