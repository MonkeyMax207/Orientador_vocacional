"""Los programas de la UAO como hechos estructurados, no como prosa que hay que buscar.

Por que existe. El 2026-11-17, hablando con el robot de verdad por primera vez, la primera
conversacion mostro dos fallas que ningun quiz habia revelado:

  * "Hola, como va todo?" recupero documentos del CAFS y respondio sobre espacios deportivos.
    79 de los 400 documentos no pertenecen a ninguna carrera (bienestar, campus, deportes) y
    secuestraban la charla.
  * "Que carreras de arte existen?" trajo Mercadeo y Negocios Internacionales; "y es dificil?"
    trajo Ingenieria Mecanica cuando se hablaba de Mercadeo. La busqueda por texto mezcla
    programas, y confundir programas era ya el modo de falla dominante del modelo.

Y la latencia: entregar 520 caracteres de folleto cuesta ~140 fichas de prompt, que en la Pi
son segundos de silencio antes de que el robot hable. Un hecho estructurado — "Psicologia: 8
periodos, 139 creditos, titulo Psicologo" — son ~20.

Como esta construido. `knowledge/cards.json` **ya es un grafo**: cada ficha trae `programa` y
`tipo` (perfil, egresado, materias, laboratorios). Lo que falta son los datos duros, que viven
dentro de la prosa de `chunks.json` y se extraen una sola vez con expresiones regulares.
Cobertura medida sobre los 30 programas: creditos, titulo y SNIES 100%, duracion 90%,
modalidad 73%.

Lo que este modulo NO hace: adivinar. Si un programa no tiene el dato, se devuelve vacio y el
robot puede decir que no lo sabe, que es mejor que inventarlo con seguridad.
"""
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
CARDS = ROOT / "knowledge" / "cards.json"
CHUNKS = ROOT / "knowledge" / "chunks.json"

# Campos duros, extraidos de la prosa. El orden importa: el primero que coincide gana.
PATRONES = {
    "duracion": [r"duracion del programa[:\s]*([0-9]{1,2})\s*(?:periodos|semestres)",
                 r"([0-9]{1,2})\s*periodos academicos"],
    "creditos": [r"([0-9]{2,3})\s*creditos academicos", r"([0-9]{2,3})\s*creditos"],
    # El tope era 45 y cortaba "disenador de la comunicacion grafica descripc" a mitad de
    # palabra, antes de que recortar() pudiera limpiarlo: se deja holgado y limpia el recorte.
    "titulo":   [r"titulo que otorga[:\s]*([a-zñ /()\.]{4,90})"],
    "modalidad": [r"modalidad[:\s]*(presencial|virtual|hibrida|dual|a distancia)"],
    "snies":    [r"snies[:\s]*([0-9]{4,7})"],
}

# Que palabras de una pregunta apuntan a que campo. Se comparan sobre texto sin tildes.
CAMPOS = {
    "duracion": ("dura", "duracion", "semestres", "periodos", "anos", "tiempo", "largo"),
    "creditos": ("creditos", "credito"),
    "titulo": ("titulo", "grado", "me gradúo", "egreso", "diploma", "llama el titulo"),
    "modalidad": ("modalidad", "virtual", "presencial", "distancia", "hibrida", "nocturno"),
    "perfil": ("perfil", "para quien", "me sirve", "me gusta", "soy bueno", "aspirante",
                "que necesito", "requisitos", "va conmigo"),
    "egresado": ("egresado", "trabajar", "trabajo", "salida", "campo laboral", "donde puedo",
                  "empleo", "sirve para", "futuro"),
    "materias": ("materia", "materias", "ver", "estudia", "plan de estudios", "pensum",
                  "semestre", "asignatura", "curriculo"),
    "laboratorios": ("laboratorio", "laboratorios", "equipos", "practica", "taller", "lab"),
}


# Palabras que en el folleto vienen pegadas despues del dato util. La extraccion por expresion
# regular no sabe donde termina un titulo, asi que se corta en la primera de estas: sin esto,
# el titulo de Psicologia salia "psicologo becas y financiacion directorio de".
CORTES = ("becas", "financiacion", "descripcion", "directorio", "clic", "nivel academico",
          "pregunta por", "modalidad", "duracion", "horarios", "snies", "plan de estudios",
          "conoce", "mas informacion", "inscripciones")


def recortar(valor: str) -> str:
    """El valor hasta la primera palabra de relleno, sin espacios ni puntos sueltos."""
    v = valor
    for corte in CORTES:
        i = v.find(corte)
        if i > 0:
            v = v[:i]
    return " ".join(v.split()).strip(" .,-")


def normalize(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", (texto or "").lower())
                   if unicodedata.category(c) != "Mn")


def build(cards_path: Path | None = None, chunks_path: Path | None = None) -> dict:
    """{programa: {campo: valor}}. Se construye en memoria al arrancar: son milisegundos."""
    cards = json.loads((cards_path or CARDS).read_text())
    chunks = json.loads((chunks_path or CHUNKS).read_text())

    grafo: dict[str, dict] = {}
    prosa: dict[str, list[str]] = {}
    for d in cards + chunks:
        programa = d.get("programa")
        if not programa:
            continue                      # bienestar, campus, deportes: no son de ninguna carrera
        nodo = grafo.setdefault(programa, {"programa": programa})
        prosa.setdefault(programa, []).append(str(d.get("texto", "")))
        tipo = d.get("tipo")
        if tipo in ("perfil", "egresado", "materias", "laboratorios"):
            # La ficha mas larga de cada tipo: las cortas suelen ser encabezados sueltos.
            texto = str(d.get("texto", "")).strip()
            if len(texto) > len(nodo.get(tipo, "")):
                nodo[tipo] = texto

    for programa, textos in prosa.items():
        plano = normalize(" ".join(textos))
        for campo, patrones in PATRONES.items():
            for rx in patrones:
                m = re.search(rx, plano)
                if m:
                    valor = recortar(m.group(1))
                    if valor:
                        grafo[programa][campo] = valor
                    break
    return grafo


def campo_de(pregunta: str) -> str | None:
    """Que dato pide la pregunta, o None si no se reconoce ninguno.

    Devolver None es una respuesta valida y util: significa "esto no es una consulta de ficha",
    y quien llama puede recurrir a la busqueda por texto en vez de forzar un campo equivocado.
    """
    p = normalize(pregunta)
    mejor, largo = None, 0
    for campo, claves in CAMPOS.items():
        for clave in claves:
            if clave in p and len(clave) > largo:   # la coincidencia mas especifica gana
                mejor, largo = campo, len(clave)
    return mejor


def programa_de(pregunta: str, grafo: dict, actual: str = "") -> str:
    """El programa que nombra la pregunta; si no nombra ninguno, el de la conversacion.

    Medido el 2026-11-16: con el programa en la consulta el acierto de programa es 100%, sin el
    40%. En una conversacion el programa casi nunca se repite ("¿y cuanto dura?"), asi que
    arrastrarlo es lo que sostiene la precision.
    """
    p = normalize(pregunta)
    candidatos = []
    for nombre in grafo:
        for palabra in normalize(nombre).split():
            if len(palabra) > 5 and palabra not in (
                    "ingenieria", "licenciatura", "tecnologia", "profesional", "comunicacion",
                    "administracion", "virtual", "modelo", "aleman") and palabra in p:
                candidatos.append((len(palabra), nombre))
    if candidatos:
        return max(candidatos)[1]
    return actual


def hecho(grafo: dict, programa: str, campo: str, max_chars: int = 260) -> str:
    """El dato pedido, ya redactado como una linea corta; vacio si no se tiene.

    Nunca se rellena con lo que haya cerca: un dato ausente tiene que poder decirse "no lo se".
    """
    nodo = grafo.get(programa)
    if not nodo or not campo:
        return ""
    v = nodo.get(campo)
    if not v:
        return ""
    if campo == "duracion":
        extra = f", {nodo['creditos']} créditos" if nodo.get("creditos") else ""
        return f"{programa}: {v} periodos académicos{extra}."
    if campo == "creditos":
        return f"{programa}: {v} créditos académicos."
    if campo == "titulo":
        return f"{programa}: el título que otorga es {v.strip().rstrip('.')}."
    if campo in ("modalidad", "snies"):
        return f"{programa}: {campo} {v}."
    return f"{programa} ({campo}): {str(v)[:max_chars]}"


# Preguntas por el catalogo entero, no por una carrera. En la conversacion real del
# 2026-11-17 aparecieron dos ("que carreras de arte existen", "que carreras no existen") y las
# dos se respondieron de memoria: una invento que Mercadeo es una carrera de arte.
MAX_LISTA = 4        # cuantos nombres se le entregan al modelo en una pregunta de catalogo
CATALOGO = ("que carreras", "cuales carreras", "que programas", "cuales programas",
            "que estudios", "que ofrecen", "que hay para estudiar", "carreras tienen",
            "carreras hay", "programas tienen", "que puedo estudiar")


def listar(grafo: dict, pregunta: str, familia: str = "") -> str:
    """Los nombres oficiales, si la pregunta es por el catalogo. Vacio si no lo es.

    Se entregan los nombres y nada mas: son el unico antidoto contra inventar carreras, que
    sigue siendo un requisito duro incumplido (3 invenciones por cada 200 respuestas).
    """
    p = normalize(pregunta)
    if not any(c in p for c in CATALOGO):
        return ""
    nombres = sorted(grafo)
    # "carreras de ingenieria" acota; sin filtro el prompt serian los 30 nombres completos.
    for pista in ("ingenieria", "diseno", "comunicacion", "administracion", "licenciatura"):
        if pista in p:
            nombres = [n for n in nombres if pista in normalize(n)] or nombres
            break
    # Se entregan pocos a proposito. Pedirle al modelo "no los leas todos" no funciono: con 30
    # nombres delante los leyo los 30, 43 palabras y 29 segundos. Un dato que no esta no se
    # puede recitar, y eso es mas fiable que una instruccion.
    muestra = nombres[:MAX_LISTA]
    cola = f" y {len(nombres) - len(muestra)} más" if len(nombres) > len(muestra) else ""
    return "Programas de la UAO: " + ", ".join(muestra) + cola + "."


def consultar(grafo: dict, pregunta: str, actual: str = "", contexto: str = "") -> tuple[str, str]:
    """(hecho, programa). Vacio cuando no hay programa en foco o no se tiene el dato.

    `pregunta` es lo que se acaba de decir y `contexto` lo anterior. La distincion importa: el
    programa puede venir de hace varios turnos, pero el CAMPO tiene que salir de la pregunta
    actual. El 2026-11-17, mezclandolos, "y cuanto dura" despues de "hablame del plan de
    estudios" devolvia las materias, porque "plan de estudios" es una coincidencia mas larga
    que "dura" y mi regla premiaba la mas larga.
    """
    programa = programa_de(f"{contexto} {pregunta}", grafo, actual)
    if not programa:
        return "", ""
    campo = campo_de(pregunta) or campo_de(contexto)
    return hecho(grafo, programa, campo), programa


if __name__ == "__main__":
    import sys
    g = build()
    print(f"{len(g)} programas en el grafo")
    completos = {c: sum(1 for n in g.values() if n.get(c))
                 for c in ("duracion", "creditos", "titulo", "modalidad", "snies",
                            "perfil", "egresado", "materias", "laboratorios")}
    for c, n in completos.items():
        print(f"  {c:<14} {n:2d}/{len(g)}")
    if len(sys.argv) > 1:
        pregunta = " ".join(sys.argv[1:])
        h, p = consultar(g, pregunta)
        print(f"\npregunta: {pregunta}\n  programa: {p or '(ninguno)'}  "
              f"campo: {campo_de(pregunta)}\n  hecho: {h or '(no se tiene)'}")
