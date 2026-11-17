"""Respuestas inmediatas, sin pasar por el modelo.

Idea de Juan Diego el 2026-11-17, despues de oir al robot tardar varios segundos en decir
"Hola, que bien": hay turnos donde el modelo no aporta nada y si cuesta. En la Pi un turno son
~7,6 s de los cuales el modelo es el 57%; aqui bajan a cero.

Dos razones, y la segunda importa mas que la latencia:

  * **Velocidad.** Un saludo no necesita 1.700 millones de parametros.
  * **Seguridad.** Los requisitos duros del proyecto — nunca dar precios, nunca inventar una
    carrera — dejan de depender de que el modelo se porte bien. Una respuesta escrita a mano
    no alucina. El 2026-11-17 el robot contesto "Si" a "¿esa carrera me sirve para hacer
    astronauta?"; lo que no pasa por el modelo no puede hacer eso.

Lo que NO va aqui: cualquier cosa que dependa de los datos de un programa. Eso es trabajo del
grafo, que si sabe. Esta tabla es solo para lo que se puede responder sin mirar nada.
"""
import random
import re
import unicodedata


def normalize(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", (texto or "").lower())
                   if unicodedata.category(c) != "Mn")


# (nombre, patron, respuestas). El orden importa: gana el primero que coincide, asi que lo
# especifico (precio) va antes que lo general (saludo).
TABLA = [
    ("precio", r"\b(cuanto (cuesta|vale|sale)|precio|precios|matricula|semestre cuesta|"
               r"valor del? (semestre|programa|carrera)|cuanto pagaria|es cara|es barata|"
               r"mas economica|mas barata|beca|becas|financiacion|credito icetex)\b",
     ["Los valores y las becas cambian cada semestre, así que prefiero no darte una cifra que "
      "pueda estar vieja. Eso lo ves con admisiones, que te dan el dato exacto.",
      "Mejor no te doy números de matrícula, que cambian seguido. Admisiones te cuenta los "
      "valores y las becas al día.",
      "De plata mejor te hablan en admisiones: ahí tienen los valores y las opciones de beca "
      "actualizadas."]),

    ("saludo", r"^(hola|buenas|buenos dias|buenas tardes|buenas noches|hey|que mas|quiubo|"
               r"hola como estas|hola que tal)[\s,\.!¡¿\?]*$",
     ["¡Hola! Soy Orienta. ¿Qué te gustaría saber de la universidad?",
      "¡Hola! Cuéntame, ¿qué carrera te está dando vueltas en la cabeza?",
      "¡Buenas! Estoy para ayudarte a escoger carrera. ¿Por dónde empezamos?"]),

    ("como_estas", r"\b(como (estas|vas|te va|andas)|que tal estas|todo bien)\b",
     ["Muy bien, gracias por preguntar. ¿Qué te trae por acá?",
      "Todo bien por acá. ¿Querés que hablemos de alguna carrera en particular?",
      "Bien, con ganas de ayudarte. ¿Qué te gustaría saber?"]),

    ("quien_eres", r"\b(quien eres|como te llamas|que eres|tu nombre|eres un robot|"
                   r"eres una maquina|eres humano)\b",
     ["Soy Orienta, el orientador de la Universidad Autónoma de Occidente. Estoy acá para "
      "ayudarte a escoger carrera.",
      "Me llamo Orienta y acompaño a quienes están decidiendo qué estudiar en la Autónoma."]),

    ("gracias", r"^(muchas gracias|gracias|listo gracias|vale gracias|te agradezco)"
                r"[\s,\.!¡¿\?]*$",
     ["Con mucho gusto. ¿Algo más que quieras saber?",
      "¡De nada! Si te queda otra duda, aquí estoy.",
      "Para eso estoy. ¿Seguimos con algo más?"]),

    ("despedida", r"^(chao|adios|hasta luego|nos vemos|ya me voy|eso es todo)[\s,\.!¡¿\?]*$",
     ["¡Listo! Que te vaya muy bien con la decisión.",
      "¡Chao! Vuelve cuando quieras seguir mirando carreras."]),
]

_COMPILADA = [(nombre, re.compile(patron), opciones) for nombre, patron, opciones in TABLA]


class Respuestas:
    """Elige de una bolsa barajada, por la misma razon que las muletillas: con random.choice
    la misma frase sale dos veces seguidas con demasiada frecuencia y suena a maquina."""

    def __init__(self):
        self._bolsas: dict[str, list[str]] = {}

    def buscar(self, texto: str) -> tuple[str, str]:
        """(respuesta, nombre_de_la_regla). Vacio si ninguna aplica, que es el caso normal."""
        plano = " ".join(normalize(texto).split())
        for nombre, patron, opciones in _COMPILADA:
            if patron.search(plano):
                bolsa = self._bolsas.setdefault(nombre, [])
                if not bolsa:
                    bolsa.extend(opciones)
                    random.shuffle(bolsa)
                return bolsa.pop(), nombre
        return "", ""


if __name__ == "__main__":
    import sys
    r = Respuestas()
    pruebas = sys.argv[1:] or [
        "hola", "hola como estas", "cuanto cuesta la carrera", "cual es la mas economica",
        "muchas gracias", "quien eres", "que materias tiene mecatronica"]
    for p in pruebas:
        texto, regla = r.buscar(p)
        print(f"  {p:<34} -> [{regla or 'al modelo'}] {texto[:70]}")
