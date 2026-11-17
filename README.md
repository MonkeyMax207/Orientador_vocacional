# Orientador vocacional por voz — Oso Andino

Un animatrónico que conversa en español con aspirantes y les ayuda a escoger carrera en la
Universidad Autónoma de Occidente. **Funciona sin internet**, entero dentro de una Raspberry
Pi 5 de 8 GB: voz, modelo de lenguaje, conocimiento y servos.

Correr local no es un capricho técnico: el robot se promueve como una solución económica, así
que depender de un servidor o de conexión anula su razón de ser.

---

## Cómo está armado

Cada pieza hace un trabajo y no el del vecino. Esa separación salió de medir, no de diseñar:

| Pieza | Qué aporta | Archivo |
|---|---|---|
| **Whisper** (`base`, int8) | Oír | `stt.py` |
| **Respuestas inmediatas** | Saludos, gracias y **precios**, sin pasar por el modelo | `respuestas.py` |
| **Grafo de programas** | El dato exacto de la carrera correcta | `graph.py` |
| **BM25 léxico** | Buscar texto cuando el grafo no tiene el dato | `lexical.py` |
| **Modelo afinado** | *Cómo* habla: registro, brevedad, negativas | `brain.py` |
| **Piper** | Voz | `tts.py` |
| **ESP32 + PCA9685** | Ojos y boca | `esp32.py`, `firmware/` |

El modelo **no sabe los datos de memoria y no debe saberlos**. Medido contra la fuente, el
modelo solo se equivocaba en el **46,5%** de las respuestas, inventando materias casi correctas
atribuidas al programa equivocado. Con recuperación baja a **19,9%**. Los hechos se recuperan;
el modelo solo los dice bien.

---

## Arrancar en la Raspberry

```bash
cd ~/Orientador_vocacional
.venv/bin/python main.py
```

No lleva argumentos: el perfil se detecta leyendo `/proc/device-tree/model`. En la Pi usa el
perfil `pi`; en la DGX, `dgx`. Si alguna vez ves que arranca a vectorizar documentos, está en
el perfil equivocado.

Antes de la primera vez:

```bash
.venv/bin/python test_logic.py        # 137 comprobaciones, ~1 min
.venv/bin/python -m lexical "titulo de psicologia"   # el buscador, sin modelo
ollama list | grep orienta            # debe estar orienta-17b-q8
```

Para el animatrónico, cuando esté enchufado:

```bash
python3 probar_oso.py                 # busca el puerto y recorre ojos, boca y gestos
```

y después poné el puerto en `config.toml`: `esp32_port = "/dev/ttyUSB0"`.

---

## Los números, sin adornos

Medidos el 2026-11-17 en la Pi 5, con el juez de `finetune/judge.py` sobre 200 preguntas
apartadas. **No es calidad de producción todavía.**

| | Medido | Umbral del proyecto |
|---|---|---|
| Exactitud estricta | **50,6%** | 80% |
| Exactitud laxa (correcta + incompleta) | 80,1% | — |
| **Tasa de error** (inventa o contradice) | **19,9%** | — |
| Programas inventados | 3 de 200 | **0** |
| Precios dados | **0 de 200** ✅ | 0 |

De cada cinco respuestas, una es falsa dicha con seguridad. Sirve para demostrar el robot y
para medir desde algo real; no para dejarlo solo con un aspirante.

**Latencia por turno**, con el historial acotado a 3 intercambios:

| Etapa | Tiempo |
|---|---|
| Esperar el fin de tu frase | 0,6 s |
| Whisper | 1,9 s |
| Modelo hasta la primera palabra | 4 – 6 s |
| Piper (primera frase) | 0,6 s |

Las muletillas (`fillers.py`) tapan ese hueco: una corta apenas dejás de hablar y una larga a
los 3,5 s. No acortan la espera real; cambian la percibida, que es lo que se podía mover.

Un saludo o una pregunta de precio responden en **0 s**: no pasan por el modelo.

---

## Las decisiones que no son obvias

**Por qué BM25 y no embeddings.** El RAG semántico necesita un segundo modelo cargado y una
petición HTTP por turno. BM25 indexa los 400 documentos en 27 ms y consulta en 0,43 ms, en
Python puro. Su debilidad —no entiende sinónimos— se cubre con expansión por prefijo, para que
*"cuánto **dura**"* alcance *"**Duración** del programa"*.

**Por qué un grafo y no solo búsqueda.** `knowledge/cards.json` ya venía estructurado: 147
fichas con `programa` y `tipo`. Un hecho del grafo son 60-70 caracteres; un párrafo de folleto,
520. En la Pi cada ficha de prompt cuesta ~23 ms, así que el tamaño del dato es latencia.

**Por qué la carrera en foco es pegajosa.** Con el programa en la consulta, la recuperación
acierta el programa el **100%** de las veces; sin él, el 40%. En una conversación nadie repite
el nombre ("¿y cuánto dura?"), así que el foco se arrastra — y se suelta en cuanto nombrás otra.

**Por qué Q8 y no Q4.** Q4 es 25% más rápido y la tasa de error sube de 19,9% a 34,6%.

**Por qué Whisper `base` y no `tiny`.** `tiny` es 1,8x más rápido y convierte *"me interesa
psicología"* en *"me interesa así con lo día"*, lo que manda la búsqueda a una carrera al azar.

---

## Dónde mirar cuando algo falla

**Cada turno queda registrado** en `logs/conversacion-<fecha>.jsonl`: la pregunta, la respuesta,
qué documentos se recuperaron y cuánto tardó cada etapa. Es la herramienta de diagnóstico más
útil del proyecto — distingue si falló el oído, la búsqueda o el modelo. Los audios quedan en
`recordings/`.

En la terminal, `[rag]` muestra lo recuperado y `[inmediata:<regla>]` cuando se saltó el modelo.

---

## Documentación

| Documento | Para qué |
|---|---|
| [ROADMAP.md](ROADMAP.md) | Lo hecho y lo que falta, con prioridades |
| [docs/GUIA-PI.md](docs/GUIA-PI.md) | Desplegar el modelo en la Raspberry, paso a paso |
| [docs/GUIA-FINE-TUNING.md](docs/GUIA-FINE-TUNING.md) | Entrenar el modelo desde cero, con 20 trampas documentadas |
| [docs/GUIA-DE-PRUEBAS.md](docs/GUIA-DE-PRUEBAS.md) | Probar cada pieza por separado |
| [docs/2026-11-16-resultados-medicion.md](docs/2026-11-16-resultados-medicion.md) | Todas las mediciones del entrenamiento |
| `docs/superpowers/specs/` | Los diseños originales de cada subproyecto |

---

## Entrenar (solo en la DGX)

El entrenamiento no corre en la Pi y no debe: `pyproject.toml` nunca debe arrastrar torch.

```bash
~/.venvs/orienta-train/bin/python -m finetune.train --model Qwen/Qwen3-1.7B --epochs 5
~/.venvs/orienta-train/bin/python -m finetune.export --nombre orienta-17b-q8 --cuantizar q8_0
~/.venvs/orienta-train/bin/python -m finetune.export --nombre orienta-17b-q8 --empaquetar ~/orienta-pi
```

Corré siempre el guardia térmico al lado: `uv run python -m finetune.thermal --pausa 90 --reanuda 82`.
