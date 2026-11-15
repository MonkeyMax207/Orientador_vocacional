# Pi autónoma: modelo afinado sin RAG — Diseño

Sub-proyecto 5. La Raspberry Pi 5 (8 GB) corre el agente completo por sí sola, con el conocimiento
de la UAO dentro de los pesos de un modelo pequeño afinado en la DGX, sin retrieval en tiempo de
inferencia.

## Motivación

El robot se quiere promover como una **solución económica**, y para eso tiene que correr de manera
puramente local: una Pi 5 de 8 GB, sin depender de la DGX, de internet ni de una red confiable.
Esto cambia el rol de la DGX: de **cerebro desplegado** (sub-proyecto 1, funcionando pero no es el
producto) a **máquina de entrenamiento**.

El perfil `[pi]` autónomo que ya existe fue descartado por medición propia (ver `README.md` y
§8 de `docs/GUIA-DE-PRUEBAS.md`): `qwen2.5:1.5b` con RAG "usa las fichas correctas pero se desvía,
se obsesiona con carreras sin motivo, adorna datos", con 2.6–6 s por respuesta y hasta 16–17 s en
conversaciones largas. En la Pi el RAG cuesta una llamada al embedder en CPU por turno más ~60
palabras de ficha en el prompt (~+1.5 s cada una).

## Decisión y alternativas consideradas

**Elegido — B: conocimiento en los pesos, cero retrieval al servir.** Dataset sintético robusto
generado por destilación, fine-tuning completo de un modelo de 0.6–1B, sin retrieval en la Pi.

Alternativas evaluadas y descartadas *para esta iteración*:

- **A: destilar persona + retrieval léxico barato.** Mismo fine-tuning, pero conservando búsqueda
  BM25 (submilisegundo, sin modelo) sobre las 147 fichas para los detalles. Más seguro en hechos,
  menos simple en tiempo de ejecución. **Queda como plan de salida medido** (ver §Compuerta).
- **C: sin fine-tuning.** Cambiar el modelo base y abaratar el RAG. Horas en vez de días, pero el
  techo de calidad es el del modelo base y la persona sigue dependiendo de un prompt largo, que es
  latencia pagada en cada turno.

El riesgo asumido con B, explícito: a 0.6–1B los modelos aprenden bien el *estilo* y difuminan los
*hechos*. El modo de falla esperado es contaminación cruzada (atribuir los laboratorios de
Mecatrónica a Mecánica) y, en el peor caso, inventar programas. El quiz factual (§Evaluación)
existe para detectarlo con números y no con opiniones.

## Presupuesto de latencia: dónde está realmente el tiempo

Derivado de las mediciones del `README.md` (fila `pi`, 4 hilos) y de la nota propia del proyecto
("el Pi 5 real será ~1.5–2.5x más lento"):

| Etapa | Simulado (4 hilos) | Pi 5 real estimado |
|---|---|---|
| STT (Whisper base int8) | 1.50 s | **2.2–3.8 s** |
| LLM 1ª frase (1.5B + 1 ficha) | 0.81 s | 1.2–2.0 s |
| TTS 1ª frase (Piper) | 0.21 s | 0.3–0.5 s |
| Silencio de fin de turno | 0.60 s | 0.6 s |
| **Total "respuesta lista"** | **3.12 s** | **~5–7 s** |

**El STT domina, no el LLM.** Quitar el RAG y acortar el prompt gana del orden de 1–2.5 s de esos
~6 s. Es una mejora real, pero este spec no resuelve el cuello de botella principal de la Pi: la
voz queda explícitamente pendiente (ver §Fuera de alcance).

## Criterios de éxito

Umbrales fijados **antes** de ver resultados, medidos siempre sobre el **GGUF cuantizado** que va
a correr en la Pi:

| Métrica | Umbral |
|---|---|
| Negativas correctas (precios, carreras inexistentes, datos de contacto) | **100%** — línea ética del proyecto, no negociable |
| Programas inventados en el replay de 11 turnos | **0** |
| Contaminación cruzada (hecho atribuido al programa equivocado) | **<5%** ideal; **>15% invalida B** tal como está diseñado |
| Forma hablada (sin markdown, viñetas ni dígitos) | **100%** |
| Precisión factual (términos esperados presentes) | **≥80%**, y no más de **10 puntos porcentuales** por debajo del baseline `qwen2.5:1.5b` + RAG |

## Alcance

**Dentro:** generación y validación del dataset, entrenamiento en la DGX, export y cuantización a
GGUF, el cambio de servido para el perfil `[pi]`, y el arnés de evaluación (quiz factual nuevo +
reutilización de `replay.py` y `bench.py`).

**Fuera:** STT (NeMo/alternativas a Whisper) y TTS (Riva/ajustes de Piper) — pendientes conocidos y
el verdadero cuello de botella; servos del animatrónico; fusión de `feature/uao-rag` a `main`. Los
perfiles `[pc]` y `[dgx]` no se tocan y siguen funcionando como hoy.

## Arquitectura

Dos máquinas, dos fases separadas en el tiempo:

```
DGX (entrenamiento, fuera de línea)
  knowledge/chunks.json (253) + cards.json (147)
      │
      ├─► maestro nemotron-nano-9b-v2 (NIM, :8001) ──► ejemplos sintéticos
      │                                                      │
      │                                   validación determinista + dedup
      │                                                      │
      └─────────────────────────► fine-tuning completo (0.6B) ──► GGUF cuantizado
                                                                        │
Pi 5 8 GB (servido, local y autónomo)  ◄────── Tailscale/scp ───────────┘
  client/main.py → Whisper base → Ollama (modelo afinado) → Piper
  sin embedder, sin fichas, prompt = solo persona
```

## Dataset

El principio que manda: inyectar conocimiento en un modelo sub-1B no se logra con *volumen* de
texto sino con **diversidad de fraseo por hecho**. Las 34.000 palabras de fuente (253 chunks) y
8.500 (147 fichas) no se entrenan tal cual: se expanden.

**Maestro:** un modelo de la familia `qwen2.5` servido por Ollama en local. Es destilación: el
maestro escribe en buen español conversacional y el 0.6B hereda ese registro, que es la cura para lo
acartonado que suele sonar un modelo tan pequeño en español. Se reutiliza el patrón existente de
`knowledge/make_cards.py` (pedir JSON al LLM sobre un texto fuente, con reintentos; ya cubierto por
`test_with_retries` y `test_ask_retries_cut_json`).

**Corrección del 2026-11-15, descubierta al ejecutar:** el plan original era usar el NIM de
`nemotron-nano-9b-v2`, pero su API key de NGC perdió permisos ("user not being a member of the
organization that owns the repo") y los pesos que había bajado en septiembre se perdieron porque el
contenedor corrió con `--rm` **sin montar volumen para el caché**. Se reemplaza por `qwen2.5` vía
Ollama, que está descargado, es local y no requiere autenticación. La muestra de calidad del 32B fue
buena (español hablado limpio, JSON válido, fraseo variado). El tamaño exacto del maestro se fija por
medición en la Task 1 del plan: el 32B rinde 22,5 tok/s agregados a concurrencia 8 (≈10,5 h para el
dataset completo), así que se elige el modelo más grande que baje de 4 h sin perder calidad. La
concurrencia exige una instancia de Ollama con `OLLAMA_NUM_PARALLEL=8` (el servicio del sistema
encola en serie: medido 10,4 tok/s con 8 peticiones, idéntico a una sola).

**Composición objetivo, ~11.000 ejemplos:**

| Bucket | Cantidad | Generación |
|---|---|---|
| QA factual sobre fichas | ~2.900 | 147 fichas × ~20 fraseos distintos |
| QA factual sobre chunks | ~3.800 | 253 chunks × ~15, con regla estricta "solo lo que dice este texto" |
| Diálogos vocacionales multi-turno | ~3.200 | ~400 conversaciones × ~8 turnos (escuchar → reaccionar → una sola pregunta → recomendar del catálogo real) |
| Negativas y rechazos | ~1.200 | precios/becas (~300), carreras que no existen (~300), datos de contacto (~200), fuera de tema (~200), entrada ininteligible (~200): transcripciones sin sentido ante las que el modelo debe pedir que le repitan, no adivinar |

Más una transformación de **ruido de STT** aplicada al ~20% de las entradas (no es un bucket
aparte): la entrada real viene de Whisper en una Pi, y el proyecto ya aprendió esto por las malas
("Pugar fútbol" → "jugar fútbol"). Entrenar con entradas sucias es entrenar donde el modelo vive.

**Validación automática y determinista de cada ejemplo, sin juez LLM.** Un dataset sucio enseña a
alucinar, así que esto es la mitad del trabajo:

- Todo nombre `Ingeniería|Licenciatura|Tecnología X` debe pertenecer a la lista oficial de 30
  programas — se reutilizan `NAME_RE` y `program_names()` de `knowledge/replay.py` y `rag.py`.
- Cero cifras de precios, matrícula o becas; cero teléfonos o correos inventados (regex).
- Forma hablada obligatoria: sin markdown, sin viñetas, sin emoji, números en palabras, 1–2 frases.
  **Crítico**: en el replay del 2026-09-29 el maestro escupía `**negritas**` y listas por todos
  lados; su salida cruda es inservible tal cual.
- Solapamiento léxico mínimo con el chunk fuente: atrapa cuando el maestro se va por su cuenta.
- Deduplicación por similitud para que ningún fraseo domine.

Los ejemplos rechazados se regeneran una vez y, si vuelven a fallar, se descartan con registro del
motivo (para poder auditar si un bucket entero está saliendo mal).

**El split se hace por fraseos, no por hechos.** Los 30 programas tienen que estar todos en
entrenamiento: el modelo debe conocerlos. Lo que se aparta para el quiz es un conjunto de
*preguntas con redacción no vista* sobre esos mismos hechos. Apartar hechos mediría generalización
a conocimiento que nunca le dimos, que no es la pregunta que nos interesa.

**Archivos nuevos:** directorio `finetune/`, siguiendo el estilo del repo (módulos planos, uno por
propósito, corribles con `uv run python -m finetune.X`):

- `finetune/make_dataset.py` — genera los ejemplos con el maestro.
- `finetune/validate.py` — las compuertas deterministas y la deduplicación.
- `finetune/data/` — el `.jsonl` grande, ignorado por git (se regenera); se comitea una muestra
  pequeña para poder revisarla a ojo.

## Entrenamiento

**Fine-tuning completo, no LoRA.** LoRA de rango bajo (8–16) es excelente para *estilo* y pobre
para *memorizar hechos nuevos*: la actualización vive en un subespacio de rango bajo, que alcanza
para "compórtate así" y no para "almacena estos 147 grupos de datos". En la GB10 la memoria no es
restricción: un 0.6B completo con AdamW son ~10 GB (pesos bf16 + gradientes + estados del
optimizador) sobre 121 GB unificados; un 1B completo ~16 GB. Teniendo el presupuesto, full FT es
estrictamente mejor para lo que estamos haciendo.

**Modelos en secuencia, no en paralelo**, para no botar trabajo:

1. **Qwen3 0.6B Instruct** — el loop más rápido y el más veloz en la Pi; a 0.6B el entrenamiento
   "pega" más fuerte respecto al conocimiento base.
2. Si falla el quiz: **Llama 3.2 1B Instruct** o **Gemma 3 1B** (ambos requieren aceptar licencia
   en HF; el token ya está configurado en la DGX).

Se parte del *instruct* y no del base para heredar el seguimiento de instrucciones. `qwen2.5:1.5b`
queda fuera: ya fue descartado por medición en este proyecto.

**Hiperparámetros que importan:**

- **Pérdida enmascarada solo a los tokens del asistente.** Si no, el modelo aprende a generar
  también las preguntas del estudiante. Es el error más común y el más silencioso.
- **3–5 épocas**, con checkpoint por época; el quiz de fraseos no vistos decide cuál se queda
  (inyectar conocimiento exige repetición, pero sobreajusta).
- **Longitud de secuencia 1024**: se entrena a la longitud en la que se va a servir.
- LR 1e-5–2e-5 con warmup y cosine, bf16.

**La trampa que más probablemente nos hunda en silencio:** la plantilla de chat del entrenamiento
tiene que ser **byte a byte idéntica** a la que Ollama aplica en la Pi. La de Qwen3 trae tokens de
*thinking*; si entrenamos con `apply_chat_template` de HF y el `Modelfile` difiere en un token o un
salto de línea, la calidad se cae de formas que parecen "el fine-tuning no funcionó". Mitigación
obligatoria: extraer la cadena exacta renderizada en entrenamiento y verificar que Ollama rinde la
idéntica para los mismos mensajes. El manejo del modo *thinking* se fija una vez y se usa igual en
ambos lados.

**Restricción de empaquetado que no se puede violar:** las dependencias de entrenamiento (torch,
trl, transformers) **nunca entran al `pyproject.toml` del proyecto**. La Pi instala desde ahí y un
`uv sync` que empiece a bajar torch en una Raspberry es inaceptable. El entrenamiento vive en su
propio venv o contenedor en la DGX, con `finetune/requirements-train.txt` aparte.

**Entorno, en orden de qué intentar primero.** La GB10 es **sm_121 (Blackwell)**, así que los
builds viejos no sirven — el contenedor `nvcr.io/nvidia/pytorch:24.06-py3` que ya está descargado
es de 2024 y no tiene esos kernels:

1. Wheels de pip para aarch64+CUDA en un venv nuevo.
2. El contenedor `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc4`, ya descargado.
3. Bajar un `nvcr.io/nvidia/pytorch:25.xx-py3`.

Stack: `transformers` + `SFTTrainer` de TRL, que ya maneja el enmascarado de completions.

**Archivo nuevo:** `finetune/train.py`.

## Export y servido en la Pi

**Regla sin excepciones: toda cifra de evaluación sale del GGUF cuantizado.** Cuantizar daña los
*hechos memorizados* mucho más que el *estilo*, y B apuesta todo a que los hechos vivan en los
pesos. Un modelo que saca buena nota en bf16 puede empezar a confundir laboratorios al bajar a Q4.

Esto invierte la costumbre de "Q4 por defecto": a 0.6B los tamaños son irrelevantes frente a los
8 GB de la Pi, así que se empieza arriba y se baja solo si la latencia obliga, midiendo el quiz en
cada nivel.

| Cuantización | Tamaño aprox. | Criterio |
|---|---|---|
| Q8_0 | ~640 MB | Punto de partida: el que menos daña los hechos |
| Q6_K | ~520 MB | Si Q8 pesa en velocidad |
| Q5_K_M | ~450 MB | — |
| Q4_K_M | ~400 MB | Último recurso |

**Pipeline** (ventaja secundaria del full FT: no hay adaptador que fusionar, el checkpoint ya es un
modelo HF normal): `convert_hf_to_gguf.py` de llama.cpp → `llama-quantize` para las variantes →
`ollama create orienta:0.6b-q8 -f Modelfile` con la plantilla clavada a la del entrenamiento →
copia a la Pi por Tailscale. El `Modelfile` se comitea (es texto, es la receta); el GGUF no
(`models/` ya está en `.gitignore`).

**El cambio en el código de servido es pequeño, lo que confirma que la frontera estaba bien
trazada.** Hoy `server.py`/`main.py` hacen `knowledge, self.find = make_finder(cfg)` y luego
`names = knowledge.program_names() if knowledge else ()`; `brain.py` arma
`SYSTEM_PROMPT + (RAG_RULES... if program_names else "")`. Es decir: si `make_finder` no devuelve
conocimiento, **el prompt se queda solo con la persona automáticamente, sin tocar `brain.py`**.

La modificación se concentra en:

- `config.toml`: `rag_enabled = true` en `[common]`, `false` en `[pi]`; `llm_model` del perfil `[pi]`
  apuntando al modelo afinado. **La clave va en `[common]` obligatoriamente**: el test
  `test_config_profiles_have_same_keys` exige que todos los perfiles expongan el mismo conjunto de
  claves (una clave que solo existe en un perfil revienta los otros con `AttributeError` en
  arranque). Esto ya rompió el build al añadir `nim_host` el 2026-09-29.
- `rag.py`: corto-circuito en `make_finder` cuando `rag_enabled = false` — no carga embedder, no lee
  los `.npz`, no hace llamadas de embedding, devuelve `knowledge = None`.

**La única pieza del retrieval que sobrevive: el guardia de precios.** `rag.py` ya tiene
`PRICE_WORDS` como una tupla de palabras clave — comparación de strings, latencia cero, ningún
modelo. "Nunca dar precios ni montos de becas" es un requisito duro, y hacerlo depender del
comportamiento *aprendido* de un 0.6B es peor que dejar diez líneas deterministas. El bucket de
rechazos se entrena igual (cinturón y tirantes), pero el guardia se queda.

`clean_for_tts`, el troceo en frases y las animaciones no cambian.

## Evaluación

**Quiz factual nuevo** (`finetune/quiz.py` + su conjunto de preguntas, ambos comiteados para poder
reejecutarlos cuando `cards.json` cambie). Se deriva de las fichas: por cada una, las preguntas con
fraseo apartado del dataset, más la lista de términos esperados extraídos del texto de la ficha
(nombres de laboratorios, materias, programa). Puntuación por coincidencia determinista de
términos.

Métricas, en orden de importancia: negativas correctas → contaminación cruzada → programas
inventados → precisión factual → forma hablada (umbrales en §Criterios de éxito).

**Tres puntos de comparación** para que los números signifiquen algo: el `qwen2.5:1.5b` + RAG
actual (lo que se reemplaza), el modelo afinado en cada nivel de cuantización, y el maestro de 9B
como techo.

**Latencia:** `bench.py`, que ya mide por etapa sobre turnos grabados, corrido **en la Pi real**.
La CPU ARM de la DGX sirve como proxy de *corrección* pero no de *latencia*: los núcleos Grace son
muchísimo más rápidos que los Cortex-A76, y cualquier tiempo medido ahí sería optimista y engañoso.
La Pi está disponible y se enciende para esta fase.

## Compuerta de go/no-go

Si el modelo cuantizado no pasa los umbrales de §Criterios de éxito, **se cae al plan A**: se
reactiva `rag_enabled` en el perfil `[pi]` con un buscador léxico barato (BM25 sobre las 147 fichas,
submilisegundo, cero modelos).

**Nada del trabajo se pierde en ese caso**, porque A usa exactamente el mismo modelo con la persona
destilada; solo se vuelve a encender el retrieval con un buscador distinto. Por eso B es seguro de
intentar: el peor caso es terminar en A con el modelo ya mejorado.

## Riesgos y desconocidos

| Riesgo | Cuándo se resuelve |
|---|---|
| torch con sm_121 en aarch64: no verificado todavía | Primera compuerta, antes de cualquier otra cosa |
| Contaminación cruzada a 0.6B: es la apuesta central de B | Quiz factual, tras el primer entrenamiento |
| Daño de la cuantización a los hechos memorizados | Quiz por nivel de cuantización |
| Desalineación de plantilla entrenamiento ↔ Ollama | Verificación byte a byte antes de evaluar |
| Markdown y listas del maestro contaminando el dataset | Compuertas de validación; se audita la muestra comiteada |
| Latencia real en la Pi: desconocida hasta medirla | `bench.py` en la Pi encendida |
| Modo *thinking* de Qwen3 | Se fija al definir la plantilla |

## Siguientes pasos fuera de este spec

- **STT**: el costo dominante en la Pi (~2.2–3.8 s). Sub-proyecto pendiente.
- **TTS**: Piper ya es rápido en la Pi; la calidad de voz quedó pendiente por decisión explícita.
- Servos del animatrónico y fusión de `feature/uao-rag` a `main`.
