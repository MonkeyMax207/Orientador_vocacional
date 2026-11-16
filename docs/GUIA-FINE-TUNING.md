# Guía de fine-tuning: meter el conocimiento de la UAO dentro de un modelo pequeño

Para rehacer esto desde cero por tu cuenta. Explica **por qué** cada paso existe, no solo qué
comando correr, porque cuando algo falle vas a necesitar el porqué para decidir.

Hermana de `GUIA-DE-PRUEBAS.md`, que cubre el agente ya funcionando. Diseño y plan completos en
`docs/superpowers/specs/2026-11-15-pi-finetune-no-rag-design.md` y
`docs/superpowers/plans/2026-11-15-pi-finetune-no-rag.md`.

---

## 0. La idea, en una página

**El objetivo:** que la Raspberry Pi 5 corra el agente sola, sin internet y sin la DGX, para poder
venderlo como solución económica. Eso obliga a un modelo de 0.6–1B, que es lo que la Pi aguanta.

**El problema:** un modelo de ese tamaño no sabe nada de la UAO. Hay dos formas de que lo sepa:

| | Cómo | Costo |
|---|---|---|
| **RAG** (lo que teníamos) | Buscar la ficha relevante y pegarla en el prompt en cada turno | Una llamada al embedder + ~60 palabras extra por turno. En la Pi eso fue 2,6–6 s y hasta 16–17 s |
| **Fine-tuning** (esto) | Meter el conocimiento en los pesos entrenando | Horas de entrenamiento una vez, y el riesgo de que difumine los datos |

**La apuesta y su riesgo honesto:** a 0.6–1B los modelos aprenden muy bien el *estilo* y difuminan
los *hechos*. El modo de falla esperado es que confunda los laboratorios de Mecatrónica con los de
Mecánica, o que invente un programa. Por eso el quiz de la sección 6 no es opcional: es el
instrumento que decide si la apuesta sirvió. Si falla, el plan de salida es volver a RAG pero con
búsqueda léxica (BM25, submilisegundo, sin modelo) en vez del embedder neuronal.

**Qué es destilación:** un modelo grande (el *maestro*) escribe miles de ejemplos de conversación
basados en tu texto fuente, y el modelo pequeño (el *estudiante*) se entrena con ellos. El pequeño
hereda el buen español del grande. Es la cura para lo acartonado que suele sonar un 0.6B.

---

## 1. La máquina: lo que hay que saber de la DGX Spark

Media docena de las trampas de esta guía vienen de aquí, así que vale la pena entenderlo antes.

**La memoria es unificada.** CPU y GPU comparten un solo pozo de 121 GB. No hay VRAM aparte. Eso
significa que:

- "memoria disponible" en `/proc/meminfo` es **la única cifra real** que decide si un modelo cabe.
- Lo que reserva CUDA **no aparece como RSS de ningún proceso**. Si mirás `ps` o `top`, Ollama con
  un modelo de 19 GB cargado se ve con 23 MB.
- `nvidia-smi --query-gpu=memory.used` responde **`[N/A]`** en esta máquina. Estás ciego por ese
  camino; usá `/proc/meminfo`.

**Consecuencia grave:** si se agota la memoria, el OOM killer del kernel no encuentra a quién
atribuirle los 120 GB y se pone a matar demonios del escritorio de 400 KB. La máquina **se
congela** en vez de matar al culpable. Nos pasó (sección 8).

**Ollama no sabe medirse.** Su log lo confesó:

```
NVML not supported for memory query, using system memory (total=130662940672, available=123650174976)
```

Presupuesta contra la memoria total del sistema y descarga todas las capas a la GPU, sin saber que
compite con algo más.

**La arquitectura es ARM (aarch64) y la GPU es Blackwell, compute capability `sm_121`.** Los builds
viejos de PyTorch no traen kernels para ella: el contenedor `pytorch:24.06` que hay descargado es
de 2024 y no sirve.

**Calienta.** Las zonas ACPI llegan a 95 °C bajo carga y el punto de apagado crítico es 104 °C.

---

## 2. El entorno, desde cero

El entrenamiento vive **fuera** del proyecto. Esto no es opcional: la Pi instala sus dependencias
desde `pyproject.toml`, y un `uv sync` que empiece a bajar PyTorch en una Raspberry es inaceptable.

```bash
# uv, si no está
curl -LsSf https://astral.sh/uv/install.sh | sh

# venv de entrenamiento, aparte del .venv del proyecto
uv venv ~/.venvs/orienta-train --python 3.11
uv pip install --python ~/.venvs/orienta-train/bin/python torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python ~/.venvs/orienta-train/bin/python transformers datasets accelerate sentencepiece protobuf numpy
```

**Verificá la compuerta antes de seguir.** Si esto no da `True` y `(12, 1)`, nada de lo demás va a
correr y hay que ir a un contenedor reciente (`tensorrt-llm/release:1.3.0rc4` o
`nvcr.io/nvidia/pytorch:25.xx-py3`):

```bash
~/.venvs/orienta-train/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# esperado: 2.14.1+cu130 True (12, 1)
```

Ojo: `uv venv` **no instala `pip`**. Hay que usar `uv pip install --python <ruta>`, no
`<venv>/bin/pip`.

---

## 3. El maestro: elegirlo midiendo, no opinando

El maestro es un modelo grande local que escribe los ejemplos. Usamos `qwen2.5` por Ollama: está
descargado, es local y no necesita autenticación.

> **Por qué no el NIM de NVIDIA:** era el plan original, pero su API key de NGC perdió permisos y
> los pesos que había bajado se perdieron porque el contenedor corrió con `--rm` **sin montar un
> volumen para el caché**. Si volvés a usar NIMs: montá el caché (`-v ~/.cache/nim:/opt/nim/.cache`)
> y guardá la key en `.env` (ya está en `.gitignore`), que `docker run --env-file .env` la lee.

**Lo que más importa del maestro no es su calidad, es su velocidad agregada**, porque hay que
generar ~11.000 ejemplos. Y aquí hay una trampa enorme:

**El servicio de Ollama del sistema atiende las peticiones en serie.** Medido: 8 peticiones
concurrentes dieron 10,4 tok/s agregados, idéntico a una sola. Las latencias escalonadas (19 s a
154 s) lo delatan. Hay que levantar una instancia con paralelismo real:

```bash
OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
OLLAMA_NUM_PARALLEL=8 \
OLLAMA_HOST=127.0.0.1:11435 \
ollama serve
```

`OLLAMA_MODELS` apuntando a los blobs del sistema evita volver a descargar los modelos (el
directorio es legible por tu usuario).

**Lo que midió esto en la práctica** (generar ~660k tokens de salida):

| Maestro | Concurrencia | Agregado | Dataset completo |
|---|---|---|---|
| `qwen2.5:32b` | 8, servicio del sistema | 10,4 tok/s | ~23 h |
| `qwen2.5:32b` | 8, instancia paralela | 22,5 tok/s | ~10 h |
| **`qwen2.5:14b`** | **16, instancia paralela** | **83,3 tok/s** | **~2,2 h** |

El 14B ganó. Razón física: un modelo de 19 GB está limitado por el ancho de banda de memoria
(~273 GB/s), así que uno más pequeño va mucho más rápido. Y para esta tarea —reescribir un texto
que ya le das en preguntas y respuestas habladas— un 14B alcanza de sobra.

**La regla de decisión, fijada antes de medir:** el modelo más grande que termine el dataset en
menos de 4 h y cuya muestra no traiga markdown, viñetas ni programas inventados.

```bash
uv run python -m finetune.teacher --probe qwen2.5:14b 16
```

---

## 4. El dataset: aquí se gana o se pierde

### El principio

**Inyectar conocimiento en un modelo sub-1B no se logra con volumen de texto, se logra con
diversidad de fraseo por hecho.** Un dato visto una vez en una redacción no se memoriza; el mismo
dato preguntado de veinte formas sí. Nuestras 34.000 palabras de fuente (253 chunks) no se entrenan
tal cual: se expanden a ~11.000 ejemplos.

### Los buckets

| Bucket | Qué es | Cómo se genera |
|---|---|---|
| `ficha` | QA sobre las 147 fichas curadas | 3 pasadas × 10 preguntas, variando el registro |
| `chunk` | QA sobre los 253 chunks de texto completo | 3 pasadas × 8, con regla estricta "solo lo que dice el texto" |
| `dialogo` | Conversaciones vocacionales de 8 turnos | Escuchar → reaccionar → **una** pregunta → recomendar solo si ya conoce a la persona |
| `neg` | Negativas y rechazos | precios, carreras que no existen, datos de contacto, fuera de tema, transcripción ininteligible |

Más **ruido de STT** aplicado al ~20% de las entradas: la entrada real viene de Whisper en una Pi,
y el proyecto ya aprendió por las malas que dice cosas como "Pugar fútbol". Entrenar con entradas
sucias es entrenar donde el modelo vive.

### Las compuertas: la mitad del trabajo

**Un dataset sucio enseña a alucinar.** Cada ejemplo generado pasa por verificaciones
deterministas, sin juez LLM (barato y reproducible). Los motivos de rechazo y por qué existe cada
uno:

| Motivo | Qué atrapa | Por qué importa |
|---|---|---|
| `programa_inventado` | Nombres de carrera que la UAO no ofrece | Es la línea roja del proyecto. **Atrapó 481** |
| `precio` | Un monto, incluso escrito en palabras ("tres millones de pesos") | El filtro de dígitos no alcanza. Decir "precios" no es dar un precio: el rechazo honesto debe pasar |
| `formato` | Markdown, viñetas, emoji, dígitos | El maestro los emite igual aunque el prompt los prohíba. Todo esto se lo come el TTS |
| `contacto` | Correos, teléfonos, URLs inventados | **Atrapó 103** |
| `sin_anclaje` | Respuestas que no comparten vocabulario con su texto fuente | Atrapa al maestro inventando |
| `longitud` | Respuestas de más de 60 palabras | Son frases habladas |
| `forma_invalida` | Ejemplos con claves raras o respuesta vacía | Pasaban todo lo demás y entrenaban al modelo a no decir nada |

**Dos detalles sutiles que cuestan caro si se hacen mal:**

1. **El bucket de negativas TIENE que nombrar carreras que no existen** para poder negarlas. Un
   chequeo ciego de nombres oficiales lo haría imposible. Por eso existe `allow_absent`, que
   invierte la compuerta y exige una negación explícita.
2. **La lista de carreras ausentes se calcula**, filtrando candidatas contra las oficiales. Yo la
   escribí a mano e incluí **Derecho**, que la UAO sí ofrece: habría entrenado al modelo a negar un
   programa real.

### El split va por fraseos, no por hechos

Todos los 30 programas tienen que estar en entrenamiento: el modelo debe conocerlos. Lo que se
aparta para el quiz es un conjunto de **preguntas con redacción no vista** sobre esos mismos hechos.
Apartar hechos mediría generalización a conocimiento que nunca le diste, que no es la pregunta.

### Correrlo

Siempre un piloto primero y **leé la salida a ojo**. Es el seguro más barato que existe: diez
ejemplos leídos a mano te ahorran horas.

```bash
# piloto: ~25 llamadas, 5 min
uv run python -m finetune.make_dataset --limit 6 --out finetune/data/piloto --model qwen2.5:14b --workers 16 --batch 16

# revisá la muestra y, sobre todo, los rechazos por bucket
python3 -c "
import json, collections
r=[json.loads(l) for l in open('finetune/data/piloto/rechazos.jsonl')]
print(collections.Counter(m for x in r for m in x['motivos']))"

# completo: ~1.750 llamadas, ~3 h
uv run python -m finetune.make_dataset --out finetune/data --model qwen2.5:14b \
  --workers 16 --batch 32 --pasadas 3 --dialogos 400 --negativas 30

# rehacer train/quiz desde lo ya generado, sin volver a llamar al maestro
uv run python -m finetune.make_dataset --out finetune/data --solo-escribir
```

**Si un bucket rechaza más del 30%, el prompt de ese bucket está mal.** Arreglalo antes de escalar.
Una tasa global de 20–25% es normal.

El generador **escribe incrementalmente y reanuda por id de trabajo**: si se corta a la hora tres,
volver a lanzar el mismo comando sigue donde quedó. En corridas de horas esto no es un lujo.

---

## 5. El entrenamiento

### Fine-tuning completo, no LoRA

LoRA de rango bajo es excelente para *estilo* y pobre para *memorizar hechos nuevos*: la
actualización vive en un subespacio de rango bajo, que alcanza para "compórtate así" y no para
"almacena estos 147 grupos de datos". Y aquí la memoria no es restricción: un 0.6B completo con
AdamW son ~10 GB de pesos, gradientes y estados del optimizador (medido: ~27 GB con activaciones).

### Lo único que de verdad hay que entender: el enmascarado

La pérdida se calcula **solo sobre los tokens del asistente**. Si no, el modelo aprende a generar
también las preguntas del estudiante. Es el error más común y el más silencioso de todo el
fine-tuning supervisado.

Vive aislado en `finetune/masking.py`, con funciones puras y probadas, porque es el corazón de la
corrección. Una versión temprana mía etiquetaba **solo el último turno** de cada diálogo: de ocho
turnos se entrenaba uno, perdiendo 7/8 de la señal.

Un número para vigilar: la **fracción de tokens con etiqueta**, que el script imprime al arrancar.
Con nuestro prompt de persona (~400 tokens) y respuestas de ~20, **5% es normal**. Si da ~0%, la
plantilla y el tokenizador no encajan y vas a entrenar horas aprendiendo nada; por eso aborta
debajo de 2%.

### La trampa de la plantilla

La plantilla de chat del entrenamiento tiene que ser **byte a byte idéntica** a la que Ollama
aplica al servir. Si difieren en un token o un salto de línea, la calidad se cae de formas que
parecen "el fine-tuning no funcionó".

Y una específica de Qwen3: **`enable_thinking=False` no elimina el bloque de razonamiento, inserta
uno vacío** (`<think>\n\n</think>`). Un robot que habla todo lo que escribe no puede aprender a
emitir etiquetas, así que usamos una plantilla mínima propia con la misma estructura
`<|im_start|>`/`<|im_end|>` que el modelo ya conoce, sin el andamiaje. Queda registrada en
`finetune/TEMPLATE.md`, que es el contrato con el `Modelfile`.

```bash
# volcar la plantilla exacta (el contrato)
~/.venvs/orienta-train/bin/python -m finetune.train --dump-template

# entrenar
~/.venvs/orienta-train/bin/python -m finetune.train \
  --data finetune/data/train.jsonl --out ~/orienta-ft --epochs 4 --bs 8 --excluir neg:fuera
```

`--excluir neg:fuera` deja fuera el bucket de preguntas ajenas al tema: sin texto fuente no hay
chequeo de anclaje y el maestro inventó charla incoherente. Entrenar persona con ejemplos flojos
degrada justo lo que ya sonaba bien.

**Checkpoints cada 200 pasos** (`save_steps`), no por época. Un crash a los 58 minutos se llevó una
hora completa porque el primer checkpoint de época no había alcanzado a escribirse.

---

## 6. Export y evaluación

### La cuantización daña los hechos

Cuantizar afecta los *hechos memorizados* mucho más que el *estilo*, y aquí todo el conocimiento
vive en los pesos. **Regla sin excepciones: toda cifra de evaluación sale del GGUF cuantizado que
va a correr en la Pi, nunca del checkpoint en fp16.**

Eso invierte la costumbre de "Q4 por defecto": a 0.6B los tamaños son irrelevantes frente a los
8 GB de la Pi (Q8_0 ≈ 640 MB), así que se empieza arriba y se baja solo si la latencia obliga,
midiendo el quiz en cada nivel.

```bash
# HF -> GGUF -> cuantizaciones -> Ollama
bash finetune/export.sh                       # convert_hf_to_gguf.py + llama-quantize
ollama create orienta:0.6b-q8 -f finetune/Modelfile
curl -s http://127.0.0.1:11434/v1/models       # verificar que está
uv run python test_logic.py                    # incluye la paridad de plantilla
```

### El quiz

Preguntas con fraseo no visto, puntuadas por coincidencia determinista de términos. **Sin juez LLM:
barato, reproducible, y no nos va a adular.** Las métricas en orden de importancia:

| Métrica | Umbral | Por qué |
|---|---|---|
| Negativas correctas (precios, carreras inexistentes, contacto) | **100%** | Línea ética, no se negocia |
| Contaminación cruzada (un hecho atribuido al programa equivocado) | **<5%**; >15% invalida el enfoque | Es *el* modo de falla de meter hechos en pesos |
| Programas inventados en el replay de 11 turnos | **0** | — |
| Precisión factual | ≥80% y no más de 10 puntos bajo el baseline con RAG | — |

```bash
uv run python -m finetune.quiz --model orienta:0.6b-q8   # y repetir con q6/q5/q4
uv run python -m knowledge.replay pi
uv run python bench.py pi                                # latencia, en la Pi encendida
```

La CPU de la DGX sirve como proxy de *corrección* pero **no de latencia**: sus núcleos son
muchísimo más rápidos que los Cortex-A76 de la Pi y cualquier tiempo medido ahí es engañoso.

---

## 7. Los guardias: no son opcionales

Dos peligros de máquina, una sola respuesta (congelar el trabajo pesado con SIGSTOP y reanudar con
SIGCONT, que es el mismo mecanismo que usa `docker pause` por debajo).

```bash
uv run python -m finetune.thermal        # vigila temperatura Y memoria, loguea en logs/
```

| Umbral | Acción | De dónde sale |
|---|---|---|
| 90 °C | No arrancar trabajo nuevo; esperar a 85 °C | Medido en `animation_creator` |
| 94 °C | Congelar; reanudar a 85 °C | Apagado crítico a 104 °C |
| 100 °C | Matar (último recurso) | — |
| < 12 GB libres | Congelar | El OOM del 15/11 |

Y una verificación **previa** de memoria (`finetune/memory.py`): exige 33 GB disponibles antes de
reservar nada, y si no los hay dice qué hacer. Nada se pierde cuando congela, porque la generación
reanuda por id y el entrenamiento tiene checkpoints.

**Antes de entrenar, descargá los modelos de Ollama** (`ollama stop <modelo>` o
`systemctl restart ollama`). Son ~30 GB que no hacen falta: el dataset ya está generado.

---

## 8. Las trampas, todas las que pisamos

La parte más útil de esta guía. Cada una nos costó tiempo real.

| Síntoma | Causa | Lección |
|---|---|---|
| La máquina **se congela** y hay que apagarla a mano | OOM global: dos instancias de Ollama con `keep_alive=-1` (los modelos nunca se descargan) más el entrenamiento, en un pozo de 121 GB. El OOM killer no ve las reservas de CUDA y mata demonios del escritorio de 400 KB | Nunca `keep_alive=-1` en algo que no esté sirviendo. Vigilá `/proc/meminfo`, no `nvidia-smi` |
| Cero diálogos sobreviven, 795 rechazos por `longitud` | La compuerta medía los 8 turnos **concatenados** (136 palabras) contra un límite pensado para una respuesta suelta (60) | Las métricas por turno se miden por turno |
| `KeyError: 'r'` al final de una corrida de horas | El maestro devuelve claves distintas (`pregunta`/`respuesta`), o parte un turno en dos objetos. El chequeo validaba el nivel de arriba pero no la forma de cada turno | Validá la forma **antes** de usar, y en cada nivel de anidamiento |
| El modelo responde con un monólogo interno visible | Nemotron y Qwen3 son modelos de razonamiento híbrido | `/no_think` en el system prompt (Nemotron) o plantilla propia sin el bloque (Qwen3) |
| Se pierde una hora de entrenamiento en un crash | Checkpoints por época, y la primera época no había cerrado | `save_steps`, no `save_strategy="epoch"` |
| No se sabe por qué falló: no hay log | Los logs estaban en `/tmp`, que se borra al reiniciar | Logs en `logs/` del repo |
| La concurrencia no acelera nada | El servicio de Ollama del sistema encola en serie | Instancia aparte con `OLLAMA_NUM_PARALLEL` |
| El contenedor NIM no arranca: *"operation requires an API key"* | `docker login` alcanza para bajar la imagen, no para que el contenedor baje los pesos | La key va como variable de entorno (`--env-file .env`) |
| El NIM vuelve a descargar 10 GB en cada arranque | Corrió con `--rm` sin volumen para el caché | Montá `~/.cache/nim:/opt/nim/.cache` |
| Un comando se mata a sí mismo | `pkill -f <patrón>` donde el patrón aparece en la propia línea de comandos del comando | Matá por PID, o leé `/proc`. Dentro de un archivo `.sh` el patrón no está en la línea de comandos |
| Un script de bash se comporta de forma errática | Lo editaste mientras corría; bash lo lee por offsets de bytes | Detenelo, editalo, relanzalo |
| `uv venv` y después `<venv>/bin/pip` falla | `uv venv` no instala `pip` | `uv pip install --python <ruta-al-python>` |
| `TrainingArguments got an unexpected keyword 'warmup_ratio'` | transformers 5.x lo quitó | `warmup_steps`, calculado desde el total de pasos |
| El entrenamiento **arranca y se queda en 0 pasos**, proceso en estado `T` | El guardia lo congeló por presión de memoria. `SIGSTOP` no devuelve ni una página, así que el entrenamiento congelado seguía reteniendo sus 110 GB y la condición que lo congeló no podía ceder nunca | Congelar sirve contra el **calor** (el cómputo para, la temperatura baja), nunca contra la **memoria**. Ante memoria escasa hay que *soltar* (terminar lo que descarga su modelo), jamás congelar el trabajo principal. Revisá `awk '/^State:/' /proc/<pid>/status`: `T` es congelado, no lento |
| `NVRM: Out of memory [NV_ERR_NO_MEMORY]` en `dmesg` y el entrenamiento se traba | El lote era demasiado grande: `--bs 16` dejó 11 GB libres de 121 y el driver ya fallaba asignaciones internas | Bajá `--bs` y subí `--ga` en la misma proporción: el **lote efectivo** (`bs × ga`) no cambia, la matemática del entrenamiento es idéntica, y el pico de activaciones se parte a la mitad |
| La máquina se apaga en seco y el kernel **no registró nada** | Pérdida de energía, no software. Un trip térmico, un pánico o un OOM *siempre* dejan rastro; un corte eléctrico no deja ninguno porque no hay tiempo de escribir | Diagnosticá por ausencia: `last -x reboot shutdown` sin línea `shutdown` entre dos arranques = corte abrupto. `journalctl -b -1 -k \| tail` te dice si el kernel alcanzó a quejarse |
| Hay checkpoints perfectos en disco y aun así toca repetir horas de GPU | `trainer.train()` sin `resume_from_checkpoint` empieza de cero | Una corrida larga necesita la bandera de reanudar **antes** de necesitarla. El checkpoint trae pesos, momentos del optimizador, posición del scheduler y estado del RNG: retomar es la misma corrida continuada, y se nota porque la pérdida sigue la curva en vez de volver a ~5,6 |

---

## 9. La secuencia completa, de cero

```bash
# 1. entorno (sección 2)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.venvs/orienta-train --python 3.11
uv pip install --python ~/.venvs/orienta-train/bin/python torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python ~/.venvs/orienta-train/bin/python transformers datasets accelerate sentencepiece protobuf numpy
~/.venvs/orienta-train/bin/python -c "import torch; print(torch.cuda.get_device_capability())"   # (12, 1)

# 2. guardias primero, siempre (sección 7)
cd ~/Orientador_vocacional
uv run python -m finetune.thermal > logs/guard.log 2>&1 &

# 3. maestro con paralelismo real (sección 3)
OLLAMA_MODELS=/usr/share/ollama/.ollama/models OLLAMA_NUM_PARALLEL=8 \
  OLLAMA_HOST=127.0.0.1:11435 ollama serve > logs/ollama-par.log 2>&1 &
uv run python -m finetune.teacher --probe qwen2.5:14b 16

# 4. dataset: piloto, leerlo, completo (sección 4)
uv run python -m finetune.make_dataset --limit 6 --out finetune/data/piloto --model qwen2.5:14b --workers 16 --batch 16
uv run python -m finetune.make_dataset --out finetune/data --model qwen2.5:14b --workers 16 --batch 32 --pasadas 3

# 5. apagar el maestro y liberar memoria antes de entrenar (sección 7)
ollama stop qwen2.5:14b                          # descarga el modelo del servicio del sistema
OLLAMA_HOST=127.0.0.1:11435 ollama stop qwen2.5:14b   # y de la instancia paralela
ps -u $USER -o pid=,cmd= | grep "[o]llama serve"      # matá la paralela por PID, no con pkill -f
uv run python -c "from finetune import memory; print(memory.check())"   # (True, '...GB disponibles')

# 6. entrenar (sección 5)
~/.venvs/orienta-train/bin/python -m finetune.train --data finetune/data/train.jsonl \
  --out ~/orienta-ft --epochs 4 --bs 8 --excluir neg:fuera > logs/train.log 2>&1 &

# 7. export y evaluación (sección 6)
bash finetune/export.sh && ollama create orienta:0.6b-q8 -f finetune/Modelfile
uv run python -m finetune.quiz --model orienta:0.6b-q8
uv run python -m knowledge.replay pi
```

---

## 10. Cómo saber si funcionó

Pasó si: **100%** de negativas correctas, **cero** programas inventados en el replay,
contaminación cruzada **bajo 5%**, precisión factual **≥80%**, y la latencia en la Pi real mejora
respecto a la fila `pi` de `README.md`.

No pasó si la contaminación cruzada supera el 15% o cualquier negativa falla. En ese caso se activa
`rag_enabled = true` en el perfil `[pi]` con búsqueda léxica BM25, y **nada del trabajo se pierde**:
el modelo afinado sigue sirviendo igual, porque la persona destilada es la misma. Por eso este
enfoque era seguro de intentar — el peor caso es terminar con RAG y un modelo mejor.
