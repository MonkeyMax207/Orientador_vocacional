# Llevar el modelo a la Raspberry Pi

Escrito para hacerlo con la Pi al frente, en orden, sin consultar nada más. Cada paso dice qué
esperar; si lo que ves no coincide, ahí está el problema y no más adelante.

Medido el 2026-11-16 en la DGX. Las cifras de aquí son contra las que vas a comparar allá.

---

## 0. Qué vas a instalar, y por qué así

Dos piezas que hacen trabajos distintos:

| Pieza | Qué aporta | Tamaño |
|---|---|---|
| `orienta-17b-q8` | **Cómo habla**: registro, brevedad, y la negativa de precios | 1,8 GB |
| `lexical.py` (BM25) | **Qué dice**: el hecho correcto del programa correcto | 0 — es código |

El modelo afinado solo, sin recuperar nada, erraba el **46,5%** de las respuestas, y su falla
dominante era inventar específicos: nombres de materias casi correctos, atribuidos al programa
equivocado. Con recuperación baja a **~26%**. Ninguna de las dos piezas hace el trabajo de la otra.

BM25 es recuperación léxica: un índice de palabras, **400 documentos en 27 ms, consulta en
0,11 ms**, Python puro. No carga ningún modelo y no hace peticiones HTTP. Por eso se puede usar
en la Pi sin la latencia que nos hizo descartar el RAG con embeddings.

---

## 1. Copiar

Del portátil o por USB:

```bash
# El modelo empaquetado (1,8 GB)
scp -r ~/orienta-pi juandi@raspberrypi.local:~/

# El repositorio
scp -r ~/Orientador_vocacional juandi@raspberrypi.local:~/
```

Se copia el GGUF **ya cuantizado**, no se vuelve a cuantizar allá: una cuantización repetida en
otra máquina no tiene por qué dar los mismos bytes, y entonces la cifra medida dejaría de
corresponder al modelo desplegado.

## 2. Registrar el modelo en Ollama

```bash
cd ~/orienta-pi && ollama create orienta-17b-q8 -f Modelfile
```

**Esperá:** `success`. El `Modelfile` ya trae la plantilla de chat y el `SYSTEM` correctos — los
generó `finetune/export.py` importándolos del código, y se verificó carácter por carácter que
coinciden con los del entrenamiento. Una plantilla que difiere en un token degrada la calidad
**sin producir ningún error**, así que no la edites a mano.

```bash
ollama list | grep orienta     # debe mostrar orienta-17b-q8, ~1.8 GB
```

## 3. Comprobar que la Pi tiene memoria

El modelo son 1,8 GB, más Whisper (`base`, int8) y Piper. En 8 GB alcanza, pero sin margen para
sorpresas:

```bash
free -h                        # disponible debería ser > 4 GB antes de arrancar
```

Si está ajustado, cerrá el escritorio gráfico: el robot no lo necesita.

## 4. Probar BM25 solo, sin modelo

Esto no necesita Ollama ni GPU, así que es la primera cosa que debe funcionar:

```bash
cd ~/Orientador_vocacional && python3 -m lexical "titulo de psicologia"
```

**Esperá** algo así:

```
400 documentos indexados en ~40 ms; consulta en ~0.2 ms

  [ 10.2] Psicología — Psicología
```

Si los milisegundos salen mucho más altos, no es grave: el índice se construye **una vez** al
arrancar, no por turno.

## 5. Probar el modelo solo

```bash
ollama run orienta-17b-q8 "hola, que carreras de ingenieria tienen?"
```

**Esperá** una respuesta corta y hablada, sin asteriscos ni listas con guiones. Si aparece
`<think>` o markup, la plantilla quedó mal: volvé al paso 2 sin editar nada.

Para medir velocidad:

```bash
ollama run --verbose orienta-17b-q8 "cuanto dura la carrera de psicologia?"
```

**Esperá** del orden de **9 tokens/s** (1,8 GB sobre ~17 GB/s de ancho de banda). Una respuesta
de 15 palabras son ~20 tokens: **~2,2 s**. Si sale muy por debajo, revisá que no esté usando
swap (`free -h` durante la respuesta).

## 6. Arrancar el robot completo

El perfil `[pi]` del `config.toml` ya quedó configurado:

```toml
[pi]
llm_model = "orienta-17b-q8"
rag_backend = "bm25"
rag_max_cards = 5
```

```bash
python3 main.py --profile pi
```

## 7. Qué medir, y contra qué

La latencia del LLM **no es el cuello de botella**: el STT domina con 2,2–3,8 s de un turno de
5–7 s. Medilo antes de optimizar nada:

```bash
python3 bench.py --profile pi
```

Y las cifras de calidad de la DGX, para comparar si corrés el quiz allá:

| Configuración | Exactitud estricta | Tasa de error |
|---|---|---|
| Afinado **sin** recuperación | 26,5% | 46,5% |
| Afinado + BM25, fp16 | 53,5% | 19,7% |
| **Afinado + BM25, Q8 + ponderacion (lo que lleva la Pi)** | **50,6%** | **19,9%** |
| Afinado + BM25, Q4 | 41,0% | 34,6% |

La configuracion de la Pi iguala a fp16 en tasa de error (19,9% contra 19,7%) con la mitad del
tamano. Lo que cerro la diferencia no fue el tamano del modelo sino ponderar el nombre del
programa en el indice: ver `PESO_PROGRAMA` en `lexical.py`.

Mas documentos recuperados **no** compra exactitud de forma confiable: k=3, k=5 y k=8 quedan
dentro de ~4 puntos entre si, aunque el recall del recuperador suba de 61,9% a 84,9%. El cuello
no es que falte informacion, es que el modelo no la aprovecha toda cuando la tiene.

**Esto no es calidad de producción todavía.** El umbral del proyecto era 80% de precisión
estricta y vamos en 50,6%. Dicho de otra forma: de cada cinco respuestas, una es falsa dicha con
seguridad. No es un robot listo para atender aspirantes solo. Sirve para demostrarlo y para medir desde algo real.

## 8. Lo que no cambió, y conviene verificar que siga así

Tres requisitos duros del proyecto. En 200 preguntas medidas:

- **Nunca dar precios ni montos de becas** — 0 de 200, en todas las configuraciones probadas
- **Nunca nombrar un programa que la UAO no ofrece** — 2 a 5 casos, sigue sin cumplirse del todo
- **Nunca hablar markup** — 6 a 11 de 200 incluyen dígitos, que el TTS lee mal

El de los dígitos se arregla mejor en la capa de voz (convertir números a palabras antes de
sintetizar) que pidiéndoselo al modelo: ya se lo pedimos en el prompt y aun así aparecen.

---

## Si algo falla

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `unsupported architecture "Qwen3ForCausalLM"` | Intentaste `ollama create` desde la carpeta de safetensors | Usá el `.gguf`, que es lo que trae `~/orienta-pi` |
| El modelo dice `<think>` o escribe listas | La plantilla del `Modelfile` no es la del entrenamiento | `ollama rm orienta-17b-q8` y repetir el paso 2 |
| Respuestas de 2 tok/s | Está en swap | Cerrá el escritorio; bajá `rag_max_cards` a 3 |
| El robot habla de otra carrera | La recuperación no recibió el tema | Verificá que el turno anterior se esté pasando: con el programa en la consulta el acierto de programa es 100%, sin él 40% |
| `No module named lexical` | Estás fuera de la raíz del repo | `cd ~/Orientador_vocacional` |
