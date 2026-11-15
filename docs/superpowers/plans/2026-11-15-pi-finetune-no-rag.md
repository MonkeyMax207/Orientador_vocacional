# Pi autónoma con modelo afinado sin RAG — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un modelo de 0.6–1B afinado con el conocimiento de la UAO en los pesos, cuantizado a GGUF, que corre en una Raspberry Pi 5 de 8 GB sin retrieval y pasa los umbrales de calidad del spec.

**Architecture:** Destilación en la DGX: un maestro local (familia `qwen2.5` vía Ollama) genera ~11.000 ejemplos conversacionales anclados en `knowledge/chunks.json` y `cards.json`; compuertas deterministas los validan; fine-tuning completo de un modelo pequeño; export a GGUF; evaluación sobre el modelo **cuantizado**. La Pi solo sirve: sin embedder, sin fichas, prompt reducido a la persona.

**Tech Stack:** Python 3.11 + stdlib (`urllib`, `json`, `re`) para generación y validación; `transformers` + TRL `SFTTrainer` + torch cu130 en venv aparte (`~/.venvs/orienta-train`); llama.cpp para GGUF; Ollama para servir.

**Spec:** `docs/superpowers/specs/2026-11-15-pi-finetune-no-rag-design.md`

## Global Constraints

- Las dependencias de entrenamiento **nunca** entran a `pyproject.toml`: la Pi instala desde ahí. Viven en `~/.venvs/orienta-train` con `finetune/requirements-train.txt`.
- Los módulos importados por `test_logic.py` usan **solo stdlib** (`finetune/validate.py`, `finetune/schema.py`). `finetune/train.py` nunca se importa desde ahí.
- Toda clave nueva de `config.toml` va en `[common]` con override por perfil: `test_config_profiles_have_same_keys` exige paridad de claves.
- Forma hablada obligatoria en todo ejemplo generado: sin markdown, sin viñetas, sin emoji, números en palabras, 1–2 frases.
- Cero cifras de precios, matrícula o becas. Cero teléfonos o correos.
- Todo nombre `Ingeniería|Licenciatura|Tecnología X` debe pertenecer a la lista oficial de 30 programas.
- Toda métrica de evaluación se mide sobre el **GGUF cuantizado**, nunca sobre el checkpoint fp16.
- El maestro corre en la instancia paralela de Ollama (`OLLAMA_MODELS=/usr/share/ollama/.ollama/models OLLAMA_NUM_PARALLEL=8 OLLAMA_HOST=127.0.0.1:11435`). El NIM de NGC **no sirve**: la API key perdió permisos (2026-11-15).

## Review Focus

- **Precios escritos con palabras** ("tres millones de pesos") pasan el filtro de dígitos y violan el requisito duro. La validación necesita detección de números en palabras, no solo `\d`.
- **El maestro devuelve markdown o viñetas** pese a las instrucciones (lo hizo nemotron el 2026-09-29): la compuerta debe rechazar el ejemplo, no limpiarlo silenciosamente, para que el registro delate si un bucket entero sale mal.
- **Nombre de programa inventado o mal escrito** ("Ingeniería Aeroespacial", "Ingeniería en Mecatrónica"): la verificación contra los 30 oficiales debe normalizar acentos y mayúsculas antes de comparar.
- **Desalineación de plantilla entrenamiento ↔ Ollama**: hunde la calidad sin ningún error visible. Necesita una comparación byte a byte explícita.
- **Pregunta duplicada entre buckets**: infla el dataset y sesga el entrenamiento hacia un fraseo. La deduplicación debe cruzar buckets, no solo dentro de cada uno.

---

### Task 1: Cliente del maestro y elección del modelo

**Files:**
- Create: `finetune/teacher.py`
- Create: `finetune/requirements-train.txt`
- Test: `test_logic.py` (añadir al final, antes del `__main__`)

**Interfaces:**
- Produces: `teacher.ask_json(messages: list[dict], *, host: str, model: str, retries: int = 2, num_predict: int = 900) -> list | dict` — POST a `/api/chat` de Ollama, extrae el primer bloque JSON de la respuesta y lo parsea; reintenta ante JSON malformado. `teacher.ask_many(jobs: list[list[dict]], *, host, model, workers: int = 8) -> list` — ejecuta en `ThreadPoolExecutor`, devuelve resultados en el orden de entrada, con `None` donde falló.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_teacher_extracts_json_from_prose():
    # El maestro suele envolver el JSON en texto ("Aquí tienes: [...]").
    raw = 'Claro, aquí tienes:\n[{"p":"¿Qué labs?","r":"El Fab-Lab."}]\nEspero que sirva.'
    assert teacher._extract_json(raw) == [{"p": "¿Qué labs?", "r": "El Fab-Lab."}]

def test_teacher_retries_malformed_json_then_succeeds():
    respuestas = iter(['{"roto": ', '[{"p":"a","r":"b"}]'])
    teacher._post = lambda host, body: {"message": {"content": next(respuestas)}}
    assert teacher.ask_json([{"role": "user", "content": "x"}], host="http://x",
                             model="m", retries=2) == [{"p": "a", "r": "b"}]

def test_teacher_ask_many_keeps_order_and_marks_failures():
    teacher._post = lambda host, body: ({"message": {"content": "[]"}}
                                         if "ok" in body["messages"][0]["content"]
                                         else {"message": {"content": "basura"}})
    jobs = [[{"role": "user", "content": "ok1"}], [{"role": "user", "content": "malo"}],
            [{"role": "user", "content": "ok2"}]]
    assert teacher.ask_many(jobs, host="http://x", model="m", workers=2) == [[], None, []]
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `uv run python test_logic.py`
Expected: FAIL — `module 'finetune.teacher' has no attribute '_extract_json'`

- [ ] **Step 3: Implementar `finetune/teacher.py`**

Solo stdlib (`urllib`, `json`, `re`, `concurrent.futures`). `_post(host, body) -> dict` hace el POST a `/api/chat` (es el punto que los tests sustituyen). `_extract_json` busca el primer `[` o `{` y recorta hasta su cierre balanceado — el maestro añade prosa alrededor.

- [ ] **Step 4: Correr y verificar que pasan**

Run: `uv run python test_logic.py`
Expected: PASS, el contador total sube en 3.

- [ ] **Step 5: Elegir el maestro con una medición**

Correr el sondeo de rendimiento (`finetune/teacher.py --probe`) sobre `qwen2.5:32b`, `:14b` y `:7b` a concurrencia 8 y 16, y una muestra de calidad de cada uno sobre un chunk real.
**Regla de decisión, fijada ahora:** el modelo más grande cuyo rendimiento agregado complete los ~11.000 ejemplos en **menos de 4 h** y cuya muestra de calidad no contenga markdown, viñetas ni nombres de programa inventados. Medido: el 32B da 22,5 tok/s agregados a concurrencia 8 (≈10,5 h) → descartado salvo que escale a concurrencia 16.
Registrar la elección y los números en `finetune/MAESTRO.md`.

- [ ] **Step 6: Commit**

```bash
git add finetune/teacher.py finetune/requirements-train.txt finetune/MAESTRO.md test_logic.py
git commit -m "feat: teacher client for dataset distillation, with measured model choice"
```

---

### Task 2: Compuertas de validación

**Files:**
- Create: `finetune/validate.py`
- Test: `test_logic.py`

**Interfaces:**
- Consumes: la lista oficial de programas vía `rag.Knowledge.program_names()`; `NAME_RE` de `knowledge/replay.py`.
- Produces: `validate.check(example: dict, *, source: str, official: list[str]) -> list[str]` — devuelve la lista de motivos de rechazo (vacía = válido). `validate.dedup(examples: list[dict]) -> list[dict]` — elimina near-duplicados por conjunto de tokens normalizados de la pregunta, cruzando todos los buckets.

- [ ] **Step 1: Escribir los tests que fallan**

```python
OFICIALES = ["Ingeniería Mecatrónica", "Ingeniería Mecánica", "Psicología"]

def test_validate_rejects_invented_program():
    ej = {"p": "¿Tienen aeroespacial?", "r": "Sí, tenemos Ingeniería Aeroespacial."}
    assert "programa_inventado" in validate.check(ej, source="", official=OFICIALES)

def test_validate_accepts_official_program_despite_accents_and_case():
    ej = {"p": "¿Qué hay?", "r": "Tenemos ingenieria mecatronica."}
    assert "programa_inventado" not in validate.check(
        ej, source="ingeniería mecatrónica", official=OFICIALES)

def test_validate_rejects_price_written_in_words():
    # Review Focus #1: el filtro de dígitos no alcanza.
    ej = {"p": "¿Cuánto cuesta?", "r": "El semestre vale tres millones de pesos."}
    assert "precio" in validate.check(ej, source="", official=OFICIALES)

def test_validate_rejects_markdown_and_bullets():
    ej = {"p": "¿Qué labs?", "r": "Tiene:\n- **Fab-Lab**\n- Automática"}
    motivos = validate.check(ej, source="Fab-Lab y Automática", official=OFICIALES)
    assert "formato" in motivos

def test_validate_rejects_invented_contact():
    ej = {"p": "¿A quién llamo?", "r": "Llama al tres dos uno, o escribe a info@uao.edu.co."}
    assert "contacto" in validate.check(ej, source="", official=OFICIALES)

def test_validate_rejects_answer_unmoored_from_source():
    ej = {"p": "¿Qué labs tiene?", "r": "Tiene un reactor nuclear y un túnel de viento."}
    assert "sin_anclaje" in validate.check(
        ej, source="Fab-Lab para impresión 3D y Laboratorio de Automática", official=OFICIALES)

def test_dedup_crosses_buckets():
    # Review Focus #5: la misma pregunta generada por dos buckets distintos.
    ejs = [{"p": "¿Qué laboratorios tiene Mecatrónica?", "r": "a", "bucket": "fichas"},
           {"p": "¿qué laboratorios tiene mecatronica?", "r": "b", "bucket": "chunks"},
           {"p": "¿Hay equipo de fútbol?", "r": "c", "bucket": "chunks"}]
    assert len(validate.dedup(ejs)) == 2
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `uv run python test_logic.py`
Expected: FAIL — `finetune.validate` no existe.

- [ ] **Step 3: Implementar `finetune/validate.py`**

Solo stdlib. Motivos exactos que devuelve: `"programa_inventado"`, `"precio"`, `"formato"`, `"contacto"`, `"sin_anclaje"`, `"longitud"`. Normalizar quitando acentos y pasando a minúsculas antes de comparar nombres. Para `"precio"`, combinar `PRICE_WORDS` de `rag.py` con una lista de números en palabras (`mil`, `millón`, `millones`…) además de `\d`. Para `"sin_anclaje"`, exigir un piso de solapamiento de tokens de contenido entre `r` y `source`.

- [ ] **Step 4: Correr y verificar que pasan**

Run: `uv run python test_logic.py`
Expected: PASS, el contador sube en 7.

- [ ] **Step 5: Commit**

```bash
git add finetune/validate.py test_logic.py
git commit -m "feat: deterministic validation gates for the distilled dataset"
```

---

### Task 3: Generación del dataset

**Files:**
- Create: `finetune/make_dataset.py`
- Create: `finetune/schema.py`
- Modify: `.gitignore` (añadir `finetune/data/`)
- Test: `test_logic.py`

**Interfaces:**
- Consumes: `teacher.ask_many`, `validate.check`, `validate.dedup`.
- Produces: `schema.to_chat(example: dict) -> dict` — convierte un par `{p, r}` o un diálogo al formato de entrenamiento `{"messages": [{"role": ...}]}`. `make_dataset.main(profile_host, model, out_dir)` — escribe `train.jsonl`, `quiz.jsonl` y `rechazos.jsonl`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_schema_to_chat_single_turn_has_system_and_masks_nothing_extra():
    d = schema.to_chat({"p": "¿Qué labs?", "r": "El Fab-Lab."})
    assert [m["role"] for m in d["messages"]] == ["system", "user", "assistant"]
    assert d["messages"][-1]["content"] == "El Fab-Lab."

def test_schema_to_chat_multiturn_preserves_order():
    d = schema.to_chat({"turnos": [{"u": "Hola", "a": "¡Hola!"}, {"u": "¿Y labs?", "a": "El Fab-Lab."}]})
    assert [m["role"] for m in d["messages"]] == ["system", "user", "assistant", "user", "assistant"]

def test_split_holds_out_phrasings_not_facts():
    # Todos los hechos van a entrenamiento; lo apartado son redacciones.
    ejs = [{"p": f"pregunta {i}", "r": "x", "fact_id": fid}
           for fid in ("a", "b") for i in range(10)]
    train, quiz = make_dataset.split_by_phrasing(ejs, quiz_frac=0.2)
    assert {e["fact_id"] for e in train} == {"a", "b"}
    assert {e["fact_id"] for e in quiz} == {"a", "b"}
    assert not ({e["p"] for e in train} & {e["p"] for e in quiz})
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `uv run python test_logic.py`
Expected: FAIL — `finetune.schema` no existe.

- [ ] **Step 3: Implementar `finetune/schema.py` y `finetune/make_dataset.py`**

El `system` de entrenamiento es la persona **corta** (sin `RAG_RULES`, sin la lista de programas: el modelo los va a saber). Buckets y cantidades exactas del spec §Dataset. Cada ejemplo lleva `fact_id` (el `id` de la ficha o chunk de origen) y `bucket`. Los rechazados se reintentan una vez y, si vuelven a fallar, se escriben en `rechazos.jsonl` con su motivo. Ruido de STT aplicado al ~20% de las entradas.

- [ ] **Step 4: Correr y verificar que pasan**

Run: `uv run python test_logic.py`
Expected: PASS, el contador sube en 3.

- [ ] **Step 5: Corrida piloto y revisión a ojo**

Run: `uv run python -m finetune.make_dataset --limit 40 --out finetune/data/piloto`
Expected: ≥30 ejemplos válidos; leer 10 a mano y confirmar español hablado natural, fraseo variado, nada inventado. Revisar `rechazos.jsonl`: si un bucket rechaza >30%, el prompt de ese bucket está mal — corregirlo antes de escalar.

- [ ] **Step 6: Corrida completa**

Run: `uv run python -m finetune.make_dataset --out finetune/data`
Expected: ~11.000 en `train.jsonl`, el quiz apartado en `quiz.jsonl`, tasa de rechazo global <15%.

- [ ] **Step 7: Commit**

```bash
git add finetune/make_dataset.py finetune/schema.py .gitignore test_logic.py finetune/data/muestra.jsonl
git commit -m "feat: generate and validate the distilled UAO dataset"
```

---

### Task 4: Entrenamiento

**Files:**
- Create: `finetune/train.py`
- Create: `finetune/TEMPLATE.md`

**Interfaces:**
- Consumes: `finetune/data/train.jsonl`.
- Produces: checkpoint HF en `~/orienta-ft/<modelo>-e<N>/`, y en `finetune/TEMPLATE.md` la cadena exacta que rinde la plantilla de chat para un par de mensajes fijo (insumo de la Task 5).

- [ ] **Step 1: Verificar la compuerta de entorno**

Run: `~/.venvs/orienta-train/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"`
Expected: `True` y `(12, 1)`. Si falla, pasar al contenedor `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc4` antes de seguir.

- [ ] **Step 2: Implementar `finetune/train.py`**

`transformers` + `SFTTrainer` de TRL. Valores fijados por el spec: fine-tuning **completo** (no LoRA), pérdida **solo sobre tokens del asistente**, `max_seq_length=1024`, 3–5 épocas con checkpoint por época, LR 1e-5–2e-5 con warmup y cosine, bf16. Modelo inicial `Qwen/Qwen3-0.6B` (instruct).

- [ ] **Step 3: Volcar la plantilla renderizada**

Run: `~/.venvs/orienta-train/bin/python -m finetune.train --dump-template`
Expected: escribe en `finetune/TEMPLATE.md` la cadena exacta (con tokens especiales visibles) para los mensajes de prueba fijos. Es el contrato que la Task 5 verifica.

- [ ] **Step 4: Entrenar**

Run: `~/.venvs/orienta-train/bin/python -m finetune.train --epochs 4`
Expected: la pérdida baja; un checkpoint por época en `~/orienta-ft/`.

- [ ] **Step 5: Commit**

```bash
git add finetune/train.py finetune/TEMPLATE.md
git commit -m "feat: full fine-tune of the small model on the distilled dataset"
```

---

### Task 5: Export a GGUF y paridad de plantilla

**Files:**
- Create: `finetune/export.sh`
- Create: `finetune/Modelfile`
- Test: `test_logic.py`

**Interfaces:**
- Consumes: el checkpoint de la Task 4 y `finetune/TEMPLATE.md`.
- Produces: tags de Ollama `orienta:0.6b-q8` (y las otras cuantizaciones), más `finetune/check_template.py` con `render_ollama(messages, host, model) -> str`.

- [ ] **Step 1: Escribir el test de paridad que falla**

```python
def test_training_template_matches_ollama_rendering():
    # Review Focus #4: una diferencia de un token hunde la calidad sin error visible.
    esperado = check_template.load_expected("finetune/TEMPLATE.md")
    rendido = check_template.render_ollama(check_template.FIXED_MESSAGES,
                                            host="http://127.0.0.1:11434", model="orienta:0.6b-q8")
    assert rendido == esperado
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run python test_logic.py`
Expected: FAIL — el modelo `orienta:0.6b-q8` no existe todavía.

- [ ] **Step 3: Implementar `finetune/export.sh` y el `Modelfile`**

`convert_hf_to_gguf.py` de llama.cpp → `llama-quantize` para `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M` → `ollama create orienta:0.6b-<q> -f Modelfile`. La plantilla del `Modelfile` se escribe para reproducir exactamente `finetune/TEMPLATE.md`.

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run python test_logic.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/export.sh finetune/Modelfile finetune/check_template.py test_logic.py
git commit -m "feat: export the fine-tune to GGUF with verified template parity"
```

---

### Task 6: Servido sin retrieval en el perfil pi

**Files:**
- Modify: `config.toml` (`[common]` y `[pi]`)
- Modify: `rag.py` (`make_finder`)
- Test: `test_logic.py`

**Interfaces:**
- Produces: `make_finder(cfg)` devuelve `(None, finder)` cuando `cfg.rag_enabled` es falso, donde `finder` solo aplica el guardia de precios.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_make_finder_disabled_loads_no_embedder_and_no_knowledge():
    cfg = SimpleNamespace(rag_enabled=False, ollama_host="http://no-existe:1",
                           embed_model="x", rag_threshold=0.3, rag_max_cards=0, rag_gate=0.33,
                           recommend_after_turns=8)
    knowledge, find = rag.make_finder(cfg)   # no debe hacer ninguna llamada HTTP
    assert knowledge is None
    assert find("Me gusta el fútbol", "") == []

def test_price_guard_survives_with_rag_disabled():
    cfg = SimpleNamespace(rag_enabled=False, ollama_host="http://no-existe:1",
                           embed_model="x", rag_threshold=0.3, rag_max_cards=0, rag_gate=0.33,
                           recommend_after_turns=8)
    _, find = rag.make_finder(cfg)
    assert [c["id"] for c in find("¿Cuánto cuesta el semestre?", "")] == ["precios"]

def test_profiles_still_share_keys_with_rag_enabled():
    assert set(vars(config.load_config("pi"))) == set(vars(config.load_config("pc")))
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `uv run python test_logic.py`
Expected: FAIL — `rag_enabled` no existe.

- [ ] **Step 3: Implementar**

`config.toml`: `rag_enabled = true` en `[common]`, `false` en `[pi]`, y `llm_model` de `[pi]` al tag del modelo afinado. `rag.py`: corto-circuito al inicio de `make_finder` — sin embedder, sin `.npz`, sin llamadas; el finder solo chequea `PRICE_WORDS` y devuelve `[PRICE_CARD]` o `[]`. Con `knowledge=None`, `brain.py` ya deja el prompt solo con la persona (no se toca).

- [ ] **Step 4: Correr y verificar que pasan**

Run: `uv run python test_logic.py`
Expected: PASS, el contador sube en 3.

- [ ] **Step 5: Commit**

```bash
git add config.toml rag.py test_logic.py
git commit -m "feat: serve the pi profile without retrieval, keeping the price guard"
```

---

### Task 7: Quiz factual y evaluación

**Files:**
- Create: `finetune/quiz.py`
- Test: `test_logic.py`

**Interfaces:**
- Consumes: `finetune/data/quiz.jsonl`, los tags de Ollama de la Task 5, `knowledge/replay.py`.
- Produces: `quiz.score(answer: str, expected_terms: list[str], *, program: str, official: list[str]) -> dict` con las claves `aciertos`, `contaminacion_cruzada`, `inventado`, `formato_ok`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_quiz_scores_expected_terms():
    r = quiz.score("Tiene el Fab-Lab y el laboratorio de Automática.",
                    ["Fab-Lab", "Automática"], program="Ingeniería Mecatrónica", official=OFICIALES)
    assert r["aciertos"] == 1.0 and not r["contaminacion_cruzada"]

def test_quiz_detects_cross_contamination():
    # La métrica central de este plan: atribuir un hecho al programa equivocado.
    r = quiz.score("Eso es del laboratorio de Mecánica.", ["Fab-Lab"],
                    program="Ingeniería Mecatrónica", official=OFICIALES)
    assert r["contaminacion_cruzada"] is True

def test_quiz_flags_invented_program_and_bad_format():
    r = quiz.score("- **Ingeniería Aeroespacial** tiene de todo.", ["Fab-Lab"],
                    program="Ingeniería Mecatrónica", official=OFICIALES)
    assert r["inventado"] and not r["formato_ok"]
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `uv run python test_logic.py`
Expected: FAIL — `finetune.quiz` no existe.

- [ ] **Step 3: Implementar `finetune/quiz.py`**

Puntuación determinista por coincidencia de términos, sin juez LLM. `--model <tag>` obligatorio y **debe ser un tag de Ollama**, no una ruta a un checkpoint: así es imposible medir el fp16 por accidente (Review Focus).

- [ ] **Step 4: Correr y verificar que pasan**

Run: `uv run python test_logic.py`
Expected: PASS, el contador sube en 3.

- [ ] **Step 5: Correr la evaluación completa**

Run: `uv run python -m finetune.quiz --model orienta:0.6b-q8` y repetir con `q6_K`, `q5_K_M`, `q4_K_M`, más el baseline `qwen2.5:1.5b` con RAG.
Run: `uv run python -m knowledge.replay pi`
Expected: comparar contra los umbrales del spec §Criterios de éxito.

- [ ] **Step 6: Medir latencia en la Pi real**

Requiere la Pi encendida. Instalar el GGUF elegido (`ollama create`), correr `uv run python bench.py pi`.
Expected: registrar por etapa; el LLM debería bajar respecto a la fila `pi` del README por el prompt corto y la ausencia de embedder.

- [ ] **Step 7: Decidir la compuerta y documentar**

Si pasa: registrar la tabla medida en `README.md` siguiendo el estilo de las otras tablas. Si falla (contaminación >15%, o cualquier negativa incorrecta, o programas inventados >0): activar el plan A — `rag_enabled = true` en `[pi]` con buscador léxico BM25 — y anotar en el README por qué, igual que la fila "Rejected: Whisper `initial_prompt`".

- [ ] **Step 8: Commit**

```bash
git add finetune/quiz.py README.md config.toml test_logic.py
git commit -m "feat: factual quiz harness and measured go/no-go for the no-RAG Pi model"
```

---

## Notas de auto-revisión

- **Cobertura del spec:** Dataset → Tasks 1–3. Entrenamiento → Task 4. Export y cuantización → Task 5. Cambio de servido → Task 6. Evaluación y compuerta → Task 7. El guardia de precios queda cubierto por el test de la Task 6.
- **Consistencia de nombres:** `validate.check`/`validate.dedup`, `teacher.ask_json`/`ask_many`, `schema.to_chat`, `make_dataset.split_by_phrasing`, `quiz.score` se usan con la misma firma donde reaparecen.
- **Review Focus:** los cinco modos de falla tienen test en la tarea dueña del código (precios en palabras y markdown → Task 2; nombres normalizados → Task 2; plantilla → Task 5; duplicados entre buckets → Task 2).
- **Desviación del spec ya conocida:** el maestro es `qwen2.5` vía Ollama y no el NIM de nemotron-9b, porque la API key de NGC perdió permisos. El spec quedó escrito antes de descubrirlo; la Task 1 fija el reemplazo con una medición.
