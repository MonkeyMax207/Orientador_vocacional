# Sub-proyecto 1 de 3: LLM de la DGX servido con NIM (en vez de Ollama)

Primer paso de la migración al ecosistema NVIDIA para el pipeline de voz. Los otros dos
sub-proyectos (STT con NeMo, TTS con Riva/NeMo) tienen su propio spec después de medir este.

## Objetivo y prioridad

Reemplazar `qwen2.5:32b` vía Ollama por el NIM `nvidia-nemotron-nano-9b-v2-dgx-spark`
(contenedor ya descargado, **hecho específicamente para el hardware GB10 de esta DGX**) para
el perfil `dgx`. Prioridad: **calidad/consistencia primero** (menos alucinación de carreras y
datos de la UAO), latencia en segundo lugar — igual criterio que ya usa este proyecto
(ver tabla de cambios medidos en `README.md`).

## Por qué NIM y no TensorRT-LLM/vLLM directo

- Ya está descargado (`docker images` → `nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant`), no hay que construir ni cuantizar nada.
- Expone una API **compatible con OpenAI** (`/v1/chat/completions`), lo que acota el cambio en código a la capa de transporte de `brain.py`, sin tocar RAG ni el resto del pipeline.
- Alternativa descartada por ahora: TensorRT-LLM directo (contenedores `tensorrt-llm/release` ya están) — requiere construir y cuantizar un engine, mucho más esfuerzo/riesgo para la primera iteración. Queda como opción de optimización futura si NIM no da suficiente velocidad.

## Alcance

**Dentro:** el camino de chat del LLM en el perfil `dgx` (`brain.py`, `server.py`, `config.toml`).
**Fuera:** embeddings de RAG (siguen en Ollama vía `OllamaEmbedder`, confirmado en `rag.py:148,166` —
usan `cfg.ollama_host`/`cfg.embed_model`, independientes del modelo de chat). STT, TTS. Perfiles `pc`/`pi`
(siguen en Ollama sin cambios).

## Arquitectura

```
server.py (dgx) ──chat──▶ NIM (docker, puerto 8001, /v1/chat/completions)
                └─embeddings──▶ Ollama (puerto 11434, sin cambios)
```

- El NIM corre en un contenedor Docker separado, puerto **8001** (11434 lo sigue usando Ollama
  para embeddings — no hay que apagar Ollama).
- `brain.py`: la clase `Brain` de hoy mezcla dos cosas: la lógica de persona/RAG (prompt, historial,
  herramienta `buscar_uao`) que **no cambia**, y el transporte HTTP hacia Ollama (NDJSON, `options`,
  `keep_alive`) que **sí cambia**. Se separa el transporte detrás de una interfaz mínima
  (`check()`, `warmup()`, `stream()` → produce tokens + tool_calls) con dos implementaciones:
  - `OllamaBackend` (código actual, sin tocar lógica, solo renombrado/movido).
  - `NimBackend` (nuevo): mismo `/v1/chat/completions` pero streaming SSE (`data: {...}\n\n`,
    sentinel `[DONE]`) en vez de NDJSON; sin `options` de Ollama (num_gpu/num_thread/keep_alive no
    aplican, el NIM ya viene configurado); el checkeo de salud usa `/v1/models` en vez de `/api/tags`.
  - El formato de `tool_calls` de OpenAI ya es el que `SEARCH_TOOL` usa hoy (Ollama lo imita), así
    que la herramienta `buscar_uao` no debería necesitar cambios — se verifica con pruebas reales.
- `config.toml`: en `[dgx]`, nuevas claves `llm_backend = "nim"` y `nim_host = "http://127.0.0.1:8001"`;
  `[pc]`/`[pi]` no declaran `llm_backend` → default `"ollama"` (sin cambios de comportamiento).
- `Brain.__init__` elige el backend según `llm_backend`; el resto de la clase (persona, RAG, historial)
  queda igual para los tres perfiles.

## Prerrequisitos (una vez, con internet)

1. Cuenta gratuita en `ngc.nvidia.com` → generar un API key.
2. `docker login nvcr.io` con ese key (el NIM hace un chequeo de licencia al arrancar aunque la imagen
   ya esté local).
3. Confirmar arranque manual: `docker run --rm --gpus all -p 8001:8000 nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant`
   y esperar a que loguee que el servidor está listo.

## Medición (mismo método que el resto del proyecto)

Comparar contra la base actual (`qwen2.5:32b` vía Ollama) usando lo que ya existe, sin escribir
herramientas nuevas:
- `uv run python brain.py dgx` — chat de texto, tiempo a primer token.
- `uv run python -m knowledge.replay dgx` — conversación completa simulada; revisar que solo
  nombre carreras reales y que use las fichas correctas (igual criterio que la tabla del README).

Si el NIM no mejora consistencia ni tiempos razonables, `llm_backend = "ollama"` revierte al instante
sin perder nada (Ollama y `qwen2.5:32b` siguen instalados).

## Riesgos / desconocidos a resolver en el plan

- Formato exacto de `tool_calls` en la respuesta streaming del NIM (SSE) puede diferir en detalle de
  Ollama; se descubre con una prueba real antes de dar por cerrada la integración.
- Uso de memoria: NIM (9B) + Ollama (embeddinggemma, pequeño) + lo que ya corre en la DGX deben caber
  en los 121 GB unificados — sobra margen amplio (hoy 114 GB libres), no debería ser problema.
- Nombre exacto del modelo que espera el campo `"model"` en el request al NIM (se obtiene de
  `GET /v1/models` una vez el contenedor esté arriba).

## Sub-proyectos siguientes (fuera de este spec, orden ya acordado)

2. **STT con NeMo** (parakeet-tdt-0.6b-v3 o canary-1b) reemplazando Whisper/faster-whisper en CPU ARM.
   Nada descargado todavía; instalar `nemo_toolkit[asr]` y medir contra la fila `pi`/`dgx` de la
   tabla de latencia del README.
3. **TTS con Riva/NeMo** (Magpie multilingüe o FastPitch+HiFi-GAN español) reemplazando Kokoro en CPU
   ARM. Nada descargado todavía; medir naturalidad y RTF contra Kokoro/Piper.

Cada uno se hace después de medir el anterior, mismo patrón que el resto de este proyecto.
