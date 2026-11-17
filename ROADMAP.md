# Roadmap

Estado al 2026-11-17. Escrito para trabajar desde la Raspberry sin tener que reconstruir el
contexto de cómo se llegó hasta acá.

Cada punto pendiente dice **por qué importa** y **cómo saber si quedó bien**, porque varias
veces en este proyecto se optimizó lo que era cómodo medir en vez de lo que fallaba.

---

## Hecho

### El cerebro

- **Modelo afinado** `orienta-17b-q8` (Qwen3-1.7B, Q8_0, 1,8 GB) con 13.372 ejemplos
  destilados de un maestro local. Aporta registro hablado, brevedad y negativas.
- **Grafo de programas** (`graph.py`): 30 carreras con duración, créditos, título, modalidad,
  SNIES, perfil, egresado, materias y laboratorios. Cobertura medida sobre los 30 programas:
  créditos, título, SNIES, perfil y egresado al 100%; duración 97%; modalidad y materias 73%;
  **laboratorios solo 43%**, que es el hueco más grande del grafo.
- **BM25 léxico** (`lexical.py`) con ponderación del nombre del programa y expansión por
  prefijo. 400 documentos en 27 ms, consulta en 0,43 ms.
- **Respuestas inmediatas** (`respuestas.py`): saludos, gracias, despedidas, "quién eres" y
  **precios**, sin pasar por el modelo. 0 s y sin riesgo de invención.
- **Foco de carrera pegajoso** que se suelta al nombrar otra, y que también escucha lo que
  responde el robot, no solo lo que dice la persona.

### La medición

- **Juez LLM** (`finetune/judge.py`): califica contra el texto fuente, sin ver la respuesta de
  referencia. Verificado contra cuatro casos conocidos antes de confiar en él.
- **Registro por turno** en `logs/conversacion-*.jsonl` con lo recuperado y los tiempos.
- **137 comprobaciones** en `test_logic.py`, que pasan en la DGX y en la Pi.

### El cuerpo

- **Firmware** no bloqueante (`firmware/oso_andino/`) con los límites calibrados originales,
  vida propia (parpadeo y mirada espontáneos) y hombre muerto en la boca.
- **Puente serie** (`esp32.py`) que trata la ausencia de hardware como normal.
- **Boca sincronizada con la voz** por amplitud, ~21 actualizaciones por segundo.

### La infraestructura

- Guardia térmico y de memoria para entrenamientos largos, probados en fuego.
- Reanudación desde checkpoint (un corte de luz costó 9 minutos en vez de 2 h 15).
- Exportación a GGUF con **verificación de paridad de plantilla**, que se niega a exportar si
  la plantilla de Ollama difiere de la del entrenamiento.

---

## Pendiente, por prioridad

### 1. Probar el animatrónico con hardware real

**Nada de `esp32.py` ni del firmware se ha probado contra servos.** Está escrito y con pruebas,
pero las pruebas solo cubren lo que se puede comprobar sin motores.

```bash
python3 probar_oso.py
```

**Qué mirar:** que los párpados no fuercen contra el tope, que la boca no zumbe cuando sigue la
voz, y que los gestos no se pisen con el habla. Si un párpado hace ruido, el límite en
microsegundos está mal y se corrige **en el firmware**, nunca desde Python.

**Riesgo conocido:** el mini servo de la boca a 21 actualizaciones por segundo puede
sobrecalentarse. Si pasa, subir `UMBRAL_BOCA` en `esp32.py` para que ignore cambios pequeños.

### 2. Subir la exactitud del 50,6%

El cuello **no es el tamaño del modelo** y tampoco falta de información: k=3, k=5 y k=8 dan
resultados dentro de 4 puntos mientras el recall del buscador sube de 61,9% a 84,9%. El modelo
no aprovecha todo lo que recibe.

Los caminos, de más a menos prometedor:

- **Ampliar el grafo.** Cada pregunta que responda el grafo es un turno instantáneo y sin
  invención. Hoy cubre 9 campos; faltan horarios, jornada, sede, convenios, intercambios,
  requisitos de admisión.
- **Rehacer el dataset** atando cada hecho a su carrera explícitamente. La sospecha es que las
  pasadas `esp*` enseñaron a *sonar* específico sin enseñar los específicos.
- **Un modelo más grande** solo si lo anterior no alcanza, y sabiendo que cuesta latencia.

**Cómo saber si funcionó:** `finetune/judge.py` sobre las respuestas guardadas. No usar la
"precisión factual" del quiz: está demostrado que califica respuestas incorrectas por encima de
respuestas ciertas pero incompletas.

### 3. Los tres programas inventados

Requisito duro incumplido: 3 de 200. El grafo ya entrega la lista oficial en preguntas de
catálogo, pero el modelo sigue inventando nombres en respuestas libres.

**Idea sin probar:** validar la respuesta antes de hablarla. `validate._mentions_absent_program`
ya existe y se usa en el quiz; aplicarlo en vivo y, si detecta una carrera inexistente,
reemplazar la respuesta por una negativa escrita a mano. Costaría milisegundos.

### 4. Latencia

4-6 s hasta la primera palabra. Las palancas grandes están gastadas: Q4 cuesta 15 puntos de
error, Whisper `tiny` rompe los nombres propios, y el historial ya está en 3 intercambios.

Lo que queda:

- Más casos resueltos por el grafo o por `respuestas.py` (cada uno es un turno a 0 s).
- Bajar `silence_ms` de 600 a 400 gana 0,2 s con riesgo de cortar pausas naturales.
- Revisar si Ollama mantiene el prompt del sistema en caché entre turnos.

### 5. Lo que nunca se midió

- **Línea base con recuperación** de un modelo sin afinar: cuánto del resultado es el
  fine-tuning y cuánto el grafo. Se prometió desde el principio y sigue sin hacerse.
- **BM25 contra errores reales del STT**. Se sabe que *"mecaltrónica"* rompe la búsqueda
  léxica, pero no con qué frecuencia pasa en conversación real. El registro por turno ya tiene
  los datos para medirlo.
- **Consumo y temperatura de la Pi** en una sesión larga con servos moviéndose.

### 6. Deuda técnica

- **24 commits sin subir y `main` muy atrás.** `origin/main` sigue en *"Add implementation plan
  for UAO knowledge RAG"*, de septiembre. La rama `feature/uao-rag` lleva 70 commits, de los
  cuales 24 solo existen en el disco de la DGX. Un disco que falle se lleva un día de trabajo.
  Esto es lo primero que haría de los pendientes de deuda.
- **La Pi y la DGX divergieron** y se reconciliaron a mano copiando archivos. No hay un
  mecanismo de despliegue: hoy es `scp`. Un `git pull` en la Pi sería más sano, y haría
  innecesario el baile de `/tmp/nuevo_/` que usamos hoy.
- La Pi conserva en `~/respaldo-pi-20261117-0833.tar.gz` sus 16 archivos modificados de antes
  de la sincronización, por si algún ajuste de audio se perdió sin que nos diéramos cuenta.
- `rag.py` conserva el camino de embeddings (`make_finder`, `make_search`) que ya no se usa en
  ningún perfil. Funciona, pero es código muerto que confunde.
- Las guías `GUIA-DE-PRUEBAS.md` y los specs de septiembre describen la arquitectura vieja
  (RAG semántico, `qwen2.5:1.5b`) y contradicen al README.

### 7. Aplazados desde el principio

- **STT con NeMo y TTS con Riva** (subproyectos 2 y 3 del plan de septiembre). El STT son 1,9 s
  de los ~7 s de un turno, así que hay margen real ahí.
- **Barge-in** (interrumpir al robot hablando). Desactivado en la Pi porque el módulo Waveshare
  no cancela eco por hardware.
- **Tailscale en la Pi.** Hoy solo está en la red local; para llegarle desde afuera hay que
  configurarlo.

---

## Trampas documentadas

Antes de depurar algo raro, mirá la tabla de `docs/GUIA-FINE-TUNING.md`: 20 trampas que ya
costaron horas. Las que más reinciden:

- Un patrón de `pgrep`/`pkill` coincide con quien lo ejecuta, y con cualquier shell que apenas
  lo mencione. Matá por PID o leé `/proc`.
- `uv venv` no instala `pip`. Es `uv pip install --python <ruta>`.
- Editar un script de bash mientras corre lo corrompe.
- Las pruebas escribían en el registro real de conversación y esos turnos falsos se leyeron
  después como reales. Ya está separado.
- Congelar un proceso con `SIGSTOP` no libera su memoria.
