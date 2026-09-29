# Guía de pruebas (para trabajar sin Claude)

Todos los comandos se corren en **PowerShell, desde la carpeta del proyecto** (`C:\Users\juand\Documentos\IA VOZ`).
`uv run` usa automáticamente el Python 3.11 y las librerías del proyecto (`.venv/`); no hace falta activar nada.

---

## 0. Antes de empezar (cada vez)

1. **Ollama debe estar abierto.** Busca el icono de la llama en la bandeja de Windows; si no está, abre "Ollama" desde el menú Inicio.
2. **Conecta el módulo USB Waveshare** antes de arrancar el agente.
3. **Cierra programas pesados en GPU** (juegos, editores de video): el perfil `pc` usa 3.8 de los 4 GB de la tarjeta.

---

## 1. Conversar con el agente

```powershell
uv run python main.py pc      # perfil PC: GPU, llama3.2:3b, voz Kokoro (Santa)
uv run python main.py pi      # perfil Pi simulado: CPU 4 hilos, qwen2.5:1.5b, voz Piper (España)
```la 

Salir: `Ctrl+C`.

**Qué significan las líneas en pantalla:**

| Línea | Significado |
|---|---|
| `Tú: …` | Lo que Whisper entendió de tu voz |
| `[rag] detalle: ingenieria-mecatronica-laboratorios` | Modo del turno y fichas de la UAO que se le dieron al modelo (`-` = ninguna) |
| `[rag] precio: precios` | Preguntaste por precios: el agente deriva a admisiones |
| `[ANIM] pensar` / `asentir` / `saludar` | La animación que dispararía el animatrónico |
| `(ruido ignorado)` | Detectó sonido pero no palabras (tos, golpe) |
| `[tiempos] primer sonido 1.00s (filler)` | Cuánto tardó en sonar algo tras dejar de hablar (meta < 2 s) |
| `respuesta lista 1.5s` | Cuándo estuvo lista la primera frase real |

Cada turno se guarda en `recordings/turn_*.wav` (no se sube a git).

> **Importante al cambiar de perfil:** Ollama mantiene como máximo 3 modelos cargados. Si quedan modelos del
> otro perfil, cada turno puede tardar ~5 s más (Ollama descarga y recarga el modelo). Antes de cambiar:
> ```powershell
> ollama ps                      # ver qué modelos están cargados
> ollama stop llama3.2:3b        # descargar los del otro perfil (o qwen2.5:1.5b, gemma3:4b, granite-embedding:278m)
> ```

---

## 2. Pruebas rápidas sin micrófono

```powershell
uv run python test_logic.py              # 35 checks de lógica; deben pasar todos ("35 checks passed")
uv run python -m knowledge.replay pc     # conversación completa simulada con la UAO (sin audio)
uv run python -m knowledge.replay pi     # lo mismo en el perfil Pi (más lento)
uv run python brain.py pc                # chatear por texto con el modelo (escribe como estudiante)
uv run python bench.py pc                # latencias sobre las grabaciones de recordings/turn_*.wav
```

En `replay`, revisa:
- que los modelos solo nombren **carreras que existen** (si aparece `⚠ revisar nombres`, léelo: puede ser una negación honesta como "no tenemos Ingeniería Aeroespacial");
- que las preguntas de laboratorios/materias/deportes usen las fichas correctas (entre corchetes);
- el tiempo `1ª frase` de cada turno.

---

## 3. Pruebas por pieza (con el módulo USB)

```powershell
uv run python list_devices.py            # debe decir OK input / OK output para "USB PnP MME"
uv run python audio_io.py                # habla 3 s: nivel pico ideal 0.1–0.8 y te escuchas de vuelta
uv run python vad.py                     # barra que sube al hablar (Ctrl+C para salir)
uv run python stt.py pc                  # transcribe recordings/mic_test.wav (el de audio_io.py)
uv run python tts.py pc                  # escucha la voz del perfil y su velocidad (RTF < 1 = más rápido que hablar)
uv run python tts.py pc "Hola, soy Orienta, tu orientador vocacional."   # frase propia
uv run python check_gpu.py               # Whisper y el LLM en la GPU
```

---

## 4. Ajustes útiles en `config.toml`

| Ajuste | Qué hace | Probar |
|---|---|---|
| `voice_rate` (por perfil) | Voz más grave y lenta ("tortuga"), sin costo de CPU | `0.9` |
| `voice` | Voz: Kokoro (`em_santa`, `em_alex`, `ef_dora`) o Piper (`es_ES-davefx-medium`, `es_ES-sharvard-medium`, `es_MX-claude-high`) | — |
| `filler_after_s` | Tras cuántos segundos sin respuesta suena "Bueno…", "A ver…" | `1.2` si suena demasiado |
| `silence_ms` | Silencio que da por terminado tu turno | subir a `800` si te corta al pensar |
| `vad_threshold` | Sensibilidad a la voz | subir a `0.6` si el ruido lo activa |
| `rag_threshold` | Qué tan parecida debe ser una ficha para usarla | `0.30` (elegido con `knowledge/eval.py`) |
| `rag_max_cards` (por perfil) | Fichas por turno (en la Pi cada una suma ~1.5 s) | pc `2`, pi `1` |
| `recommend_after_turns` | En qué turno recomienda carreras aunque no se lo pidan | `6` |
| `barge_in` | Permitir interrumpirlo hablando (solo con audífonos) | `true` |

Tras cambiar la voz, regenera los fillers: `uv run python fillers.py pc` (o `pi`).

---

## 5. Conocimiento de la UAO (fichas)

Las fichas están en `knowledge/cards.json` (147, revisadas). Puedes **editarlas a mano**: cambia `texto`,
borra una ficha, o agrega una nueva copiando el formato. Una ficha con `"revisar": true` **nunca se usa**
hasta que la pongas en `false`. Los vectores de búsqueda se recalculan solos al arrancar si el archivo cambió.

```powershell
uv run python -m knowledge.eval embeddinggemma    # calidad de la búsqueda (hoy 21/25 con umbral 0.30)
```

**Actualizar desde la web de la UAO** (cuando cambien los programas; necesita internet, ~15 min en total):
```powershell
Remove-Item -Recurse -Force knowledge\raw         # borra las páginas descargadas antes
uv run python -m knowledge.scrape                 # descarga (espera 2 s entre páginas)
ollama stop llama3.2:3b                           # libera la GPU para gemma3:4b
uv run python -m knowledge.make_cards             # regenera todas las fichas
uv run python -m knowledge.make_cards psicologia  # o solo una carrera / página (se mezcla con las demás)
git diff knowledge/cards.json                     # revisa qué cambió antes de hacer commit
```
Al final `make_cards` lista las fichas marcadas para revisar y cómo reintentar las que fallaron.

---

## 6. Problemas comunes

| Síntoma | Solución |
|---|---|
| `Ollama no responde…` | Abre la app Ollama y vuelve a correr |
| `Falta el modelo 'X'. Ejecuta: ollama pull X` | Corre ese comando (necesita internet una vez) |
| `Falla el modelo de embeddings…` | `ollama pull embeddinggemma` |
| `Whisper falló en 'cuda'…` | En `config.toml` → `[pc]`: `whisper_device = "cpu"`, `whisper_compute = "int8"` |
| `Micrófono 'USB PnP MME': …` | Conecta el módulo; corre `list_devices.py` y ajusta `input_device`/`output_device` |
| Cada respuesta tarda ~5 s extra | Modelos de otro perfil cargados: `ollama ps` y `ollama stop …` |
| La voz sale entrecortada | Cierra programas que usen la GPU; revisa `ollama ps` |
| Respuestas raras en `pi` | Limitación conocida del modelo de 1.5B (ver pendientes) |

**Modelos necesarios** (una vez, con internet): `ollama pull llama3.2:3b`, `ollama pull qwen2.5:1.5b`,
`ollama pull embeddinggemma`, y solo para regenerar fichas `ollama pull gemma3:4b`.
Archivos de voz y Whisper: `uv run python download_models.py`.

---

## 7. Guardar tu trabajo en GitHub

```powershell
git status                                  # qué cambió
git add config.toml knowledge/cards.json    # lo que quieras guardar
git commit -m "Describe el cambio"
git push
```
El trabajo del RAG está en la rama `feature/uao-rag` (todavía sin fusionar a `main`).

---

## 8. Pendientes (estado al 2026-09-27)

- **Revisión final** de la rama `feature/uao-rag` y fusión a `main`.
- **Decidir el cerebro de la Pi:** `qwen2.5:1.5b` usa las fichas correctas pero se desvía (se obsesiona con
  carreras sin motivo, adorna datos) y en conversaciones largas llegó a 16–17 s por respuesta.
- **Huecos de contenido** que la búsqueda no cubre bien: biblioteca (la página se llama "CRAI"), apoyo
  psicológico, investigación. Se arreglan agregando o editando fichas en `cards.json`.
- Sub-proyecto 3 (instalar en la Raspberry Pi) y 4 (servos del animatrónico).

---

## 9. PC como cerebro, Pi como voz (WebSocket)

El PC corre Whisper, RAG, LLM y Kokoro (`server.py`); la Pi solo escucha, dice "Jum…" y reproduce (`client.py`).

**En el PC (una vez):** abrir el puerto 8765 en el firewall (PowerShell **como administrador**) y ver la IP:
```powershell
New-NetFirewallRule -DisplayName "Orientador WS" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
ipconfig        # busca "Dirección IPv4" de tu Wi-Fi/Ethernet, p. ej. 192.168.1.35
```
**En la Pi (una vez):** en `config.toml` pon esa IP en `server_url = "ws://192.168.1.35:8765"`, y deja el módulo
USB como dispositivo de audio predeterminado (icono de sonido del escritorio de la Pi).

**Cada vez:**
```powershell
uv run python server.py          # en el PC: espera "Servidor listo…"
```
```bash
uv run python client.py          # en la Pi: "Conectado a ws://…". Habla.
```
Ambos deben estar en la misma red. Si la Pi no conecta: revisa la IP, el firewall y que el servidor esté corriendo.

---

## 10. DGX Spark (casa) como cerebro, Pi (universidad) como voz

**Red (una vez, en la DGX y en la Pi):** Tailscale crea una red privada entre ambas sin abrir puertos
(el servidor no tiene contraseña: nunca lo expongas directo a internet).
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up           # inicia sesión con la MISMA cuenta en las dos
tailscale ip -4             # en la DGX: 100.x.y.z  →  en la Pi: server_url = "ws://100.x.y.z:8765"
```

**Instalación en la DGX (una vez):**
```bash
git clone -b feature/uao-rag https://github.com/MonkeyMax207/Orientador_vocacional.git && cd Orientador_vocacional
uv sync && uv run python download_models.py
ollama pull qwen2.5:32b && ollama pull embeddinggemma
uv run python fillers.py dgx
```
**Cada vez:** `uv run python server.py dgx` en la DGX, `uv run python client.py` en la Pi.
La Pi necesita internet en la universidad (ya no es 100 % offline).

### 10.1 Prueba inicial: DGX como cerebro, PC como voz (misma red de casa)

Antes de meter la Pi: el PC habla con la DGX por la misma red local, sin Tailscale.

**En la DGX:** `uv run python server.py dgx` (espera "Servidor listo…").
**En el PC** (PowerShell, repo ya clonado con `uv sync` hecho): en `config.toml`, `server_url = "ws://<IP-LAN-de-la-DGX>:8765"`
(`ip -4 addr` en la DGX para verla; hoy es `192.168.1.16`) — luego `uv run python client.py pc`.

Si el PC no conecta: confirma que están en la misma red Wi-Fi/Ethernet, y que nada bloquea el puerto 8765
en la DGX (`sudo ufw allow 8765/tcp` si `ufw` está activo).

### 10.2 LLM en la DGX vía NIM (`nemotron-nano-9b-v2-dgx-spark`) en vez de Ollama

Ya soportado en el código (`brain.py: make_brain`, `config.toml: [dgx].llm_backend`). Para activarlo:

1. **API key de NGC** (una vez): `https://org.ngc.nvidia.com/setup/api-keys` → "Generate API Key".
   `docker login nvcr.io` (usuario `$oauthtoken`, contraseña = la key) alcanza para *bajar* la imagen,
   pero el contenedor necesita la key otra vez, como variable de entorno, para bajar los pesos del modelo
   la primera vez que arranca (falla si falta: "operation requires an API key, but none was found").
2. **Arrancar el contenedor** (la imagen ya está descargada localmente, no vuelve a bajarla):
   ```bash
   export NGC_API_KEY=nvapi-...
   docker run --rm --gpus all -p 8001:8000 -e NGC_API_KEY --name orienta-nim \
     nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant
   ```
   Espera el log de "listo" (baja pesos la primera vez, puede tardar unos minutos).
3. **Averiguar el id exacto del modelo:** `curl -s http://127.0.0.1:8001/v1/models | python3 -m json.tool`
   (2026-09-29: `nvidia/nemotron-nano-9b-v2`, ya puesto en `config.toml`).
4. **Activar:** en `config.toml` bajo `[dgx]`, pon `llm_backend = "nim"` y `llm_model` = el id del paso 3.
   Ya hecho — el perfil `dgx` usa el NIM por defecto.
5. **Medir:** `uv run python brain.py dgx` (chat de texto) y `uv run python -m knowledge.replay dgx`,
   comparar contra la fila de Ollama en `README.md`. Si no mejora, `llm_backend = "ollama"` revierte al
   instante (no hace falta tocar código).

**Dos cosas que costó descubrir (2026-09-29, ya arregladas en el código):**
- Nemotron es un modelo de razonamiento híbrido: sin nada especial, cada respuesta sale como un
  monólogo interno (`<think>...`) en vez de una respuesta hablable. `NimBrain` agrega `/no_think`
  al final del system prompt — es el interruptor documentado de NVIDIA para respuestas directas.
- El *warmup* (una petición mínima al arrancar para cargar el modelo) falla con "400 Bad Request"
  si el mensaje es solo de sistema, sin turno de usuario — a diferencia de Ollama. `NimBrain.warmup()`
  ya manda un mensaje de usuario ("Hola") de relleno.
- La primera vez que arranca `server.py dgx`, calcula los embeddings de las 253 fichas de texto
  completo (`knowledge/chunks.json`) en la CPU de la DGX: ~1.4 s por ficha, ~6 min en total. Se
  guarda en caché (`models/chunk_vectors.npz`); los siguientes arranques son instantáneos. Por eso
  el timeout en `rag.py` subió de 300 a 900 s.

**Próximos pasos para velocidad y calidad en la DGX (por evaluar, medir antes de cambiar):**
- **STT en GPU con NeMo:** `nvidia/parakeet-tdt-0.6b-v3` (multilingüe, incluye español) o `canary-1b`,
  reemplazando Whisper en CPU (faster-whisper no tiene GPU en ARM). Sub-proyecto 2, spec pendiente.
- **TTS con NeMo / Riva:** Magpie TTS multilingüe (español) vía Riva/NIM, o FastPitch+HiFi-GAN en español;
  comparar voz y latencia contra Kokoro. Sub-proyecto 3, spec pendiente.
