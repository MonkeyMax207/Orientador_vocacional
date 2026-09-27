# UAO Knowledge (RAG) — Design (Sub-project 2 of 4)

## Goal

Ground the voice agent in real information about the Universidad Autónoma de Occidente
(UAO, Cali). The agent should recommend real undergraduate programs that match the student's
interests, and answer detail questions about programs and campus life, without inventing
anything.

**Success criteria**
- Every program the agent names exists in `knowledge/cards.json`. In the first live test it
  invented "Ingeniería de Automática"; that must never happen.
- Detail questions (courses per stage, labs, sports, library, culture) are answered from
  cards, or the agent says honestly that it doesn't know and suggests a human advisor.
- Prices, tuition and scholarships are never answered. They are always handed to admissions staff.
- Pi budget: at most ~150 words of card text per turn. Turns that use no cards cost nothing extra.
- Runtime stays fully offline. Scraping happens once, with internet; its result lives in the repo.

## Decisions (agreed)

| Topic | Decision | Why |
|---|---|---|
| Scope | Programs (description, applicant profile, graduate profile, courses, labs, modality) + campus life (labs, wellbeing, arts & culture, sports, library, campus, internationalization) | User choice "B + infrastructure and wellbeing" |
| Prices | Excluded, handed to humans | They change yearly; a wrong price to an applicant is harmful |
| Approach | Offline "fact cards" + semantic search, injected only when relevant | Measured: on the pi profile every 150 words of context cost +2.4 s on this PC (~5–7 s on a real Pi); on the pc GPU it's ~free |
| Knowledge file | `knowledge/cards.json`, committed, human-reviewable | Staff can verify and correct what the robot says |
| Embeddings | Small multilingual model in Ollama, always on CPU | Keeps the 4 GB GPU for the LLM; ~50–100 ms per query |
| Scraping etiquette | `robots.txt` allows `/programa/` and `/wp-content/uploads/`; ~2 s between requests; run manually, rarely | Be a good citizen of the university's site |

## Sources

- Program list: `https://www.uao.edu.co/programas-de-pregrado/`. It links 30 pages at `/programa/<slug>/`.
- Each program page has HTML sections: "Descripción del programa", "Perfil del aspirante",
  "Perfil del egresado" (empty in some pages, probably loaded by JS; handled as missing),
  "Plan de estudios" (credits, modality) and "Conoce nuestros laboratorios". It also links PDFs:
  the full study plan (several versions; take the most recent `PLAN-DE-ESTUDIO*`/`Plan-de-estudios*`),
  the PEP and an electives list.
- The study-plan PDFs contain extractable text (not scans), but in scrambled layout order
  (course names split across lines, mixed with course codes).
- Campus pages: `/uao-labs/laboratorios-uao/`, `/bienestar-universitario-1/`, `/arte-y-cultura/`
  (and its sub-pages `danza-contemporanea`, `expresion-artistica`, `iniciacion-musical`,
  `salsa-cabaret`, `teatro`), `/deporte-y-recreacion/`, `/biblioteca/`, `/nuestro-campus/`,
  `/opciones-de-internacionalizacion/`, `/grupos-de-investigacion/`.
- Never fetched: tuition/price PDFs (e.g. `Res-CS-757-Valor-Matricula-*`), URLs with `?`.

## Components

### Build (once, on the PC, with internet)

| File | Job |
|---|---|
| `knowledge/scrape.py` | Reads the program list and fetches each program page, its latest study-plan PDF and the campus pages into `knowledge/raw/` (git-ignored). Waits 2 s between requests, skips files already downloaded, and uses a descriptive User-Agent. |
| `knowledge/make_cards.py` | Extracts useful text (drops nav, footer, scripts) per section. Asks the pc LLM (temperature 0) to condense each source into cards of 50–70 words of spoken Spanish. Checks the cards against their sources and writes `knowledge/cards.json`. |
| `knowledge/cards.json` | The robot's knowledge (committed). |
| `knowledge/eval.py` | Retrieval evaluation (see Testing). |

**Cards per program** (a card is omitted when its source section is empty):
- `<slug>-perfil`: who the program is for (from "Descripción" + "Perfil del aspirante"). Used for recommendations.
- `<slug>-egresado`: what graduates do.
- `<slug>-materias`: courses grouped by stage (first semesters / middle / final), plus credits and modality. From the PDF, falling back to the HTML.
- `<slug>-laboratorios`: the program's labs.

**Campus cards:** one or more per campus page (for example `deportes`, `biblioteca`, `arte-cultura-danza`).

**Card schema**
```json
{"id": "ingenieria-mecatronica-perfil", "tipo": "perfil|egresado|materias|laboratorios|campus",
 "programa": "Ingeniería Mecatrónica", "titulo": "Mecatrónica: para quién es",
 "texto": "…50–70 words of spoken Spanish…", "fuente": "https://www.uao.edu.co/programa/ingenieria-mecatronica/",
 "revisar": false}
```
`programa` is `null` for campus cards.

**Source check:** every course or lab name the LLM puts in `materias` or `laboratorios` cards
must appear (accent- and case-insensitive) in the source text. Otherwise the card gets
`revisar: true`, and `make_cards.py` prints the list for a human to review.

### Runtime (offline; PC and Pi)

| File | Job |
|---|---|
| `knowledge.py` | Loads the cards and embeds them with the Ollama embedding model (CPU). Caches the vectors in `models/knowledge_vectors.npz`, keyed by a hash of `cards.json`, and recomputes when the file changes. Provides `search(query, k, kinds=None) -> list[card]` (cosine similarity; returns nothing below `rag_threshold`) and `program_names()`. |
| `brain.py` (changed) | `stream_reply(user_text, cards=())`. The system prompt gets a fixed extra block: the rules below plus the list of the 30 official program names. It is built once, so Ollama's prompt cache keeps working. Cards for the current turn are prepended to the **current** user message only, as "Información verificada de la UAO: …". The history stores the plain user text, so context doesn't grow. |
| `main.py` (changed) | Chooses the retrieval mode per turn and passes the cards to the brain. |
| `config.toml` (changed) | `embed_model`, `rag_threshold`, `rag_max_cards` (2), `recommend_after_turns` (e.g. 6). |

**Retrieval modes per turn**
1. **Detail:** query = the agent's last question + the student's text. Up to `rag_max_cards` cards above the threshold.
2. **Recommendation:** triggered when the student asks for one (keyword match on phrases like
   "qué me recomiendas", "qué carrera", "qué puedo estudiar") or when `recommend_after_turns`
   is reached. Query = everything the student has said. Searches `perfil` cards only; top 3.
3. **Prices:** keyword match on "precio", "cuánto cuesta", "matrícula", "beca", "valor". No
   search. A card-like instruction tells the agent to hand over to an admissions advisor.

**Prompt rules added to the system prompt**
- Mention only programs, courses, labs and services that appear in the verified information or in the official program list.
- If something isn't there, say so honestly and suggest talking to a UAO advisor.
- Never give prices, tuition or scholarship amounts.

## Latency budget

| | pc | pi (estimated on a real Pi 5) |
|---|---|---|
| Embedding search | ~50–100 ms (CPU) | ~100–200 ms |
| Reading 2 cards (~150 words) | ~0.2 s | ~3–5 s |
| Turns without cards | +0 | +0 |

On the Pi, the short filler (at 1.0 s) and the slow filler (at 4.0 s) cover card turns. If that
is still too slow, make the cards shorter or use `rag_max_cards = 1`. No redesign is needed.

## Testing

**Logic checks** in `test_logic.py` (no network, no models):
- `search()` with fake vectors: ordering, threshold (nothing returned when nothing is similar), `k` limit, `kinds` filter.
- Brain injection: cards appear in the request's last user message, the stored history holds only the plain text, and the system prompt is identical across turns.
- Mode selection: recommendation phrases and the turn count trigger recommendation; price phrases trigger the hand-over; normal turns trigger detail.
- Vector cache: recomputed when the `cards.json` hash changes, reused otherwise.
- Source check: a course name missing from the source text sets `revisar: true`.
- HTML section extraction on a saved copy of the Mecatrónica page (checked in under `knowledge/fixtures/`).

**Retrieval evaluation** (`knowledge/eval.py`):
- About 25 real Spanish questions, each with the card id(s) expected in the top 2, plus
  price and off-topic questions expected to return nothing.
- Reports top-2 hit rate per embedding model and threshold. Used to choose between
  `embeddinggemma` and multilingual `granite-embedding`, and to set `rag_threshold`.

**End to end:** replay the last live conversation plus detail questions on `pc` and `pi`. Check
that every program named exists in `cards.json`, and report the added seconds on `pi`.

## Out of scope

- Prices, tuition, scholarships (handled by humans)
- Graduate programs
- Scheduled or automatic re-scraping
- Pi installation (sub-project 3) and servo control (sub-project 4)
