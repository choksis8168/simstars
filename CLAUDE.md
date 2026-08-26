# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit Conventions
- NEVER include Claude/AI-related comments, attributions, or signatures in PR descriptions, commit messages, or code comments.
- Do not add "Co-Authored-By" lines to any Git metadata.

## What this is

SimStars: a user defines characters and a world, and autonomous character agents (hidden goals,
memory, only-what-they've-witnessed knowledge) simulate a self-contained dramatic story inside
it — no plot required from the user. A director agent biases pacing toward real dramatic shape,
a critic pass checks the result actually has one, and a production pipeline turns the finished
script into a voiced, scored audio movie via ElevenLabs. Full design rationale (including why
each piece exists) lives in `docs/design.md` — read it before making non-trivial changes to the
simulation/critic/production pipeline.

**Two callers of the same engine** (`pipeline.py`'s `new_session`/`generate`/`play`, unmodified
by either): a Typer CLI (`cli.py`) and a React web app (`frontend/` + `api.py`) that's the
intended way to actually use SimStars — see docs/design.md's web-app plan. The CLI's `serve`
command starts the backend, which also serves the built frontend; that's the only CLI
interaction the web-app workflow needs.

## Commands

```
brew services start postgresql@14             # one-time (or however Postgres is already running locally)
createdb simstars                             # one-time - see db.py; DATABASE_URL defaults to this local db
uv sync                                       # install/update dependencies
cp .env.example .env                          # then fill in ANTHROPIC_API_KEY and ELEVENLABS_API_KEY
cd frontend && npm install && npm run build   # one-time (or after frontend changes)

uv run simstars serve                         # start the web app: http://127.0.0.1:8000
uv run simstars new                           # CLI equivalent: define characters + world, creates a session
uv run simstars script <session>              # CLI equivalent: generate + critique only, print transcript. No ElevenLabs cost.
uv run simstars play <session> [--note "..."] # CLI equivalent: full generate -> critique -> produce -> release

uv run pytest                                 # run the full Python test suite
uv run pytest tests/test_simulation.py        # run one test file
uv run pytest tests/test_simulation.py::test_character_only_witnesses_events_in_its_own_location  # single test
uv run pytest -k branching                    # run tests matching a keyword
```

No lint/format/type-check tooling is configured for the Python side. The frontend has no test
suite (a small local-only app; manual browser verification is the norm - see docs/design.md
web-app plan's Verification section).

The Python test suite never touches a real Anthropic or ElevenLabs API, so `uv run pytest` never
costs money and never needs `.env` populated. It does need a **local Postgres instance running**
(with the `vector` extension installed - see db.py/pyproject.toml's `onnxruntime` pin note below)
since DB-touching tests use the `temp_db` fixture in `tests/conftest.py`, which creates and drops
a fresh throwaway database per test, never the real local `simstars` database.
`tests/test_memory_store.py` additionally exercises the real local fastembed embedding model
(no API key, no network call) - still free, just slower on first run while the model loads.
Real Anthropic/ElevenLabs API calls only happen through `simstars serve`/the CLI commands above.

Two environment gotchas worth knowing about, both specific to this dev machine (Intel Mac):
`pyproject.toml` pins `onnxruntime==1.23.2` because `fastembed`'s default onnxruntime versions
dropped macOS x86_64 wheels entirely - don't drop that pin without checking wheel availability
first. And Homebrew's `pgvector` bottle only ships extension files for postgresql@17/@18, not the
postgresql@14 this project runs against - it had to be built from source against `pg_config
=/usr/local/opt/postgresql@14/bin/pg_config` (`make && make install` from a pgvector release
checkout) rather than a plain `brew install pgvector`. Neither of these should matter on a
different machine/architecture, but if `CREATE EXTENSION vector` or a `uv sync` involving
`fastembed` ever fails mysteriously here, this is why.

## Architecture

### Pipeline

`pipeline.py` is the only library entrypoint the CLI calls (`new_session` / `generate` / `play`)
— it holds no business logic itself, just orchestration, so a future web UI can be a second
caller of the same three functions without touching the engine. The flow:

```
new_session()  characters + world -> enrichment.py (hidden secrets/wounds/goals, forcing
               mechanic) -> production.cast_voices() (ElevenLabs voice assigned per character,
               persisted, reused across regenerates) -> persisted Session/Character rows

generate()     outline.generate_outline() (once per call, not per retry attempt) ->
               simulation.simulate() (GENERATE) -> critic.evaluate() (grade) -> reroll on
               failure up to MAX_CRITIC_RETRIES -> _select_best_attempt() picks the
               highest-scoring attempt (not just the last one tried) -> screenplay.build_screenplay()
               -> persisted Run row (transcript_json/screenplay_json/outline filled, no audio yet)

play()         generate() -> production.produce() (TTS + SFX + music + mixing) -> final_audio_path
```

### Simulation engine (`simulation.py`)

- `WorldState` holds the full cross-location event log plus a `character -> location` map.
  `CharacterAgent` (Haiku) only ever sees `memory_for(name)` — events witnessed at its own
  location at the time they happened, never the full log. `DirectorAgent` (Sonnet) sees
  everything. This asymmetry is what makes secrets-kept-from-one-person and misunderstandings
  possible as a conflict source, not just clashing goals — do not give characters access to the
  full transcript.
- `Event` is the single atomic unit — a character or the director logs one per turn (dialogue /
  action / movement / director-injected). There's no separate "beat" wrapper; a `Transcript` is
  just an ordered list of `Event`s.
- **Audio-only constraint**: only `EventType.DIALOGUE` is ever voiced in production. Action and
  director-injected events are unspoken stage directions. Both agent prompts and the critic
  (`dialogue_carries_the_story` criterion) actively guard against plot-critical reveals landing
  only in unvoiced text — this was a real, repeatedly-reproduced failure mode in live testing
  (see docs/design.md verification notes), not a hypothetical. There's also a mechanical
  backstop in `_run_turns`: the turn immediately after a director-injected event is force-routed
  to a witnessing character, regardless of what the director's own decision said.
- **Pre-generation outline (`outline.py`)**: before `simulate()` runs a single turn, one Sonnet
  call sketches a rough 3-5 beat dramatic arc from the cast's hidden secrets/wounds/goals - not a
  script, just guidance fed into `DirectorAgent`'s cached prompt context (director-only, same
  omniscience boundary as hidden material - `CharacterAgent` never sees it). Exists because
  branching/critic both only react to flatness *after* something was generated; this gives the
  director something to steer toward from turn one instead of discovering a shape only through
  trial. One outline per `generate()` call, not per retry attempt - persisted on `Run.outline`.
- **Branching lookahead, orchestrated as a LangGraph `StateGraph`**: `simulate()` doesn't generate
  linearly. The turn budget is grouped into segments (`SEGMENT_LENGTH`); at each boundary,
  `BRANCH_FACTOR` short previews (`PREVIEW_LENGTH` turns, not the full segment — keeps cost down)
  are generated in parallel — dispatched via LangGraph's `Send` API to the `generate_previews`
  node, fanned back in through an `Annotated[list, operator.add]` reducer field — from the same
  cloned `WorldState`, `critic.compare_previews()` picks the most dramatically promising one, and
  only the winner is carried forward. If even the best preview is still flat, one bounded
  re-preview round (`MAX_SEGMENT_ROUNDS`) runs with the failing reasoning fed back to the director
  as explicit guidance. This exists because some linear runs landed a strong resolved story while
  others went flat and burned all critic retries — branching catches that locally instead of only
  judging a finished transcript after the fact. The graph's nodes (`plan_segment`,
  `generate_previews`, `compare_and_resolve`, `finish_segment_linearly`, `finish` — see
  `_build_graph()`) are the orchestration shell only; `DirectorAgent`, `CharacterAgent`,
  `_run_turns`, `_resolve_round()`, and `_partition_preview_results()` are plain functions/classes
  called from those nodes, unit-testable without any graph or concurrency machinery. No
  checkpointer is configured — the graph runs once in-process per `simulate()` call.
- **Character memory retrieval (pgvector)**: `CharacterAgent.decide_action` uses the full
  chronological `memory_for(name)` verbatim while a character has witnessed
  `MEMORY_RETRIEVAL_THRESHOLD` (8) or fewer events — identical behavior to before this existed,
  including the prompt-caching benefit of a stable append-only prefix. Past that, it calls
  `memory_store.retrieve_relevant_memories()` instead: local embeddings (fastembed,
  `BAAI/bge-small-en-v1.5`) narrow to a candidate pool by pgvector cosine distance, then a blended
  recency+relevance re-rank picks the final top-K, returned in chronological order. Scoping is
  deliberately per-call, not per session/attempt — see `memory_store.py`'s module docstring for
  why (branch previews would otherwise leak provisional events across siblings). `DirectorAgent`
  never uses retrieval — its full cross-location omniscience is the invariant above.
- Guardrails (`config.py`): 3-5 characters, 2-4 locations, ~24-38 turn budget, capped critic
  retries and segment re-preview rounds — tune here, not scattered through the engine.

### Critic (`critic.py`)

`evaluate()` grades a finished transcript (conflict / escalation / resolution / dialogue-carries)
against pass/fail criteria. `compare_previews()` is the finer-grained sibling used by branching —
one comparative call across all candidate previews rather than independent absolute scores per
candidate (cheaper, more reliable than calibrating scores that then get compared).

### Production (`production.py`)

Voice casting happens once at session creation (persisted, reused across regenerate runs — same
character should sound the same movie to movie): `cast_voices()` infers each character's likely
gender from name/role/traits (one batched call, `_infer_genders`) and passes it to ElevenLabs'
`voices.search(gender=...)` — searching on role/traits text alone had no gender signal at all, a
real bug found via live usage. `cast_narrator_voice()` separately casts one voice from the
`narrative_story` use case for scene-setting narration, stored on `Session.narrator_voice_id`.
`_synthesize_line` passes explicit `voice_settings` (`config.TTS_STABILITY/TTS_STYLE/...`) —
previously none were passed, so ElevenLabs fell back to each library voice's conservative stored
defaults, which read as flat/robotic; another real live-usage complaint.

`screenplay.build_screenplay()`'s cue-generation call also produces a short `narration` line per
scene (read by the narrator, ahead of that scene's dialogue in `produce()`) and asks for 2-4 SFX
cues per scene rather than an unspecified count — both added in response to feedback that scenes
were hard to follow without context and felt sparse on sound design.

TTS/SFX/music generation are parallelized via `asyncio.to_thread` over the synchronous ElevenLabs
SDK calls. Mixing is scene-level, not per-line — a known simplification, not a bug. The
transcript/screenplay are persisted *before* production starts specifically so a mid-production
API failure never loses the (expensive) simulation result.

### Data model (`models.py`)

`sqlmodel` (PostgreSQL) rather than JSON files — see `db.py`/`config.DATABASE_URL` (defaults to a
local Homebrew instance) and `memory_store.py`'s `MemoryEmbedding` table, which lives in the same
database via `pgvector.sqlalchemy.Vector`. `Character` is a single class with
optional hidden-enrichment fields (`secret`, `wound`, `hidden_goal`) filled in by `enrichment.py`
after creation — there's no separate "enriched" subtype. "Hidden" means hidden from the user at
session-creation time only; it's expected and fine for a secret to surface diegetically once a
story is running. Detached-instance gotcha: SQLModel relationships (e.g. `session.characters`)
must be accessed once while the DB session is still open (see the `len(session.characters)` call
in `pipeline.new_session`) or they raise `DetachedInstanceError` later.

### Web app (`api.py`, `jobs.py`, `frontend/`)

Single local user, no auth, polling instead of websockets for progress — see docs/design.md's
web-app plan for the full scoping rationale. `generate`/`play` can take minutes, so
`POST /api/sessions/{id}/generate|play` hand the pipeline call to a small `ThreadPoolExecutor`
(`jobs.py`; a task queue would be overkill here) and return a `Job` id immediately;
`GET /api/jobs/{id}` is what the frontend polls. `POST /api/sessions` (session creation) stays a
plain blocking call — enrichment/voice-casting is only ~10-30s, tolerable for a spinner.

**Hard rule, not a style preference**: API responses never return SQLModel instances directly —
hand-written Pydantic response schemas in `api.py` (`CharacterOut`, etc.) explicitly whitelist
public fields, because `Character.secret`/`.wound`/`.hidden_goal`/`.relationship_seeds` must never
reach the browser (same hidden-at-creation-time boundary as `models.py`). Returning an ORM object
directly would leak them via FastAPI's default encoder. `tests/test_api.py` asserts this
recursively on every session/character response.

`api.py`'s catch-all route serves the built `frontend/dist/` and falls back to `index.html` for
any unmatched path (so React Router's client-side routes survive a refresh/deep link) - a
hand-written route, not `StaticFiles(html=True)`, which only handles directory requests, not
arbitrary SPA paths. `frontend/vite.config.ts`'s dev-server proxy (`/api` -> `:8000`) is a
dev-only convenience for `npm run dev`; the shipped path is always the FastAPI-served build.

### Testing conventions

Tests mock at the `call_structured` boundary (`llm.py`'s single chokepoint for all Anthropic
calls) rather than mocking higher-level agent methods — see `tests/test_simulation_branching.py`'s
`FakeLLM`, which dispatches on `tool_name` and must be patched onto *both*
`simstars.simulation.call_structured` and `simstars.critic.call_structured` (each module holds
its own imported reference). When adding a new code path, prefer extracting the pure decision
logic into its own function (see `_resolve_round`, `_select_best_attempt`,
`_validate_session_input`) so it's testable without mocking concurrency/DB/API — this pattern is
used throughout specifically because the async branching machinery is otherwise very hard to
test deterministically.
