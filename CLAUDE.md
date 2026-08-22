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

The Python test suite is all pure-logic/mocked — no test touches a real Anthropic or ElevenLabs
API, so `uv run pytest` never costs money and never needs `.env` populated (DB-touching tests use
the `temp_db` fixture in `tests/conftest.py`, never the real local `sessions/simstars.db`). Real
API calls only happen through `simstars serve`/the CLI commands above.

## Architecture

### Pipeline

`pipeline.py` is the only library entrypoint the CLI calls (`new_session` / `generate` / `play`)
— it holds no business logic itself, just orchestration, so a future web UI can be a second
caller of the same three functions without touching the engine. The flow:

```
new_session()  characters + world -> enrichment.py (hidden secrets/wounds/goals, forcing
               mechanic) -> production.cast_voices() (ElevenLabs voice assigned per character,
               persisted, reused across regenerates) -> persisted Session/Character rows

generate()     simulation.simulate() (GENERATE) -> critic.evaluate() (grade) -> reroll on
               failure up to MAX_CRITIC_RETRIES -> _select_best_attempt() picks the
               highest-scoring attempt (not just the last one tried) -> screenplay.build_screenplay()
               -> persisted Run row (transcript_json/screenplay_json filled, no audio yet)

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
- **Branching lookahead**: `simulate()` doesn't generate linearly. The turn budget is grouped
  into segments (`SEGMENT_LENGTH`); at each boundary, `BRANCH_FACTOR` short previews
  (`PREVIEW_LENGTH` turns, not the full segment — keeps cost down) are generated in parallel via
  `asyncio.gather`/`asyncio.to_thread` from the same cloned `WorldState`, `critic.compare_previews()`
  picks the most dramatically promising one, and only the winner is carried forward. If even the
  best preview is still flat, one bounded re-preview round (`MAX_SEGMENT_ROUNDS`) runs with the
  failing reasoning fed back to the director as explicit guidance. This exists because some
  linear runs landed a strong resolved story while others went flat and burned all critic
  retries — branching catches that locally instead of only judging a finished transcript after
  the fact. `_resolve_round()` and `WorldState.clone()` are pulled out as pure functions
  specifically so this logic is unit-testable without async/concurrency machinery.
- Guardrails (`config.py`): 3-5 characters, 2-4 locations, ~24-38 turn budget, capped critic
  retries and segment re-preview rounds — tune here, not scattered through the engine.

### Critic (`critic.py`)

`evaluate()` grades a finished transcript (conflict / escalation / resolution / dialogue-carries)
against pass/fail criteria. `compare_previews()` is the finer-grained sibling used by branching —
one comparative call across all candidate previews rather than independent absolute scores per
candidate (cheaper, more reliable than calibrating scores that then get compared).

### Production (`production.py`)

Voice casting happens once at session creation (persisted, reused across regenerate runs — same
character should sound the same movie to movie). TTS/SFX/music generation are parallelized via
`asyncio.to_thread` over the synchronous ElevenLabs SDK calls. Mixing is scene-level, not
per-line — a known simplification, not a bug. The transcript/screenplay are persisted *before*
production starts specifically so a mid-production API failure never loses the (expensive)
simulation result.

### Data model (`models.py`)

`sqlmodel` (SQLite) rather than JSON files, chosen so a future multi-user web app is a storage
swap (SQLite -> Postgres) rather than a data-layer rewrite. `Character` is a single class with
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
