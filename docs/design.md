# Autonomous Movie Simulator — MVP CLI Prototype

## Context

The goal is a system where a user defines characters and a world, presses
"play," and a finished audio movie emerges — without the user ever writing
a plot. Drama has to come from the characters and world alone: the system
must guarantee conflict exists (via hidden character enrichment and a
"pressure cooker" world) and a director agent must actively bias the
simulation toward dramatic shape, not just referee it.

Key product decisions already made in discussion:
- User supplies characters (surface-level) + world; the system secretly
  enriches characters with hidden secrets/wounds/conflicting wants —
  enrichment stays hidden from the user, guaranteeing combustible material
  even from thin input.
- One self-contained story per session (not serialized/ongoing).
- Regenerate = same cast/world, new run. Default is free variance; user can
  optionally supply a text "producer's note" to nudge (not dictate) the
  next run.
- Pipeline is generate → produce → release (not live/streamed) — the full
  text simulation runs first (cheap to retry/reroll), then a separate
  production pass turns the finished transcript into audio via ElevenLabs.
- First buildable version is a **CLI/script prototype**, not a web app —
  the goal is to validate whether the simulation+director loop actually
  produces good drama before investing in any UI.
- Stack choice is delegated to Claude: **Python**, chosen for best fit with
  LLM agent orchestration and audio tooling.

This is a greenfield project (working directory is currently empty) — no
existing code/patterns to reuse yet.

## Architecture

```
[Cast + World input]
      │
      ▼
[Hidden Enrichment]   Claude call per character → adds secret/wound/
      │                conflicting want, invisible to the user. Also
      │                checks the world has a forcing mechanic (bounded
      │                space / ticking clock / scarce resource) and adds
      │                one if the user's world description lacks it.
      ▼
[Simulation Loop]     GENERATE phase — pure text, cheap to retry.
      │                Each turn, Director Agent either picks the next
      │                actor or injects an event itself; whoever acts logs
      │                an Event (dialogue/action/movement), tagged with
      │                location, to the transcript; Character Agents act
      │                from a memory filtered to only what they've
      │                witnessed (their location) + hidden goals; Director
      │                sees the full event log across all locations, tracks
      │                story shape (setup → escalation → climax →
      │                resolution), nudges itself toward resolving as the
      │                turn budget nears, and calls "cut" on completion or
      │                budget.
      ▼
[Critic Pass]          Separate evaluation: did this produce a real story
      │                (real conflict, escalation, a resolution)? If not,
      │                reroll the simulation (bounded retries).
      ▼
[Screenplay Formatter] Converts the transcript into structured scenes:
      │                dialogue lines, stage directions, SFX/music cues.
      ▼
[Production Pipeline]  PRODUCE phase — ElevenLabs.
      │                Voice casting per character → TTS per line → SFX
      │                generation → music generation → mix/assemble into
      │                one timeline (dialogue + ducked music + SFX).
      ▼
[final_movie.mp3]      RELEASE — finished, self-contained audio file.
```

Regenerate re-runs the Simulation Loop → Critic → Screenplay → Production
pipeline against the same `characters.json`/`world.json`, optionally with a
producer's note merged into the director's brief. Each run gets its own
subfolder so prior runs aren't overwritten.

### Data model (pydantic)
- `Character`: user-authored fields required (name, role, traits, vibe,
  starting `location`); hidden-enrichment fields optional/`None` until the
  enrichment step fills them in (`secret`, `wound`, `hidden_goal`,
  `relationship_seeds`). One class, not a separate "enriched" subtype —
  enrichment just populates the optional fields in place. Hidden means
  hidden *at creation* — not shown back to the user after `movie new`. It's
  expected and fine for a secret to surface diegetically once the story is
  running (that's the drama); the boundary is "user doesn't see it before
  pressing play," not "never appears anywhere."
- `World`: setting description, forcing mechanic, `locations` (a handful of
  named/described spaces characters can occupy and move between — this is
  what makes partial knowledge meaningful; a single-location world is just
  a `World` with one location, still valid).
- Producer's note is **not** a model — just an `Optional[str]` passed
  directly into the run call (`movie.play(session, note: str | None = None)`),
  since it's a transient steer, not a persisted domain object.
- `Event`: whatever a character or the director logs when it's their turn —
  actor, type (dialogue / action / movement / director-injected), content,
  target(s), `location`, index. Not a separate abstraction from the
  transcript: this is simply what gets appended, in order, each turn. The
  `location` tag is what makes perception filtering possible.
- `Transcript` = ordered list of `Event`s + end reason. No wrapper object
  in between — agents log an `Event`, it goes straight into the transcript.
- `Screenplay` (scenes → lines/stage-directions/SFX & music cues) — scenes
  map naturally onto locations/time-spans, matching standard screenplay
  scene-heading conventions ("INT. KITCHEN — NIGHT").

### Simulation engine
- `WorldState`: the full event log across all locations (this is what the
  Director sees) plus a `character -> current_location` map. Movement is
  just an `Event` of type `movement` that updates the map.
- `CharacterAgent`: holds a memory stream filtered to events tagged with
  the character's location *at the time of that event* (chronological; no
  vector retrieval needed yet — movies are short), plus public/hidden
  goals. A character can only act on what they've directly witnessed —
  this is what makes secrets-kept-from-one-person and misunderstandings
  possible as a conflict source, not just clashing goals.
  `decide_action(filtered_memory) -> Event` via Claude (Haiku — many
  calls/movie, needs speed over depth).
- `DirectorAgent`: `pick_next(state)`, `maybe_inject_event(state)`
  (location-scoped, e.g. "the phone rings in the kitchen," or global, e.g.
  a broadcast everyone hears), `evaluate_arc(state) -> continue | escalate |
  cut` via Claude (Sonnet — few calls, needs judgment, and needs the *full*
  event log since it's the only agent tracking the whole story). As the
  turn budget nears, the director's own prompt includes how many turns are
  left so it can steer toward a resolution beat instead of just getting cut
  off mid-escalation. Enforces the turn budget as a hard stop regardless.
- Loop: each turn, director acts itself or picks an actor → whoever acted
  logs an `Event` (with location) → world state and only the relevant
  characters' memories are updated with it → director evaluates → break on
  cut.

### Critic pass
Separate Sonnet call scoring the finished transcript against story-shape
criteria; on failure, reroll the simulation loop (max 2 retries), then
ship the best attempt regardless so a run never silently fails to produce
output.

### Production pipeline
- Voice casting: at session-creation time (not per-run), match each
  character to an existing ElevenLabs library voice based on described
  traits — fast, free, no extra latency (vs. Voice Design, which generates
  a bespoke voice per character but costs extra time/money per session;
  deferred past v1). The assignment is persisted on the session and reused
  across regenerate runs, so a character sounds the same movie to movie —
  it's the same person living a different story, not a different person.
- TTS: one ElevenLabs call per line, parallelized via `asyncio`.
- SFX + music: ElevenLabs sound-effects/music generation driven by the
  screenplay's cues.
- Mixing: `pydub` (+ ffmpeg) assembles dialogue/SFX/music into one timeline
  with basic music ducking under dialogue; exports `final_movie.mp3`.
- Resilience: the transcript and screenplay are persisted to the DB
  *before* production starts, so a mid-production failure (a dropped
  ElevenLabs call) never loses the (expensive) simulation result — it can
  be resumed/retried from the screenplay without re-running generation.
  Per-call retry with backoff on transient API failures.

### Persistence
`sqlmodel` (pydantic + SQLite) instead of raw JSON files — same amount of
effort to set up now, but avoids a migration later: a future web app needs
concurrent sessions/multiple users, and SQLite-via-sqlmodel gets there
without changing the data layer, just swapping SQLite for Postgres.
Generated audio artifacts (per-line clips, final mix) still live as plain
files on disk, referenced by path from the DB row:
```
sessions/<session_id>/
  runs/<run_id>/
    audio/               # per-line + SFX/music clips
    final_movie.mp3
```

### CLI (Typer)
- `movie new` — define characters + world (interactive or from file) → saves session.
- `movie script <session>` — run generate+critic only, print the
  transcript/screenplay. No ElevenLabs cost — fast loop for validating the
  simulation/director quality before producing audio.
- `movie play <session> [--note "..."]` — full generate → critic →
  screenplay → produce → release; creates a new run under the session.

The CLI is a thin caller of a plain Python library (`enrichment`,
`simulation`, `production` modules) — it holds no logic of its own beyond
argument parsing. This is deliberate: when a web UI comes later, it's a
second caller of the same library (e.g. FastAPI endpoints wrapping the same
`new_session()`/`run_movie()` calls, feeding a React frontend), not a
rewrite of the engine.

## Tech stack

- **Python 3.11+**
- **Anthropic Claude API**: Haiku for character-agent per-turn calls,
  Sonnet for director, critic, and enrichment calls.
- **ElevenLabs SDK**: text-to-speech, sound-effects generation, music
  generation.
- **pydantic**: typed data models throughout.
- **pydub + ffmpeg**: audio assembly/mixing.
- **Typer**: CLI.
- **sqlmodel (SQLite)**: session/run persistence — chosen over flat JSON
  specifically so a later web app (multi-user, concurrent sessions) is a
  storage swap (SQLite → Postgres) rather than a rewrite.
- **asyncio**: parallelizes independent TTS/SFX production calls (the
  generate-phase LLM calls stay sequential — they're inherently turn-based).
- **python-dotenv**: loads `ANTHROPIC_API_KEY`/`ELEVENLABS_API_KEY` from a
  local `.env` (not committed) — the one setup/config piece the earlier
  draft omitted entirely.

Deliberately deferred for v1: a graph-based agent framework (e.g.
LangGraph) — a custom turn-taking loop gives the director more direct
control than forcing this into a graph abstraction; a vector DB for
character memory — only needed once sessions get long or persist across
episodes (out of scope for "one self-contained story per session"); and
FastAPI/React for a web UI — the engine is built as a library from day one
(see CLI section) specifically so this can be added later without
disturbing the core.

### Scope guardrails (cost/complexity control)
- 3–5 characters per movie.
- 2–4 locations per world (enough for partial-knowledge/movement to matter
  without needing to track a sprawling map).
- ~15–25 turn budget per run (~5–12 minutes of finished audio).
- Max 2 critic-triggered rerolls of the generate phase before shipping the
  best attempt anyway.

### Content boundary
Director and enrichment prompts constrain drama to interpersonal/emotional
conflict — secrets, betrayal, jealousy, moral compromise, high-stakes
pressure — not gratuitous violence or hate content. This is a hard
constraint regardless of preference (both for output quality and because
the underlying models won't produce that content anyway), so it's stated
explicitly here rather than left implicit.

## Verification

1. Run `movie script` against a handful of test casts/worlds and read the
   raw transcripts — confirm the simulation+director loop actually
   produces dramatic, coherent, well-shaped stories (real conflict,
   escalation, a resolution) before wiring up any audio spend.
2. Run `movie play` end-to-end on a small cast; confirm the final audio
   file plays, voices are distinct and intelligible, and music/SFX don't
   drown dialogue.
3. Test regenerate: run the same session twice with no note (outputs
   should meaningfully differ) and once with a producer's note (output
   should visibly reflect the nudge without becoming a literal script).
4. Confirm a run still ships usable output even when the critic pass fails
   both retries (the "ship best attempt" fallback actually fires).
5. Confirm partial knowledge actually works: read a transcript and check
   that a character in Location A never acts on something that only
   happened in Location B — and, ideally, that this produces a genuine
   misunderstanding or dramatic-irony moment at least some of the time.
6. Kill an ElevenLabs call mid-production (simulate a failure) and confirm
   the run recovers/resumes from the persisted screenplay instead of
   re-running the simulation from scratch.
