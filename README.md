# SimStars

> Create the characters. Set the world. Press play. Watch a movie emerge.

Define characters and a world, and autonomous character agents (with hidden goals, memory, and
only-what-they've-witnessed knowledge) simulate a self-contained dramatic story inside it — no
plot required from the user. A director agent biases pacing toward real dramatic shape (using a
branching lookahead, not just a single linear pass, to catch a story going flat before it derails
the whole run), a critic pass checks the result actually has one, and a production pipeline turns
the finished script into a voiced, scored audio movie via ElevenLabs.

See `docs/design.md` for the full design.

## Setup

```
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY and ELEVENLABS_API_KEY
cd frontend && npm install && npm run build && cd ..   # one-time; re-run after frontend changes
```

## Usage

The web app is the intended way to use SimStars — start it once, then everything (creating
characters, generating a script, producing the movie) is a click in the browser, never another
terminal command:

```
uv run simstars serve
```

then open `http://127.0.0.1:8000`.

### CLI (scripting/debugging)

The original CLI commands still work standalone, against the same database as the web app:

```
uv run simstars new                           # define characters + world
uv run simstars script <session>              # text-only debug run, no ElevenLabs cost
uv run simstars play <session> [--note "..."] # full generate -> produce -> release
```
