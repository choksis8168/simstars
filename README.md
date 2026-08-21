# SimStars

> Create the characters. Set the world. Press play. Watch a movie emerge.

An MVP CLI prototype: define characters and a world, and autonomous
character agents (with hidden goals, memory, and only-what-they've-witnessed
knowledge) simulate a self-contained dramatic story inside it — no plot
required from the user. A director agent biases pacing toward real dramatic
shape, a critic pass checks the result actually has one, and a production
pipeline turns the finished script into a voiced, scored audio movie via
ElevenLabs.

See `docs/design.md` for the full design.

## Setup

```
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY and ELEVENLABS_API_KEY
```

## Usage

```
uv run simstars new                          # define characters + world
uv run simstars script <session>             # text-only debug run, no ElevenLabs cost
uv run simstars play <session> [--note "..."] # full generate -> produce -> release
```
