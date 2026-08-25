"""Core data model.

Kept intentionally small: one `Character` class (not a separate "enriched"
subtype — enrichment just fills in optional fields in place), a `World`
with a handful of locations, and `Event` as the single unit that both the
simulation loop and the transcript are built from — there is no separate
"beat" wrapper. See docs/design.md for the reasoning behind each of these.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, Relationship, SQLModel


def _id() -> str:
    return uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EventType(str, Enum):
    DIALOGUE = "dialogue"
    ACTION = "action"
    MOVEMENT = "movement"
    DIRECTOR = "director"  # a director-injected event, not attributed to a character


class EndReason(str, Enum):
    RESOLVED = "resolved"           # director called "cut" — story reached a resolution
    TURN_BUDGET = "turn_budget"     # hit the hard turn cap without a clean cut


class JobKind(str, Enum):
    GENERATE = "generate"
    PLAY = "play"
    PRODUCE = "produce"  # retry/redo just the PRODUCE phase against an already-persisted run's screenplay


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Character(SQLModel, table=True):
    """User-authored fields are required at creation. The enrichment fields
    are optional and start as None — the enrichment step fills them in
    in-place. "Hidden" means hidden from the user at creation time only;
    it's expected (the point, even) for a secret to surface diegetically
    once the story is running.
    """

    id: str = Field(default_factory=_id, primary_key=True)
    session_id: str = Field(foreign_key="session.id", index=True)

    # user-authored
    name: str
    role: str
    traits: str  # free-text description the user supplies
    starting_location: str
    voice_id: Optional[str] = None  # ElevenLabs voice, assigned once at session creation

    # hidden enrichment (filled in by enrichment.py, never echoed back to the user)
    secret: Optional[str] = None
    wound: Optional[str] = None
    hidden_goal: Optional[str] = None
    relationship_seeds: Optional[str] = None  # free-text: pre-existing ties to other characters

    session: "Session" = Relationship(back_populates="characters")

    @property
    def is_enriched(self) -> bool:
        return self.secret is not None


class Session(SQLModel, table=True):
    """A cast + world definition. Long-lived — regenerating a movie creates
    a new Run under the same Session, reusing the same characters/world/voices.
    """

    id: str = Field(default_factory=_id, primary_key=True)
    created_at: datetime = Field(default_factory=_now)

    world_description: str
    forcing_mechanic: Optional[str] = None  # filled in by enrichment if the user didn't supply one
    locations: str  # comma-separated location names; kept simple, no separate Location table for v1
    narrator_voice_id: Optional[str] = None  # ElevenLabs voice for scene-setting narration, cast once like character voices

    characters: list[Character] = Relationship(back_populates="session")
    runs: list["Run"] = Relationship(back_populates="session")

    def location_list(self) -> list[str]:
        return [loc.strip() for loc in self.locations.split(",") if loc.strip()]


class Run(SQLModel, table=True):
    """One generated movie attempt for a session. Regenerate = new Run."""

    id: str = Field(default_factory=_id, primary_key=True)
    session_id: str = Field(foreign_key="session.id", index=True)
    created_at: datetime = Field(default_factory=_now)

    producer_note: Optional[str] = None
    end_reason: Optional[EndReason] = None
    critic_attempts: int = 0
    critic_reasoning: Optional[str] = None  # reasoning from the winning attempt's grade - debugging visibility
    branch_rounds_used: int = 0  # total re-preview rounds across all segments (see simulation.simulate)

    # Anthropic usage across every attempt this generate() call made (not
    # just the winning one - discarded attempts/branches still cost real
    # money) - see llm.py's usage tracking. estimated_cost_usd is a rough
    # estimate from published rates, not the real bill.
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cache_read_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # persisted before production starts, so a production failure never
    # loses the (expensive) simulation result
    transcript_json: Optional[str] = None   # serialized list[Event]
    screenplay_json: Optional[str] = None   # serialized Screenplay

    final_audio_path: Optional[str] = None

    session: Session = Relationship(back_populates="runs")


class Job(SQLModel, table=True):
    """Tracks a background generate/play call so the web app can poll it -
    see docs/design.md web-app plan "Background jobs". Deliberately not
    ORM-linked to Session/Run via Relationship() - it's a lightweight
    tracking row, queried directly by session_id/id, not part of the core
    domain model the CLI depends on.
    """

    id: str = Field(default_factory=_id, primary_key=True)
    session_id: str = Field(foreign_key="session.id", index=True)
    kind: JobKind
    status: JobStatus = JobStatus.PENDING
    error_message: Optional[str] = None
    result_run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


# --- Non-persisted, in-memory shapes used during a single simulation run ---
# These don't need `table=True`: they live only for the duration of one
# generate/produce pass and get serialized into Run.transcript_json /
# Run.screenplay_json at the end, not queried individually.


class Event(SQLModel):
    """The atomic unit of the simulation loop. A character or the director
    logs one of these each turn; the transcript is just an ordered list of
    them — no separate wrapper object.
    """

    index: int
    type: EventType
    actor: str  # character name, or "director" for EventType.DIRECTOR
    location: str
    content: str  # dialogue text, action description, or movement destination
    target: Optional[str] = None  # who the dialogue/action is directed at, if anyone


class Scene(SQLModel):
    location: str
    heading: str  # e.g. "INT. KITCHEN — NIGHT"
    lines: list[str]  # formatted dialogue/action lines, for display (e.g. `movie script`)
    events: list[Event]  # the underlying events, kept structured for production (TTS needs actor+text)
    sfx_cues: list[str] = []
    music_cue: Optional[str] = None
    narration: Optional[str] = None  # short voiced scene-setting line, read by the narrator - see screenplay.py


class Screenplay(SQLModel):
    scenes: list[Scene]
