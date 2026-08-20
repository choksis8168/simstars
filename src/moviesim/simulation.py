"""The GENERATE phase: WorldState + CharacterAgent + DirectorAgent, wired
together by `simulate()` into the turn-taking loop described in
docs/design.md "Simulation engine".

Key property this module exists to preserve: a CharacterAgent only ever
sees events that happened in its own location at the time they happened.
The DirectorAgent alone sees the full cross-location event log. That
asymmetry is what makes secrets-kept-from-one-person and misunderstandings
possible as a conflict source, not just clashing goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from moviesim.config import (
    CHARACTER_MODEL,
    DIRECTOR_MODEL,
    DIRECTOR_WRAP_UP_WINDOW,
)
from moviesim.llm import call_structured
from moviesim.models import Character, EndReason, Event, EventType, Session

GLOBAL = "global"  # pseudo-location: an event here is witnessed by every character regardless of where they are


@dataclass
class WorldState:
    """Full cross-location truth. Only DirectorAgent reads this directly;
    CharacterAgent only ever sees `memory_for(name)`.
    """

    character_locations: dict[str, str]
    events: list[Event] = field(default_factory=list)
    _memories: dict[str, list[Event]] = field(default_factory=dict)

    @classmethod
    def initial(cls, characters: list[Character]) -> "WorldState":
        state = cls(character_locations={c.name: c.starting_location for c in characters})
        state._memories = {c.name: [] for c in characters}
        return state

    def memory_for(self, name: str) -> list[Event]:
        return self._memories[name]

    def present_at(self, location: str) -> list[str]:
        return [name for name, loc in self.character_locations.items() if loc == location]

    def apply_event(self, event: Event) -> None:
        witnesses = (
            list(self.character_locations)
            if event.location == GLOBAL
            else self.present_at(event.location)
        )
        self.events.append(event)
        for name in witnesses:
            self._memories[name].append(event)
        if event.type == EventType.MOVEMENT:
            self.character_locations[event.actor] = event.content


def _format_log(events: list[Event]) -> str:
    if not events:
        return "(nothing has happened yet)"
    lines = []
    for e in events:
        where = "everywhere" if e.location == GLOBAL else e.location
        who = "The scene itself" if e.type == EventType.DIRECTOR else e.actor
        lines.append(f"[{e.index}] ({where}) {who}: {e.content}")
    return "\n".join(lines)


class CharacterAgent:
    """Speaks only from what its character has personally witnessed."""

    def __init__(self, character: Character, session: Session):
        self.character = character
        self.session = session

    def decide_action(self, state: WorldState, turn_index: int) -> Event:
        c = self.character
        location = state.character_locations[c.name]
        others_here = [n for n in state.present_at(location) if n != c.name]
        system = (
            f"You are {c.name}, {c.role}, in an unscripted dramatic simulation. "
            "Stay fully in character. You only know what you have personally "
            "witnessed - never reveal or act on anything you weren't present for. "
            "Respond with exactly one beat: a line of dialogue, a physical action, "
            "or a move to a different location. Keep it natural and specific, not "
            "expository. Keep content interpersonal/emotional - no gratuitous "
            "violence.\n\n"
            "This is being produced as an AUDIO drama: the audience only hears "
            "dialogue. Physical action beats are internal stage directions for the "
            "written record - they are never spoken aloud and the audience will "
            "never know they happened. So: use action for texture/realism, but "
            "never let anything the audience needs to understand - a discovery, a "
            "reveal, a decision, an emotional turn - live only in an action beat. "
            "If it matters, say it, react to it, or ask about it out loud instead."
        )
        user = (
            f"Who you are: {c.traits}\n"
            f"Your secret (only you know this): {c.secret}\n"
            f"Your wound: {c.wound}\n"
            f"What you actually want: {c.hidden_goal}\n"
            f"Your history with others: {c.relationship_seeds}\n\n"
            f"World: {self.session.world_description}\n"
            f"Why you can't just leave: {self.session.forcing_mechanic}\n"
            f"All locations: {', '.join(self.session.location_list())}\n\n"
            f"You are currently at: {location}\n"
            f"Also here right now: {', '.join(others_here) or 'no one else'}\n\n"
            f"Everything you have personally witnessed so far:\n{_format_log(state.memory_for(c.name))}\n\n"
            "What do you do on this beat?"
        )
        result = call_structured(
            model=CHARACTER_MODEL,
            system=system,
            user=user,
            tool_name="act",
            tool_description="Take one beat of action as this character.",
            input_schema={
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["dialogue", "action", "movement"]},
                    "content": {
                        "type": "string",
                        "description": (
                            "The dialogue line or action description. If type is "
                            "'movement', this must be exactly one of the listed "
                            "location names - the destination."
                        ),
                    },
                    "target": {
                        "type": "string",
                        "description": "Who this is directed at, if anyone. Omit if not applicable.",
                    },
                },
                "required": ["type", "content"],
            },
        )
        return Event(
            index=turn_index,
            type=EventType(result["type"]),
            actor=c.name,
            location=location,
            content=result["content"],
            target=result.get("target"),
        )


class DirectorAgent:
    """Sees everything. Picks who acts, can inject events, and decides
    when the story is done - with turn-budget awareness so it steers
    toward a resolution beat instead of just getting cut off.
    """

    def __init__(self, session: Session, characters: list[Character]):
        self.session = session
        self.characters = characters

    def decide_turn(
        self,
        state: WorldState,
        turn_index: int,
        turns_remaining: int,
        producer_note: str | None,
    ) -> dict:
        cast_summary = "\n".join(
            f"- {c.name} ({c.role}): secret={c.secret}; wound={c.wound}; "
            f"wants={c.hidden_goal}; currently at {state.character_locations[c.name]}"
            for c in self.characters
        )
        final_stretch = max(2, DIRECTOR_WRAP_UP_WINDOW // 2)
        if turns_remaining <= final_stretch:
            wrap_up = (
                f"\nCRITICAL: only {turns_remaining} turns remain. The climax must "
                "happen now if it hasn't already, and you must actively steer to "
                "resolution - do not just keep escalating. If a real resolution "
                "beat has just landed IN DIALOGUE (a decision spoken, a truth said "
                "out loud, a line drawn in words) - not just implied by an action - "
                "use the 'cut' action THIS turn rather than risk running out of "
                "turns first. An ending that lands one beat early is far better "
                "than one that never lands at all."
            )
        elif turns_remaining <= DIRECTOR_WRAP_UP_WINDOW:
            wrap_up = (
                f"\nOnly {turns_remaining} turns remain in this story - if the "
                "central conflict hasn't climaxed yet, escalate hard now so there "
                "is still room left to resolve it. Do not let this end "
                "mid-escalation."
            )
        else:
            wrap_up = ""
        note = f"\nProducer's note for this run: {producer_note}" if producer_note else ""
        system = (
            "You are the director of an unscripted dramatic simulation. You do "
            "not write dialogue yourself except when injecting an event. Your "
            "job is to shape a real story out of autonomous characters: bias "
            "everything toward dramatic conflict, escalation, and a genuine "
            "resolution - never let the scene stay comfortable for long. You "
            "have full knowledge of every character's secrets; the characters "
            "themselves do not share this knowledge with each other or with you "
            "unless they choose to act on it. Keep content interpersonal/"
            "emotional - no gratuitous violence or hate content.\n\n"
            "This is being produced as an AUDIO drama: the audience only hears "
            "dialogue - your injected events are never spoken aloud, so treat "
            "them as sound design (a phone buzzing, a door slamming, a bell "
            "ringing) rather than as a way to narrate plot to the audience. Never "
            "let a reveal live only in an injected event's text: an event can "
            "create the *opportunity* for a discovery, but the discovery itself "
            "only lands for the audience once a character says it or reacts to it "
            "out loud - so make sure the character you pick next actually voices "
            "what just happened."
        )
        user = (
            f"World: {self.session.world_description}\n"
            f"Forcing mechanic: {self.session.forcing_mechanic}\n"
            f"Locations: {', '.join(self.session.location_list())}\n\n"
            f"Cast (full knowledge, including hidden material):\n{cast_summary}\n\n"
            f"Full transcript so far (turn {turn_index} of this run, "
            f"{turns_remaining} turns remaining):\n{_format_log(state.events)}"
            f"{wrap_up}{note}\n\n"
            "Decide this turn: let a specific character act, inject an event "
            "yourself, or (only if the story has genuinely reached a resolution) "
            "cut."
        )
        return call_structured(
            model=DIRECTOR_MODEL,
            system=system,
            user=user,
            tool_name="direct",
            tool_description="Decide what happens on this turn of the simulation.",
            max_tokens=1024,
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["character", "event", "cut"]},
                    "character_name": {
                        "type": "string",
                        "description": "Required if action == character. Must exactly match a cast name.",
                    },
                    "location": {
                        "type": "string",
                        "description": (
                            "Required if action == event. One of the listed locations, "
                            "or 'global' for something every character perceives "
                            "regardless of where they are (e.g. a broadcast)."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Required if action == event: what happens.",
                    },
                    "target": {"type": "string"},
                    "cut_reason": {
                        "type": "string",
                        "description": "Required if action == cut: why the story is complete.",
                    },
                },
                "required": ["action"],
            },
        )


def simulate(
    session: Session,
    characters: list[Character],
    max_turns: int,
    producer_note: str | None = None,
) -> tuple[list[Event], EndReason]:
    state = WorldState.initial(characters)
    director = DirectorAgent(session, characters)
    agents = {c.name: CharacterAgent(c, session) for c in characters}

    for turn in range(1, max_turns + 1):
        turns_remaining = max_turns - turn + 1
        decision = director.decide_turn(state, turn, turns_remaining, producer_note)
        action = decision["action"]

        if action == "cut":
            return state.events, EndReason.RESOLVED

        if action == "character":
            name = decision.get("character_name")
            if name not in agents:
                # Model drift guard: fall back to whichever character has
                # gone longest without acting, rather than crashing the run.
                name = min(
                    agents,
                    key=lambda n: max((e.index for e in state.events if e.actor == n), default=0),
                )
            event = agents[name].decide_action(state, turn)
        else:  # "event"
            event = Event(
                index=turn,
                type=EventType.DIRECTOR,
                actor="director",
                location=decision.get("location", GLOBAL),
                content=decision.get("content", ""),
                target=decision.get("target"),
            )

        state.apply_event(event)

    return state.events, EndReason.TURN_BUDGET
