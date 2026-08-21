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

import asyncio
from dataclasses import dataclass, field

from simstars.config import (
    BRANCH_FACTOR,
    CHARACTER_MODEL,
    DIRECTOR_MODEL,
    DIRECTOR_WRAP_UP_WINDOW,
    MAX_SEGMENT_ROUNDS,
    PREVIEW_LENGTH,
    SEGMENT_LENGTH,
)
from simstars.critic import compare_previews
from simstars.llm import call_structured
from simstars.models import Character, EndReason, Event, EventType, Session

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

    def clone(self) -> "WorldState":
        """Cheap clone for branch previews: shallow-copy the containers so
        appending to a clone's event/memory lists never affects the
        original. `Event` objects themselves are never mutated after
        creation, so sharing references to them across clones is safe.
        """
        return WorldState(
            character_locations=dict(self.character_locations),
            events=list(self.events),
            _memories={name: list(events) for name, events in self._memories.items()},
        )

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
            + (
                "The most recent thing you witnessed was an event, not "
                "someone speaking. If it's significant to you, this is your "
                "moment to react out loud - name what you noticed, ask about "
                "it, confront someone about it. Don't let it pass in silence.\n\n"
                if state.memory_for(c.name) and state.memory_for(c.name)[-1].type == EventType.DIRECTOR
                else ""
            )
            + "What do you do on this beat?"
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
        prior_feedback: str | None = None,
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
        feedback = (
            f"\nA previous attempt at this exact stretch of story stalled: "
            f"{prior_feedback}\nDo not repeat that - force real movement this time."
            if prior_feedback
            else ""
        )
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
            "ringing) rather than as a way to narrate plot to the audience.\n\n"
            "STRICT RULE: an injected event's content may describe a sensory "
            "trigger (a phone buzzing, an envelope sliding into view, a knock) "
            "but must NEVER itself spell out the payload of a reveal - no "
            "caller ID names, no letterhead text, no read-aloud document "
            "contents, nothing a character hasn't actually said yet. That "
            "specific information must not exist anywhere in the transcript "
            "until a character speaks it. If you want a phone call to reveal "
            "who's calling, the event is just 'the phone rings' - the caller's "
            "identity only becomes real once a character reads it and says it "
            "out loud on a later turn."
        )
        user = (
            f"World: {self.session.world_description}\n"
            f"Forcing mechanic: {self.session.forcing_mechanic}\n"
            f"Locations: {', '.join(self.session.location_list())}\n\n"
            f"Cast (full knowledge, including hidden material):\n{cast_summary}\n\n"
            f"Full transcript so far (turn {turn_index} of this run, "
            f"{turns_remaining} turns remaining):\n{_format_log(state.events)}"
            f"{wrap_up}{note}{feedback}\n\n"
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


def _run_turns(
    state: WorldState,
    director: DirectorAgent,
    agents: dict[str, "CharacterAgent"],
    start_turn: int,
    num_turns: int,
    max_turns: int,
    producer_note: str | None,
    prior_feedback: str | None,
) -> EndReason | None:
    """Runs up to `num_turns` turns starting at `start_turn` against `state`
    IN PLACE. Returns EndReason.RESOLVED if the director cut during this
    stretch, else None (the stretch just ran its course). Used both for
    short branch previews and for finishing a segment linearly after a
    preview wins - see `simulate()`.
    """
    for turn in range(start_turn, min(start_turn + num_turns, max_turns + 1)):
        turns_remaining = max_turns - turn + 1
        decision = director.decide_turn(state, turn, turns_remaining, producer_note, prior_feedback)
        action = decision["action"]

        # Mechanical backstop for the reveals-only-in-unvoiced-events failure
        # mode (see docs/design.md verification notes): prompting the director
        # to always follow an injected event with a character reacting in
        # dialogue was not reliable on its own across live testing. If the
        # previous beat was a director-injected event, force this turn to be
        # a witnessing character, regardless of what the model chose - it
        # doesn't guarantee *what* they say, but it guarantees someone gets
        # the chance to voice the reveal right away instead of the director
        # wandering off into unrelated business.
        last_event = state.events[-1] if state.events else None
        if last_event is not None and last_event.type == EventType.DIRECTOR and action != "character":
            witnesses = (
                list(state.character_locations)
                if last_event.location == GLOBAL
                else state.present_at(last_event.location)
            )
            if witnesses:
                action = "character"
                decision = {
                    "action": "character",
                    "character_name": min(
                        witnesses,
                        key=lambda n: max((e.index for e in state.events if e.actor == n), default=0),
                    ),
                }

        if action == "cut":
            return EndReason.RESOLVED

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

    return None


async def _run_preview(
    state: WorldState,
    director: DirectorAgent,
    agents: dict[str, "CharacterAgent"],
    start_turn: int,
    num_turns: int,
    max_turns: int,
    producer_note: str | None,
    prior_feedback: str | None,
) -> tuple[WorldState, EndReason | None]:
    """Runs a short branch preview in a thread (the underlying Anthropic
    calls are synchronous; `asyncio.to_thread` is the same pattern
    production.py already uses to parallelize synchronous SDK calls).
    Operates on `state`, which the caller must have already cloned - each
    preview gets its own independent WorldState.
    """
    end_reason = await asyncio.to_thread(
        _run_turns, state, director, agents, start_turn, num_turns, max_turns, producer_note, prior_feedback
    )
    return state, end_reason


async def simulate(
    session: Session,
    characters: list[Character],
    max_turns: int,
    producer_note: str | None = None,
) -> tuple[list[Event], EndReason, int]:
    """GENERATE phase, structured as branching lookahead over segments (see
    docs/design.md follow-on plan "Story-Quality Variance Fix"): the turn
    budget is grouped into segments; at each segment boundary BRANCH_FACTOR
    short previews (PREVIEW_LENGTH turns) are generated in parallel from the
    same committed state, a comparative evaluator picks the most
    dramatically promising one, and only the winner is carried forward -
    the rest are discarded. This catches a story going flat locally, before
    it derails the whole run, rather than only judging the finished
    transcript after the fact.

    Returns (events, end_reason, branch_rounds_used) - the extra int is how
    many times a segment needed a re-preview round because even the best
    candidate was still judged flat (observability signal persisted on Run).
    """
    state = WorldState.initial(characters)
    director = DirectorAgent(session, characters)
    agents = {c.name: CharacterAgent(c, session) for c in characters}

    turn = 1
    total_rerounds = 0

    while turn <= max_turns:
        segment_len = min(SEGMENT_LENGTH, max_turns - turn + 1)
        preview_len = min(PREVIEW_LENGTH, segment_len)

        prior_feedback: str | None = None
        winning_state: WorldState | None = None
        winning_end_reason: EndReason | None = None

        for round_num in range(MAX_SEGMENT_ROUNDS):
            base_count = len(state.events)
            clones = [state.clone() for _ in range(BRANCH_FACTOR)]
            results = await asyncio.gather(
                *[
                    _run_preview(clone, director, agents, turn, preview_len, max_turns, producer_note, prior_feedback)
                    for clone in clones
                ]
            )
            previews_new_events = [s.events[base_count:] for s, _ in results]

            comparison = compare_previews(state.events, previews_new_events)
            best_index = comparison["best_index"]
            if not (0 <= best_index < len(results)):
                best_index = 0  # model drift guard
            winning_state, winning_end_reason = results[best_index]

            already_resolved = winning_end_reason == EndReason.RESOLVED
            if already_resolved or not comparison["still_flat"] or round_num == MAX_SEGMENT_ROUNDS - 1:
                break
            prior_feedback = comparison["reasoning"]
            total_rerounds += 1

        state = winning_state

        if winning_end_reason == EndReason.RESOLVED:
            return state.events, EndReason.RESOLVED, total_rerounds

        remaining = segment_len - preview_len
        if remaining > 0:
            end_reason = await asyncio.to_thread(
                _run_turns, state, director, agents, turn + preview_len, remaining, max_turns, producer_note, None
            )
            if end_reason == EndReason.RESOLVED:
                return state.events, EndReason.RESOLVED, total_rerounds

        turn += segment_len

    return state.events, EndReason.TURN_BUDGET, total_rerounds
