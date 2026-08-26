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
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from simstars.config import (
    BRANCH_FACTOR,
    CHARACTER_MODEL,
    DIRECTOR_MODEL,
    DIRECTOR_WRAP_UP_WINDOW,
    MAX_SEGMENT_ROUNDS,
    MEMORY_RETRIEVAL_THRESHOLD,
    PREVIEW_LENGTH,
    SEGMENT_LENGTH,
)
from simstars.critic import compare_previews
from simstars.llm import cached_block, call_structured
from simstars.memory_store import retrieve_relevant_memories
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
        witnessed = state.memory_for(c.name)

        if len(witnessed) <= MEMORY_RETRIEVAL_THRESHOLD:
            memory_text = _format_log(witnessed)
        else:
            # Past the threshold, retrieve a relevant subset instead of the
            # whole growing log - see memory_store.py. This is a real
            # tradeoff, stated plainly: retrieval-selected memories aren't a
            # stable growing prefix, so the prompt-caching benefit below
            # stops helping for this character once retrieval kicks in -
            # accepted, since blocking retrieval to preserve caching would
            # make the feature not really work.
            query_text = (
                f"Currently at {location}. Also here right now: {', '.join(others_here) or 'no one'}. "
                f"Most recent thing witnessed: {witnessed[-1].content}"
            )
            selected = retrieve_relevant_memories(
                memories=[(e.index, _format_log([e])) for e in witnessed],
                query_text=query_text,
                current_turn=turn_index,
            )
            memory_text = "\n".join(selected) if selected else "(nothing especially relevant comes to mind right now)"

        # Static for this character across every call this run - never
        # varies once c.name/c.role are set - so it's safe as a cached
        # block even though it's built fresh each call.
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
        # Cached block: identity (never changes) + this character's witnessed
        # memory (append-only - grows by whatever it witnessed since the
        # last call). Current location/who's-present are kept OUT of this
        # block since they can change turn to turn as characters move.
        cacheable_context = (
            f"Who you are: {c.traits}\n"
            f"Your secret (only you know this): {c.secret}\n"
            f"Your wound: {c.wound}\n"
            f"What you actually want: {c.hidden_goal}\n"
            f"Your history with others: {c.relationship_seeds}\n\n"
            f"World: {self.session.world_description}\n"
            f"Why you can't just leave: {self.session.forcing_mechanic}\n"
            f"All locations: {', '.join(self.session.location_list())}\n\n"
            f"Everything you have personally witnessed so far:\n{memory_text}"
        )
        volatile_context = (
            f"\n\nYou are currently at: {location}\n"
            f"Also here right now: {', '.join(others_here) or 'no one else'}\n\n"
            + (
                "The most recent thing you witnessed was an event, not "
                "someone speaking. If it's significant to you, this is your "
                "moment to react out loud - name what you noticed, ask about "
                "it, confront someone about it. Don't let it pass in silence.\n\n"
                if witnessed and witnessed[-1].type == EventType.DIRECTOR
                else ""
            )
            + "What do you do on this beat?"
        )
        result = call_structured(
            model=CHARACTER_MODEL,
            system=[cached_block(system)],
            user=[cached_block(cacheable_context), {"type": "text", "text": volatile_context}],
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


_DIRECTOR_SYSTEM = (
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
    "out loud on a later turn.\n\n"
    "Every turn also carries an 'intensity' value (0.0-1.0) tracking how "
    "emotionally charged the scene has become - not per line in isolation, "
    "but the arc of the scene so far. This becomes real vocal performance "
    "in production (delivery and pacing both key off it), so it needs to "
    "actually build across an escalating exchange rather than resetting "
    "each turn, and actually drop after a blowup or a resolution rather "
    "than staying pinned high out of habit."
)  # fully static across every call, every session - see cache_control below


class DirectorAgent:
    """Sees everything. Picks who acts, can inject events, and decides
    when the story is done - with turn-budget awareness so it steers
    toward a resolution beat instead of just getting cut off.
    """

    def __init__(self, session: Session, characters: list[Character], outline: str | None = None):
        self.session = session
        self.characters = characters
        # Pre-generation beat sketch (see outline.py) - guidance only,
        # director-only (matches the same omniscience boundary hidden
        # secrets/wounds/goals already get; characters never see this).
        # Fixed for the whole run, so it lives in the cached prefix below
        # alongside world/forcing-mechanic, not the per-turn volatile part.
        self.outline = outline

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
        outline_block = (
            f"Intended dramatic arc for this run (guidance from a story editor, not a "
            f"script - steer toward this, but let exactly how it plays out emerge through "
            f"actual dialogue; deviate if the story you're actually getting calls for it):\n"
            f"{self.outline}\n\n"
            if self.outline
            else ""
        )
        # Cached block: only world/forcing-mechanic/locations/outline (all
        # fixed for the whole run) and the transcript itself (append-only -
        # grows by exactly what the previous call didn't have yet). Turn-
        # count and cast_summary (character *locations* can change turn to
        # turn) are kept OUT of this block and appended after, uncached -
        # mixing per-call-varying text into the cached prefix would
        # invalidate the cache on every single call instead of extending it.
        cacheable_context = (
            f"World: {self.session.world_description}\n"
            f"Forcing mechanic: {self.session.forcing_mechanic}\n"
            f"Locations: {', '.join(self.session.location_list())}\n\n"
            f"{outline_block}"
            f"Full transcript so far:\n{_format_log(state.events)}"
        )
        volatile_context = (
            f"\n\n(turn {turn_index} of this run, {turns_remaining} turns remaining)\n\n"
            f"Cast (full knowledge, including hidden material):\n{cast_summary}"
            f"{wrap_up}{note}{feedback}\n\n"
            "Decide this turn: let a specific character act, inject an event "
            "yourself, or (only if the story has genuinely reached a resolution) "
            "cut."
        )
        return call_structured(
            model=DIRECTOR_MODEL,
            system=[cached_block(_DIRECTOR_SYSTEM)],
            user=[cached_block(cacheable_context), {"type": "text", "text": volatile_context}],
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
                    "intensity": {
                        "type": "number",
                        "description": (
                            "0.0 (calm) to 1.0 (peak emotional intensity) for the beat you're "
                            "choosing right now, tracking how tense/heated this scene has become "
                            "so far - not just this one line in isolation. A scene escalating "
                            "toward a confrontation should climb turn over turn, not reset; a "
                            "quiet beat after a blowup should drop back down. This shapes vocal "
                            "delivery and pacing in production, so be honest about where the "
                            "scene actually is emotionally, not just this line's surface content."
                        ),
                    },
                },
                "required": ["action", "intensity"],
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
        # Captured before the backstop below can replace `decision` wholesale
        # - the director's read on this turn's emotional intensity is still
        # meaningful even when the backstop overrides *who* acts.
        intensity = decision.get("intensity")

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

        event.intensity = intensity
        state.apply_event(event)

    return None


def _partition_preview_results(raw_results: list) -> list[tuple[WorldState, EndReason | None]]:
    """Separates successful branch previews from ones that failed even
    after call_structured's own retries (see llm.py/retry.py). Pulled out
    of simulate() as a pure function, matching _resolve_round, so this is
    unit-testable without needing a real concurrent failure under
    asyncio.gather - `raw_results` is whatever asyncio.gather(...,
    return_exceptions=True) returns: a mix of (WorldState, EndReason|None)
    successes and BaseException failures, one per branch.

    Without this, asyncio.gather would propagate a single failed branch's
    exception and cancel the other BRANCH_FACTOR-1 branches too, even if
    they'd have succeeded - so only give up (raise) if every branch in the
    round failed; otherwise the caller compares whatever did succeed.
    """
    results = [r for r in raw_results if not isinstance(r, BaseException)]
    if not results:
        failures = [r for r in raw_results if isinstance(r, BaseException)]
        raise RuntimeError(f"All {len(raw_results)} preview branches failed; first error: {failures[0]}") from failures[0]
    return results


def _resolve_round(
    results: list[tuple[WorldState, EndReason | None]], comparison: dict
) -> tuple[WorldState, EndReason | None, bool]:
    """Given the BRANCH_FACTOR preview results and the comparator's verdict,
    picks the winning (state, end_reason) and whether this round's outcome
    should stop the round-retry loop. Pulled out of simulate() as a pure
    function so this decision logic - including the model-drift guard on an
    out-of-range best_index, and a resolved-but-not-chosen branch being
    correctly ignored - is unit-testable without the async branch-generation
    machinery around it.

    Stops (returns should_stop=True) if the winner resolved (a "cut" is
    authoritative regardless of what still_flat says - resolving is never
    flat) or if the comparator judged it not flat. Otherwise the caller
    should re-preview this same segment with `comparison["reasoning"]` fed
    back as guidance.
    """
    best_index = comparison["best_index"]
    if not (0 <= best_index < len(results)):
        best_index = 0  # model drift guard
    winning_state, winning_end_reason = results[best_index]
    should_stop = winning_end_reason == EndReason.RESOLVED or not comparison["still_flat"]
    return winning_state, winning_end_reason, should_stop


class _SimState(TypedDict):
    """LangGraph state for one `simulate()` call. `world_state` and the
    scalar counters are plain fields - overwritten by whichever single node
    wrote them last, which is always safe here because only `plan_segment`/
    `compare_and_resolve`/`finish_segment_linearly` ever write them, never
    the fanned-out preview branches. `branch_results` is the one field
    multiple parallel branches write to concurrently in the same superstep,
    so it needs the `operator.add` reducer to merge those writes instead of
    one clobbering another - see the `Send`/reducer verification this
    module's rewrite was built against. It grows for the whole run (every
    round of every segment appends to it, never resets), so nodes that read
    "this round's" results slice off exactly the last BRANCH_FACTOR entries
    rather than clearing it - LangGraph reducer fields have no in-run reset.
    """

    world_state: WorldState
    director: DirectorAgent
    agents: dict[str, CharacterAgent]
    max_turns: int
    producer_note: str | None
    turn: int
    segment_len: int
    preview_len: int
    round_num: int
    prior_feedback: str | None
    total_rerounds: int
    branch_results: Annotated[list, operator.add]
    pending_end_reason: EndReason | None
    route: str
    final_events: list[Event] | None
    final_end_reason: EndReason | None


def _fan_out_previews(state: _SimState) -> list[Send]:
    """Dispatches BRANCH_FACTOR parallel preview branches, each against its
    own clone of the currently-committed world state - the `Send` fan-out
    this rewrite is built around (see the module docstring reference above).
    Used both as the routing function straight out of `plan_segment` and,
    looped back into, out of `compare_and_resolve` when a round is judged
    still flat and rounds remain.
    """
    return [
        Send(
            "generate_previews",
            {
                "world_state": state["world_state"].clone(),
                "director": state["director"],
                "agents": state["agents"],
                "turn": state["turn"],
                "preview_len": state["preview_len"],
                "max_turns": state["max_turns"],
                "producer_note": state["producer_note"],
                "prior_feedback": state["prior_feedback"],
            },
        )
        for _ in range(BRANCH_FACTOR)
    ]


def _plan_segment(state: _SimState) -> dict:
    turn = state["turn"]
    max_turns = state["max_turns"]
    segment_len = min(SEGMENT_LENGTH, max_turns - turn + 1)
    preview_len = min(PREVIEW_LENGTH, segment_len)
    return {
        "segment_len": segment_len,
        "preview_len": preview_len,
        "round_num": 0,
        "prior_feedback": None,
    }


async def _generate_previews_node(branch: dict) -> dict:
    """One `Send`-dispatched branch: runs a short preview on its own cloned
    WorldState in a thread (the underlying Anthropic calls are synchronous;
    `asyncio.to_thread` is the same pattern production.py uses to
    parallelize synchronous SDK calls). Exceptions are caught rather than
    left to propagate - LangGraph has no built-in equivalent of
    `asyncio.gather(..., return_exceptions=True)`, so this branch stores its
    own failure into `branch_results` in the same shape
    `_partition_preview_results` already expects (a mix of successes and
    bare exceptions), keeping that fail-soft resilience without changing it.
    """
    clone: WorldState = branch["world_state"]
    try:
        end_reason = await asyncio.to_thread(
            _run_turns,
            clone,
            branch["director"],
            branch["agents"],
            branch["turn"],
            branch["preview_len"],
            branch["max_turns"],
            branch["producer_note"],
            branch["prior_feedback"],
        )
        return {"branch_results": [(clone, end_reason)]}
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        return {"branch_results": [exc]}


def _compare_and_resolve(state: _SimState) -> dict:
    this_round_raw = state["branch_results"][-BRANCH_FACTOR:]
    results = _partition_preview_results(this_round_raw)

    world_state = state["world_state"]
    base_count = len(world_state.events)
    previews_new_events = [s.events[base_count:] for s, _ in results]
    comparison = compare_previews(world_state.events, previews_new_events)
    winning_state, winning_end_reason, should_stop = _resolve_round(results, comparison)

    round_num = state["round_num"]
    is_last_round = round_num >= MAX_SEGMENT_ROUNDS - 1

    if should_stop or is_last_round:
        if winning_end_reason == EndReason.RESOLVED:
            return {"world_state": winning_state, "pending_end_reason": EndReason.RESOLVED, "route": "finish"}
        return {"world_state": winning_state, "route": "finish_segment_linearly"}

    return {
        "prior_feedback": comparison["reasoning"],
        "total_rerounds": state["total_rerounds"] + 1,
        "round_num": round_num + 1,
        "route": "repreview",
    }


def _route_after_compare(state: _SimState) -> str | list[Send]:
    if state["route"] == "repreview":
        return _fan_out_previews(state)
    return state["route"]  # "finish" or "finish_segment_linearly"


async def _finish_segment_linearly(state: _SimState) -> dict:
    remaining = state["segment_len"] - state["preview_len"]
    world_state = state["world_state"]
    end_reason = None
    if remaining > 0:
        end_reason = await asyncio.to_thread(
            _run_turns,
            world_state,
            state["director"],
            state["agents"],
            state["turn"] + state["preview_len"],
            remaining,
            state["max_turns"],
            state["producer_note"],
            None,
        )

    if end_reason == EndReason.RESOLVED:
        return {"world_state": world_state, "pending_end_reason": EndReason.RESOLVED, "route": "finish"}

    next_turn = state["turn"] + state["segment_len"]
    if next_turn > state["max_turns"]:
        return {
            "world_state": world_state,
            "turn": next_turn,
            "pending_end_reason": EndReason.TURN_BUDGET,
            "route": "finish",
        }
    return {"world_state": world_state, "turn": next_turn, "route": "plan_segment"}


def _route_after_linear_finish(state: _SimState) -> str:
    return state["route"]  # "finish" or "plan_segment"


def _finish(state: _SimState) -> dict:
    return {"final_events": state["world_state"].events, "final_end_reason": state["pending_end_reason"]}


def _build_graph() -> Any:
    """Mirrors the pre-LangGraph imperative control flow directly (see
    docs/design.md "Simulation engine"): `plan_segment` computes one
    segment's lengths, `generate_previews` is the `Send`-fanned branch
    node, `compare_and_resolve` is the fan-in that picks a winner and
    routes to either another preview round, the linear segment finish, or
    straight to `finish` if a branch already resolved the story;
    `finish_segment_linearly` runs the segment's remaining turns and routes
    to either `finish` or the next segment's `plan_segment`. No checkpointer
    is configured - this graph runs once per `simulate()` call, in-process,
    with no need to resume across restarts.
    """
    graph = StateGraph(_SimState)
    graph.add_node("plan_segment", _plan_segment)
    graph.add_node("generate_previews", _generate_previews_node)
    graph.add_node("compare_and_resolve", _compare_and_resolve)
    graph.add_node("finish_segment_linearly", _finish_segment_linearly)
    graph.add_node("finish", _finish)

    graph.add_edge(START, "plan_segment")
    graph.add_conditional_edges("plan_segment", _fan_out_previews, ["generate_previews"])
    graph.add_edge("generate_previews", "compare_and_resolve")
    graph.add_conditional_edges(
        "compare_and_resolve", _route_after_compare, ["generate_previews", "finish_segment_linearly", "finish"]
    )
    graph.add_conditional_edges("finish_segment_linearly", _route_after_linear_finish, ["plan_segment", "finish"])
    graph.add_edge("finish", END)

    return graph.compile()


_GRAPH = _build_graph()


async def simulate(
    session: Session,
    characters: list[Character],
    max_turns: int,
    producer_note: str | None = None,
    outline: str | None = None,
) -> tuple[list[Event], EndReason, int]:
    """GENERATE phase, structured as branching lookahead over segments (see
    docs/design.md follow-on plan "Story-Quality Variance Fix") and run as a
    LangGraph `StateGraph` (see `_build_graph`): the turn budget is grouped
    into segments; at each segment boundary BRANCH_FACTOR short previews
    (PREVIEW_LENGTH turns) are generated in parallel via `Send` from the
    same committed state, a comparative evaluator picks the most
    dramatically promising one, and only the winner is carried forward -
    the rest are discarded. This catches a story going flat locally, before
    it derails the whole run, rather than only judging the finished
    transcript after the fact.

    `outline` is the pre-generation beat sketch from outline.py (see its
    module docstring) - passed straight through to DirectorAgent, which is
    the only place it's read; CharacterAgent never sees it.

    Returns (events, end_reason, branch_rounds_used) - the extra int is how
    many times a segment needed a re-preview round because even the best
    candidate was still judged flat (observability signal persisted on Run).
    """
    initial_state: _SimState = {
        "world_state": WorldState.initial(characters),
        "director": DirectorAgent(session, characters, outline),
        "agents": {c.name: CharacterAgent(c, session) for c in characters},
        "max_turns": max_turns,
        "producer_note": producer_note,
        "turn": 1,
        "segment_len": 0,
        "preview_len": 0,
        "round_num": 0,
        "prior_feedback": None,
        "total_rerounds": 0,
        "branch_results": [],
        "pending_end_reason": None,
        "route": "",
        "final_events": None,
        "final_end_reason": None,
    }
    result = await _GRAPH.ainvoke(initial_state)
    return result["final_events"], result["final_end_reason"], result["total_rerounds"]
