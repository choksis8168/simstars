"""Integration tests against the real local Postgres+pgvector instance (via
the `temp_db` fixture - an isolated throwaway database, not the real local
`simstars` one) and the real local fastembed model - no Anthropic/
ElevenLabs calls, so still free, just slower than the pure-mocked suite
(the model loads once per process; see memory_store._model's lazy
singleton). See memory_store.py's module docstring for why scoping is
self-contained per call rather than persisted across a whole simulate()
attempt - these tests verify that shape directly: a call cleans up after
itself and never sees another call's rows.
"""

from simstars.memory_store import MemoryEmbedding, retrieve_relevant_memories
from sqlmodel import select


def test_retrieve_returns_semantically_relevant_memories_over_irrelevant_ones(temp_db):
    memories = [
        (1, "Ana found a bloodstained letter hidden in the drawer"),
        (2, "Ben made himself a cup of coffee"),
        (3, "Ben commented on the weather being unusually warm today"),
        (4, "Ana confronted Ben about the letter and the secret it revealed"),
        (5, "Ben watered the plants on the windowsill"),
    ]

    results = retrieve_relevant_memories(
        memories=memories,
        query_text="What does Ana know about the bloodstained letter and its secret?",
        current_turn=5,
        top_k=2,
    )

    # the two memories directly about the letter/secret should be selected
    # over the unrelated coffee/weather/plants small talk
    assert any("letter" in r for r in results)


def test_retrieve_returns_results_in_chronological_order(temp_db):
    # Deliberately inserted out of chronological order (by event_index) -
    # the returned order should still be by event_index, not insertion or
    # relevance-score order.
    memories = [
        (5, "Ana mentioned the letter again, briefly"),
        (1, "Ana found a bloodstained letter hidden in the drawer"),
        (9, "Ana finally confronted Ben about the letter"),
    ]

    results = retrieve_relevant_memories(
        memories=memories, query_text="the letter", current_turn=9, top_k=3,
    )

    assert results == [
        "Ana found a bloodstained letter hidden in the drawer",
        "Ana mentioned the letter again, briefly",
        "Ana finally confronted Ben about the letter",
    ]


def test_retrieve_respects_top_k(temp_db):
    memories = [(i, f"event number {i} happened") for i in range(1, 11)]

    results = retrieve_relevant_memories(memories=memories, query_text="an event", current_turn=10, top_k=3)

    assert len(results) == 3


def test_retrieve_with_no_memories_returns_empty_without_touching_the_db(temp_db):
    results = retrieve_relevant_memories(memories=[], query_text="anything", current_turn=1)

    assert results == []


def test_retrieve_cleans_up_its_scope_row_after_returning(temp_db):
    retrieve_relevant_memories(
        memories=[(1, "something happened")], query_text="something", current_turn=1,
    )

    from simstars.db import get_session

    with get_session() as db:
        remaining = db.exec(select(MemoryEmbedding)).all()

    assert remaining == []


def test_second_call_never_sees_the_first_call_s_memories(temp_db):
    # Each call embeds/queries/deletes its own fresh scope (see module
    # docstring) - a completely unrelated second call, with a query that
    # would only match the first call's content, must come back empty
    # rather than accidentally matching leftover rows.
    retrieve_relevant_memories(memories=[(1, "Ana found a bloodstained letter")], query_text="letter", current_turn=1)

    results = retrieve_relevant_memories(
        memories=[(1, "Ben watered the plants")], query_text="Ana's bloodstained letter", current_turn=1,
    )

    assert all("letter" not in r for r in results)
