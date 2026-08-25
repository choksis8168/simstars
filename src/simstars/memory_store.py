"""pgvector-backed live character memory retrieval - see docs/design.md
"Simulation engine" and CLAUDE.md's architecture notes.

`CharacterAgent.decide_action` (simulation.py) calls
`retrieve_relevant_memories` once a character has witnessed more than
`MEMORY_RETRIEVAL_THRESHOLD` events - below that, the full chronological
memory is used verbatim, unchanged from before this existed. `DirectorAgent`
never calls into this module - its full cross-location omniscience is a
deliberate, documented invariant (see simulation.py's module docstring);
retrieval-filtering it would undermine the exact thing that makes
secrets-kept-from-one-person possible.

**Scoping is fully self-contained per call, not per simulate() attempt.**
The branching lookahead (see simulation.py) runs several preview branches
in parallel off cloned WorldStates, most of which are discarded - if
memory rows were written incrementally as events happened and scoped to a
shared attempt id, a losing branch's provisional events would either leak
into a sibling branch's retrieval mid-round or (if scoped per branch
instead) need careful lineage bookkeeping to reconcile which branch won.
Sidestepping that entirely: each call embeds whatever the *caller's own*
`WorldState.memory_for(name)` already holds right now (always the correct,
already-witnessed-only truth for that exact branch) into a fresh
throwaway scope, queries it, and deletes the scope before returning - so
there is never any state here that could leak across characters, branches,
or attempts, and no cleanup needs to be threaded through simulate()'s
control flow at all.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from fastembed import TextEmbedding
from pgvector.sqlalchemy import Vector
from sqlmodel import Field, SQLModel, delete, select

from simstars.config import EMBEDDING_DIM, EMBEDDING_MODEL, MEMORY_RETRIEVAL_TOP_K
from simstars.db import get_session

_embedder: TextEmbedding | None = None


def _model() -> TextEmbedding:
    # Lazy singleton: loading the ONNX model has real (~seconds) startup
    # cost, so it happens once per process on first actual use, not at
    # import time - a run that never crosses MEMORY_RETRIEVAL_THRESHOLD for
    # any character never pays it at all.
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


class MemoryEmbedding(SQLModel, table=True):
    """One memory line, embedded, scoped to one throwaway `scope_id` -
    written, queried, and deleted within a single `retrieve_relevant_memories`
    call. See module docstring for why scoping is this narrow.
    """

    __tablename__ = "memory_embedding"

    id: Optional[int] = Field(default=None, primary_key=True)
    scope_id: str = Field(index=True)
    event_index: int
    content: str  # the formatted memory line - same shape simulation._format_log produces
    embedding: list[float] = Field(sa_type=Vector(EMBEDDING_DIM))


def retrieve_relevant_memories(
    memories: list[tuple[int, str]],
    query_text: str,
    current_turn: int,
    top_k: int = MEMORY_RETRIEVAL_TOP_K,
) -> list[str]:
    """`memories` is the character's full witnessed-event list as
    (event_index, formatted_content) pairs. Embeds and inserts all of them
    under a fresh scope, narrows to a candidate pool by cosine distance
    against `query_text` (cheap, index-backed), then re-ranks that pool
    with a blended recency+relevance score - matching the Generative
    Agents/Smallville memory architecture referenced in docs/design.md, so
    an old but highly relevant memory and a recent-but-mildly-relevant one
    both get a fair shot rather than pure nearest-neighbor letting recency
    drown out relevance or vice versa. Returns the selected lines in
    chronological order (not score order), so what the character "recalls"
    still reads as a coherent timeline, matching the full-memory path.
    """
    if not memories:
        return []

    scope_id = uuid4().hex
    try:
        vectors = list(_model().embed([content for _, content in memories] + [query_text]))
        memory_vectors, query_vector = vectors[:-1], vectors[-1]

        with get_session() as db:
            db.add_all(
                MemoryEmbedding(scope_id=scope_id, event_index=idx, content=content, embedding=vec.tolist())
                for (idx, content), vec in zip(memories, memory_vectors)
            )
            db.commit()

        distance_col = MemoryEmbedding.embedding.cosine_distance(query_vector.tolist()).label("distance")
        candidate_pool = max(top_k * 4, top_k)
        with get_session() as db:
            stmt = (
                select(MemoryEmbedding, distance_col)
                .where(MemoryEmbedding.scope_id == scope_id)
                .order_by(distance_col)
                .limit(candidate_pool)
            )
            rows = db.exec(stmt).all()

        if not rows:
            return []

        oldest_index = min(memory.event_index for memory, _ in rows)
        span = max(current_turn - oldest_index, 1)  # avoid a divide-by-zero when everything just happened

        def blended_score(row: tuple) -> float:
            memory, distance = row
            relevance = 1.0 - float(distance)
            recency = 1.0 - (current_turn - memory.event_index) / span
            return 0.5 * relevance + 0.5 * recency

        top = sorted(rows, key=blended_score, reverse=True)[:top_k]
        chronological = sorted((memory for memory, _ in top), key=lambda m: m.event_index)
        return [m.content for m in chronological]
    finally:
        with get_session() as db:
            db.exec(delete(MemoryEmbedding).where(MemoryEmbedding.scope_id == scope_id))
            db.commit()
