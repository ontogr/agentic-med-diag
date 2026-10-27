"""Local context around matched chunks: their entities and neighbouring chunks."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.methods.base import entity_label
from am_diag.retrieval.methods.chunk_expand import parse_chunk_id
from am_diag.retrieval.retrievers.chunk import chunk_from_payload


async def local_context(
    graph_store: GraphStore,
    vector_store: VectorStoreBase,
    collection: str,
    chunk_ids: list[str],
) -> list[SearchResult]:
    """Return the entities in, and the chunks next to, each matched chunk.

    Args:
        graph_store: Graph database client.
        vector_store: Vector store that holds the chunk collection.
        collection: Chunk collection name.
        chunk_ids: Matched chunk id strings, best first.

    Returns:
        Entity results for the entities the chunks mention, then the previous and
        next chunks. Neither list repeats an item, and matched chunks are skipped.
    """
    if not chunk_ids:
        return []

    rows = await asyncio.gather(
        *(
            graph_store.execute_query(
                load("retrieval/chunk_neighbors"),
                {"chunk_id": chunk_id},
                read_only=True,
            )
            for chunk_id in chunk_ids
        ),
    )

    entities: dict[str, Entity] = {}
    neighbour_ids: dict[str, None] = {}
    for result in rows:
        for row in result:
            for entity in _row_entities(row):
                entities.setdefault(str(entity.id), entity)
            for key in ("prev_chunk_id", "next_chunk_id"):
                if parse_chunk_id(row.get(key)) and str(row[key]) not in chunk_ids:
                    neighbour_ids[str(row[key])] = None

    results = [
        SearchResult(
            item=entity,
            score=1.0 / (1 + rank),
            source="local_context",
            matched_on="local",
        )
        for rank, entity in enumerate(entities.values())
    ]
    if neighbour_ids:
        records = await vector_store.retrieve(collection, list(neighbour_ids))
        results += [
            SearchResult(
                item=chunk_from_payload(record.id, record.payload),
                score=1.0 / (1 + rank),
                source="local_context",
                matched_on="local",
            )
            for rank, record in enumerate(records)
        ]
    return results


def _row_entities(row: dict[str, Any]) -> list[Entity]:
    """Build entities from the ``entities`` column of a neighbour row."""
    entities: list[Entity] = []
    for raw in row.get("entities") or []:
        entity_id = parse_chunk_id(raw.get("id"))
        if entity_id is None:
            continue
        entities.append(
            Entity(
                id=UUID(str(entity_id)),
                name=raw.get("name") or "",
                label=entity_label(raw.get("labels")),
            ),
        )
    return entities


__all__ = ["local_context"]
