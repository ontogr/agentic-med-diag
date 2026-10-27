"""Chunk expanders: passages that mention, or sit next to, the seed entities."""

from __future__ import annotations

from uuid import UUID

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.search import SearchResult
from am_diag.common.data_models.vector_record import VectorRecord
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.rerank.cosine import cosine
from am_diag.retrieval.retrievers.chunk import chunk_from_payload


def _chunk_result(
    record: VectorRecord,
    score: float,
    source: str,
    filters: SearchFilters | None,
) -> SearchResult | None:
    """Wrap a record as a chunk result, or None if the filters exclude it."""
    sources = filters.document_sources if filters else None
    if sources and record.payload.get("document_source") not in sources:
        return None
    return SearchResult(
        item=chunk_from_payload(record.id, record.payload),
        score=score,
        source=source,
        matched_on=source,
    )


async def chunks_mentioning(
    graph_store: GraphStore,
    vector_store: VectorStoreBase,
    collection: str,
    seed_ids: list[str],
    query_vector: list[float],
    *,
    per_entity: int,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Return the chunks that mention each seed entity, closest to the query first.

    Chunk vectors come from the vector store, so no text is embedded.

    Args:
        graph_store: Graph database client.
        vector_store: Vector store that holds the chunk collection.
        collection: Chunk collection name.
        seed_ids: Entity id strings, best first.
        query_vector: The query embedding.
        per_entity: Maximum chunks kept for each entity.
        filters: Optional search filters.

    Returns:
        Chunk results grouped by seed entity, each scored by cosine similarity.
    """
    if not seed_ids:
        return []

    rows = await graph_store.execute_query(
        load("retrieval/entity_chunk_ids"),
        {"ids": seed_ids},
        read_only=True,
    )
    if not rows:
        return []

    chunk_ids = sorted({str(row["chunk_id"]) for row in rows})
    records = {
        str(record.id): record
        for record in await vector_store.retrieve(
            collection, chunk_ids, with_vectors=True
        )
    }

    results: list[SearchResult] = []
    for entity_id in seed_ids:
        mine = [
            records[str(row["chunk_id"])]
            for row in rows
            if str(row["entity_id"]) == entity_id and str(row["chunk_id"]) in records
        ]
        mine.sort(key=lambda record: cosine(query_vector, record.vector), reverse=True)
        for record in mine[:per_entity]:
            result = _chunk_result(
                record, cosine(query_vector, record.vector), "chunks_mentions", filters
            )
            if result:
                results.append(result)
    return results


async def chunks_two_hop(
    graph_store: GraphStore,
    vector_store: VectorStoreBase,
    collection: str,
    seed_ids: list[str],
    *,
    limit: int,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Return chunks that mention a neighbour of a seed entity.

    Args:
        graph_store: Graph database client.
        vector_store: Vector store that holds the chunk collection.
        collection: Chunk collection name.
        seed_ids: Entity id strings.
        limit: Maximum chunks returned.
        filters: Optional search filters.

    Returns:
        Chunk results in graph order, scored by rank.
    """
    if not seed_ids:
        return []

    rows = await graph_store.execute_query(
        load("retrieval/two_hop_chunk_ids"),
        {"ids": seed_ids, "limit": limit},
        read_only=True,
    )
    chunk_ids = [str(row["chunk_id"]) for row in rows]
    if not chunk_ids:
        return []

    records = {
        str(record.id): record
        for record in await vector_store.retrieve(collection, chunk_ids)
    }
    results: list[SearchResult] = []
    for chunk_id in chunk_ids:
        record = records.get(chunk_id)
        if record is None:
            continue
        result = _chunk_result(
            record, 1.0 / (1 + len(results)), "chunks_two_hop", filters
        )
        if result:
            results.append(result)
    return results


def parse_chunk_id(value: object) -> UUID | None:
    """Return ``value`` as a UUID, or None if it is empty or not a UUID."""
    if not value:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


__all__ = ["chunks_mentioning", "chunks_two_hop"]
