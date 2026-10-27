"""Tests for chunk expanders (am_diag/retrieval/methods/chunk_expand.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from am_diag.common.data_models.vector_record import VectorRecord
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.chunk_expand import chunks_mentioning, chunks_two_hop


def _record(chunk_id: UUID, vector: list[float], source: str = "s") -> VectorRecord:
    return VectorRecord(
        id=chunk_id,
        vector=vector,
        payload={
            "text": str(chunk_id)[:4],
            "document_id": str(uuid4()),
            "document_source": source,
        },
    )


def _stores(rows: list[dict], records: list[VectorRecord]):
    graph = MagicMock(spec=GraphStore)
    graph.execute_query = AsyncMock(return_value=rows)
    vectors = MagicMock(spec=VectorStoreBase)
    vectors.retrieve = AsyncMock(return_value=records)
    return graph, vectors


class TestChunksMentioning:
    """Tests for chunks_mentioning."""

    async def test_keeps_the_closest_chunks_per_entity(self) -> None:
        """Each entity keeps its top chunks by cosine similarity to the query."""
        ids = [uuid4() for _ in range(4)]
        # cosine to [1, 0]: 1.0, 0.8, 0.6, 0.0
        vectors = [[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0]]
        rows = [{"entity_id": "e1", "chunk_id": str(i)} for i in ids]
        records = [_record(i, v) for i, v in zip(ids, vectors, strict=True)]
        graph, store = _stores(rows, records)

        results = await chunks_mentioning(
            graph, store, "chunks", ["e1"], [1.0, 0.0], per_entity=3
        )

        assert [str(r.item.id) for r in results] == [str(i) for i in ids[:3]]
        assert [round(r.score, 2) for r in results] == [1.0, 0.8, 0.6]

    async def test_returns_nothing_without_mentions(self) -> None:
        """No mentioning chunks means the vector store is not queried."""
        graph, store = _stores([], [])

        results = await chunks_mentioning(
            graph, store, "chunks", ["e1"], [1.0], per_entity=3
        )

        assert results == []
        store.retrieve.assert_not_awaited()

    async def test_document_source_filter_applies_to_chunks(self) -> None:
        """Chunks from other document sources are left out."""
        keep, drop = uuid4(), uuid4()
        rows = [{"entity_id": "e1", "chunk_id": str(i)} for i in (keep, drop)]
        graph, store = _stores(
            rows, [_record(keep, [1.0], "pubmed"), _record(drop, [1.0], "other")]
        )

        results = await chunks_mentioning(
            graph,
            store,
            "chunks",
            ["e1"],
            [1.0],
            per_entity=3,
            filters=SearchFilters(document_sources=["pubmed"]),
        )

        assert [str(r.item.id) for r in results] == [str(keep)]


class TestChunksTwoHop:
    """Tests for chunks_two_hop."""

    async def test_returns_chunks_in_graph_order(self) -> None:
        """Chunks come back in the order the graph returned them."""
        first, second = uuid4(), uuid4()
        rows = [{"chunk_id": str(first)}, {"chunk_id": str(second)}]
        graph, store = _stores(rows, [_record(second, []), _record(first, [])])

        results = await chunks_two_hop(graph, store, "chunks", ["e1"], limit=20)

        assert [str(r.item.id) for r in results] == [str(first), str(second)]

    async def test_no_seeds_runs_no_query(self) -> None:
        """Without seeds there is nothing to look up."""
        graph, store = _stores([], [])

        assert await chunks_two_hop(graph, store, "chunks", [], limit=20) == []
        graph.execute_query.assert_not_awaited()
