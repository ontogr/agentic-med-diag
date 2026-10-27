"""Tests for local context (am_diag/retrieval/methods/local_context.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from am_diag.common.data_models.vector_record import VectorRecord
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.methods.local_context import local_context


class TestLocalContext:
    """Tests for local_context."""

    async def test_returns_entities_and_neighbouring_chunks(self) -> None:
        """Entities in the chunk and its previous and next chunks are returned."""
        matched, previous, following, entity = uuid4(), uuid4(), uuid4(), uuid4()
        row = {
            "entities": [
                {"id": str(entity), "name": "asthma", "labels": ["Entity", "Disease"]},
                {"id": None, "name": None, "labels": None},
            ],
            "prev_chunk_id": str(previous),
            "next_chunk_id": str(following),
        }
        graph = MagicMock(spec=GraphStore)
        graph.execute_query = AsyncMock(return_value=[row])
        store = MagicMock(spec=VectorStoreBase)
        store.retrieve = AsyncMock(
            return_value=[
                VectorRecord(
                    id=chunk_id,
                    payload={
                        "text": "t",
                        "document_id": str(uuid4()),
                        "document_source": "s",
                    },
                )
                for chunk_id in (previous, following)
            ]
        )

        results = await local_context(graph, store, "chunks", [str(matched)])

        kinds = [(type(r.item).__name__, str(r.item.id)) for r in results]
        assert kinds == [
            ("Entity", str(entity)),
            ("Chunk", str(previous)),
            ("Chunk", str(following)),
        ]
        assert results[0].item.label == "Disease"

    async def test_skips_neighbours_that_were_already_matched(self) -> None:
        """A neighbour that is itself a match is not fetched again."""
        first, second = uuid4(), uuid4()
        graph = MagicMock(spec=GraphStore)
        graph.execute_query = AsyncMock(
            side_effect=[
                [{"entities": [], "prev_chunk_id": None, "next_chunk_id": str(second)}],
                [{"entities": [], "prev_chunk_id": str(first), "next_chunk_id": None}],
            ]
        )
        store = MagicMock(spec=VectorStoreBase)
        store.retrieve = AsyncMock(return_value=[])

        results = await local_context(graph, store, "chunks", [str(first), str(second)])

        assert results == []
        store.retrieve.assert_not_awaited()

    async def test_no_chunks_runs_no_query(self) -> None:
        """Without matched chunks there is no context."""
        graph = MagicMock(spec=GraphStore)
        graph.execute_query = AsyncMock()

        assert await local_context(graph, MagicMock(), "chunks", []) == []
        graph.execute_query.assert_not_awaited()
