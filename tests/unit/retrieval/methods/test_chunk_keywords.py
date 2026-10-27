"""Tests for chunk keyword search (am_diag/retrieval/methods/chunk_keywords.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from am_diag.common.data_models.vector_record import VectorHit
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.methods.chunk_keywords import chunk_keyword_search
from am_diag.retrieval.methods.keywords import Keywords
from am_diag.vector.embedding.base import Embedder


def _hit(chunk_id, text: str, score: float) -> VectorHit:
    return VectorHit(
        id=chunk_id,
        score=score,
        payload={"text": text, "document_id": str(uuid4()), "document_source": "s"},
    )


class TestChunkKeywordSearch:
    """Tests for chunk_keyword_search."""

    async def test_merges_chunks_found_by_several_queries(self) -> None:
        """A chunk found twice keeps its best score; results are best first."""
        first, second = uuid4(), uuid4()
        by_query = {
            "does asthma recur": [_hit(first, "A", 0.5)],
            "asthma": [_hit(first, "A", 0.9), _hit(second, "B", 0.4)],
        }

        async def hybrid_search(collection, vector, text, **kwargs):
            return by_query.get(text, [])

        store = MagicMock(spec=VectorStoreBase)
        store.hybrid_search = AsyncMock(side_effect=hybrid_search)
        embedder = MagicMock(spec=Embedder)
        embedder.embed = AsyncMock(return_value=[[0.1]])

        results = await chunk_keyword_search(
            store,
            "chunks",
            embedder,
            "does asthma recur",
            Keywords(llm=["asthma"], simple=[]),
            keyword_queries=6,
            simple_queries=4,
            top_k=5,
        )

        assert [(str(r.item.id), r.score) for r in results] == [
            (str(first), 0.9),
            (str(second), 0.4),
        ]

    async def test_queries_only_the_question_without_keywords(self) -> None:
        """With no keywords the question is the only query."""
        store = MagicMock(spec=VectorStoreBase)
        store.hybrid_search = AsyncMock(return_value=[_hit(uuid4(), "A", 0.5)])
        embedder = MagicMock(spec=Embedder)
        embedder.embed = AsyncMock(return_value=[[0.1]])

        results = await chunk_keyword_search(
            store,
            "chunks",
            embedder,
            "q",
            Keywords(),
            keyword_queries=6,
            simple_queries=4,
            top_k=5,
        )

        assert len(results) == 1
