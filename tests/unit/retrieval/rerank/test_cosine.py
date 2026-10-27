"""Tests for cosine reranking (am_diag/retrieval/rerank/cosine.py)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.search import SearchResult
from am_diag.retrieval.rerank.cosine import cosine, cosine_rerank


def _chunk_result(text: str) -> SearchResult:
    chunk = Chunk(
        id=uuid4(),
        text=text,
        document_id=uuid4(),
        document_source="s",
        chunk_index=0,
        chunk_size=len(text),
        cut_type="x",
    )
    return SearchResult(item=chunk, score=0.0, source="chunk")


class TestCosineRerank:
    """Tests for cosine_rerank."""

    def test_orders_by_similarity_to_the_query(self) -> None:
        """Chunks are ranked by cosine and take the cosine as their score."""
        a, b, c = _chunk_result("a"), _chunk_result("b"), _chunk_result("c")
        vectors = {
            str(a.item.id): [1.0, 0.0],
            str(b.item.id): [0.0, 1.0],
            str(c.item.id): [0.6, 0.8],
        }

        ranked = cosine_rerank([1.0, 0.0], [b, c, a], vectors)

        assert [r.item.text for r in ranked] == ["a", "c", "b"]
        assert [r.score for r in ranked] == pytest.approx([1.0, 0.6, 0.0])

    def test_leaves_out_results_without_a_vector(self) -> None:
        """A result with no stored vector is dropped."""
        a, b = _chunk_result("a"), _chunk_result("b")

        ranked = cosine_rerank([1.0, 0.0], [a, b], {str(a.item.id): [1.0, 0.0]})

        assert [r.item.text for r in ranked] == ["a"]

    def test_top_k_limits_the_output(self) -> None:
        """Only the best top_k results are returned."""
        a, b = _chunk_result("a"), _chunk_result("b")
        vectors = {str(a.item.id): [1.0, 0.0], str(b.item.id): [0.0, 1.0]}

        ranked = cosine_rerank([1.0, 0.0], [b, a], vectors, top_k=1)

        assert [r.item.text for r in ranked] == ["a"]


class TestCosine:
    """Tests for cosine."""

    def test_zero_vector_scores_zero(self) -> None:
        """A zero vector has no direction, so similarity is 0."""
        assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
