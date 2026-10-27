"""Cosine-similarity reranking with caller-supplied vectors."""

from __future__ import annotations

import math
from collections.abc import Sequence

from am_diag.common.data_models.search import SearchResult


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, or 0.0 if either has no length."""
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(x * x for x in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(left, right, strict=False))
    return dot / (left_norm * right_norm)


def cosine_rerank(
    query_vector: Sequence[float],
    results: list[SearchResult],
    vectors: dict[str, list[float]],
    top_k: int | None = None,
) -> list[SearchResult]:
    """Order ``results`` by cosine similarity to ``query_vector``.

    The function makes no embedding calls. A result whose item id has no entry
    in ``vectors`` is left out.

    Args:
        query_vector: The query embedding.
        results: Candidate results whose items have an ``id``.
        vectors: Embedding for each item id, as a string.
        top_k: If set, only the top ``k`` results are returned.

    Returns:
        Results with ``score`` set to the cosine similarity, best first.
    """
    scored: list[tuple[float, SearchResult]] = []
    for result in results:
        vector = vectors.get(str(getattr(result.item, "id", "")))
        if vector is not None:
            scored.append((cosine(query_vector, vector), result))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    reranked: list[SearchResult] = []
    for score, result in scored[:top_k]:
        result.score = score
        reranked.append(result)
    return reranked


__all__ = ["cosine", "cosine_rerank"]
