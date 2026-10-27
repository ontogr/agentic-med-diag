"""RRF (Reciprocal Rank Fusion): merge ranked result lists by item id.

Score formula: ``score(item) = sum(1 / (k + rank))`` across all input lists.
"""

from __future__ import annotations

from collections import defaultdict

from am_diag.common.data_models.search import CypherRow, Fact, SearchResult


def item_key(item: object) -> tuple:
    """Return a stable identity key for a result item.

    Two objects that describe the same graph element get the same key, so
    results from different retrievers merge. Items without an ``id`` fall back
    to object identity.
    """
    if isinstance(item, CypherRow):
        return ("cypher", tuple(sorted((k, str(v)) for k, v in item.values.items())))
    if isinstance(item, Fact):
        return ("fact", item.head_id, item.type, item.tail_id)
    item_id = getattr(item, "id", None)
    if item_id is None:
        return (type(item).__name__, id(item))
    return (type(item).__name__, str(item_id))


def rrf(
    result_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """Fuse multiple ranked result lists via reciprocal rank fusion.

    Each result's ``item`` is deduplicated by :func:`item_key`, so the same
    entity returned by two retrievers counts once. When the same item appears in
    multiple lists, its RRF score is the sum of ``1 / (k + rank)``.

    Args:
        result_lists: One or more ranked result lists.
        k: RRF rank constant (default 60).

    Returns:
        Results ordered by descending fused RRF score. Duplicate items
        (by id) are collapsed into a single entry with the max score.
    """
    if not result_lists:
        return []

    scores: dict[tuple, float] = defaultdict(float)
    items: dict[tuple, SearchResult] = {}

    for ranked in result_lists:
        for rank, result in enumerate(ranked, start=1):
            idx = item_key(result.item)
            scores[idx] += 1.0 / (k + rank)
            if idx not in items or result.score > items[idx].score:
                items[idx] = result

    # Sort descending by fused score, update score on result
    sorted_indices = sorted(scores, key=lambda i: -scores[i])
    for _i, idx in enumerate(sorted_indices):
        items[idx].score = scores[idx]

    return [items[idx] for idx in sorted_indices]


__all__ = ["rrf"]
