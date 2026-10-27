"""Relevance filter for relation facts."""

from __future__ import annotations

from am_diag.common.data_models.search import SearchResult


def filter_facts(
    facts: list[SearchResult],
    *,
    min_score: float,
    max_facts: int,
    min_keep: int,
) -> list[SearchResult]:
    """Keep the facts that score at least ``min_score``.

    When fewer than ``min_keep`` facts pass, the top ``min_keep`` are kept anyway
    so the model still gets some evidence.

    Args:
        facts: Fact results scored by similarity to the query.
        min_score: Lowest score that passes.
        max_facts: Maximum facts returned.
        min_keep: Minimum facts returned when that many exist.

    Returns:
        Facts in descending score order.
    """
    ranked = sorted(facts, key=lambda result: result.score, reverse=True)
    kept = [result for result in ranked if result.score >= min_score]
    if len(kept) < min_keep:
        kept = ranked[:min_keep]
    return kept[:max_facts]


__all__ = ["filter_facts"]
