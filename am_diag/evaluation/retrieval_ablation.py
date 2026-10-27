"""Retrieval ablation: score recipes on questions that have gold item ids."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Mapping, Sequence

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.evaluation import (
    QuestionResult,
    RecipeSummary,
    RetrievalQuestion,
)
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.search import CypherRow, Fact, SearchResult
from am_diag.retrieval.config import SearchConfig
from am_diag.retrieval.search import SearchEngine


def first_gold_rank(results: Sequence[SearchResult], gold_ids: frozenset[str]) -> int:
    """Return the 1-based rank of the first gold result, or 0.

    An entity, chunk or community report is gold when its id is. A fact is gold
    only when both of its endpoint entities are, so an expansion fact does not
    count just because it starts at a gold seed. A relation is gold when its own
    id is, or when both of its endpoints are. A Cypher row is never gold.
    """
    for rank, result in enumerate(results, start=1):
        if _is_gold(result.item, gold_ids):
            return rank
    return 0


def _is_gold(
    item: Entity | Relation | Chunk | CommunityReport | CypherRow | Fact,
    gold_ids: frozenset[str],
) -> bool:
    """Return True when ``item`` counts as a gold result."""
    if isinstance(item, CypherRow):
        return False
    if isinstance(item, Fact):
        return {item.head_id, item.tail_id} <= gold_ids
    if str(item.id) in gold_ids:
        return True
    if isinstance(item, Relation) and item.head_id and item.tail_id:
        return {str(item.head_id), str(item.tail_id)} <= gold_ids
    return False


def hit_at_k(rank: int, k: int) -> bool:
    """Return True when a gold item appears within the first ``k`` results."""
    return 0 < rank <= k


def reciprocal_rank(rank: int) -> float:
    """Return ``1 / rank``, or 0.0 when no gold item was found."""
    return 1.0 / rank if rank > 0 else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of ``values`` for a fraction in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(recipe: str, rows: Sequence[QuestionResult]) -> RecipeSummary:
    """Average the per-question results of one recipe."""
    latencies = [row.latency_ms for row in rows]
    return RecipeSummary(
        recipe=recipe,
        recall_at_k=statistics.fmean(row.hit_at_k for row in rows) if rows else 0.0,
        mean_reciprocal_rank=(
            statistics.fmean(row.reciprocal_rank for row in rows) if rows else 0.0
        ),
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
    )


async def run_ablation(
    engine: SearchEngine,
    questions: Sequence[RetrievalQuestion],
    recipes: Mapping[str, SearchConfig],
    *,
    k: int = 10,
    repeats: int = 3,
) -> list[QuestionResult]:
    """Run every recipe on every question ``repeats`` times.

    Build the recipe set with ``SearchConfig.model_copy`` to compare a method
    alone, or a recipe with one method removed.

    Args:
        engine: Search engine to evaluate.
        questions: Questions with gold item ids.
        recipes: Recipes to compare, by name.
        k: Cut-off for the hit measure.
        repeats: Runs per recipe and question, to show run-to-run spread.

    Returns:
        One result per recipe, question and repeat.
    """
    rows: list[QuestionResult] = []
    for name, recipe in recipes.items():
        for repeat in range(repeats):
            for question in questions:
                started = time.perf_counter()
                results = await engine.search(question.text, recipe=recipe)
                latency_ms = (time.perf_counter() - started) * 1000
                rank = first_gold_rank(results, question.gold_ids)
                rows.append(
                    QuestionResult(
                        recipe=name,
                        question_id=question.question_id,
                        repeat=repeat,
                        hit_at_k=hit_at_k(rank, k),
                        reciprocal_rank=reciprocal_rank(rank),
                        latency_ms=latency_ms,
                    )
                )
    return rows


__all__ = [
    "first_gold_rank",
    "hit_at_k",
    "percentile",
    "reciprocal_rank",
    "run_ablation",
    "summarize",
]
