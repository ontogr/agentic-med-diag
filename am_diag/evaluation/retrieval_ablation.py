"""Retrieval ablation: score recipes on questions that have gold item ids."""

from __future__ import annotations

import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from am_diag.common.data_models.search import SearchResult
from am_diag.retrieval.config import SearchConfig
from am_diag.retrieval.search import SearchEngine


@dataclass(frozen=True)
class RetrievalQuestion:
    """A question with the ids of the items that answer it."""

    question_id: str
    text: str
    gold_ids: frozenset[str]


@dataclass(frozen=True)
class QuestionResult:
    """Outcome of one recipe run on one question."""

    recipe: str
    question_id: str
    repeat: int
    hit_at_k: bool
    reciprocal_rank: float
    latency_ms: float


@dataclass(frozen=True)
class RecipeSummary:
    """Scores for one recipe, averaged over its questions and repeats."""

    recipe: str
    recall_at_k: float
    mean_reciprocal_rank: float
    latency_p50_ms: float
    latency_p95_ms: float


def first_gold_rank(results: Sequence[SearchResult], gold_ids: frozenset[str]) -> int:
    """Return the 1-based rank of the first result whose item id is gold, or 0."""
    for rank, result in enumerate(results, start=1):
        item = result.item
        if str(getattr(item, "id", "")) in gold_ids:
            return rank
        head, tail = getattr(item, "head_id", None), getattr(item, "tail_id", None)
        if {str(head), str(tail)} & gold_ids:
            return rank
    return 0


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
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
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
    "QuestionResult",
    "RecipeSummary",
    "RetrievalQuestion",
    "first_gold_rank",
    "hit_at_k",
    "percentile",
    "reciprocal_rank",
    "run_ablation",
    "summarize",
]
