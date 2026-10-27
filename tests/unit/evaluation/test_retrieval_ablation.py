"""Tests for retrieval ablation scoring (am_diag/evaluation/retrieval_ablation.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.evaluation import QuestionResult, RetrievalQuestion
from am_diag.common.data_models.search import SearchResult
from am_diag.evaluation.retrieval_ablation import (
    first_gold_rank,
    hit_at_k,
    percentile,
    reciprocal_rank,
    run_ablation,
    summarize,
)
from am_diag.retrieval.config import SearchConfig
from am_diag.retrieval.search import SearchEngine


def _result(entity_id: UUID) -> SearchResult:
    return SearchResult(
        item=Entity(id=entity_id, name="x", label="Drug"), score=1.0, source="entity"
    )


class TestMetrics:
    """Tests for the rank metrics."""

    def test_rank_of_first_gold_result(self) -> None:
        """The first result with a gold id sets the rank."""
        gold = uuid4()
        results = [_result(uuid4()), _result(gold), _result(uuid4())]

        assert first_gold_rank(results, frozenset({str(gold)})) == 2

    def test_rank_is_zero_when_no_gold_result(self) -> None:
        """No gold result gives rank 0."""
        assert first_gold_rank([_result(uuid4())], frozenset({"x"})) == 0

    def test_reciprocal_rank(self) -> None:
        """Rank 2 scores one half and a miss scores zero."""
        assert reciprocal_rank(2) == 0.5
        assert reciprocal_rank(0) == 0.0

    @pytest.mark.parametrize(
        ("rank", "expected"), [(10, True), (11, False), (0, False), (1, True)]
    )
    def test_hit_at_k(self, rank: int, expected: bool) -> None:
        """A hit needs a gold result within the first k."""
        assert hit_at_k(rank, 10) is expected

    def test_percentile_uses_nearest_rank(self) -> None:
        """The median and 95th percentile of ten values."""
        values = [float(i) for i in range(1, 11)]

        assert percentile(values, 0.5) == 5.0
        assert percentile(values, 0.95) == 10.0

    def test_summary_averages_questions(self) -> None:
        """Recall and reciprocal rank are means over the rows."""
        rows = [
            QuestionResult("r", "q1", 0, True, 1.0, 10.0),
            QuestionResult("r", "q2", 0, False, 0.0, 30.0),
        ]

        summary = summarize("r", rows)

        assert summary.recall_at_k == 0.5
        assert summary.mean_reciprocal_rank == 0.5


class TestRunAblation:
    """Tests for run_ablation."""

    async def test_runs_each_recipe_question_and_repeat(self) -> None:
        """One result is produced per recipe, question and repeat."""
        gold = uuid4()
        engine = MagicMock(spec=SearchEngine)
        engine.search = AsyncMock(return_value=[_result(gold)])
        questions = [RetrievalQuestion("q1", "text", frozenset({str(gold)}))]
        recipes = {
            "a": SearchConfig(methods=["entity"]),
            "b": SearchConfig(methods=["chunk"]),
        }

        rows = await run_ablation(engine, questions, recipes, k=5, repeats=2)

        assert [(r.recipe, r.repeat) for r in rows] == [
            ("a", 0),
            ("a", 1),
            ("b", 0),
            ("b", 1),
        ]
        assert all(r.hit_at_k and r.reciprocal_rank == 1.0 for r in rows)
