"""Tests for the fact filter (am_diag/retrieval/rerank/facts.py)."""

from __future__ import annotations

from am_diag.common.data_models.search import Fact, SearchResult
from am_diag.retrieval.rerank.facts import filter_facts


def _facts(*scores: float) -> list[SearchResult]:
    return [
        SearchResult(
            item=Fact(
                head_id=str(i), head_name="a", type="T", tail_id="z", tail_name="b"
            ),
            score=score,
            source="relation",
        )
        for i, score in enumerate(scores)
    ]


class TestFilterFacts:
    """Tests for filter_facts."""

    def test_keeps_facts_at_or_above_min_score(self) -> None:
        """Facts below the minimum score are dropped."""
        kept = filter_facts(
            _facts(0.9, 0.5, 0.3, 0.2, 0.1), min_score=0.25, max_facts=12, min_keep=3
        )

        assert [r.score for r in kept] == [0.9, 0.5, 0.3]

    def test_keeps_the_top_facts_when_too_few_pass(self) -> None:
        """When few facts pass, the best ones are kept anyway."""
        kept = filter_facts(
            _facts(0.9, 0.2, 0.1, 0.05), min_score=0.25, max_facts=12, min_keep=3
        )

        assert [r.score for r in kept] == [0.9, 0.2, 0.1]

    def test_caps_the_number_of_facts(self) -> None:
        """No more than max_facts are returned."""
        kept = filter_facts(
            _facts(*[0.9] * 20), min_score=0.25, max_facts=12, min_keep=3
        )

        assert len(kept) == 12

    def test_returns_what_exists_when_below_min_keep(self) -> None:
        """Fewer facts than min_keep are all returned."""
        kept = filter_facts(_facts(0.1), min_score=0.25, max_facts=12, min_keep=3)

        assert [r.score for r in kept] == [0.1]
