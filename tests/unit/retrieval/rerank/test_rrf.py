"""Tests for RRF (Reciprocal Rank Fusion)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import Fact, SearchResult
from am_diag.retrieval.rerank.rrf import rrf


_METFORMIN = Entity(id=uuid4(), name="metformin", label="Drug")
_DIABETES = Entity(id=uuid4(), name="diabetes", label="Disease")
_ASPIRIN = Entity(id=uuid4(), name="aspirin", label="Drug")


class TestRRF:
    """Tests for RRF (Reciprocal Rank Fusion)."""

    def test_empty_lists_returns_empty(self) -> None:
        """Verify RRF returns empty when given no input lists."""
        assert rrf([]) == []

    def test_single_list_returns_same_order(self) -> None:
        """Verify a single result list retains its relative ordering."""
        results = [
            SearchResult(item=_METFORMIN, score=0.9, source="test"),
            SearchResult(item=_DIABETES, score=0.8, source="test"),
        ]
        fused = rrf([results])
        assert len(fused) == 2
        assert fused[0].score >= fused[1].score

    def test_fuses_two_lists_with_same_item(self) -> None:
        """Verify the same item in two lists is deduped with a combined RRF score."""
        r1 = SearchResult(item=_METFORMIN, score=0.9, source="entity")
        r2 = SearchResult(item=_METFORMIN, score=0.8, source="relation")
        fused = rrf([[r1], [r2]])
        assert len(fused) == 1  # deduped by item id
        # Fused score = 1/(60+1) + 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(fused[0].score - expected) < 1e-6

    def test_higher_ranked_items_get_higher_score(self) -> None:
        """Verify items appearing in multiple lists get higher fused scores."""
        r1 = SearchResult(item=_METFORMIN, score=0.9, source="e1")
        r2 = SearchResult(item=_DIABETES, score=0.8, source="e1")
        r3 = SearchResult(item=_METFORMIN, score=0.9, source="e2")
        r4 = SearchResult(item=_ASPIRIN, score=0.7, source="e2")
        fused = rrf([[r1, r2], [r3, r4]])
        metformin_score = 2.0 / 61.0
        diabetes_score = 1.0 / 62.0
        aspirin_score = 1.0 / 62.0
        assert abs(fused[0].score - metformin_score) < 1e-6
        assert fused[1].score in (diabetes_score, aspirin_score)

    def test_custom_k_value(self) -> None:
        """Verify a custom k constant affects RRF score calculation."""
        r1 = SearchResult(item=_METFORMIN, score=0.9, source="e1")
        r2 = SearchResult(item=_METFORMIN, score=0.9, source="e2")
        fused = rrf([[r1], [r2]], k=10)
        expected = 2.0 / 11.0
        assert abs(fused[0].score - expected) < 1e-6


class TestRRFIdentity:
    """RRF merges items by id, not by Python object."""

    def test_same_entity_as_two_objects_merges(self) -> None:
        """Two Entity objects with one id count as one item."""
        entity_id = uuid4()
        first = SearchResult(
            item=Entity(id=entity_id, name="x", label="Disease"),
            score=0.9,
            source="entity",
        )
        second = SearchResult(
            item=Entity(id=entity_id, name="x", label="Disease"),
            score=0.8,
            source="bfs",
        )

        fused = rrf([[first], [second]], k=60)

        assert len(fused) == 1
        assert fused[0].score == pytest.approx(2 / 61)

    def test_same_fact_from_two_lists_merges(self) -> None:
        """Facts with one head, type and tail are one item."""
        fact = Fact(head_id="a", head_name="a", type="T", tail_id="b", tail_name="b")

        fused = rrf(
            [
                [SearchResult(item=fact, score=0.9, source="relation")],
                [
                    SearchResult(
                        item=fact.model_copy(), score=0.5, source="relations_1hop"
                    )
                ],
            ]
        )

        assert len(fused) == 1

    def test_different_entities_stay_separate(self) -> None:
        """Entities with different ids are not merged."""
        fused = rrf(
            [
                [SearchResult(item=_METFORMIN, score=0.9, source="entity")],
                [SearchResult(item=_DIABETES, score=0.9, source="entity")],
            ]
        )

        assert len(fused) == 2

    def test_two_hop_facts_with_different_middles_stay_separate(self) -> None:
        """Paths that differ only in the entity between the ends are two items."""

        def path(via_id: str) -> SearchResult:
            fact = Fact(
                head_id="a",
                head_name="a",
                type="T1>T2",
                tail_id="b",
                tail_name="b",
                hops=2,
                via_id=via_id,
                via_name=via_id,
            )
            return SearchResult(item=fact, score=0.5, source="relations_2hop")

        assert len(rrf([[path("m1"), path("m2")]])) == 2
