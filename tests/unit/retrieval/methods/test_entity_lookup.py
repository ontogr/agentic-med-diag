"""Tests for keyword entity lookup (am_diag/retrieval/methods/entity_lookup.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.entity_lookup import entity_lookup


def _row(name: str, entity_id=None) -> dict:
    return {
        "id": str(entity_id or uuid4()),
        "name": name,
        "labels": ["Entity", "Disease"],
        "description": None,
        "curie": None,
    }


def _graph(exact: list[dict], contains: list[dict]) -> MagicMock:
    store = MagicMock(spec=GraphStore)
    store.execute_query = AsyncMock(side_effect=[exact, contains])
    return store


class TestEntityLookup:
    """Tests for entity_lookup."""

    async def test_exact_matches_rank_above_substring_matches(self) -> None:
        """Exact name, alias and synonym hits come before substring hits."""
        graph = _graph([_row("diabetes mellitus")], [_row("diabetes insipidus")])

        results = await entity_lookup(
            graph, ["diabetes mellitus"], exact_per_keyword=3, contains_per_keyword=5
        )

        assert [(r.item.name, r.matched_on) for r in results] == [
            ("diabetes mellitus", "exact"),
            ("diabetes insipidus", "contains"),
        ]
        assert results[0].score > results[1].score

    async def test_an_entity_found_twice_appears_once(self) -> None:
        """The exact hit wins when the substring query returns the same entity."""
        shared = uuid4()
        graph = _graph([_row("asthma", shared)], [_row("asthma", shared)])

        results = await entity_lookup(
            graph, ["asthma"], exact_per_keyword=3, contains_per_keyword=5
        )

        assert [r.matched_on for r in results] == ["exact"]

    async def test_quotas_and_filters_reach_the_queries(self) -> None:
        """Per-keyword quotas are passed to the database for each query."""
        graph = _graph([], [])

        await entity_lookup(
            graph,
            ["a", "b"],
            exact_per_keyword=3,
            contains_per_keyword=5,
            filters=SearchFilters(labels=["Disease"]),
        )

        exact_call, contains_call = graph.execute_query.call_args_list
        assert exact_call.args[1]["per_keyword"] == 3
        assert contains_call.args[1]["per_keyword"] == 5
        assert "AND (e:Disease)" in exact_call.args[0]

    async def test_no_keywords_runs_no_query(self) -> None:
        """An empty keyword list returns nothing without a query."""
        graph = _graph([], [])

        results = await entity_lookup(
            graph, [], exact_per_keyword=3, contains_per_keyword=5
        )

        assert results == []
        graph.execute_query.assert_not_awaited()
