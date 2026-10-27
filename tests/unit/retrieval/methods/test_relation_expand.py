"""Tests for relation expansion (am_diag/retrieval/methods/relation_expand.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.relation_expand import expand_relations


def _graph(rows: list[dict]) -> MagicMock:
    store = MagicMock(spec=GraphStore)
    store.execute_query = AsyncMock(return_value=rows)
    return store


_TEXT_EDGE = {
    "head_id": "a",
    "head_name": "Metformin",
    "type": "REDUCES",
    "tail_id": "b",
    "tail_name": "Glucose",
    "description": "lowers blood glucose",
    "structural_label": None,
    "sources": [],
    "chunk_ids": ["c1"],
}
_RELEASED_EDGE = {
    "head_id": "a",
    "head_name": "Metformin",
    "type": "INDICATION",
    "tail_id": "d",
    "tail_name": "Type 2 diabetes",
    "description": None,
    "structural_label": "DRG-DIS",
    "sources": ["DrugBank"],
    "chunk_ids": None,
}


async def _facts(rows: list[dict], **kwargs: Any) -> list[Any]:
    results = await expand_relations(
        _graph(rows),
        ["a"],
        hops=kwargs.pop("hops", 1),
        per_seed=10,
        total_limit=kwargs.pop("total_limit", 100),
        exclude_types=[],
        **kwargs,
    )
    return [r.item for r in results]


class TestExpandRelations:
    """Tests for expand_relations."""

    async def test_text_edge_keeps_its_description(self) -> None:
        """A text-extracted edge renders with its sentence."""
        (fact,) = await _facts([_TEXT_EDGE])

        assert fact.text() == "Metformin —[REDUCES]→ Glucose: lowers blood glucose"
        assert fact.chunk_ids == ["c1"]

    async def test_released_edge_renders_from_structure(self) -> None:
        """An edge with no sentence shows the source databases instead."""
        (fact,) = await _facts([_RELEASED_EDGE])

        assert fact.text() == (
            "Metformin —[INDICATION]→ Type 2 diabetes (sources: DrugBank)"
        )
        assert fact.structural_label == "DRG-DIS"

    async def test_two_hop_path_names_the_middle_entity(self) -> None:
        """A two-edge path carries both relation types and the entity between."""
        row = {
            "head_id": "a",
            "head_name": "Metformin",
            "first_type": "INDICATION",
            "via_name": "Type 2 diabetes",
            "second_type": "PHENOTYPE_PRESENT",
            "tail_id": "p",
            "tail_name": "Polyuria",
        }

        (fact,) = await _facts([row], hops=2)

        assert fact.text() == (
            "Metformin —[INDICATION]→ Type 2 diabetes —[PHENOTYPE_PRESENT]→ Polyuria"
        )

    async def test_drops_repeated_edges_and_caps_total(self) -> None:
        """The same edge seen from both endpoints counts once."""
        other = {**_TEXT_EDGE, "type": "TREATS", "tail_id": "z"}

        facts = await _facts([_TEXT_EDGE, _TEXT_EDGE, other], total_limit=1)

        assert [f.type for f in facts] == ["REDUCES"]

    async def test_excluded_types_reach_the_query(self) -> None:
        """Hierarchy edge types are passed to the database to skip."""
        graph = _graph([])

        await expand_relations(
            graph,
            ["a"],
            hops=1,
            per_seed=3,
            total_limit=10,
            exclude_types=["PARENT", "IS_A"],
        )

        params = graph.execute_query.call_args.args[1]
        assert params["exclude"] == ["PARENT", "IS_A"]
        assert params["per_seed"] == 3

    async def test_label_filter_constrains_the_entity_reached(self) -> None:
        """A label filter becomes a condition on the neighbour, not the seed."""
        graph = _graph([])

        await expand_relations(
            graph,
            ["a"],
            hops=1,
            per_seed=3,
            total_limit=10,
            exclude_types=[],
            filters=SearchFilters(labels=["Drug"]),
        )

        query = graph.execute_query.call_args.args[0]
        assert "AND (b:Drug)" in query

    async def test_no_seeds_runs_no_query(self) -> None:
        """Without seeds there is nothing to expand."""
        graph = _graph([])

        results = await expand_relations(
            graph, [], hops=1, per_seed=3, total_limit=10, exclude_types=[]
        )

        assert results == []
        graph.execute_query.assert_not_awaited()

    async def test_rejects_unsupported_hop_count(self) -> None:
        """Only one and two hops exist."""
        with pytest.raises(ValueError, match="hops"):
            await expand_relations(
                _graph([]), ["a"], hops=3, per_seed=1, total_limit=1, exclude_types=[]
            )

    async def test_filter_property_cannot_replace_a_query_parameter(self) -> None:
        """A filter key that matches a query parameter name leaves the query intact."""
        graph = _graph([])

        await expand_relations(
            graph,
            ["a"],
            hops=1,
            per_seed=3,
            total_limit=10,
            exclude_types=[],
            filters=SearchFilters(properties={"ids": "other"}),
        )

        assert graph.execute_query.call_args.args[1]["ids"] == ["a"]

    async def test_unsafe_filter_label_never_reaches_the_query(self) -> None:
        """A label that is not a plain PascalCase name is refused."""
        graph = _graph([])

        with pytest.raises(ValueError, match="label"):
            await expand_relations(
                graph,
                ["a"],
                hops=1,
                per_seed=3,
                total_limit=10,
                exclude_types=[],
                filters=SearchFilters(labels=["Drug) OR (b"]),
            )

        graph.execute_query.assert_not_awaited()
