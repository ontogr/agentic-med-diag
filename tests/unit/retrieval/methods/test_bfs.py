"""Tests for BFS graph expansion (am_diag/retrieval/methods/bfs.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from am_diag.common.data_models.entity import Entity
from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.bfs import bfs_expand


def _graph(rows: list[dict]) -> MagicMock:
    store = MagicMock(spec=GraphStore)
    store.execute_query = AsyncMock(return_value=rows)
    return store


class TestBfsExpand:
    """Tests for bfs_expand."""

    @pytest.mark.parametrize(("depth", "bound"), [(99, 5), (0, 1), (2, 2)])
    async def test_bounds_traversal_depth(self, depth: int, bound: int) -> None:
        """Traversal uses a bounded literal rather than an invalid parameter."""
        graph = _graph([])

        assert await bfs_expand(graph, ["a"], depth=depth) == []

        query, params = graph.execute_query.call_args.args
        assert f"*1..{bound}]" in query
        assert params["depth"] == bound

    async def test_returns_no_results_without_seeds(self) -> None:
        """No seeds means no query and no results."""
        graph = _graph([])

        assert await bfs_expand(graph, []) == []
        graph.execute_query.assert_not_called()

    async def test_neighbours_become_typed_entities(self) -> None:
        """Each neighbour row becomes an Entity labelled by its type label."""
        node_id = uuid4()
        graph = _graph(
            [
                {
                    "id": str(node_id),
                    "label": "Drug",
                    "name": "metformin",
                    "hop": 2,
                }
            ]
        )

        results = await bfs_expand(graph, [str(uuid4())])

        item = results[0].item
        assert isinstance(item, Entity)
        assert (item.id, item.name, item.label) == (node_id, "metformin", "Drug")
        assert results[0].score == 1.0 / 3
        assert results[0].matched_on == "hop_2"

    async def test_node_without_type_label_falls_back_to_entity(self) -> None:
        """A node with only the base label is labelled Entity."""
        graph = _graph([{"id": str(uuid4()), "label": None, "name": "x", "hop": 1}])

        results = await bfs_expand(graph, [str(uuid4())])

        assert results[0].item.label == "Entity"

    async def test_filters_join_the_existing_hop_predicate(self) -> None:
        """Filter conditions extend the template WHERE instead of adding one."""
        graph = _graph([])

        await bfs_expand(graph, ["a"], filters=SearchFilters(labels=["A", "B"]))

        query, params = graph.execute_query.call_args.args
        assert query.count("WHERE") == 1
        assert "WHERE hop <= $depth AND (n:A OR n:B)" in query
        assert params["depth"] == 2
