"""Tests for federated (cross-graph) project search.

Two graphs are registered in a project and backed by real Graph objects with
fake SearchEngines returning distinct entities, so cross-graph fusion, provenance
tagging, subset targeting, and membership validation can be checked offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from am_diag.api.errors import GraphNotFoundError
from am_diag.api.graph import Graph
from am_diag.api.registry import Registry
from am_diag.api.workspace import Workspace
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.project import ProjectRecord
from am_diag.common.data_models.search import CypherRow, SearchResult
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA


_COLLECTIONS = {"entity": "e", "relation": "r", "chunk": "c", "community": "k"}


def _graph_record(graph_id: str) -> GraphRecord:
    return GraphRecord(
        graph_id=graph_id,
        neo4j=GraphConnection(uri="bolt://h", user_env="U", password_env="P"),
        vector=VectorConnection(url_env="URL"),
    )


def _fake_graph(tmp_path, graph_id: str, entity_name: str) -> Graph:
    result = SearchResult(
        item=Entity(name=entity_name, label="Disease"), score=1.0, source="entity"
    )
    return Graph(
        graph_store=SimpleNamespace(close=AsyncMock()),
        vector_store=SimpleNamespace(close=AsyncMock()),
        embedder=SimpleNamespace(model_id="fake-model", close=AsyncMock()),
        search_engine=SimpleNamespace(search=AsyncMock(return_value=[result])),
        schema=MEDICAL_GRAPHRAG_SCHEMA,
        collections=_COLLECTIONS,
        graph_id=graph_id,
        state_dir=tmp_path / "state",
    )


def _setup(tmp_path):
    """Register graphs a, b in project p; preload real graphs over fakes."""
    registry = Registry.open(tmp_path / "registry.json")
    client = Workspace(registry, state_dir=tmp_path / "state")
    registry.add_graph(_graph_record("a"))
    registry.add_graph(_graph_record("b"))
    registry.add_graph(_graph_record("unreachable"))
    registry.add_project(ProjectRecord(project_id="p", graph_ids=["a", "b"]))
    registry.add_project(ProjectRecord(project_id="q", graph_ids=["a", "unreachable"]))
    client.attach_graph(_fake_graph(tmp_path, "a", "from_a"))
    client.attach_graph(_fake_graph(tmp_path, "b", "from_b"))
    return client


class TestFederatedSearch:
    """Tests for Project.search federation."""

    async def test_search_fuses_both_graphs(self, tmp_path):
        """Verify results come from both graphs with distinct provenance."""
        client = _setup(tmp_path)
        results = await client.project("p").search("q")
        graph_ids = {r.source_graph_id for r in results}
        assert graph_ids == {"a", "b"}
        names = {r.item.name for r in results}
        assert names == {"from_a", "from_b"}

    async def test_subset_restricts_targets(self, tmp_path):
        """Verify a graphs subset only queries the named graphs."""
        client = _setup(tmp_path)
        results = await client.project("p").search("q", graphs=["a"])
        assert {r.source_graph_id for r in results} == {"a"}
        client.graph("b").search_engine.search.assert_not_awaited()

    async def test_subset_does_not_connect_other_members(self, tmp_path):
        """Verify members outside the subset are never built from the registry."""
        client = _setup(tmp_path)
        results = await client.project("q").search("q", graphs=["a"])
        assert {r.source_graph_id for r in results} == {"a"}

    async def test_limit_raises_per_graph_candidate_limit(self, tmp_path):
        """Verify a federated limit above the recipe cap reaches each graph."""
        client = _setup(tmp_path)
        await client.project("p").search("q", limit=100)
        recipe = client.graph("a").search_engine.search.await_args.kwargs["recipe"]
        assert recipe.limit == 100

    async def test_non_member_graph_raises(self, tmp_path):
        """Verify requesting a non-member graph raises."""
        client = _setup(tmp_path)
        with pytest.raises(GraphNotFoundError):
            await client.project("p").search("q", graphs=["ghost"])

    async def test_context_packs_federated_results(self, tmp_path):
        """Verify context() packs the federated results within budget."""
        client = _setup(tmp_path)
        ctx = await client.project("p").context("q", max_characters=500)
        assert len(ctx) <= 500
        assert "from_a" in ctx
        assert "from_b" in ctx


class TestFederatedCypherRows:
    """Cypher rows skip cross-graph fusion."""

    async def test_rows_come_first_and_keep_their_graph_id(self, tmp_path):
        """Verify each graph's Cypher rows lead the result, unfused."""
        client = _setup(tmp_path)
        for graph_id in ("a", "b"):
            row = SearchResult(
                item=CypherRow(values={"graph": graph_id}),
                score=0.01,
                source="cypher",
            )
            entity = SearchResult(
                item=Entity(name=f"from_{graph_id}", label="Disease"),
                score=1.0,
                source="entity",
            )
            client.graph(graph_id).search_engine.search = AsyncMock(
                return_value=[row, entity]
            )

        results = await client.project("p").search("q", limit=1)

        assert [r.source for r in results[:2]] == ["cypher", "cypher"]
        assert {(r.item.values["graph"], r.source_graph_id) for r in results[:2]} == {
            ("a", "a"),
            ("b", "b"),
        }
        assert len(results) == 3
