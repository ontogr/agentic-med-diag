"""Tests for the Workspace admin namespaces.

Graph creation builds the live graph eagerly, so the Neo4j driver is patched at
its construction boundary and connection env vars are injected. The vector store
and embedder are lazy and never connect during these tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from am_diag.api.admin import GraphAdmin
from am_diag.api.errors import ConnectionError as WorkspaceConnectionError
from am_diag.api.errors import GraphNotFoundError, RegistryError
from am_diag.api.graph import Graph
from am_diag.api.workspace import Workspace
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.schema import GraphSchema
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.errors import OntologyVersionMismatch
from am_diag.db.graph.settings import Neo4jSettings
from am_diag.db.vector.settings import QdrantSettings


_DRIVER_TARGET = "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver"


@pytest.fixture
def mock_driver() -> MagicMock:
    """Return a MagicMock configured as a healthy Neo4j async driver."""
    driver = MagicMock()
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))
    return driver


@pytest.fixture
def client(tmp_path) -> Workspace:
    """Return an Workspace backed by a temp registry and state dir."""
    return Workspace.open(
        registry_path=tmp_path / "registry.json", state_dir=tmp_path / "state"
    )


def _neo4j() -> GraphConnection:
    return GraphConnection(
        uri="bolt://h:7687",
        user_env="MED_NEO4J_USER",
        password_env="MED_NEO4J_PASSWORD",
    )


def _vector() -> VectorConnection:
    return VectorConnection(url_env="MED_QDRANT_URL", api_key_env="MED_QDRANT_KEY")


def _set_env(monkeypatch) -> None:
    monkeypatch.setenv("MED_NEO4J_USER", "neo4j")
    monkeypatch.setenv("MED_NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("MED_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MED_QDRANT_KEY", "")


def _ontology_store() -> MagicMock:
    store = MagicMock(spec=GraphStore)
    store.health_check = AsyncMock(return_value=True)
    store.setup_schema = AsyncMock()
    store.register_ontology = AsyncMock()
    return store


class TestGraphAdmin:
    """Tests for client.graphs create/list/get and rollback behaviour."""

    async def test_create_round_trips(self, client, mock_driver, monkeypatch):
        """Verify a created graph appears in list and validates connectivity."""
        _set_env(monkeypatch)
        with patch(_DRIVER_TARGET, return_value=mock_driver):
            rec = await client.graphs.create(
                "med", neo4j=_neo4j(), vector=_vector(), name="Medical"
            )
        assert rec.graph_id == "med"
        assert mock_driver.verify_connectivity.await_count == 1
        listed = await client.graphs.list()
        assert [g.graph_id for g in listed] == ["med"]
        await client.aclose()

    async def test_missing_env_var_rolls_back(self, client, mock_driver, monkeypatch):
        """Verify a missing secret env var raises and leaves no dangling record."""
        monkeypatch.delenv("MED_NEO4J_USER", raising=False)
        with (
            patch(_DRIVER_TARGET, return_value=mock_driver),
            pytest.raises(WorkspaceConnectionError),
        ):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        assert await client.graphs.list() == []

    async def test_duplicate_graph_id_raises(self, client, mock_driver, monkeypatch):
        """Verify creating the same graph id twice raises RegistryError."""
        _set_env(monkeypatch)
        with patch(_DRIVER_TARGET, return_value=mock_driver):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
            with pytest.raises(RegistryError):
                await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        await client.aclose()

    async def test_unhealthy_backend_rolls_back_and_closes_driver(
        self, client, mock_driver, monkeypatch
    ):
        """Verify a failed health check removes the record and closes the driver."""
        _set_env(monkeypatch)
        mock_driver.verify_connectivity = AsyncMock(side_effect=OSError("down"))
        with patch(_DRIVER_TARGET, return_value=mock_driver), pytest.raises(OSError):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        assert await client.graphs.list() == []
        mock_driver.close.assert_awaited()

    async def test_ontology_conflict_rolls_back_and_closes_store(
        self, client, monkeypatch
    ):
        """Verify an ontology conflict removes the record and closes the store."""
        _set_env(monkeypatch)
        store = MagicMock(spec=GraphStore)
        store.health_check = AsyncMock(return_value=True)
        store.setup_schema = AsyncMock()
        store.register_ontology = AsyncMock(
            side_effect=OntologyVersionMismatch(stored="3.2", requested="unversioned")
        )
        with (
            patch("am_diag.api.graph.create_graph_store", return_value=store),
            pytest.raises(OntologyVersionMismatch),
        ):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        assert await client.graphs.list() == []
        store.close.assert_awaited()

    async def test_ontology_update_conflict_keeps_record_and_live_graph(
        self, client, monkeypatch
    ):
        """Verify a conflicting schema update keeps the record and the live graph."""
        _set_env(monkeypatch)
        store = _ontology_store()
        with patch("am_diag.api.graph.create_graph_store", return_value=store):
            await client.graphs.create(
                "med",
                neo4j=_neo4j(),
                vector=_vector(),
                schema=GraphSchema(version="1.0"),
            )
        store.register_ontology.side_effect = OntologyVersionMismatch(
            stored="1.0", requested="3.0"
        )

        with pytest.raises(OntologyVersionMismatch):
            await client.graphs.update("med", schema=GraphSchema(version="3.0"))

        assert (await client.graphs.get("med")).graph_schema.version == "1.0"
        store.close.assert_not_awaited()
        await client.aclose()

    async def test_ontology_update_persists_schema_and_evicts_live_graph(
        self, client, monkeypatch
    ):
        """Verify a schema update registers the schema, persists it, and evicts."""
        _set_env(monkeypatch)
        store = _ontology_store()
        with patch("am_diag.api.graph.create_graph_store", return_value=store):
            await client.graphs.create(
                "med",
                neo4j=_neo4j(),
                vector=_vector(),
                schema=GraphSchema(version="1.0"),
            )
        store.register_ontology.reset_mock()
        replacement = GraphSchema(version="2.0")

        await client.graphs.update("med", schema=replacement)

        store.register_ontology.assert_awaited_once_with(replacement)
        assert (await client.graphs.get("med")).graph_schema.version == "2.0"
        store.close.assert_awaited_once()
        await client.aclose()

    async def test_create_rejects_id_of_attached_graph(
        self, client, mock_driver, monkeypatch
    ):
        """Verify create() will not shadow a live graph attached under the same id."""
        _set_env(monkeypatch)
        with patch(_DRIVER_TARGET, return_value=mock_driver):
            attached = Graph.connect(
                neo4j=Neo4jSettings(uri="bolt://h:7687", user="u", password="p"),
                vector=QdrantSettings(url="http://localhost:6333"),
                graph_id="med",
                embedder=SimpleNamespace(close=AsyncMock()),
            )
            client.attach_graph(attached)
            with pytest.raises(RegistryError):
                await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        assert await client.graphs.list() == []

    async def test_delete_unknown_graph_does_not_wipe_data(self, client):
        """Verify delete(drop_data=True) validates the record before touching data."""
        with pytest.raises(GraphNotFoundError):
            await client.graphs.delete("ghost", drop_data=True)

    async def test_drop_data_uses_backend_wipe(self):
        """Graph deletion delegates data removal to the selected backend."""
        registry = MagicMock()
        record = MagicMock(graph_id="med", collections={})
        registry.get_graph.return_value = record
        registry.list_graphs.return_value = [record]
        graph_store = SimpleNamespace(wipe=AsyncMock())
        graph = SimpleNamespace(graph_store=graph_store, collections={})
        pool = MagicMock(evict_graph=AsyncMock())
        pool.graph.return_value = graph

        await GraphAdmin(registry, pool).delete("med", drop_data=True)

        graph_store.wipe.assert_awaited_once_with()
        registry.remove_graph.assert_called_once_with("med")

    async def test_drop_data_refuses_shared_collections(
        self, client, mock_driver, monkeypatch
    ):
        """Verify drop_data is refused when another graph uses the same collections."""
        _set_env(monkeypatch)
        with patch(_DRIVER_TARGET, return_value=mock_driver):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
            await client.graphs.create("drug", neo4j=_neo4j(), vector=_vector())
            with pytest.raises(RegistryError, match="shared"):
                await client.graphs.delete("med", drop_data=True)
        assert {g.graph_id for g in await client.graphs.list()} == {"med", "drug"}
        await client.aclose()


class TestProjectAdmin:
    """Tests for client.projects and project membership management."""

    async def test_create_project_and_add_graph(self, client, mock_driver, monkeypatch):
        """Verify a project can be created and a graph added to it."""
        _set_env(monkeypatch)
        with patch(_DRIVER_TARGET, return_value=mock_driver):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        await client.projects.create("clinical")
        await client.project("clinical").add_graph("med")
        assert await client.project("clinical").graphs() == ["med"]
        await client.aclose()

    async def test_create_project_unknown_graph_raises(self, client):
        """Verify referencing an unknown graph at project creation raises."""
        with pytest.raises(RegistryError):
            await client.projects.create("clinical", graph_ids=["ghost"])
