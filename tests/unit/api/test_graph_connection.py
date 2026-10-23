"""Check backend identity and registry credential resolution without network I/O."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from am_diag.api.connection_pool import build_graph
from am_diag.api.errors import ConnectionError as WorkspaceConnectionError
from am_diag.api.graph import Graph, _derive_graph_id
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.settings import FalkorDBSettings, Neo4jSettings
from am_diag.db.vector.base import VectorStoreBase


class TestGraphConnection:
    """Graph construction preserves backend identity and environment references."""

    def test_neo4j_identity_keeps_existing_digest(self):
        """Retain the cache shard used by existing Neo4j deployments."""
        settings = Neo4jSettings(uri="bolt://h", database="g")
        assert _derive_graph_id(settings) == "auto-491db74612b2"

    def test_falkordb_identity_is_stable_and_backend_specific(self):
        """Different graph keys and servers must not share a content cache."""
        settings = FalkorDBSettings(host="h", graph="g")
        graph_id = _derive_graph_id(settings)
        assert graph_id == _derive_graph_id(settings)
        assert graph_id != _derive_graph_id(Neo4jSettings(uri="bolt://h", database="g"))
        assert graph_id != _derive_graph_id(
            settings.model_copy(update={"graph": "other"})
        )
        assert graph_id != _derive_graph_id(settings.model_copy(update={"port": 6380}))

    def test_connect_accepts_live_store(self):
        """The existing bring-your-own-store contract needs no settings."""
        graph = Graph.connect(
            neo4j=MagicMock(spec=GraphStore),
            vector=MagicMock(spec=VectorStoreBase),
            embedder=SimpleNamespace(),
            graph_id="custom",
        )
        assert graph.graph_id == "custom"

    def test_connect_accepts_falkordb_settings(self):
        """Build a FalkorDB graph through the shared settings factory."""
        settings = FalkorDBSettings(host="h", graph="g")
        store = MagicMock(spec=GraphStore)
        with patch(
            "am_diag.api.graph.create_graph_store", return_value=store
        ) as factory:
            graph = Graph.connect(
                neo4j=settings,
                vector=MagicMock(spec=VectorStoreBase),
                embedder=SimpleNamespace(),
            )
        factory.assert_called_once_with(settings)
        assert graph.graph_id == _derive_graph_id(settings)

    @pytest.mark.parametrize("credentials", [False, True])
    def test_pool_resolves_falkordb_settings(self, monkeypatch, tmp_path, credentials):
        """Resolve env refs or use empty credentials when refs are absent."""
        monkeypatch.setenv("TEST_USER", "env-user")
        monkeypatch.setenv("TEST_PASS", "env-password")
        monkeypatch.setenv("TEST_VECTOR", "http://localhost:6333")
        record = GraphRecord(
            graph_id="medical",
            neo4j=GraphConnection(
                backend="falkordb",
                uri="rediss://host:6380",
                database="medical",
                user_env="TEST_USER" if credentials else None,
                password_env="TEST_PASS" if credentials else None,
            ),
            vector=VectorConnection(url_env="TEST_VECTOR"),
        )
        with patch.object(Graph, "connect") as connect:
            build_graph(record, state_dir=tmp_path, cache=MagicMock(), job_registry={})
        settings = connect.call_args.kwargs["neo4j"]
        assert isinstance(settings, FalkorDBSettings)
        assert (settings.host, settings.port, settings.ssl, settings.graph) == (
            "host",
            6380,
            True,
            "medical",
        )
        assert settings.username == ("env-user" if credentials else "")
        assert settings.password.get_secret_value() == (
            "env-password" if credentials else ""
        )

    def test_pool_rejects_missing_optional_credential_ref(self, monkeypatch, tmp_path):
        """A supplied credential reference must resolve even for FalkorDB."""
        monkeypatch.delenv("UNSET_FALKOR_PASSWORD", raising=False)
        record = GraphRecord(
            graph_id="medical",
            neo4j=GraphConnection(
                backend="falkordb",
                uri="redis://host",
                password_env="UNSET_FALKOR_PASSWORD",
            ),
            vector=VectorConnection(url_env="TEST_VECTOR"),
        )
        with pytest.raises(WorkspaceConnectionError, match="UNSET_FALKOR_PASSWORD"):
            build_graph(record, state_dir=tmp_path, cache=MagicMock(), job_registry={})
