"""Settings select backend clients without opening FalkorDB connections."""

from unittest.mock import MagicMock

import pytest

from am_diag.db.graph.factory import create_graph_store
from am_diag.db.graph.falkordb import FalkorDBClient
from am_diag.db.graph.neo4j import Neo4jClient
from am_diag.db.graph.settings import FalkorDBSettings, Neo4jSettings


class TestGraphStoreFactory:
    """Dispatch preserves Neo4j construction and keeps FalkorDB lazy."""

    def test_selects_falkordb_without_connecting(self, monkeypatch):
        """Construct the FalkorDB adapter without connecting to Redis."""
        driver = MagicMock()
        monkeypatch.setattr("falkordb.asyncio.FalkorDB", driver)
        assert isinstance(create_graph_store(FalkorDBSettings()), FalkorDBClient)
        driver.assert_not_called()

    def test_selects_neo4j(self, monkeypatch):
        """Keep Neo4j settings mapped to the existing driver adapter."""
        monkeypatch.setattr(
            "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver", MagicMock()
        )
        assert isinstance(create_graph_store(Neo4jSettings()), Neo4jClient)

    def test_rejects_unsupported_settings(self):
        """Reject unsupported settings instead of silently choosing a backend."""
        with pytest.raises(TypeError, match="Unsupported graph settings"):
            create_graph_store(None)
