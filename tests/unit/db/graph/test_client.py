"""Tests for Neo4jClient and create_neo4j_client."""

from __future__ import annotations

import inspect
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import RoutingControl

from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.common.data_models.schema import EntityType, GraphSchema
from am_diag.db.graph.neo4j import (
    Neo4jClient,
    Neo4jTransaction,
    create_neo4j_client,
)
from am_diag.db.graph.settings import Neo4jSettings


_PATCH_TARGET = "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver"


@pytest.fixture
def settings() -> Neo4jSettings:
    """Return Neo4jSettings with dummy credentials for testing."""
    return Neo4jSettings(uri="bolt://localhost:7687", user="neo4j", password="pass")


@pytest.fixture
def mock_driver() -> MagicMock:
    """Return a MagicMock configured as a Neo4j async driver."""
    driver = MagicMock()
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.execute_query = AsyncMock()
    return driver


@pytest.fixture
def client(settings: Neo4jSettings, mock_driver: MagicMock) -> Neo4jClient:
    """Return a Neo4jClient with a patched driver for testing."""
    with patch(_PATCH_TARGET, return_value=mock_driver):
        return Neo4jClient(settings)


class TestNeo4jClient:
    """Tests for Neo4jClient query execution and lifecycle."""

    async def test_execute_query_returns_list_of_dicts(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Verify execute_query returns a list of dicts from records."""
        record = MagicMock()
        record.data.return_value = {"name": "Alice", "age": 30}
        mock_result = MagicMock()
        mock_result.records = [record]
        mock_driver.execute_query.return_value = mock_result

        results = await client.execute_query(
            "MATCH (n:Person) RETURN n.name AS name, n.age AS age",
        )

        assert results == [{"name": "Alice", "age": 30}]

    async def test_execute_query_passes_cypher_and_params(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Verify Cypher and parameters are forwarded to the driver."""
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        await client.execute_query("MATCH (n {id: $id}) RETURN n", {"id": "abc"})

        call_args = mock_driver.execute_query.call_args
        assert call_args.kwargs["parameters_"] == {"id": "abc"}
        assert call_args.kwargs["database_"] == "neo4j"

    async def test_execute_query_defaults_to_write_routing(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Verify queries use write routing unless read-only is requested."""
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        await client.execute_query("MATCH (n) RETURN n")

        routing = mock_driver.execute_query.call_args.kwargs["routing_"]
        assert routing == RoutingControl.WRITE

    async def test_execute_query_read_only_uses_read_routing(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Verify read_only routes the query as a read transaction."""
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        await client.execute_query("MATCH (n) RETURN n", read_only=True)

        routing = mock_driver.execute_query.call_args.kwargs["routing_"]
        assert routing == RoutingControl.READ

    async def test_execute_query_returns_empty_list_on_no_results(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Verify query with no results returns an empty list."""
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        results = await client.execute_query("MATCH (n) RETURN n LIMIT 0")

        assert results == []

    async def test_execute_query_returns_multiple_records(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Verify query returning multiple records produces correct results."""
        records = [MagicMock(), MagicMock()]
        records[0].data.return_value = {"id": 1}
        records[1].data.return_value = {"id": 2}
        mock_result = MagicMock()
        mock_result.records = records
        mock_driver.execute_query.return_value = mock_result

        results = await client.execute_query("MATCH (n) RETURN id(n) AS id")

        assert results == [{"id": 1}, {"id": 2}]

    async def test_close_delegates_to_driver(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Verify close delegates to the underlying driver."""
        await client.close()
        mock_driver.close.assert_called_once()

    async def test_health_check_calls_verify_connectivity(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Verify health_check calls verify_connectivity and returns True."""
        result = await client.health_check()
        mock_driver.verify_connectivity.assert_called_once()
        assert result is True

    async def test_async_context_manager_closes_on_exit(
        self,
        settings: Neo4jSettings,
        mock_driver: MagicMock,
    ) -> None:
        """Verify async context manager closes the driver on exit."""
        with patch(_PATCH_TARGET, return_value=mock_driver):
            async with Neo4jClient(settings):
                pass
        mock_driver.close.assert_called_once()

    async def test_async_context_manager_returns_client(
        self,
        settings: Neo4jSettings,
        mock_driver: MagicMock,
    ) -> None:
        """Verify async context manager returns the Neo4jClient instance."""
        with patch(_PATCH_TARGET, return_value=mock_driver):
            async with Neo4jClient(settings) as c:
                assert isinstance(c, Neo4jClient)

    def test_database_defaults_to_neo4j(
        self,
        settings: Neo4jSettings,
        mock_driver: MagicMock,
    ) -> None:
        """Verify default database name is 'neo4j'."""
        with patch(_PATCH_TARGET, return_value=mock_driver):
            c = Neo4jClient(settings)
        assert c._database == "neo4j"

    def test_custom_database_stored_on_client(self, mock_driver: MagicMock) -> None:
        """Verify custom database name from settings is stored on the client."""
        settings = Neo4jSettings(
            uri="bolt://localhost:7687",
            user="u",
            password="p",
            database="mydb",
        )
        with patch(_PATCH_TARGET, return_value=mock_driver):
            c = Neo4jClient(settings)
        assert c._database == "mydb"

    @pytest.mark.parametrize("constraint", ["onto_entity_label", "onto_meta_key"])
    async def test_setup_schema_creates_ontology_mirror_constraints(
        self,
        client: Neo4jClient,
        mock_driver: MagicMock,
        constraint: str,
    ) -> None:
        """Verify setup_schema issues the uniqueness constraints for the mirror."""
        await client.setup_schema(GraphSchema())

        executed = [call.args[0] for call in mock_driver.execute_query.call_args_list]
        assert any(f"CREATE CONSTRAINT {constraint} " in cypher for cypher in executed)


class TestCreateNeo4jClient:
    """Tests for create_neo4j_client factory function."""

    def test_returns_neo4j_client_instance(
        self,
        settings: Neo4jSettings,
        mock_driver: MagicMock,
    ) -> None:
        """Verify create_neo4j_client returns a Neo4jClient instance."""
        with patch(_PATCH_TARGET, return_value=mock_driver):
            client = create_neo4j_client(settings)
        assert isinstance(client, Neo4jClient)

    def test_explicit_settings_used(self, mock_driver: MagicMock) -> None:
        """Verify explicit settings are used when provided."""
        settings = Neo4jSettings(
            uri="bolt://custom:7687",
            user="u",
            password="p",
            database="db",
        )
        with patch(_PATCH_TARGET, return_value=mock_driver):
            client = create_neo4j_client(settings)
        assert client._database == "db"

    def test_loads_settings_from_env_when_none(self, mock_driver: MagicMock) -> None:
        """Verify settings are loaded from env when none are provided."""
        with patch(_PATCH_TARGET, return_value=mock_driver):
            client = create_neo4j_client()
        assert isinstance(client, Neo4jClient)


@pytest.fixture
def mock_transaction(mock_driver: MagicMock):
    """Wire a mock session and transaction onto mock_driver for transaction() tests."""
    mock_tx = AsyncMock()
    mock_tx.commit = AsyncMock()
    mock_tx.rollback = AsyncMock()

    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[])
    mock_tx.run = AsyncMock(return_value=mock_result)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin_transaction = AsyncMock(return_value=mock_tx)

    mock_driver.session = MagicMock(return_value=mock_session)

    return mock_session, mock_tx


class TestTransaction:
    """Tests for Neo4jClient.transaction context manager."""

    async def test_commits_on_success(self, client, mock_driver, mock_transaction):
        """Verify successful transaction commits."""
        _, mock_tx = mock_transaction
        async with client.transaction():
            pass
        mock_tx.commit.assert_called_once()
        mock_tx.rollback.assert_not_called()

    async def test_rolls_back_on_exception(self, client, mock_driver, mock_transaction):
        """Verify transaction rolls back on exception."""
        _, mock_tx = mock_transaction
        with pytest.raises(RuntimeError):
            async with client.transaction():
                raise RuntimeError("boom")
        mock_tx.rollback.assert_called_once()
        mock_tx.commit.assert_not_called()

    async def test_yields_neo4j_transaction(
        self,
        client,
        mock_driver,
        mock_transaction,
    ):
        """Verify transaction context yields a Neo4jTransaction instance."""
        async with client.transaction() as tx:
            assert isinstance(tx, Neo4jTransaction)

    async def test_tx_run_returns_records(self, client, mock_driver, mock_transaction):
        """Verify Neo4jTransaction.run returns records."""
        mock_session, mock_tx = mock_transaction
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"id": 1}])
        mock_tx.run = AsyncMock(return_value=mock_result)

        async with client.transaction() as tx:
            results = await tx.run("RETURN 1 AS id")

        assert results == [{"id": 1}]

    async def test_session_uses_configured_database(
        self,
        client,
        mock_driver,
        mock_transaction,
    ):
        """Verify session uses the configured database name."""
        async with client.transaction():
            pass
        mock_driver.session.assert_called_once_with(database="neo4j")


class TestExecuteBatch:
    """Tests for Neo4jClient.execute_batch batching logic."""

    async def test_batch_splits_large_list(self, client, mock_driver):
        """Verify large record lists are split into multiple batches."""
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        records = [{"id": str(i)} for i in range(1200)]
        await client.execute_batch(
            "UNWIND $records AS r MERGE (n {id: r.id})",
            records,
            batch_size=500,
        )

        # 1200 / 500 = 3 batches (500 + 500 + 200)
        assert mock_driver.execute_query.call_count == 3

    async def test_batch_sizes_are_correct(self, client, mock_driver):
        """Verify each batch contains the expected number of records."""
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        records = [{"id": str(i)} for i in range(1200)]
        await client.execute_batch(
            "UNWIND $records AS r MERGE (n {id: r.id})",
            records,
            batch_size=500,
        )

        calls = mock_driver.execute_query.call_args_list
        assert len(calls[0].kwargs["parameters_"]["records"]) == 500
        assert len(calls[1].kwargs["parameters_"]["records"]) == 500
        assert len(calls[2].kwargs["parameters_"]["records"]) == 200

    async def test_empty_records_is_noop(self, client, mock_driver):
        """Verify empty records list results in no Cypher calls."""
        await client.execute_batch("UNWIND $records AS r MERGE (n {id: r.id})", [])
        mock_driver.execute_query.assert_not_called()

    async def test_under_batch_limit_single_call(self, client, mock_driver):
        """Verify records under the batch limit make a single call."""
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        records = [{"id": str(i)} for i in range(10)]
        await client.execute_batch(
            "UNWIND $records AS r MERGE (n {id: r.id})",
            records,
            batch_size=500,
        )

        mock_driver.execute_query.assert_called_once()


class TestCypherLoadedFromFiles:
    """Guard: verify no multi-line Cypher literal strings exist in db/*.py."""

    def test_no_embedded_cypher_in_neo4j_module(self):
        """Verify db/*.py has no embedded multi-line Cypher strings."""
        import am_diag.db.graph.neo4j as module  # noqa: PLC0415

        src = inspect.getsource(module)
        # Multi-line strings containing Cypher DML keywords are a red flag.
        cypher_pattern = re.compile(
            r'""".*?(MATCH|MERGE|CREATE|UNWIND).*?"""',
            re.DOTALL,
        )
        assert not cypher_pattern.search(src), (
            "Found embedded Cypher string in neo4j.py — load from .cypher files instead"
        )

    async def test_upsert_documents_loads_cypher_file(self, client, mock_driver):
        """Verify upsert_documents calls execute_query with file-loaded cypher."""
        from am_diag.common.cypher.loader import load  # noqa: PLC0415
        from am_diag.common.data_models.document import Document  # noqa: PLC0415

        expected_cypher = load("ingest/upsert_documents")

        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        doc = Document(text="t", source="s", external_id="e1")
        await client.upsert_documents([doc])

        call_cypher = mock_driver.execute_query.call_args.args[0]
        assert call_cypher == expected_cypher


def _make_tracker(call_order: list, name: str):
    """Return an AsyncMock that appends name to call_order when awaited."""

    async def _track(*args, **kwargs):  # noqa: ANN002, ANN003
        call_order.append(name)

    return _track


class TestWriteKnowledgeGraphOrdering:
    """Verify write_knowledge_graph: constraints then nodes then rels."""

    async def test_setup_schema_called_before_upserts(self, client, mock_driver):
        """Verify setup_schema is called before all upsert operations."""
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query.return_value = mock_result

        schema = GraphSchema(
            entity_types=[
                EntityType(label="Drug", gliner_label="drug", description="d"),
            ],
        )
        kg = KnowledgeGraph(graph_schema=schema)

        call_order: list[str] = []
        method_names = [
            "setup_schema",
            "upsert_documents",
            "upsert_chunks",
            "upsert_entities",
            "upsert_communities",
            "upsert_reports",
            "link_part_of",
            "link_next_chunk",
            "link_has_entity",
            "upsert_relations",
            "link_in_community",
            "link_parent_community",
            "link_has_report",
        ]

        patches = [
            patch.object(client, name, side_effect=_make_tracker(call_order, name))
            for name in method_names
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            patches[11],
            patches[12],
        ):
            await client.write_knowledge_graph(kg)

        node_methods = {
            "upsert_documents",
            "upsert_chunks",
            "upsert_entities",
            "upsert_communities",
            "upsert_reports",
        }
        rel_methods = {
            "link_part_of",
            "link_next_chunk",
            "link_has_entity",
            "upsert_relations",
            "link_in_community",
            "link_parent_community",
            "link_has_report",
        }

        schema_idx = call_order.index("setup_schema")
        for nm in node_methods:
            assert call_order.index(nm) > schema_idx, (
                f"{nm} was called before setup_schema"
            )
        last_node = max(call_order.index(nm) for nm in node_methods)
        for rm in rel_methods:
            assert call_order.index(rm) > last_node, (
                f"{rm} was called before node upserts"
            )
