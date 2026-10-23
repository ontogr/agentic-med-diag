"""Driver-boundary checks for nullable parameters, retries and index lifecycle.

Failures include malformed headers, rejected writes, permanent query errors,
exhausted connection retries, stale index state after deletion and dirty props.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.schema import GraphSchema
from am_diag.db.graph.errors import (
    GraphQueryRejected,
    GraphUnavailableError,
    GraphWriteRejected,
)
from am_diag.db.graph.falkordb import FalkorDBClient
from am_diag.db.graph.settings import FalkorDBSettings


@pytest.fixture
def driver_boundary(monkeypatch):
    """Replace only the SDK and Redis constructors, leaving transport logic real."""
    graph = SimpleNamespace(
        name="unit_graph",
        query=AsyncMock(return_value=SimpleNamespace(header=[], result_set=[])),
        ro_query=AsyncMock(return_value=SimpleNamespace(header=[], result_set=[])),
        execute_command=AsyncMock(),
        delete=AsyncMock(),
    )
    connection = SimpleNamespace(ping=AsyncMock(return_value=True), aclose=AsyncMock())
    driver = SimpleNamespace(
        select_graph=MagicMock(return_value=graph), connection=connection
    )
    constructor = MagicMock(return_value=driver)
    pool = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr("falkordb.asyncio.FalkorDB", constructor)
    monkeypatch.setattr("redis.asyncio.ConnectionPool", MagicMock(return_value=pool))
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    return graph, constructor, connection, pool


class TestFalkorDBClient:
    """Exercise public transport methods against a fake SDK boundary."""

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ([[1, b"a"], [1, "b"]], {"a": 1, "b": ["x"]}),
            (["a", "b"], {"a": 1, "b": ["x"]}),
            (None, {"0": 1, "1": ["x"]}),
        ],
    )
    async def test_normalizes_headers_and_values(
        self, driver_boundary, header, expected
    ):
        """Decode names and nested bytes even when column names are absent."""
        graph, *_ = driver_boundary
        graph.ro_query.return_value = SimpleNamespace(
            header=header, result_set=[[1, [b"x"]]]
        )
        client = FalkorDBClient(FalkorDBSettings())
        assert await client.execute_query("RETURN 1", read_only=True) == [expected]
        graph.query.assert_not_awaited()

    async def test_cleans_properties_without_removing_nullable_bindings(
        self, driver_boundary
    ):
        """Preserve cursor nulls while making node properties server-safe."""
        graph, *_ = driver_boundary
        client = FalkorDBClient(FalkorDBSettings())
        await client.execute_query(
            "RETURN $cursor",
            {
                "cursor": None,
                "props": {
                    "a": 1,
                    "b": None,
                    "s": "x\x00y",
                    "items": [1, {}, None, "a\x01b"],
                    "nested": {"x": 1, "absent": None},
                },
                "records": [{"id": "a", "props": {"text": "a\x7fb", "absent": None}}],
            },
        )
        assert graph.query.await_args.args[1] == {
            "cursor": None,
            "props": {"a": 1, "s": "xy", "items": [1, "ab"], "nested": '{"x": 1}'},
            "records": [{"id": "a", "props": {"text": "ab"}}],
        }

    async def test_retries_connections_and_returns_eventual_result(
        self, driver_boundary
    ):
        """A transient failure can recover within the retry budget."""
        graph, *_ = driver_boundary
        graph.query.side_effect = [
            RedisConnectionError("offline"),
            SimpleNamespace(header=["x"], result_set=[[2]]),
        ]
        client = FalkorDBClient(FalkorDBSettings())
        assert await client.execute_query("RETURN 2") == [{"x": 2}]
        assert graph.query.await_count == 2

    @pytest.mark.parametrize(
        ("error", "expected", "attempts"),
        [
            (ResponseError("Invalid input"), GraphQueryRejected, 1),
            (
                ResponseError(
                    "graph.RO_QUERY is to be executed only on read-only queries"
                ),
                GraphWriteRejected,
                1,
            ),
            (RedisConnectionError("offline"), GraphUnavailableError, 3),
        ],
    )
    async def test_maps_errors_and_bounds_retries(
        self, driver_boundary, error, expected, attempts
    ):
        """Retry unavailable connections but never repeat server refusals."""
        graph, *_ = driver_boundary
        graph.ro_query.side_effect = error
        client = FalkorDBClient(FalkorDBSettings())
        with pytest.raises(expected):
            await client.execute_query("RETURN 1", read_only=True)
        assert graph.ro_query.await_count == attempts

    async def test_explain_uses_async_command_and_maps_syntax_errors(
        self, driver_boundary
    ):
        """Validate through EXPLAIN without executing the supplied Cypher."""
        graph, *_ = driver_boundary
        client = FalkorDBClient(FalkorDBSettings())
        await client.explain("RETURN 1")
        graph.execute_command.assert_awaited_once_with(
            "GRAPH.EXPLAIN", "unit_graph", "RETURN 1"
        )
        graph.execute_command.side_effect = ResponseError("Invalid input")
        with pytest.raises(GraphQueryRejected):
            await client.explain("bad query")
        graph.query.assert_not_awaited()

    async def test_indexes_precede_writes_and_are_recreated_after_wipe(
        self, driver_boundary
    ):
        """Create every required index before bounded batches, including after wipe."""
        graph, *_ = driver_boundary
        client = FalkorDBClient(FalkorDBSettings(batch_size=1))
        records = [{"id": "a"}, {"id": "b"}]
        await client.execute_batch("UNWIND $records AS r RETURN r", records)
        queries = [call.args[0] for call in graph.query.await_args_list]
        assert queries[:9] == [
            "CALL db.indexes()",
            "CREATE INDEX FOR (n:`Entity`) ON (n.id)",
            "CREATE INDEX FOR (n:`Document`) ON (n.id)",
            "CREATE INDEX FOR (n:`Chunk`) ON (n.id)",
            "CREATE INDEX FOR (n:`Community`) ON (n.id)",
            "CREATE INDEX FOR (n:`CommunityReport`) ON (n.id)",
            "CREATE INDEX FOR (n:`Entity`) ON (n.label)",
            "CREATE INDEX FOR ()-[n:`RELATES_TO`]-() ON (n.id)",
            "CREATE INDEX FOR ()-[n:`RELATES_TO`]-() ON (n.type)",
        ]
        assert [call.args[1] for call in graph.query.await_args_list[9:]] == [
            {"records": [{"id": "a"}]},
            {"records": [{"id": "b"}]},
        ]
        graph.query.reset_mock()
        await client.execute_batch("RETURN $records", records)
        assert graph.query.await_count == 2
        await client.wipe()
        graph.query.reset_mock()
        await client.execute_batch("RETURN $records", records)
        assert graph.query.await_count == 11

    async def test_existing_index_errors_are_successful_and_memoized(
        self, driver_boundary
    ):
        """Repeated index creation is safe when another client created the index."""
        graph, *_ = driver_boundary

        async def query(text, params, **kwargs):
            if text.startswith("CREATE INDEX"):
                raise ResponseError("already indexed")
            return SimpleNamespace(header=[], result_set=[])

        graph.query.side_effect = query
        client = FalkorDBClient(FalkorDBSettings())
        await client.execute_batch("RETURN $records", [{"id": "a"}])
        graph.query.reset_mock()
        await client.execute_batch("RETURN $records", [{"id": "b"}])
        assert graph.query.await_count == 1

    async def test_dynamic_entity_label_is_indexed(self, driver_boundary):
        """Labels absent from initial setup are indexed before entity writes."""
        graph, *_ = driver_boundary
        client = FalkorDBClient(FalkorDBSettings())
        await client.upsert_entities([Entity(name="test", label="Disease")])
        queries = [call.args[0] for call in graph.query.await_args_list]
        assert "CREATE INDEX FOR (n:`Disease`) ON (n.id)" in queries
        assert "MERGE (n:Entity" in queries[-1]

    @pytest.mark.parametrize("message", ["empty key", "invalid graph operation"])
    async def test_wipe_accepts_missing_graph(self, driver_boundary, message):
        """Deleting a missing graph still allows the next indexed batch write."""
        graph, *_ = driver_boundary
        graph.delete.side_effect = ResponseError(message)
        client = FalkorDBClient(FalkorDBSettings())
        await client.wipe()
        await client.execute_batch("RETURN $records", [{"id": "a"}])
        assert graph.query.await_count == 10

    async def test_health_and_close_release_resources(self, driver_boundary):
        """Close releases resources and later health checks reconnect."""
        graph, constructor, connection, pool = driver_boundary
        graph.query.return_value = SimpleNamespace(header=["ok"], result_set=[[1]])
        client = FalkorDBClient(FalkorDBSettings())
        constructor.assert_not_called()
        assert await client.health_check() is True
        await client.close()
        connection.aclose.assert_awaited_once()
        pool.aclose.assert_awaited_once()
        assert await client.health_check() is True
        assert constructor.call_count == 2

    async def test_setup_schema_is_repeatable(self, driver_boundary):
        """Repeated schema setup does not issue redundant DDL."""
        graph, *_ = driver_boundary
        client = FalkorDBClient(FalkorDBSettings())
        schema = GraphSchema(entity_types=[], relation_types=[])
        await client.setup_schema(schema)
        graph.query.reset_mock()
        await client.setup_schema(schema)
        graph.query.assert_not_awaited()

    async def test_discovers_existing_node_and_edge_indexes(self, driver_boundary):
        """Existing index metadata avoids redundant DDL on fresh connections."""
        graph, *_ = driver_boundary
        graph.query.side_effect = [
            SimpleNamespace(
                header=["label", "properties", "entitytype"],
                result_set=[
                    ["Entity", ["id", "label"], "NODE"],
                    ["RELATES_TO", ["id", "type"], "RELATIONSHIP"],
                ],
            ),
            *[SimpleNamespace(header=[], result_set=[]) for _ in range(5)],
        ]
        client = FalkorDBClient(FalkorDBSettings())
        await client.execute_batch("RETURN $records", [{"id": "a"}])
        assert [call.args[0] for call in graph.query.await_args_list] == [
            "CALL db.indexes()",
            "CREATE INDEX FOR (n:`Document`) ON (n.id)",
            "CREATE INDEX FOR (n:`Chunk`) ON (n.id)",
            "CREATE INDEX FOR (n:`Community`) ON (n.id)",
            "CREATE INDEX FOR (n:`CommunityReport`) ON (n.id)",
            "RETURN $records",
        ]

    async def test_reads_after_wipe_return_empty_without_writable_fallback(
        self, driver_boundary
    ):
        """Treat the server's missing graph response as no rows for read calls."""
        graph, *_ = driver_boundary
        graph.ro_query.side_effect = ResponseError(
            "Invalid graph operation on empty key"
        )
        client = FalkorDBClient(FalkorDBSettings())
        assert await client.list_entities() == []
        graph.query.assert_not_awaited()

    async def test_timeout_override_reaches_driver(self, driver_boundary):
        """Allow long finalization queries without changing connection settings."""
        graph, *_ = driver_boundary
        client = FalkorDBClient(FalkorDBSettings())
        await client.execute_query("RETURN 1", timeout=90_000)
        assert graph.query.await_args.kwargs["timeout"] == 90_000

    async def test_missing_optional_driver_fails_only_on_use(self, monkeypatch):
        """Construct settings and clients without installing the optional extra."""
        monkeypatch.setitem(sys.modules, "falkordb.asyncio", None)
        client = FalkorDBClient(FalkorDBSettings())
        with pytest.raises(ImportError, match=r"am_diag\[falkordb\]"):
            await client.execute_query("RETURN 1")

    def test_transactions_are_explicitly_unsupported(self):
        """Do not imply atomic multi-statement writes are available."""
        with pytest.raises(NotImplementedError):
            FalkorDBClient(FalkorDBSettings()).transaction()
