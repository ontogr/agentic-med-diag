"""Async FalkorDB transport with bounded retries and indexed batch writes."""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Awaitable, Callable
from importlib import import_module
from typing import Any

from am_diag.common.cypher._identifiers import validate_label
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.schema import GraphSchema
from am_diag.db.graph.base import GraphCapabilities
from am_diag.db.graph.cypher_store import CypherGraphStore
from am_diag.db.graph.errors import (
    GraphQueryRejected,
    GraphStoreError,
    GraphUnavailableError,
    GraphWriteRejected,
)
from am_diag.db.graph.settings import FalkorDBSettings


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_BASE_LABELS = ("Entity", "Document", "Chunk", "Community", "CommunityReport")


def _clean_value(value: Any, *, properties: bool = False) -> Any:
    """Clean record maps without converting structural record maps to JSON."""
    if isinstance(value, str):
        return _CONTROL_CHARS.sub("", value)
    if isinstance(value, dict):
        return {
            key: (
                json.dumps(_clean_value(item), ensure_ascii=False)
                if properties and isinstance(item, dict)
                else _clean_value(item, properties=properties or key == "props")
            )
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        if properties:
            return [
                _clean_value(item)
                for item in value
                if isinstance(item, (str, int, float, bool))
            ]
        return [_clean_value(item) for item in value if item is not None]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _CONTROL_CHARS.sub("", str(value))


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Preserve nullable query bindings while removing null record properties."""
    return {
        key: _clean_value(value, properties=key == "props")
        for key, value in (params or {}).items()
    }


def _plain(value: Any) -> Any:
    """Convert graph values and nested byte strings into plain Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, dict):
        return {_plain(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "properties"):
        return _plain(value.properties)
    return value


def _rows(result: Any) -> list[dict[str, Any]]:
    """Normalize paired, bare and missing result headers."""
    header = result.header if hasattr(result, "header") else []
    names = [
        str(_plain(entry[-1] if isinstance(entry, (list, tuple)) and entry else entry))
        for entry in (header or [])
    ]
    return [
        {
            names[index] if index < len(names) else str(index): _plain(value)
            for index, value in enumerate(row)
        }
        for row in result.result_set
    ]


def _map_error(exc: Exception) -> GraphStoreError:
    """Separate server refusals from retryable connection failures."""
    errors = import_module("redis.exceptions")
    message = str(exc)
    lowered = message.lower()
    if any(text in lowered for text in ("read-only", "read only", "graph.ro_query")):
        return GraphWriteRejected(message)
    if isinstance(
        exc,
        (
            errors.ConnectionError,
            errors.TimeoutError,
            errors.BusyLoadingError,
            OSError,
            TimeoutError,
        ),
    ):
        return GraphUnavailableError(message)
    return GraphQueryRejected(message)


class FalkorDBClient(CypherGraphStore):
    """FalkorDB graph client; independent batches are not atomic together.

    Args:
        settings: Connection settings and write limits. The optional driver is
            imported on first use, so construction does not contact a server.
    """

    dialect = "falkordb"
    capabilities = GraphCapabilities(gds_leiden=False)

    def __init__(self, settings: FalkorDBSettings) -> None:
        """Store settings without importing or constructing the optional driver."""
        self._settings = settings
        self._driver: Any = None
        self._graph: Any = None
        self._pool: Any = None
        self._connect_lock = asyncio.Lock()
        self._index_lock = asyncio.Lock()
        self._indexed: set[tuple[str, str, bool]] = set()
        self._base_indexes_ready = False

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        read_only: bool = False,
        timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run a query, enforcing read-only execution at the server.

        Args:
            query: Parameterized Cypher.
            params: Bindings; top-level null bindings are preserved.
            read_only: Reject writes through ``GRAPH.RO_QUERY``.
            timeout: Optional server timeout in milliseconds, overriding settings.

        Returns:
            Rows with decoded column names and plain values.

        Raises:
            GraphUnavailableError: If transient failures exhaust three attempts.
            GraphQueryRejected: If the server refuses the query.
            GraphWriteRejected: If read-only execution refuses a write.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")

        async def run() -> Any:
            graph = await self._ensure_graph()
            execute = graph.ro_query if read_only else graph.query
            return await execute(
                query,
                _clean_params(params),
                timeout=timeout
                if timeout is not None
                else self._settings.query_timeout_ms,
            )

        try:
            return _rows(await self._call(run))
        except GraphQueryRejected as exc:
            # A deleted graph has no rows; never fall back to writable execution.
            if read_only and str(exc).lower() == "invalid graph operation on empty key":
                return []
            raise

    async def execute_batch(
        self, query: str, records: list[dict[str, Any]], batch_size: int = 500
    ) -> None:
        """Create structural indexes before independently retryable write slices."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not records:
            return
        await self._ensure_base_indexes()
        size = self._settings.batch_size if batch_size == 500 else batch_size
        await super().execute_batch(query, records, size)

    async def upsert_entities(self, entities: list[Entity]) -> None:
        """Index dynamic type labels before delegating shared entity writes."""
        labels = {validate_label(entity.label) for entity in entities if entity.id}
        if labels:
            await self._ensure_base_indexes()
            await self._create_indexes([(label, "id", False) for label in labels])
        await super().upsert_entities(entities)

    async def setup_schema(self, schema: GraphSchema) -> None:
        """Create node and relationship indexes; repeated setup is safe."""
        labels = [validate_label(entity.label) for entity in schema.entity_types]
        await self._ensure_base_indexes()
        await self._create_indexes([(label, "id", False) for label in labels])

    async def explain(self, cypher: str) -> None:
        """Validate Cypher without executing it."""

        async def run() -> Any:
            graph = await self._ensure_graph()
            # The driver's inherited explain helper constructs a synchronous plan.
            return await graph.execute_command("GRAPH.EXPLAIN", graph.name, cypher)

        await self._call(run)

    async def wipe(self) -> None:
        """Delete the graph and forget indexes, including for a missing graph."""
        async with self._index_lock:

            async def run() -> Any:
                graph = await self._ensure_graph()
                return await graph.delete()

            try:
                await self._call(run)
            except GraphQueryRejected as exc:
                if not any(
                    marker in str(exc).lower() for marker in ("empty", "invalid")
                ):
                    raise
            finally:
                self._indexed.clear()
                self._base_indexes_ready = False

    async def health_check(self) -> bool:
        """Check Redis connectivity and graph query execution."""

        async def ping() -> Any:
            await self._ensure_graph()
            return await self._driver.connection.ping()

        await self._call(ping)
        return await self.execute_query("RETURN 1 AS ok") == [{"ok": 1}]

    def transaction(self) -> Any:
        """Reject multi-statement transactions, which this backend does not offer."""
        raise NotImplementedError("FalkorDB does not support client transactions")

    async def close(self) -> None:
        """Release the Redis pool; a later query reconnects with fresh index state."""
        try:
            if self._driver is not None:
                await self._driver.connection.aclose()
        finally:
            if self._pool is not None:
                await self._pool.aclose()
            self._pool = self._driver = self._graph = None
            self._indexed.clear()
            self._base_indexes_ready = False

    async def _ensure_graph(self) -> Any:
        """Build the driver off-loop because its constructor probes synchronously."""
        async with self._connect_lock:
            if self._graph is not None:
                return self._graph
            try:
                sdk = import_module("falkordb.asyncio")
                redis = import_module("redis.asyncio")
                connections = import_module("redis.asyncio.connection")
            except ImportError as exc:
                raise ImportError("Install am_diag[falkordb] to use FalkorDB") from exc

            if self._pool is None:
                kwargs: dict[str, Any] = {
                    "host": self._settings.host,
                    "port": self._settings.port,
                    "username": self._settings.username or None,
                    "password": self._settings.password.get_secret_value() or None,
                    "max_connections": self._settings.max_connections,
                    "decode_responses": True,
                    "socket_connect_timeout": 5,
                    "socket_timeout": None,
                }
                if self._settings.ssl:
                    kwargs["connection_class"] = connections.SSLConnection
                self._pool = redis.ConnectionPool(**kwargs)
            self._driver = await asyncio.to_thread(
                sdk.FalkorDB, connection_pool=self._pool
            )
            self._graph = self._driver.select_graph(self._settings.graph)
            return self._graph

    async def _call(self, call: Callable[[], Awaitable[Any]]) -> Any:
        """Retry only unavailable-server failures, preserving permanent refusals."""
        for attempt in range(3):
            try:
                return await call()
            except ImportError:
                raise
            except Exception as exc:
                mapped = exc if isinstance(exc, GraphStoreError) else _map_error(exc)
                if not isinstance(mapped, GraphUnavailableError) or attempt == 2:
                    raise mapped from exc
                await asyncio.sleep(2**attempt + random.uniform(0, 0.25))
        raise RuntimeError("Retry loop exhausted")

    async def _ensure_base_indexes(self) -> None:
        """Discover existing indexes once and ensure all structural indexes."""
        async with self._index_lock:
            if self._base_indexes_ready:
                return
            # GRAPH.RO_QUERY cannot inspect a graph key before it exists.
            existing = await self.execute_query("CALL db.indexes()")
            for row in existing:
                edge = str(row.get("entitytype", "NODE")).upper() == "RELATIONSHIP"
                for prop in row.get("properties", []):
                    self._indexed.add((row["label"], prop, edge))
            indexes = [(label, "id", False) for label in _BASE_LABELS]
            indexes += [("Entity", "label", False)]
            indexes += [("RELATES_TO", prop, True) for prop in ("id", "type")]
            indexes += [("__OntoEntity", "label", False), ("__OntoMeta", "key", False)]
            await self._create_indexes_unlocked(indexes)
            self._base_indexes_ready = True

    async def _create_indexes(self, indexes: list[tuple[str, str, bool]]) -> None:
        async with self._index_lock:
            await self._create_indexes_unlocked(indexes)

    async def _create_indexes_unlocked(
        self, indexes: list[tuple[str, str, bool]]
    ) -> None:
        """Memoize successful DDL, including indexes another client already created."""
        for label, prop, edge in indexes:
            if (label, prop, edge) in self._indexed:
                continue
            pattern = f"()-[n:`{label}`]-()" if edge else f"(n:`{label}`)"
            try:
                await self.execute_query(f"CREATE INDEX FOR {pattern} ON (n.{prop})")
            except GraphQueryRejected as exc:
                if "already indexed" not in str(exc).lower():
                    raise
            self._indexed.add((label, prop, edge))
