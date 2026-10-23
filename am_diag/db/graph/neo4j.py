"""Neo4j transport, schema management, and driver error translation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

from neo4j import AsyncGraphDatabase, RoutingControl
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
from typing_extensions import LiteralString

from am_diag.common.cypher._identifiers import validate_label
from am_diag.common.cypher.loader import load, render
from am_diag.common.data_models.schema import GraphSchema
from am_diag.db.graph.base import GraphCapabilities
from am_diag.db.graph.cypher_store import (
    MAX_NEIGHBOR_DEPTH as MAX_NEIGHBOR_DEPTH,  # noqa: PLC0414
)
from am_diag.db.graph.cypher_store import (
    CypherGraphStore,
)
from am_diag.db.graph.errors import (
    GraphQueryRejected,
    GraphStoreError,
    GraphUnavailableError,
    GraphWriteRejected,
)
from am_diag.db.graph.settings import Neo4jSettings


_WRITE_REJECTION_CODES = frozenset({"Neo.ClientError.Statement.AccessMode"})


def _map_error(
    exc: Neo4jError | ServiceUnavailable | SessionExpired,
) -> GraphStoreError:
    """Translate driver failures without exposing driver types to callers."""
    if isinstance(exc, (ServiceUnavailable, SessionExpired)):
        return GraphUnavailableError(str(exc))
    if exc.code in _WRITE_REJECTION_CODES:
        return GraphWriteRejected(str(exc))
    return GraphQueryRejected(str(exc))


class Neo4jTransaction:
    """A Neo4j transaction handle yielded by ``Neo4jClient.transaction``."""

    def __init__(self, tx: Any) -> None:
        """Wrap an open driver transaction."""
        self._tx = tx

    async def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Run Cypher with keyword parameter bindings in this transaction."""
        try:
            result = await self._tx.run(cypher, **params)
            return await result.data()
        except (Neo4jError, ServiceUnavailable, SessionExpired) as exc:
            raise _map_error(exc) from exc


class Neo4jClient(CypherGraphStore):
    """Async Neo4j adapter using the official driver.

    Args:
        settings: Connection settings (URI, user, password, database).
    """

    dialect = "neo4j"
    capabilities = GraphCapabilities(gds_leiden=True)

    def __init__(self, settings: Neo4jSettings) -> None:
        """Open a driver configured for the requested database."""
        self._driver = AsyncGraphDatabase.driver(
            settings.uri, auth=(settings.user, settings.password)
        )
        self._database = settings.database

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        read_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute parameterized Cypher and return plain records.

        Raises:
            GraphUnavailableError: If the backend is unreachable.
            GraphWriteRejected: If read-only execution rejects a write.
            GraphQueryRejected: If the server rejects the query.
        """
        try:
            result = await self._driver.execute_query(
                cast(LiteralString, query),
                parameters_=params,
                database_=self._database,
                routing_=RoutingControl.READ if read_only else RoutingControl.WRITE,
            )
            return [r.data() for r in result.records]
        except (Neo4jError, ServiceUnavailable, SessionExpired) as exc:
            raise _map_error(exc) from exc

    async def explain(self, cypher: str) -> None:
        """Validate Cypher using a read-only EXPLAIN query."""
        await self.execute_query(f"EXPLAIN {cypher}", read_only=True)

    async def wipe(self) -> None:
        """Remove graph data using the Neo4j wipe query."""
        await self.execute_query(load("admin/wipe_graph"), {})

    async def close(self) -> None:
        """Close the underlying driver and release connections."""
        await self._driver.close()

    async def __aenter__(self) -> Neo4jClient:
        """Enter the client context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the client when its context exits."""
        await self.close()

    async def health_check(self) -> bool:
        """Return True when Neo4j is reachable, otherwise raise a store error."""
        try:
            await self._driver.verify_connectivity()
        except (Neo4jError, ServiceUnavailable, SessionExpired) as exc:
            raise _map_error(exc) from exc
        return True

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Neo4jTransaction, None]:
        """Commit on clean exit and roll back on any exception."""
        try:
            async with self._driver.session(database=self._database) as session:
                tx = await session.begin_transaction()
                try:
                    yield Neo4jTransaction(tx)
                    await tx.commit()
                except BaseException:
                    await tx.rollback()
                    raise
        except (Neo4jError, ServiceUnavailable, SessionExpired) as exc:
            raise _map_error(exc) from exc

    async def setup_schema(self, schema: GraphSchema) -> None:
        """Create idempotent constraints for entity and infrastructure labels."""
        all_labels = [et.label for et in schema.entity_types] + [
            "Document",
            "Chunk",
            "Community",
            "CommunityReport",
        ]
        for label in all_labels:
            validate_label(label)
            await self.execute_query(
                render("schema/constraints", Label=label, Constraint=f"{label}_id"), {}
            )
        for statement in load("schema/indexes").split(";"):
            if statement.strip():
                await self.execute_query(statement, {})


def create_neo4j_client(settings: Neo4jSettings | None = None) -> Neo4jClient:
    """Create a client, reading environment settings when none are supplied."""
    return Neo4jClient(settings or Neo4jSettings())
