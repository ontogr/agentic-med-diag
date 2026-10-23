"""Driver-boundary checks for backend-independent graph failures."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from am_diag.db.graph.errors import (
    GraphQueryRejected,
    GraphUnavailableError,
    GraphWriteRejected,
)
from am_diag.db.graph.neo4j import Neo4jClient
from am_diag.db.graph.settings import Neo4jSettings


class TestNeo4jErrorMapping:
    """Storage callers receive neutral failures with the driver cause intact."""

    @pytest.mark.parametrize(
        ("failure", "expected"),
        [
            (
                Neo4jError._hydrate_neo4j(
                    code="Neo.ClientError.Statement.AccessMode", message="write"
                ),
                GraphWriteRejected,
            ),
            (
                Neo4jError._hydrate_neo4j(
                    code="Neo.ClientError.Statement.SyntaxError", message="syntax"
                ),
                GraphQueryRejected,
            ),
            (ServiceUnavailable("offline"), GraphUnavailableError),
            (SessionExpired("expired"), GraphUnavailableError),
        ],
    )
    async def test_translates_query_failures(self, failure, expected) -> None:
        """Driver failures retain their cause and expose a storage exception."""
        driver = MagicMock(execute_query=AsyncMock(side_effect=failure))
        with patch(
            "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver", return_value=driver
        ):
            store = Neo4jClient(Neo4jSettings())

        with pytest.raises(expected) as caught:
            await store.execute_query("RETURN 1", read_only=True)

        assert caught.value.__cause__ is failure

    async def test_explain_uses_read_only_driver_execution(self) -> None:
        """Explain validates without allowing a write transaction."""
        driver = MagicMock(execute_query=AsyncMock(return_value=MagicMock(records=[])))
        with patch(
            "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver", return_value=driver
        ):
            store = Neo4jClient(Neo4jSettings())

        await store.explain("RETURN 1")

        assert driver.execute_query.call_args.args == ("EXPLAIN RETURN 1",)
        assert driver.execute_query.call_args.kwargs["routing_"] == "r"
