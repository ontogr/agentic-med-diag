"""Tests for Text2CypherRetriever (am_diag/retrieval/retrievers/text2cypher.py)."""

from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.errors import GraphQueryRejected, GraphWriteRejected
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.retrievers.text2cypher import Text2CypherRetriever


_BAML_TARGET = "am_diag.llm.baml_client.b"


@pytest.fixture(autouse=True)
def baml_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock generation without loading the external BAML runtime."""
    monkeypatch.setitem(
        sys.modules, "am_diag.llm.baml_client", SimpleNamespace(b=MagicMock())
    )


def _retriever(graph: MagicMock, **config: object) -> Text2CypherRetriever:
    return Text2CypherRetriever(
        RetrievalConfig(**config), graph, MEDICAL_GRAPHRAG_SCHEMA
    )


def _baml(*cyphers: str) -> MagicMock:
    baml = MagicMock()
    baml.GenerateCypherQuery = AsyncMock(
        side_effect=[SimpleNamespace(cypher=c) for c in cyphers]
    )
    return baml


@pytest.fixture
def graph() -> MagicMock:
    """Return a graph store whose queries succeed with one arbitrary row."""
    store = MagicMock(spec=GraphStore)
    store.execute_query = AsyncMock(return_value=[{"drug": "metformin", "n": 3}])
    return store


class TestText2CypherRetriever:
    """Tests for generated-Cypher retrieval."""

    async def test_runs_generated_query_as_read_only(self, graph):
        """Verify the generated query is validated and executed read-only."""
        with patch(_BAML_TARGET, _baml("MATCH (n:Entity) RETURN n.name AS name")):
            results = await _retriever(graph).retrieve("what treats diabetes")

        calls = graph.execute_query.call_args_list
        assert [c.kwargs.get("read_only") for c in calls] == [True]
        graph.explain.assert_awaited_once_with("MATCH (n:Entity) RETURN n.name AS name")
        assert [r.item.values for r in results] == [{"drug": "metformin", "n": 3}]

    async def test_returns_empty_when_read_only_execution_fails(self, graph):
        """Verify a write rejected by the server yields no results, not a crash."""
        graph.execute_query.side_effect = GraphWriteRejected("write")
        with patch(_BAML_TARGET, _baml("MATCH (n) DETACH DELETE n")):
            results = await _retriever(graph).retrieve("wipe everything")

        assert results == []

    async def test_retries_after_explain_rejection(self, graph):
        """A rejected plan is replaced before execution."""
        graph.explain.side_effect = [GraphQueryRejected("syntax"), None]
        with patch(_BAML_TARGET, _baml("invalid", "RETURN 1")):
            results = await _retriever(graph).retrieve("anything")

        graph.execute_query.assert_awaited_once_with("RETURN 1", read_only=True)
        assert [r.item.values for r in results] == [{"drug": "metformin", "n": 3}]

    async def test_never_executes_queries_rejected_by_explain(self, graph):
        """Exhausted validation attempts cannot execute an invalid query."""
        graph.explain.side_effect = GraphQueryRejected("syntax")
        with patch(_BAML_TARGET, _baml("invalid", "still invalid")):
            results = await _retriever(graph, text2cypher_max_attempts=2).retrieve("q")

        assert results == []
        graph.execute_query.assert_not_awaited()

    async def test_reraises_unexpected_execution_failure(self, graph):
        """Verify outages are not reported as empty search results."""
        graph.execute_query.side_effect = RuntimeError("connection lost")
        with (
            patch(_BAML_TARGET, _baml("MATCH (n) RETURN n")),
            pytest.raises(RuntimeError, match="connection lost"),
        ):
            await _retriever(graph).retrieve("anything")

    async def test_converts_rows_to_json_safe_values(self, graph):
        """Verify non-JSON values such as dates become primitives."""
        graph.execute_query.return_value = [{"d": date(2020, 1, 2)}]
        with patch(_BAML_TARGET, _baml("MATCH (n) RETURN n.d AS d")):
            results = await _retriever(graph).retrieve("when")

        assert results[0].item.values == {"d": "2020-01-02"}

    async def test_returns_empty_when_disabled(self, graph):
        """Verify no query is generated when text2cypher is turned off."""
        results = await _retriever(graph, use_text_to_cypher=False).retrieve("q")

        assert results == []
        graph.execute_query.assert_not_called()
