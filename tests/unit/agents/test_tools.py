"""Retrieval recipe tools — serialization and factory."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from am_diag.agents.tools import _serialize_results, make_retrieval_tools
from am_diag.common.data_models import SearchResult
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import Fact
from am_diag.retrieval.search import SearchEngine


class TestRetrievalTools:
    """Retrieval tools — serialization, factory, and engine interaction."""

    def test_entity_serialization(self) -> None:
        """Serializes Entity to dict with expected fields."""
        entity = Entity(name="metformin", label="Drug")
        result = SearchResult(item=entity, score=0.95, source="test")
        serialized = _serialize_results([result])
        assert len(serialized) == 1
        assert serialized[0]["score"] == 0.95
        assert serialized[0]["source"] == "test"
        assert serialized[0]["item"]["name"] == "metformin"

    def test_fact_serialization_includes_rendered_text(self) -> None:
        """A Fact carries its one-line text for the model."""
        fact = Fact(
            head_id="a",
            head_name="budesonide",
            type="INDICATION",
            tail_id="b",
            tail_name="asthma",
            sources=["DrugBank"],
        )
        serialized = _serialize_results(
            [SearchResult(item=fact, score=1.0, source="relations_1hop")]
        )
        assert serialized[0]["item"]["text"] == (
            "budesonide \u2014[INDICATION]\u2192 asthma (sources: DrugBank)"
        )

    def test_empty_list(self) -> None:
        """Serializing an empty result list returns an empty list."""
        assert _serialize_results([]) == []

    def test_returns_eleven_tools(self) -> None:
        """Returns exactly 11 tools."""
        engine = MagicMock(spec=SearchEngine)
        engine.search = AsyncMock(return_value=[])
        tools = make_retrieval_tools(engine)
        assert len(tools) == 11

    def test_all_tools_have_names(self) -> None:
        """All tools have expected names."""
        engine = MagicMock(spec=SearchEngine)
        engine.search = AsyncMock(return_value=[])
        tools = make_retrieval_tools(engine)
        names = {t.name for t in tools}
        assert "entity_search" in names
        assert "relation_search" in names
        assert "chunk_search" in names
        assert "community_search" in names
        assert "hybrid_search" in names
        assert "cypher_search" in names
        assert "global_map_reduce" in names
        assert {"entity_lookup", "relation_expand"} <= names
        assert {"enumerate_entities", "multi_path"} <= names

    def test_entity_search_calls_engine(self) -> None:
        """entity_search tool invokes engine.search."""
        engine = MagicMock(spec=SearchEngine)
        engine.search = AsyncMock(return_value=[])
        tools = make_retrieval_tools(engine)
        entity_search = [t for t in tools if t.name == "entity_search"][0]
        asyncio.run(entity_search.ainvoke({"query": "diabetes"}))
        engine.search.assert_called()
