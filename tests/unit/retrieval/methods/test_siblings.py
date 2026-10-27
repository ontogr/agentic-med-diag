"""Tests for sibling expansion (am_diag/retrieval/methods/siblings.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.methods.siblings import is_enumeration_query, sibling_entities


class TestIsEnumerationQuery:
    """Tests for is_enumeration_query."""

    @pytest.mark.parametrize(
        "query",
        [
            "List all drugs that treat asthma",
            "Which of the following causes cough?",
            "What is the differential for chest pain?",
        ],
    )
    def test_detects_list_questions(self, query: str) -> None:
        """Questions that ask for a list are detected."""
        assert is_enumeration_query(query)

    def test_ignores_single_answer_questions(self) -> None:
        """A question with one answer is not a list question."""
        assert not is_enumeration_query("What treats asthma?")


class TestSiblingEntities:
    """Tests for sibling_entities."""

    async def test_returns_siblings_in_graph_order(self) -> None:
        """Each row becomes an entity result, best first."""
        graph = MagicMock(spec=GraphStore)
        graph.execute_query = AsyncMock(
            return_value=[
                {
                    "id": str(uuid4()),
                    "name": "budesonide",
                    "labels": ["Entity", "Drug"],
                },
                {
                    "id": str(uuid4()),
                    "name": "fluticasone",
                    "labels": ["Entity", "Drug"],
                },
            ]
        )

        results = await sibling_entities(graph, ["seed"], max_siblings=20)

        assert [r.item.name for r in results] == ["budesonide", "fluticasone"]
        assert results[0].score > results[1].score

    async def test_no_seeds_runs_no_query(self) -> None:
        """Without seeds there are no siblings."""
        graph = MagicMock(spec=GraphStore)
        graph.execute_query = AsyncMock()

        assert await sibling_entities(graph, [], max_siblings=20) == []
        graph.execute_query.assert_not_awaited()
