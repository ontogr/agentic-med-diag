"""Tests for source document names (retrieval/methods/document_names.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.methods.document_names import attach_document_names


def _chunk_result() -> SearchResult:
    chunk = Chunk(
        id=uuid4(),
        text="t",
        document_id=uuid4(),
        document_source="pubmed",
        chunk_index=0,
        chunk_size=1,
        cut_type="x",
    )
    return SearchResult(item=chunk, score=1.0, source="chunk")


def _graph(rows: list[dict]) -> MagicMock:
    store = MagicMock(spec=GraphStore)
    store.execute_query = AsyncMock(return_value=rows)
    return store


class TestAttachDocumentNames:
    """Tests for attach_document_names."""

    async def test_uses_title_then_external_id(self) -> None:
        """A document title wins; an untitled document falls back to its id."""
        titled, untitled, missing = _chunk_result(), _chunk_result(), _chunk_result()
        graph = _graph(
            [
                {
                    "chunk_id": str(titled.item.id),
                    "title": "Asthma",
                    "external_id": "1",
                },
                {"chunk_id": str(untitled.item.id), "title": None, "external_id": "42"},
            ]
        )

        await attach_document_names(graph, [titled, untitled, missing])

        assert titled.item.properties["document_name"] == "Asthma"
        assert untitled.item.properties["document_name"] == "42"
        assert "document_name" not in missing.item.properties

    async def test_runs_no_query_without_chunks(self) -> None:
        """Results with no chunks need no lookup."""
        graph = _graph([])

        await attach_document_names(graph, [])

        graph.execute_query.assert_not_awaited()
