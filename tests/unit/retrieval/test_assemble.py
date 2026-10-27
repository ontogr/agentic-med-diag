"""Tests for context assembly (am_diag/retrieval/assemble.py)."""

from __future__ import annotations

from uuid import uuid4

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import CypherRow, Fact, SearchResult
from am_diag.retrieval.assemble import assemble_context


def _result(item) -> SearchResult:
    return SearchResult(item=item, score=1.0, source="test")


class TestAssembleContext:
    """Tests for assemble_context."""

    def test_renders_sections_in_order(self) -> None:
        """Hint, rows, entities, relationships and passages appear in order."""
        results = [
            _result(Entity(id=uuid4(), name="asthma", label="Disease")),
            _result(
                Fact(
                    head_id="a",
                    head_name="budesonide",
                    type="INDICATION",
                    tail_id="b",
                    tail_name="asthma",
                    sources=["DrugBank"],
                )
            ),
            _result(
                Chunk(
                    id=uuid4(),
                    text="Asthma is chronic.",
                    document_id=uuid4(),
                    document_source="pubmed",
                    chunk_index=0,
                    chunk_size=18,
                    cut_type="x",
                )
            ),
            _result(CypherRow(values={"drugs": 4})),
        ]

        context = assemble_context(
            results, hint="Answer format: give the number first.", scope="Counts:"
        )

        assert context.text == (
            "Answer format: give the number first.\n\n"
            "## Graph query results\nCounts:\n- drugs: 4\n\n"
            "## Key entities\n- asthma (Disease)\n\n"
            "## Relationships\n"
            "- budesonide —[INDICATION]→ asthma (sources: DrugBank)\n\n"
            "## Passages\n- [Source: pubmed] Asthma is chronic."
        )

    def test_empty_results_give_empty_text(self) -> None:
        """No results and no hint produce no text."""
        assert assemble_context([]).text == ""
