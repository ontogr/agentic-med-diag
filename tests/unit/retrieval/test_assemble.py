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

        assert context == (
            "Answer format: give the number first.\n\n"
            "## Graph query results\nCounts:\n- drugs: 4\n\n"
            "## Key entities\n- asthma (Disease)\n\n"
            "## Relationships\n"
            "- budesonide —[INDICATION]→ asthma (sources: DrugBank)\n\n"
            "## Passages\n- [Source: pubmed] Asthma is chronic."
        )

    def test_passage_source_prefers_the_document_name(self) -> None:
        """A chunk with a document name shows it instead of the corpus source."""
        chunk = Chunk(
            id=uuid4(),
            text="Asthma is chronic.",
            document_id=uuid4(),
            document_source="pubmed",
            chunk_index=0,
            chunk_size=18,
            cut_type="x",
            properties={"document_name": "Asthma review"},
        )

        context = assemble_context([_result(chunk)])

        assert context == "## Passages\n- [Source: Asthma review] Asthma is chronic."

    def test_empty_results_give_empty_text(self) -> None:
        """No results and no hint produce no text."""
        assert assemble_context([]) == ""

    def test_stops_adding_lines_at_the_character_budget(self) -> None:
        """Lines past the budget are dropped and the text never exceeds it."""
        entities = [
            _result(Entity(id=uuid4(), name=f"entity{i}", label="Drug"))
            for i in range(10)
        ]

        context = assemble_context(entities, max_characters=66)

        assert context == (
            "## Key entities\n- entity0 (Drug)\n- entity1 (Drug)\n- entity2 (Drug)"
        )
        assert len(context) <= 66

    def test_leaves_out_a_section_whose_first_line_does_not_fit(self) -> None:
        """A section with no room for one line is dropped with its header."""
        entity = _result(Entity(id=uuid4(), name="asthma", label="Disease"))
        row = _result(CypherRow(values={"drugs": 4}))

        context = assemble_context([row, entity], max_characters=40)

        assert context == "## Key entities\n- asthma (Disease)"
