"""Group search results into labelled sections of one context string."""

from __future__ import annotations

from dataclasses import dataclass

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import CypherRow, Fact, SearchResult


_DEFAULT_SCOPE = "Rows a graph query returned for the question:"


@dataclass(frozen=True)
class SectionedContext:
    """Context text built from search results."""

    text: str


def assemble_context(
    results: list[SearchResult],
    *,
    hint: str | None = None,
    scope: str | None = None,
) -> SectionedContext:
    """Build a sectioned context string from ``results``.

    Sections appear in this order: hint, graph query results, key entities,
    relationships, passages and community reports. Empty sections are left out.

    Args:
        results: Search results, in rank order.
        hint: Optional answer-format hint placed first.
        scope: Sentence placed above the graph query rows. An aggregate such as a
            count has no scope of its own, so this says what the rows describe.

    Returns:
        The assembled context.
    """
    sections: list[str] = []
    if hint:
        sections.append(hint)

    rows = [r for r in results if isinstance(r.item, CypherRow)]
    if rows:
        lines = "\n".join(f"- {r.item}" for r in rows)
        sections.append(f"## Graph query results\n{scope or _DEFAULT_SCOPE}\n{lines}")

    for title, item_type in (
        ("## Key entities", Entity),
        ("## Relationships", Fact),
        ("## Passages", Chunk),
        ("## Community reports", CommunityReport),
    ):
        picked = [r for r in results if isinstance(r.item, item_type)]
        if picked:
            lines = "\n".join(f"- {_line(r.item)}" for r in picked)
            sections.append(f"{title}\n{lines}")
    return SectionedContext("\n\n".join(sections))


def _line(item: object) -> str:
    """Render one item as a single line."""
    if isinstance(item, Entity):
        return f"{item.name} ({item.label})"
    if isinstance(item, Fact):
        return item.text()
    if isinstance(item, Chunk):
        return f"[Source: {item.document_source}] {item.text}"
    if isinstance(item, CommunityReport):
        return f"{item.title}: {item.summary}"
    return str(item)


__all__ = ["SectionedContext", "assemble_context"]
