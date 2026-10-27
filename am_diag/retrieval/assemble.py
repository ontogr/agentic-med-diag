"""Group search results into labelled sections of one context string."""

from __future__ import annotations

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import CypherRow, Fact, SearchResult


_DEFAULT_SCOPE = "Rows a graph query returned for the question:"
_DEFAULT_MAX_CHARACTERS = 4000
_SECTION_SEPARATOR = "\n\n"


def assemble_context(
    results: list[SearchResult],
    *,
    hint: str | None = None,
    scope: str | None = None,
    max_characters: int = _DEFAULT_MAX_CHARACTERS,
) -> str:
    """Build a sectioned context string from ``results``.

    Sections appear in this order: hint, graph query results, key entities,
    relationships, passages and community reports. Empty sections are left out.
    Lines are added in rank order until ``max_characters`` would be exceeded, so
    the text never overruns the budget. A section whose first line does not fit is
    left out, header included, and so is a hint that does not fit.

    Args:
        results: Search results, in rank order.
        hint: Optional answer-format hint placed first.
        scope: Sentence placed above the graph query rows. An aggregate such as a
            count has no scope of its own, so this says what the rows describe.
        max_characters: Hard upper bound on the length of the returned text.

    Returns:
        The assembled context.
    """
    sections: list[str] = []

    def add(header: str, lines: list[str]) -> None:
        """Append ``header`` with as many ``lines`` as fit in the budget."""
        used = len(_SECTION_SEPARATOR.join(sections))
        separator = len(_SECTION_SEPARATOR) if sections else 0
        text = header
        for line in lines:
            candidate = f"{text}\n{line}"
            if used + separator + len(candidate) > max_characters:
                break
            text = candidate
        if lines and text == header:
            return
        if used + separator + len(text) > max_characters:
            return
        sections.append(text)

    if hint:
        add(hint, [])

    rows = [f"- {r.item}" for r in results if isinstance(r.item, CypherRow)]
    if rows:
        add(f"## Graph query results\n{scope or _DEFAULT_SCOPE}", rows)

    for title, item_type in (
        ("## Key entities", Entity),
        ("## Relationships", Fact),
        ("## Passages", Chunk),
        ("## Community reports", CommunityReport),
    ):
        lines = [f"- {_line(r.item)}" for r in results if isinstance(r.item, item_type)]
        if lines:
            add(title, lines)
    return _SECTION_SEPARATOR.join(sections)


def _line(item: object) -> str:
    """Render one item as a single line."""
    if isinstance(item, Entity):
        return f"{item.name} ({item.label})"
    if isinstance(item, Fact):
        return item.text()
    if isinstance(item, Chunk):
        source = item.properties.get("document_name") or item.document_source
        return f"[Source: {source}] {item.text}"
    if isinstance(item, CommunityReport):
        return f"{item.title}: {item.summary}"
    return str(item)


__all__ = ["assemble_context"]
