"""Pack fused search results into a labelled, character-budgeted context block.

Used by ``Graph.context`` and ``Project.context`` to turn heterogeneous fused
``SearchResult`` items into a single string for agents or plain-RAG callers,
grouped by kind and truncated to a character budget.
"""

from __future__ import annotations

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.search import CypherRow, Fact, SearchResult


def _render_item(
    item: Entity | Relation | Chunk | CommunityReport | CypherRow | Fact,
) -> str:
    """Render one fused item as a single context line."""
    if isinstance(item, Entity):
        desc = f": {item.description}" if item.description else ""
        return f"{item.name} ({item.label}){desc}"
    if isinstance(item, Relation):
        desc = f" — {item.description}" if item.description else ""
        return f"{item.head_name} {item.type} {item.tail_name}{desc}"
    if isinstance(item, Chunk):
        return item.text
    if isinstance(item, (CypherRow, Fact)):
        return str(item)
    title = f"{item.title}: " if item.title else ""
    return f"{title}{item.summary}"


_SECTION_TITLES = {
    Entity: "Entities",
    Relation: "Facts",
    Chunk: "Passages",
    CommunityReport: "Community summaries",
    CypherRow: "Query results",
}


def pack_context(results: list[SearchResult], max_characters: int = 2500) -> str:
    """Render fused results into a labelled block within a character budget.

    Items are grouped into Entities, Facts, Passages, Community summary, and
    Query results sections (in that order), each rendered one item per line.
    Lines are appended until ``max_characters`` would be exceeded; the budget is
    never overrun.

    Args:
        results: Fused, ranked search results.
        max_characters: Hard upper bound on the returned string length.

    Returns:
        The packed context string (possibly empty if the budget is tiny).
    """
    if max_characters <= 0:
        return ""
    sections: dict[type, list[str]] = {
        Entity: [],
        Relation: [],
        Chunk: [],
        CommunityReport: [],
        CypherRow: [],
    }
    for result in results:
        # Facts share the Facts section with stored relations.
        kind = Relation if isinstance(result.item, Fact) else type(result.item)
        bucket = sections.get(kind)
        if bucket is not None:
            bucket.append(_render_item(result.item))

    out: list[str] = []
    length = 0
    for kind, title in _SECTION_TITLES.items():
        lines = sections[kind]
        if not lines:
            continue
        header = f"## {title}"
        block_added = False
        for candidate in (header, *(f"- {line}" for line in lines)):
            addition = (1 if out else 0) + len(candidate)
            if length + addition > max_characters:
                break
            out.append(candidate)
            length += addition
            block_added = block_added or candidate != header
        if not block_added and out and out[-1] == header:
            out.pop()
            length -= len(header) + (1 if out else 0)
    return "\n".join(out)
