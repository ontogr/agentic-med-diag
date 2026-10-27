"""Sibling entities for questions that ask for a list."""

from __future__ import annotations

import re
from uuid import UUID

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.base import entity_label
from am_diag.retrieval.methods.scoped_cypher import apply_node_filters


_ENUMERATION = re.compile(
    r"\b(list|name|enumerate|all|every|each|which of the following|differential|"
    r"types of|kinds of|causes of)\b",
    re.IGNORECASE,
)


def is_enumeration_query(query: str) -> bool:
    """Return True when ``query`` asks for a list of things."""
    return _ENUMERATION.search(query) is not None


async def sibling_entities(
    graph_store: GraphStore,
    seed_ids: list[str],
    *,
    max_siblings: int,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Find entities that share a neighbour and a label with the seeds.

    Args:
        graph_store: Graph database client.
        seed_ids: Entity id strings to find siblings of.
        max_siblings: Maximum entities returned.
        filters: Optional search filters.

    Returns:
        Sibling entities in graph order, none of them a seed.
    """
    if not seed_ids:
        return []

    cypher, params = apply_node_filters(load("retrieval/siblings"), filters, "s")
    rows = await graph_store.execute_query(
        cypher,
        {**params, "ids": seed_ids, "max_siblings": max_siblings},
        read_only=True,
    )
    return [
        SearchResult(
            item=Entity(
                id=UUID(str(row["id"])),
                name=row["name"],
                label=entity_label(row.get("labels")),
            ),
            score=1.0 / (1 + rank),
            source="siblings",
            matched_on="sibling",
        )
        for rank, row in enumerate(rows)
    ]


__all__ = ["is_enumeration_query", "sibling_entities"]
