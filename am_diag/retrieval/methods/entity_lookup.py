"""Entity lookup by keyword: exact name, alias or synonym, then substring."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.base import entity_label
from am_diag.retrieval.methods.scoped_cypher import apply_node_filters


async def entity_lookup(
    graph_store: GraphStore,
    keywords: list[str],
    *,
    exact_per_keyword: int,
    contains_per_keyword: int,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Find entities by keyword text instead of by embedding.

    Exact matches come first. They compare the lower-cased keyword with the
    entity name, canonical name, aliases and synonyms. Substring matches follow,
    shortest name first. Each keyword has its own quota, so a keyword that
    matches many entities cannot crowd out a rare one.

    Args:
        graph_store: Graph database client.
        keywords: Keyword strings to look up.
        exact_per_keyword: Maximum exact matches per keyword.
        contains_per_keyword: Maximum substring matches per keyword.
        filters: Optional search filters.

    Returns:
        One result per distinct entity, exact matches first.
    """
    if not keywords:
        return []

    results: dict[str, SearchResult] = {}
    for template, per_keyword, base_score, matched_on in (
        ("retrieval/entity_exact", exact_per_keyword, 1.0, "exact"),
        ("retrieval/entity_contains", contains_per_keyword, 0.5, "contains"),
    ):
        cypher, params = apply_node_filters(load(template), filters, "e")
        rows = await graph_store.execute_query(
            cypher,
            {"keywords": keywords, "per_keyword": per_keyword, **params},
            read_only=True,
        )
        for rank, row in enumerate(rows):
            results.setdefault(
                str(row["id"]),
                SearchResult(
                    item=_row_to_entity(row),
                    score=base_score - rank * 0.001,
                    source="entity_lookup",
                    matched_on=matched_on,
                ),
            )
    return list(results.values())


def _row_to_entity(row: dict[str, Any]) -> Entity:
    """Build an ``Entity`` from a lookup row."""
    return Entity(
        id=UUID(str(row["id"])),
        name=row["name"],
        label=entity_label(row.get("labels")),
        description=row.get("description"),
        curie=row.get("curie"),
    )


__all__ = ["entity_lookup"]
