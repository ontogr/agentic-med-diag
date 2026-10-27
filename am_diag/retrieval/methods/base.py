"""Shared types for expanders: seed collection and the expansion context.

Expanders run after the recipe's methods. They take the entities and chunks the
methods found and look for more evidence around them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.search import CypherRow, Fact, SearchResult, Seeds
from am_diag.retrieval.filters import SearchFilters


Expander = Callable[
    [str, Seeds, SearchFilters | None, list[float] | None],
    Awaitable[list[SearchResult]],
]
"""An expander takes the query, seeds, filters and query vector (or None)."""


def entity_label(labels: list[str] | None) -> str:
    """Return the entity type from a node's labels.

    Entity nodes carry the shared ``Entity`` label plus one type label, so the
    type is the first label that is not ``Entity``.
    """
    for label in labels or []:
        if label != "Entity":
            return label
    return "Entity"


def _uuid_values(row: CypherRow) -> list[str]:
    """Return the values of ``row`` that are UUID strings."""
    ids: list[str] = []
    for value in row.values.values():
        if not isinstance(value, str):
            continue
        try:
            ids.append(str(UUID(value)))
        except ValueError:
            continue
    return ids


def collect_seeds(results: list[SearchResult], limit: int = 15) -> Seeds:
    """Collect entity and chunk ids from method results, in rank order.

    Entities seed themselves, relations and facts seed both endpoints, and chunks
    seed the chunk list. A Cypher row seeds every value in it that is a UUID,
    since the generated query returns entity ids. Ids repeat across methods, so
    each list keeps the first occurrence only.

    Args:
        results: Results from the recipe's methods, in rank order.
        limit: Maximum ids kept in each list.

    Returns:
        The collected seeds.
    """
    entity_ids: list[str] = []
    chunk_ids: list[str] = []
    for result in results:
        item = result.item
        if isinstance(item, Entity):
            entity_ids.append(str(item.id))
        elif isinstance(item, Relation):
            entity_ids += [str(i) for i in (item.head_id, item.tail_id) if i]
        elif isinstance(item, Fact):
            entity_ids += [i for i in (item.head_id, item.tail_id) if i]
        elif isinstance(item, Chunk):
            chunk_ids.append(str(item.id))
        elif isinstance(item, CypherRow):
            entity_ids += _uuid_values(item)
    return Seeds(
        entity_ids=list(dict.fromkeys(entity_ids))[:limit],
        chunk_ids=list(dict.fromkeys(chunk_ids))[:limit],
    )


__all__ = [
    "Expander",
    "collect_seeds",
    "entity_label",
]
