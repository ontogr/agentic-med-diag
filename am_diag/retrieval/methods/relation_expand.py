"""One-hop and two-hop relation expansion that keeps the evidence on each edge."""

from __future__ import annotations

from typing import Any

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.search import Fact, SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.scoped_cypher import apply_node_filters


async def expand_relations(
    graph_store: GraphStore,
    seed_ids: list[str],
    *,
    hops: int,
    per_seed: int,
    total_limit: int,
    exclude_types: list[str],
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Return the edges around ``seed_ids`` as :class:`Fact` results.

    Edges count in both directions because the stored direction depends on the
    extractor or source. A two-hop path is read outward from its seed. Filters
    constrain the entity reached, not the seed.

    Args:
        graph_store: Graph database client.
        seed_ids: Entity id strings to expand from.
        hops: ``1`` for single edges, ``2`` for two-edge paths.
        per_seed: Maximum edges or paths kept for each seed. A cap per seed stops
            one high-degree entity from taking the whole result.
        total_limit: Maximum facts returned.
        exclude_types: Relation ``type`` values to skip.
        filters: Optional search filters.

    Returns:
        Facts in rank order, without repeats.

    Raises:
        ValueError: If ``hops`` is not 1 or 2.
    """
    if hops not in (1, 2):
        raise ValueError(f"hops must be 1 or 2, got {hops}")
    if not seed_ids:
        return []

    name = f"retrieval/expand_relations_{hops}hop"
    var = "b" if hops == 1 else "c"
    cypher, params = apply_node_filters(load(name), filters, var)
    rows = await graph_store.execute_query(
        cypher,
        {**params, "ids": seed_ids, "per_seed": per_seed, "exclude": exclude_types},
        read_only=True,
    )

    facts: dict[tuple[str, str, str, str | None], Fact] = {}
    for row in rows:
        fact = _row_to_fact(row, hops)
        key = (fact.head_id, fact.type, fact.tail_id, fact.via_id)
        facts.setdefault(key, fact)

    source = f"relations_{hops}hop"
    return [
        SearchResult(
            item=fact,
            score=1.0 / (1 + rank),
            source=source,
            matched_on="expansion",
        )
        for rank, fact in enumerate(list(facts.values())[:total_limit])
    ]


def relation_to_fact(relation: Relation) -> Fact:
    """Convert a stored :class:`Relation` into a :class:`Fact`."""
    provenance = relation.provenance
    return Fact(
        head_id=str(relation.head_id or ""),
        head_name=relation.head_name,
        type=relation.type,
        tail_id=str(relation.tail_id or ""),
        tail_name=relation.tail_name,
        description=relation.description,
        structural_label=relation.structural_label,
        sources=list(relation.sources),
        chunk_ids=[str(c) for c in provenance.chunk_ids] if provenance else [],
    )


def _row_to_fact(row: dict[str, Any], hops: int) -> Fact:
    """Build a :class:`Fact` from one expansion row."""
    if hops == 2:
        return Fact(
            head_id=str(row["head_id"]),
            head_name=row["head_name"] or "",
            type=f"{row['first_type']}>{row['second_type']}",
            tail_id=str(row["tail_id"]),
            tail_name=row["tail_name"] or "",
            hops=2,
            via_name=row["via_name"] or "",
            via_id=str(row["via_id"]),
        )
    return Fact(
        head_id=str(row["head_id"]),
        head_name=row["head_name"] or "",
        type=row["type"] or "",
        tail_id=str(row["tail_id"]),
        tail_name=row["tail_name"] or "",
        description=row.get("description") or None,
        structural_label=row.get("structural_label"),
        sources=list(row.get("sources") or []),
        chunk_ids=[str(c) for c in row.get("chunk_ids") or []],
    )


__all__ = ["expand_relations", "relation_to_fact"]
