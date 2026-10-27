"""Splice ``SearchFilters`` into retrieval Cypher templates."""

from __future__ import annotations

from typing import Any

from am_diag.retrieval.filters import SearchFilters


_MARKER = "// FILTERS"


def apply_node_filters(
    cypher: str,
    filters: SearchFilters | None,
    var: str,
) -> tuple[str, dict[str, Any]]:
    """Replace the ``// FILTERS`` marker with conditions on node ``var``.

    The marker must sit inside a ``WHERE`` clause after at least one condition.
    Only the node-level filters apply (``labels`` and ``properties``). Relation
    types, document sources and community levels do not describe an entity node.

    Args:
        cypher: Query text that contains the marker.
        filters: Optional search filters.
        var: Cypher variable of the node to constrain.

    Returns:
        The query and the parameters its conditions need.
    """
    if filters is None:
        return cypher.replace(_MARKER, ""), {}
    node_filters = filters.model_copy(
        update={
            "relation_types": None,
            "document_sources": None,
            "community_level": None,
        },
    )
    where_clause, params = node_filters.to_cypher_where(var)
    if not where_clause:
        return cypher.replace(_MARKER, ""), {}
    # Parentheses keep an OR over several labels from binding to the AND.
    conditions = where_clause.removeprefix("WHERE ")
    return cypher.replace(_MARKER, f"AND ({conditions})"), params


__all__ = ["apply_node_filters"]
