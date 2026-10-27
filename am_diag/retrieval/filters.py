"""SearchFilters — typed metadata filtering for vector and graph searches.

Converts a high-level typed filter into both a vector-store payload filter dict
and a Cypher WHERE fragment for graph traversal.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from am_diag.common.cypher._identifiers import validate_label
from am_diag.common.cypher.loader import safe_identifier


class SearchFilters(BaseModel):
    """Typed metadata filter for retrieval.

    Each field maps to a payload key in the vector store or a label/property
    in the graph. ``None`` means "no filter on this dimension".
    """

    labels: list[str] | None = None
    relation_types: list[str] | None = None
    document_sources: list[str] | None = None
    community_level: int | None = None
    properties: dict[str, Any] | None = None

    def to_payload_filter(self) -> dict[str, Any] | None:
        """Convert to a flat filter dict for vector-store ``search``.

        Combined with AND semantics: all non-None fields are included.
        Scalar values become exact matches; list values become "any-of".
        """
        out: dict[str, Any] = {}
        if self.labels:
            out["label"] = self.labels if len(self.labels) > 1 else self.labels[0]
        if self.relation_types:
            out["type"] = (
                self.relation_types
                if len(self.relation_types) > 1
                else self.relation_types[0]
            )
        if self.document_sources:
            out["document_source"] = (
                self.document_sources
                if len(self.document_sources) > 1
                else self.document_sources[0]
            )
        if self.community_level is not None:
            out["level"] = self.community_level
        if self.properties:
            for k, v in self.properties.items():
                out[k] = v
        return out or None

    def to_cypher_where(self, var: str = "n") -> tuple[str, dict[str, Any]]:
        """Build a WHERE clause fragment and parameter dict.

        Args:
            var: The Cypher variable name to qualify properties on.
                Defaults to "n".

        Returns:
            A ``(where_string, params)`` tuple. ``where_string`` is empty
            when no filters are set.

        Raises:
            ValueError: If a label or property key is not a safe Cypher
                identifier. Both are written into the query text, so they
                are checked before they are used.
        """
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if self.labels:
            clauses.append(
                " OR ".join(f"{var}:{validate_label(lab)}" for lab in self.labels),
            )
        if self.relation_types:
            clauses.append(f"type({var}) IN $relation_types")
            params["relation_types"] = self.relation_types
        if self.document_sources:
            clauses.append(f"{var}.document_source IN $document_sources")
            params["document_sources"] = self.document_sources
        if self.community_level is not None:
            clauses.append(f"{var}.level = $community_level")
            params["community_level"] = self.community_level
        if self.properties:
            for k, v in self.properties.items():
                safe_identifier(k)
                clauses.append(f"{var}.{k} = ${k}")
                params[k] = v

        if not clauses:
            return "", {}
        if len(clauses) == 1:
            return f"WHERE {clauses[0]}", params
        return f"WHERE {' AND '.join(clauses)}", params


__all__ = ["SearchFilters"]
