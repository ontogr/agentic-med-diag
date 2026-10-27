"""Checks and fixes for generated Cypher before it reaches the database.

The database's read-only execution is the main protection against writes. These
checks catch mistakes earlier and give the model a precise error to retry with.
"""

from __future__ import annotations

import re

from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.data_models.search import GuardedCypher


_STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_WRITE_CLAUSE = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|CALL\s+db\.idx)\b",
    re.IGNORECASE,
)
_NODE_LABELS = re.compile(r"\(\s*(?:[A-Za-z_]\w*)?\s*((?::\s*`?[A-Za-z_]\w*`?\s*)+)")
_RELATIONSHIP_TYPES = re.compile(
    r"\[\s*(?:[A-Za-z_]\w*)?\s*:\s*`?([A-Za-z_]\w*(?:\s*\|\s*:?\s*[A-Za-z_]\w*)*)"
)
_TYPE_EQUALS = re.compile(r"\btype\s*(?::|=)\s*'([^']+)'", re.IGNORECASE)
_TYPE_IN = re.compile(r"\btype\s+IN\s*\[([^\]]*)\]", re.IGNORECASE)
_QUOTED = re.compile(r"'([^']+)'")
# A LIMIT that is a whole number and not the start of an expression such as
# "5 + 5000". A property named limit (n.limit) is not a LIMIT clause.
_LIMIT = re.compile(r"(?<![.\w])LIMIT\s+(\d+)\b(?!\s*[-+*/%^(])", re.IGNORECASE)
_ANY_LIMIT = re.compile(r"(?<![.\w])LIMIT\b", re.IGNORECASE)
_RETURN = re.compile(r"\bRETURN\b", re.IGNORECASE)
_PATH_FEATURES = re.compile(
    r"\b(?:allShortestPaths|shortestPath)\s*\(|\bMATCH\s+[A-Za-z_]\w*\s*=",
    re.IGNORECASE,
)

# Labels and relationship types the loaders write for every graph, whatever its
# schema.
_STRUCTURAL_LABELS = frozenset(
    {"Entity", "Document", "Chunk", "Community", "CommunityReport"}
)
_STRUCTURAL_RELATIONSHIPS = frozenset(
    {
        "RELATES_TO",
        "HAS_ENTITY",
        "PART_OF",
        "NEXT_CHUNK",
        "IN_COMMUNITY",
        "PARENT_COMMUNITY",
        "HAS_REPORT",
    }
)


def guard_cypher(
    cypher: str,
    schema: GraphSchema,
    *,
    dialect: str,
    default_limit: int,
    max_limit: int,
) -> GuardedCypher:
    """Check generated ``cypher`` against the schema and cap its result size.

    Text inside string literals is ignored, so a value such as ``'SET'`` passes.

    Args:
        cypher: Query text from the model.
        schema: Graph schema that lists the valid labels and relation types.
        dialect: Graph store dialect, such as ``"neo4j"`` or ``"falkordb"``.
        default_limit: ``LIMIT`` added when the query has none.
        max_limit: Largest ``LIMIT`` allowed.

    Returns:
        The query to run and the errors to send back to the model.
    """
    errors: list[str] = []
    bare = _STRING_LITERAL.sub("''", cypher)

    if _WRITE_CLAUSE.search(bare):
        errors.append("The query contains a write clause; use only MATCH and RETURN.")

    entity_labels = schema.entity_type_names()
    labels = {
        label.strip(" `")
        for match in _NODE_LABELS.finditer(bare)
        for label in match.group(1).split(":")
        if label.strip(" `")
    }
    bad_labels = sorted(labels - entity_labels - _STRUCTURAL_LABELS)
    if bad_labels:
        errors.append(
            f"Unknown node labels {bad_labels}; valid labels: {sorted(entity_labels)}."
        )

    relationships = {
        name.strip(" `:")
        for match in _RELATIONSHIP_TYPES.finditer(bare)
        for name in match.group(1).split("|")
    }
    bad_relationships = sorted(relationships - _STRUCTURAL_RELATIONSHIPS)
    if bad_relationships:
        errors.append(
            f"Unknown relationship types {bad_relationships}; every edge is "
            ":RELATES_TO with a `type` property."
        )

    type_values = {m.group(1) for m in _TYPE_EQUALS.finditer(cypher)}
    for match in _TYPE_IN.finditer(cypher):
        type_values.update(_QUOTED.findall(match.group(1)))
    relation_types = schema.relation_type_names()
    bad_values = sorted(type_values - relation_types)
    if bad_values:
        errors.append(
            f"Unknown relation type values {bad_values}; valid values: "
            f"{sorted(relation_types)}."
        )

    if len(_ANY_LIMIT.findall(bare)) != len(_LIMIT.findall(bare)):
        errors.append(
            "LIMIT must be a whole number; expressions and parameters are not allowed."
        )

    if dialect == "falkordb" and _PATH_FEATURES.search(bare):
        errors.append(
            "This database does not support shortestPath or path variables; "
            "match a fixed-length pattern instead."
        )

    return GuardedCypher(
        cypher=_ensure_limit(cypher, default_limit, max_limit), errors=errors
    )


def _ensure_limit(cypher: str, default_limit: int, max_limit: int) -> str:
    """Bound the rows the query returns.

    A ``LIMIT`` of the outermost query is lowered to ``max_limit`` if larger. If
    the outermost query has none, one is added. Text inside string literals is
    never changed, and a ``LIMIT`` inside a ``CALL`` or ``COLLECT`` subquery does
    not count as the outer limit.
    """
    stripped = cypher.strip().rstrip(";").rstrip()
    bare = _STRING_LITERAL.sub(lambda match: " " * len(match.group(0)), stripped)

    parts: list[str] = []
    last = 0
    for match in _LIMIT.finditer(bare):
        parts.append(stripped[last : match.start()])
        parts.append(f"LIMIT {min(int(match.group(1)), max_limit)}")
        last = match.end()
    parts.append(stripped[last:])
    bounded = "".join(parts)

    if _has_outer_limit(bare):
        return bounded
    return f"{bounded}\nLIMIT {default_limit}"


def _has_outer_limit(bare: str) -> bool:
    """Return True when the outermost query ends with a ``LIMIT``.

    Args:
        bare: Query text with string literals blanked out.
    """
    depths: list[int] = []
    depth = 0
    for char in bare:
        depths.append(depth)
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

    outer_return_end = 0
    for match in _RETURN.finditer(bare):
        if depths[match.start()] == 0:
            outer_return_end = match.end()
    return any(
        depths[match.start()] == 0 for match in _LIMIT.finditer(bare, outer_return_end)
    )


__all__ = ["guard_cypher"]
