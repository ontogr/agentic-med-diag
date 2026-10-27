"""Pick a recipe for callers that have no agent to choose one."""

from __future__ import annotations

import re

from am_diag.retrieval.methods.siblings import is_enumeration_query


_AGGREGATE = re.compile(r"\b(how many|count|average|number of)\b", re.IGNORECASE)
_RELATIONSHIP = re.compile(
    r"\b(between|connect(?:ed|ion)?|related to|pathway from)\b", re.IGNORECASE
)


def route(query: str) -> str:
    """Return the name of the recipe that suits ``query``.

    The first rule that matches wins: aggregate questions go to ``text2cypher``,
    questions that ask for a list go to ``enumerate``, questions about how things
    connect go to ``relation_expand``, and everything else goes to ``multi_path``.
    """
    if _AGGREGATE.search(query):
        return "text2cypher"
    if is_enumeration_query(query):
        return "enumerate"
    if _RELATIONSHIP.search(query):
        return "relation_expand"
    return "multi_path"


__all__ = ["route"]
