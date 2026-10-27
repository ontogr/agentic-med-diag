"""Question-type detection and the answer-format hint for each type."""

from __future__ import annotations

import re


_MULTIPLE_CHOICE = re.compile(
    r"(?:^|\s)\(?[a-e]\)\s|\boptions?:|\bwhich of the following\b", re.IGNORECASE
)
_CLINICAL_DECISION = re.compile(
    r"\b(most likely diagnosis|next best step|best initial|most appropriate)\b",
    re.IGNORECASE,
)
_YES_NO = re.compile(
    r"^\s*(is|are|does|do|can|could|was|were|will|would|did|has|have)\b",
    re.IGNORECASE,
)
_COUNT = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b", re.IGNORECASE)

_HINTS = {
    "multiple_choice": "Answer format: pick one option and name its letter.",
    "clinical_decision": (
        "Answer format: give the single best answer first, then the reasoning."
    ),
    "yes_no": "Answer format: start with Yes or No, then explain briefly.",
    "count": "Answer format: give the number first, then what was counted.",
    "who": "Answer format: name the person or group first.",
    "when": "Answer format: give the time or age first.",
    "where": "Answer format: give the place or body site first.",
}


def detect_question_type(question: str) -> str | None:
    """Return the type of ``question``, or None when it fits no known type.

    Types are ``multiple_choice``, ``clinical_decision``, ``yes_no``, ``count``,
    ``who``, ``when`` and ``where``. The first match wins in that order.
    """
    if _MULTIPLE_CHOICE.search(question):
        return "multiple_choice"
    if _CLINICAL_DECISION.search(question):
        return "clinical_decision"
    if _YES_NO.match(question):
        return "yes_no"
    if _COUNT.search(question):
        return "count"
    for word in ("who", "when", "where"):
        if re.match(rf"\s*{word}\b", question, re.IGNORECASE):
            return word
    return None


def question_hint(question: str) -> str | None:
    """Return the answer-format hint for ``question``, or None."""
    question_type = detect_question_type(question)
    return _HINTS.get(question_type) if question_type else None


__all__ = ["detect_question_type", "question_hint"]
