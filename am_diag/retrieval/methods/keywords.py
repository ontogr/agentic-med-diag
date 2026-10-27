"""Keyword extraction for a clinical question."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from baml_py.errors import BamlError

from am_diag.common.data_models.schema import GraphSchema


logger = logging.getLogger(__name__)

_QUESTION_WORDS = frozenset(
    {
        "what", "which", "who", "whom", "when", "where", "why", "how", "does",
        "do", "did", "can", "could", "would", "will", "is", "are", "was", "were",
        "the", "an", "of", "in", "on", "for", "with", "and", "or", "to", "by",
        "from", "this", "that", "these", "those", "has", "have", "had", "not",
        "patient", "patients", "most", "likely", "following", "next", "best",
        "step", "diagnosis", "options", "option",
    }
)  # fmt: skip
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
_SIMPLE_LIMIT = 12


@dataclass(frozen=True)
class Keywords:
    """Search keywords for one question.

    Attributes:
        llm: Clinical entity names and terms from the model, best first.
        simple: Content words from the question, in question order.
    """

    llm: list[str] = field(default_factory=list)
    simple: list[str] = field(default_factory=list)


def _is_content_word(word: str) -> bool:
    """Return True when ``word`` can be a search keyword."""
    if word.lower() in _QUESTION_WORDS:
        return False
    return len(word) > 2 or (len(word) == 2 and word.isupper())


def simple_keywords(question: str) -> list[str]:
    """Return the content words of ``question``.

    Common question words and words of one or two characters are dropped, except
    two-letter all-caps abbreviations such as MI, BP, IV and CT.

    Args:
        question: The question text.

    Returns:
        At most 12 words, in question order.
    """
    words = [word for word in _WORD.findall(question) if _is_content_word(word)]
    return words[:_SIMPLE_LIMIT]


async def extract_keywords(
    question: str,
    schema: GraphSchema | None = None,
    limit: int = 10,
) -> Keywords:
    """Extract search keywords from ``question``.

    If the model call fails with a BAML client or validation error, only the simple
    keywords are returned, so search still runs without them. Other errors
    propagate.

    Args:
        question: The question text.
        schema: Graph schema that tells the model which entity types exist.
        limit: Maximum model keywords.

    Returns:
        The model keywords and the simple keywords.
    """
    from am_diag.llm.baml_client import b  # noqa: PLC0415

    try:
        extracted = await b.ExtractQueryKeywords(
            question,
            schema.as_text() if schema else "",
        )
        names = [k.strip() for k in [*extracted.entities, *extracted.terms]]
        llm = list(dict.fromkeys(name for name in names if name))[:limit]
    except BamlError:
        logger.warning(
            "keyword extraction failed; using simple keywords", exc_info=True
        )
        llm = []
    return Keywords(llm=llm, simple=simple_keywords(question))


__all__ = ["Keywords", "extract_keywords", "simple_keywords"]
