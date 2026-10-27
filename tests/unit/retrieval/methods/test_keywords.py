"""Tests for keyword extraction (am_diag/retrieval/methods/keywords.py)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from baml_py.errors import BamlClientError

from am_diag.retrieval.methods.keywords import extract_keywords, simple_keywords


def _baml(monkeypatch: pytest.MonkeyPatch, **kwargs) -> None:
    baml = MagicMock()
    baml.ExtractQueryKeywords = AsyncMock(**kwargs)
    monkeypatch.setitem(sys.modules, "am_diag.llm.baml_client", SimpleNamespace(b=baml))


class TestSimpleKeywords:
    """Tests for simple_keywords."""

    def test_drops_question_words_and_short_words(self) -> None:
        """Only content words remain, in question order."""
        question = "What is the next best step for a patient with chest pain?"

        assert simple_keywords(question) == ["chest", "pain"]

    def test_keeps_two_letter_abbreviations(self) -> None:
        """All-caps abbreviations stay; other short words are dropped."""
        question = "Does IV fluid help to treat MI, BP or CT findings?"

        assert simple_keywords(question) == [
            "IV",
            "fluid",
            "help",
            "treat",
            "MI",
            "BP",
            "CT",
            "findings",
        ]

    def test_caps_the_list(self) -> None:
        """No more than 12 words are returned."""
        question = " ".join(f"term{i}" for i in range(30))

        assert len(simple_keywords(question)) == 12


class TestExtractKeywords:
    """Tests for extract_keywords."""

    async def test_orders_entities_before_terms_without_repeats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Entity names come first and a repeated name is kept once."""
        _baml(
            monkeypatch,
            return_value=SimpleNamespace(
                entities=["T2DM", "metformin"], terms=["metformin", "dose"]
            ),
        )

        keywords = await extract_keywords("metformin dose in T2DM")

        assert keywords.llm == ["T2DM", "metformin", "dose"]

    async def test_limits_model_keywords(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The limit applies to model keywords."""
        _baml(
            monkeypatch,
            return_value=SimpleNamespace(entities=["a1", "b1", "c1"], terms=[]),
        )

        keywords = await extract_keywords("q", limit=2)

        assert keywords.llm == ["a1", "b1"]

    async def test_falls_back_to_simple_keywords_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed model call leaves the simple keywords in place."""
        _baml(monkeypatch, side_effect=BamlClientError("down"))

        keywords = await extract_keywords("chest pain")

        assert keywords.llm == []
        assert keywords.simple == ["chest", "pain"]

    async def test_programming_errors_are_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A result without the expected fields is a bug, not a fallback case."""
        _baml(monkeypatch, return_value=SimpleNamespace())

        with pytest.raises(AttributeError):
            await extract_keywords("chest pain")
