"""Tests for question-type detection (am_diag/retrieval/question_type.py)."""

from __future__ import annotations

import pytest

from am_diag.retrieval.question_type import detect_question_type, question_hint


class TestDetectQuestionType:
    """Tests for detect_question_type."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("A 5-year-old has fever. (A) viral (B) bacterial", "multiple_choice"),
            ("Which of the following is first-line?", "multiple_choice"),
            ("What is the most likely diagnosis?", "clinical_decision"),
            ("What is the next best step in management?", "clinical_decision"),
            ("Is metformin safe in renal failure?", "yes_no"),
            ("How many genes are linked to asthma?", "count"),
            ("Who first described asthma?", "who"),
            ("When does type 1 diabetes usually present?", "when"),
            ("Where is the pancreas located?", "where"),
            ("Describe the pathophysiology of asthma.", None),
        ],
    )
    def test_detects_type(self, question: str, expected: str | None) -> None:
        """Each question form maps to its type."""
        assert detect_question_type(question) == expected


class TestQuestionHint:
    """Tests for question_hint."""

    def test_yes_no_hint(self) -> None:
        """A yes/no question gets the yes/no answer format."""
        assert (
            question_hint("Is aspirin an NSAID?")
            == "Answer format: start with Yes or No, then explain briefly."
        )

    def test_no_hint_for_open_questions(self) -> None:
        """A question with no known type gets no hint."""
        assert question_hint("Describe asthma.") is None
