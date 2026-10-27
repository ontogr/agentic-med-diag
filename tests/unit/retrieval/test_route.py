"""Tests for recipe routing (am_diag/retrieval/route.py)."""

from __future__ import annotations

import pytest

from am_diag.retrieval.route import route


class TestRoute:
    """Tests for route."""

    @pytest.mark.parametrize(
        ("query", "recipe"),
        [
            ("How many genes are associated with asthma?", "text2cypher"),
            ("List all drugs for asthma", "enumerate"),
            ("How is metformin connected to lactic acidosis?", "relation_expand"),
            ("What is metformin?", "multi_path"),
        ],
    )
    def test_picks_recipe(self, query: str, recipe: str) -> None:
        """Each question form maps to its recipe."""
        assert route(query) == recipe
