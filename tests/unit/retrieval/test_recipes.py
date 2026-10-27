"""Tests for named search recipes — data-only SearchConfig constants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from am_diag.retrieval import recipes
from am_diag.retrieval.config import SearchConfig


class TestRecipeConstants:
    """Tests for the named search recipe constants (SearchConfig instances)."""

    def test_entity_recipe(self) -> None:
        """Verify ENTITY recipe uses entity method with RRF reranker."""
        assert recipes.ENTITY.methods == ["entity"]
        assert recipes.ENTITY.reranker == "rrf"
        assert recipes.ENTITY.limit == 20

    def test_relation_recipe(self) -> None:
        """Verify RELATION recipe uses relation method with default limit."""
        assert recipes.RELATION.methods == ["relation"]
        assert recipes.RELATION.limit == 20

    def test_chunk_recipe(self) -> None:
        """Verify CHUNK recipe uses chunk method."""
        assert recipes.CHUNK.methods == ["chunk"]

    def test_community_recipe(self) -> None:
        """Verify COMMUNITY recipe uses community method."""
        assert recipes.COMMUNITY.methods == ["community"]

    def test_hybrid_rrf_includes_all(self) -> None:
        """Verify HYBRID_RRF recipe includes all four search methods."""
        assert sorted(recipes.HYBRID_RRF.methods) == sorted(
            ["entity", "relation", "chunk", "community"],
        )
        assert recipes.HYBRID_RRF.reranker == "rrf"
        assert recipes.HYBRID_RRF.bfs is False

    def test_hybrid_cross_encoder_uses_bfs(self) -> None:
        """Verify HYBRID_CROSS_ENCODER recipe uses cross_encoder with BFS enabled."""
        assert recipes.HYBRID_CROSS_ENCODER.reranker == "cross_encoder"
        assert recipes.HYBRID_CROSS_ENCODER.bfs is True

    def test_bfs_expand(self) -> None:
        """Verify BFS_EXPAND recipe enables BFS with depth 2 and limit 50."""
        assert recipes.BFS_EXPAND.bfs is True
        assert recipes.BFS_EXPAND.bfs_depth == 2
        assert recipes.BFS_EXPAND.limit == 50

    def test_text2cypher_recipe(self) -> None:
        """Verify TEXT2CYPHER recipe uses text2cypher method."""
        assert recipes.TEXT2CYPHER.methods == ["text2cypher"]

    def test_all_recipes_has_eight_entries(self) -> None:
        """Verify ALL_RECIPES contains at least eight named recipes."""
        names = [name for name, _ in recipes.ALL_RECIPES]
        assert len(recipes.ALL_RECIPES) >= 8
        assert "hybrid_rrf" in names
        assert "text2cypher" in names

    def test_adding_new_recipe_needs_no_code_change(self) -> None:
        """Verify a new recipe can be added as a SearchConfig constant."""
        new_recipe = SearchConfig(
            methods=["entity", "chunk"],
            reranker="cosine",
            limit=15,
        )
        assert new_recipe.limit == 15
        assert "entity" in new_recipe.methods
        assert new_recipe.reranker == "cosine"


class TestRecipeValidation:
    """Tests for recipe names and reranker choices."""

    def test_unimplemented_reranker_is_rejected(self) -> None:
        """A reranker the engine cannot run fails when the recipe is built."""
        with pytest.raises(ValidationError):
            SearchConfig(methods=["entity"], reranker="mmr")  # ty: ignore[invalid-argument-type]

    def test_recipe_names_are_unique(self) -> None:
        """Verify no recipe name appears twice in ALL_RECIPES."""
        names = [name for name, _ in recipes.ALL_RECIPES]
        assert len(names) == len(set(names))

    def test_every_exported_recipe_is_listed(self) -> None:
        """Verify each recipe constant in __all__ is reachable by name."""
        listed = [recipe for _, recipe in recipes.ALL_RECIPES]
        constants = [
            getattr(recipes, name) for name in recipes.__all__ if name != "ALL_RECIPES"
        ]
        assert all(any(c is r for r in listed) for c in constants)

    def test_multi_path_cypher_adds_only_text2cypher(self) -> None:
        """Verify MULTI_PATH_CYPHER is MULTI_PATH plus text2cypher."""
        assert recipes.MULTI_PATH_CYPHER.methods == [
            *recipes.MULTI_PATH.methods,
            "text2cypher",
        ]
        assert recipes.MULTI_PATH_CYPHER.expanders == recipes.MULTI_PATH.expanders
