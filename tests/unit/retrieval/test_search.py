"""Tests for SearchEngine — top-level recipe-driven retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import CypherRow, Fact, SearchResult
from am_diag.common.data_models.vector_record import VectorRecord
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval import recipes
from am_diag.retrieval.config import RetrievalConfig, SearchConfig
from am_diag.retrieval.search import SearchEngine, _resolve_recipe
from am_diag.vector.embedding.base import Embedder


class TestResolveRecipe:
    """Tests for _resolve_recipe helper."""

    def test_known_recipe(self) -> None:
        """Verify a known recipe name resolves to the correct SearchConfig."""
        config = _resolve_recipe("entity")
        assert config.methods == ["entity"]

    def test_unknown_recipe_raises(self) -> None:
        """Verify an unknown recipe name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown recipe"):
            _resolve_recipe("nonexistent")

    def test_hybrid_rrf_resolves(self) -> None:
        """Verify hybrid_rrf recipe resolves to include all search methods."""
        config = _resolve_recipe("hybrid_rrf")
        assert "entity" in config.methods
        assert "community" in config.methods

    def test_text2cypher_resolves(self) -> None:
        """Verify text2cypher recipe resolves to the text2cypher method."""
        config = _resolve_recipe("text2cypher")
        assert config.methods == ["text2cypher"]


class TestSearchEngine:
    """Tests for SearchEngine — top-level recipe-driven retrieval."""

    @pytest.fixture
    def engine(self) -> SearchEngine:
        """Return a SearchEngine with mocked dependencies."""
        return SearchEngine(
            config=RetrievalConfig(),
            vector_store=MagicMock(spec=VectorStoreBase),
            graph_store=MagicMock(spec=GraphStore),
            embedder=MagicMock(spec=Embedder),
        )

    async def test_empty_methods_returns_empty(self, engine: SearchEngine) -> None:
        """Verify no search methods produce an empty result list."""
        results = await engine.search(
            "query",
            recipe=SearchConfig(methods=[], reranker="rrf"),
        )
        assert results == []

    async def test_search_with_string_recipe(self, engine: SearchEngine) -> None:
        """Verify searching with a string recipe name works correctly."""
        engine._entity.retrieve = AsyncMock(return_value=[])
        results = await engine.search("query", recipe="entity")
        assert results == []

    async def test_search_with_hybrid_rrf(self, engine: SearchEngine) -> None:
        """Verify hybrid_rrf recipe routes to all four retrievers."""
        for retriever in [
            engine._entity,
            engine._relation,
            engine._chunk,
            engine._community,
        ]:
            retriever.retrieve = AsyncMock(return_value=[])

        results = await engine.search("query", recipe="hybrid_rrf")
        assert results == []
        engine._entity.retrieve.assert_awaited_once()
        engine._relation.retrieve.assert_awaited_once()
        engine._chunk.retrieve.assert_awaited_once()
        engine._community.retrieve.assert_awaited_once()

    async def test_search_fuses_results(self, engine: SearchEngine) -> None:
        """Verify results from multiple retrievers are fused into a single list."""
        entity = Entity(id=uuid4(), name="metformin", label="Drug")
        engine._entity.retrieve = AsyncMock(
            return_value=[SearchResult(item=entity, score=0.9, source="entity")],
        )
        engine._relation.retrieve = AsyncMock(return_value=[])
        engine._chunk.retrieve = AsyncMock(return_value=[])
        engine._community.retrieve = AsyncMock(return_value=[])

        results = await engine.search("metformin", recipe="hybrid_rrf")
        assert len(results) > 0
        assert results[0].source == "entity"

    async def test_bfs_flags_forwarded_to_retrievers(
        self, engine: SearchEngine
    ) -> None:
        """Verify bfs and bfs_depth flags are passed to the retriever."""
        for retriever in [
            engine._entity,
            engine._relation,
            engine._chunk,
            engine._community,
        ]:
            retriever.retrieve = AsyncMock(return_value=[])

        recipe = SearchConfig(methods=["entity"], reranker="rrf", bfs=True, bfs_depth=3)
        await engine.search("q", recipe=recipe)

        engine._entity.retrieve.assert_awaited_once()
        kwargs = engine._entity.retrieve.await_args.kwargs
        assert kwargs["bfs"] is True
        assert kwargs["bfs_depth"] == 3

    async def test_filters_results_by_score(self, engine: SearchEngine) -> None:
        """Verify results below the reranker_min_score threshold are filtered out."""
        entity = Entity(id=uuid4(), name="test", label="Drug")
        engine._entity.retrieve = AsyncMock(
            return_value=[SearchResult(item=entity, score=0.05, source="entity")],
        )
        engine._relation.retrieve = AsyncMock(return_value=[])
        engine._chunk.retrieve = AsyncMock(return_value=[])
        engine._community.retrieve = AsyncMock(return_value=[])
        engine._config.reranker_min_score = 0.1

        results = await engine.search("test", recipe="hybrid_rrf")
        assert len(results) == 0


def _entity_result(name: str, source: str = "entity", score: float = 0.9):
    entity = Entity(id=uuid4(), name=name, label="Drug")
    return SearchResult(item=entity, score=score, source=source)


def _row_result(score: float = 1.0) -> SearchResult:
    return SearchResult(
        item=CypherRow(values={"n": 3}),
        score=score,
        source="cypher",
        matched_on="text2cypher",
    )


class TestResolveRecipeCoverage:
    """Every named recipe resolves."""

    @pytest.mark.parametrize("name", [name for name, _ in recipes.ALL_RECIPES])
    def test_every_listed_recipe_resolves(self, name: str) -> None:
        """Verify each recipe in ALL_RECIPES is found by name."""
        assert _resolve_recipe(name) is dict(recipes.ALL_RECIPES)[name]


class TestSearchEngineStages:
    """Cypher rows, expanders and cosine ordering in SearchEngine.search."""

    @pytest.fixture
    def graph(self) -> MagicMock:
        """Return a mocked graph store."""
        return MagicMock(spec=GraphStore)

    @pytest.fixture
    def vectors(self) -> MagicMock:
        """Return a mocked vector store."""
        return MagicMock(spec=VectorStoreBase)

    @pytest.fixture
    def embedder(self) -> MagicMock:
        """Return a mocked embedder."""
        return MagicMock(spec=Embedder)

    @pytest.fixture
    def engine(self, graph, vectors, embedder) -> SearchEngine:
        """Return a SearchEngine with mocked dependencies and a text2cypher stub."""
        engine = SearchEngine(
            config=RetrievalConfig(),
            vector_store=vectors,
            graph_store=graph,
            embedder=embedder,
        )
        engine._text2cypher = MagicMock()
        return engine

    async def test_cypher_rows_come_first_and_stay_outside_the_limit(
        self, engine: SearchEngine
    ) -> None:
        """Rows lead the result and the other results are cut to the limit."""
        entities = [_entity_result("a"), _entity_result("b")]
        engine._entity.retrieve = AsyncMock(return_value=entities)
        engine._text2cypher.retrieve = AsyncMock(return_value=[_row_result(0.01)])
        recipe = SearchConfig(methods=["entity", "text2cypher"], limit=1)

        results = await engine.search("q", recipe=recipe)

        assert [r.source for r in results] == ["cypher", "entity"]
        assert results[0].score == 0.01
        assert results[0].item.values == {"n": 3}

    async def test_other_results_match_a_run_without_text2cypher(
        self, engine: SearchEngine
    ) -> None:
        """Adding text2cypher does not change how the rest are ranked."""
        names = ["a", "b"]
        engine._entity.retrieve = AsyncMock(
            side_effect=lambda *a, **k: [_entity_result(n) for n in names]
        )
        engine._text2cypher.retrieve = AsyncMock(return_value=[_row_result()])

        without = await engine.search("q", recipe=SearchConfig(methods=["entity"]))
        with_rows = await engine.search(
            "q", recipe=SearchConfig(methods=["entity", "text2cypher"])
        )

        assert [r.item.name for r in with_rows[1:]] == [r.item.name for r in without]

    async def test_same_entity_from_two_methods_is_merged(
        self, engine: SearchEngine
    ) -> None:
        """Two method results for one entity id fuse into one result."""
        entity_id = uuid4()
        first = SearchResult(
            item=Entity(id=entity_id, name="x", label="Drug"),
            score=0.9,
            source="entity",
        )
        second = SearchResult(
            item=Entity(id=entity_id, name="x", label="Drug"),
            score=0.8,
            source="bfs",
        )
        engine._entity.retrieve = AsyncMock(return_value=[first])
        engine._relation.retrieve = AsyncMock(return_value=[second])

        results = await engine.search(
            "q", recipe=SearchConfig(methods=["entity", "relation"])
        )

        assert len(results) == 1
        assert results[0].score == pytest.approx(2 / 61)

    async def test_expanders_receive_entities_found_by_methods(
        self, engine: SearchEngine, graph: MagicMock
    ) -> None:
        """A 1-hop expander runs on the ids the entity method found."""
        seed = _entity_result("seed")
        engine._entity.retrieve = AsyncMock(return_value=[seed])
        graph.execute_query = AsyncMock(
            return_value=[
                {
                    "head_id": str(seed.item.id),
                    "head_name": "seed",
                    "type": "INDICATION",
                    "tail_id": "t",
                    "tail_name": "asthma",
                    "description": None,
                    "structural_label": None,
                    "sources": ["DrugBank"],
                    "chunk_ids": [],
                }
            ]
        )
        recipe = SearchConfig(methods=["entity"], expanders=["relations_1hop"])

        results = await engine.search("q", recipe=recipe)

        facts = [r.item for r in results if isinstance(r.item, Fact)]
        assert [f.text() for f in facts] == [
            "seed —[INDICATION]→ asthma (sources: DrugBank)"
        ]

    async def test_siblings_expander_skips_questions_that_are_not_lists(
        self, engine: SearchEngine, graph: MagicMock
    ) -> None:
        """Siblings only run when the question asks for a list."""
        engine._entity.retrieve = AsyncMock(
            side_effect=lambda *a, **k: [_entity_result("seed")]
        )
        graph.execute_query = AsyncMock(
            return_value=[
                {"id": str(uuid4()), "name": "sib", "labels": ["Entity", "Drug"]}
            ]
        )
        recipe = SearchConfig(methods=["entity"], expanders=["siblings"])

        plain = await engine.search("What treats asthma?", recipe=recipe)
        listed = await engine.search("List all drugs for asthma", recipe=recipe)

        assert [r.item.name for r in plain] == ["seed"]
        assert {r.item.name for r in listed} == {"seed", "sib"}

    async def test_cosine_orders_chunks_by_similarity(
        self, engine: SearchEngine, vectors: MagicMock, embedder: MagicMock
    ) -> None:
        """Chunks are ordered by cosine to the query using stored vectors."""
        near, far = uuid4(), uuid4()

        def chunk(chunk_id, text):
            return SearchResult(
                item=Chunk(
                    id=chunk_id,
                    text=text,
                    document_id=uuid4(),
                    document_source="s",
                    chunk_index=0,
                    chunk_size=1,
                    cut_type="x",
                ),
                score=0.5,
                source="chunk",
            )

        # RRF alone would put "far" first because it is listed first.
        engine._chunk.retrieve = AsyncMock(
            return_value=[chunk(far, "far"), chunk(near, "near")]
        )
        embedder.embed = AsyncMock(return_value=[[1.0, 0.0]])
        vectors.retrieve = AsyncMock(
            return_value=[
                VectorRecord(id=near, vector=[1.0, 0.0]),
                VectorRecord(id=far, vector=[0.0, 1.0]),
            ]
        )

        results = await engine.search(
            "q", recipe=SearchConfig(methods=["chunk"], reranker="cosine")
        )

        assert [r.item.text for r in results] == ["near", "far"]
        assert [r.score for r in results] == pytest.approx([1.0, 0.0])
