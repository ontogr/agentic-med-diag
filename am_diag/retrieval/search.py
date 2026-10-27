"""Top-level ``search()`` entry point.

A search runs in three stages. The recipe's methods run first, in parallel.
Its expanders then run in parallel on the entities and chunks the methods found.
The remaining results are fused with the recipe's reranker. Rows from generated
Cypher skip fusion: their scores are not on the same scale as vector scores, so
they are returned first and do not count against the recipe's ``limit``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.data_models.search import (
    CypherRow,
    Keywords,
    SearchResult,
    Seeds,
)
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval import recipes
from am_diag.retrieval.config import RetrievalConfig, SearchConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.base import (
    Expander,
    collect_seeds,
)
from am_diag.retrieval.methods.chunk_expand import chunks_mentioning, chunks_two_hop
from am_diag.retrieval.methods.chunk_keywords import chunk_keyword_search
from am_diag.retrieval.methods.chunk_vectors import fetch_chunk_vectors
from am_diag.retrieval.methods.document_names import attach_document_names
from am_diag.retrieval.methods.entity_lookup import entity_lookup
from am_diag.retrieval.methods.keywords import extract_keywords
from am_diag.retrieval.methods.local_context import local_context
from am_diag.retrieval.methods.relation_expand import (
    expand_relations,
    relation_to_fact,
)
from am_diag.retrieval.methods.siblings import is_enumeration_query, sibling_entities
from am_diag.retrieval.rerank import cross_encoder_rerank, rrf
from am_diag.retrieval.rerank.cosine import cosine_rerank
from am_diag.retrieval.rerank.facts import filter_facts
from am_diag.retrieval.retrievers.base import HybridRetriever
from am_diag.retrieval.retrievers.chunk import ChunkRetriever
from am_diag.retrieval.retrievers.community import CommunityRetriever
from am_diag.retrieval.retrievers.entity import EntityRetriever
from am_diag.retrieval.retrievers.relation import RelationRetriever
from am_diag.retrieval.retrievers.text2cypher import Text2CypherRetriever
from am_diag.vector.embedding.base import Embedder
from am_diag.vector.reranking.base import Reranker


_KEYWORD_METHODS = frozenset({"entity_lookup", "chunk_keywords"})


Method = Callable[
    [str, SearchConfig, SearchFilters | None, Keywords | None],
    Awaitable[list[SearchResult]],
]


class SearchEngine:
    """Orchestrates multi-strategy retrieval with recipe-driven fan-out.

    Constructed once with all dependencies; call ``search()`` per query. The
    engine keeps no per-query state, so concurrent searches are safe.

    Args:
        config: Retrieval configuration.
        vector_store: Vector store (all four collections).
        graph_store: Graph database client.
        embedder: Embedder for query vectorisation.
        schema: Optional GraphSchema for the Text2Cypher retriever and for
            keyword extraction.
        reranker: Optional cross-encoder reranker (needed for
            ``HYBRID_CROSS_ENCODER`` recipe).
    """

    def __init__(
        self,
        config: RetrievalConfig,
        vector_store: VectorStoreBase,
        graph_store: GraphStore,
        embedder: Embedder,
        schema: GraphSchema | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize search engine with config and dependencies."""
        self._config = config
        self._reranker = reranker
        self._vector_store = vector_store
        self._graph = graph_store
        self._embedder = embedder
        self._schema = schema

        # Build retrievers once
        self._entity = EntityRetriever(config, vector_store, graph_store, embedder)
        self._relation = RelationRetriever(config, vector_store, graph_store, embedder)
        self._chunk = ChunkRetriever(config, vector_store, graph_store, embedder)
        self._community = CommunityRetriever(
            config,
            vector_store,
            graph_store,
            embedder,
        )
        self._text2cypher: Text2CypherRetriever | None = (
            Text2CypherRetriever(config, graph_store, schema) if schema else None
        )

        self._methods: dict[str, Method] = {
            "entity": lambda q, r, f, _: self._retrieve(self._entity, q, r, f),
            "relation": self._relation_method,
            "chunk": lambda q, r, f, _: self._retrieve(self._chunk, q, r, f),
            "community": lambda q, r, f, _: self._retrieve(self._community, q, r, f),
            "text2cypher": self._text2cypher_method,
            "entity_lookup": self._entity_lookup_method,
            "chunk_keywords": self._chunk_keywords_method,
        }
        self._expanders: dict[str, Expander] = {
            "relations_1hop": self._relations_1hop,
            "relations_2hop": self._relations_2hop,
            "chunks_mentions": self._chunks_mentions,
            "chunks_two_hop": self._chunks_two_hop,
            "siblings": self._siblings,
            "local_context": self._local_context,
        }

    async def search(
        self,
        query: str,
        recipe: SearchConfig | str = "hybrid_rrf",
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Execute ``recipe`` against ``query`` and return fused results.

        Args:
            query: Natural-language query.
            recipe: ``SearchConfig`` instance or recipe name (key in
                ``am_diag.retrieval.recipes``).
            filters: Optional metadata filters.

        Returns:
            Rows from generated Cypher first, in query order, then the fused
            results truncated to ``recipe.limit``. Tell the two apart by
            ``SearchResult.source == "cypher"``.
        """
        if isinstance(recipe, str):
            recipe = _resolve_recipe(recipe)

        if not recipe.methods:
            return []

        keywords = (
            await extract_keywords(query, self._schema, self._config.keyword_limit)
            if _KEYWORD_METHODS.intersection(recipe.methods)
            else None
        )
        method_lists = await asyncio.gather(
            *(
                self._methods[name](query, recipe, filters, keywords)
                for name in recipe.methods
            ),
        )
        cypher_rows, vector_lists = _split_cypher(method_lists)

        query_vector = (
            (await self._embedder.embed([query]))[0]
            if recipe.reranker == "cosine" or "chunks_mentions" in recipe.expanders
            else None
        )
        expansions: list[list[SearchResult]] = []
        if recipe.expanders:
            seeds = collect_seeds(
                [*cypher_rows, *(r for lst in vector_lists for r in lst)]
            )
            expansions = list(
                await asyncio.gather(
                    *(
                        self._expanders[name](query, seeds, filters, query_vector)
                        for name in recipe.expanders
                    ),
                ),
            )

        fused = await self._fuse(
            query, recipe, [*vector_lists, *expansions], query_vector
        )
        await attach_document_names(self._graph, fused)
        return [*cypher_rows[: self._config.text2cypher_limit], *fused]

    # -- methods ------------------------------------------------------
    async def _retrieve(
        self,
        retriever: HybridRetriever,
        query: str,
        recipe: SearchConfig,
        filters: SearchFilters | None,
    ) -> list[SearchResult]:
        """Run one hybrid retriever with the recipe's BFS flags."""
        return await retriever.retrieve(
            query,
            filters=filters,
            bfs=recipe.bfs,
            bfs_depth=recipe.bfs_depth,
        )

    async def _relation_method(
        self,
        query: str,
        recipe: SearchConfig,
        filters: SearchFilters | None,
        keywords: Keywords | None,
    ) -> list[SearchResult]:
        """Search relations, as facts when the recipe asks for them."""
        results = await self._retrieve(self._relation, query, recipe, filters)
        if not recipe.facts:
            return results

        facts = [
            SearchResult(
                item=relation_to_fact(r.item),
                score=r.score,
                source=r.source,
                matched_on=r.matched_on,
            )
            for r in results
            if isinstance(r.item, Relation)
        ]
        others = [r for r in results if not isinstance(r.item, Relation)]
        kept = filter_facts(
            facts,
            min_score=self._config.fact_min_score,
            max_facts=self._config.fact_max,
            min_keep=self._config.fact_min_keep,
        )
        return [*kept, *others]

    async def _text2cypher_method(
        self,
        query: str,
        recipe: SearchConfig,
        filters: SearchFilters | None,
        keywords: Keywords | None,
    ) -> list[SearchResult]:
        """Run text-to-Cypher, or nothing when no schema was given."""
        if self._text2cypher is None:
            return []
        return await self._text2cypher.retrieve(query, filters=filters)

    async def _entity_lookup_method(
        self,
        query: str,
        recipe: SearchConfig,
        filters: SearchFilters | None,
        keywords: Keywords | None,
    ) -> list[SearchResult]:
        """Look entities up by the question's keywords."""
        if keywords is None:
            return []
        return await entity_lookup(
            self._graph,
            keywords.llm[: self._config.lookup_keywords],
            exact_per_keyword=self._config.exact_per_keyword,
            contains_per_keyword=self._config.contains_per_keyword,
            filters=filters,
        )

    async def _chunk_keywords_method(
        self,
        query: str,
        recipe: SearchConfig,
        filters: SearchFilters | None,
        keywords: Keywords | None,
    ) -> list[SearchResult]:
        """Search chunks once per keyword."""
        if keywords is None:
            return []
        return await chunk_keyword_search(
            self._vector_store,
            self._config.chunk_collection,
            self._embedder,
            query,
            keywords,
            keyword_queries=self._config.chunk_keyword_queries,
            simple_queries=self._config.chunk_simple_queries,
            top_k=self._config.chunk_keyword_top_k,
            filters=filters,
        )

    # -- expanders ----------------------------------------------------
    def _excluded_relation_types(self) -> list[str]:
        """Relation types that expansion skips."""
        if self._config.exclude_hierarchy_edges:
            return list(self._config.hierarchy_edge_types)
        return []

    async def _relations_1hop(
        self,
        query: str,
        seeds: Seeds,
        filters: SearchFilters | None,
        query_vector: list[float] | None,
    ) -> list[SearchResult]:
        """Expand seed entities by one hop."""
        return await expand_relations(
            self._graph,
            seeds.entity_ids[: self._config.hop1_seeds],
            hops=1,
            per_seed=self._config.hop1_per_seed,
            total_limit=self._config.hop1_limit,
            exclude_types=self._excluded_relation_types(),
            filters=filters,
        )

    async def _relations_2hop(
        self,
        query: str,
        seeds: Seeds,
        filters: SearchFilters | None,
        query_vector: list[float] | None,
    ) -> list[SearchResult]:
        """Expand seed entities by two hops."""
        return await expand_relations(
            self._graph,
            seeds.entity_ids[: self._config.hop2_seeds],
            hops=2,
            per_seed=self._config.hop2_per_seed,
            total_limit=self._config.hop2_limit,
            exclude_types=self._excluded_relation_types(),
            filters=filters,
        )

    async def _chunks_mentions(
        self,
        query: str,
        seeds: Seeds,
        filters: SearchFilters | None,
        query_vector: list[float] | None,
    ) -> list[SearchResult]:
        """Find chunks that mention the seed entities."""
        if query_vector is None:
            return []
        return await chunks_mentioning(
            self._graph,
            self._vector_store,
            self._config.chunk_collection,
            seeds.entity_ids[: self._config.mentions_seeds],
            query_vector,
            per_entity=self._config.mentions_per_entity,
            filters=filters,
        )

    async def _chunks_two_hop(
        self,
        query: str,
        seeds: Seeds,
        filters: SearchFilters | None,
        query_vector: list[float] | None,
    ) -> list[SearchResult]:
        """Find chunks that mention neighbours of the seed entities."""
        return await chunks_two_hop(
            self._graph,
            self._vector_store,
            self._config.chunk_collection,
            seeds.entity_ids[: self._config.two_hop_chunk_seeds],
            limit=self._config.two_hop_chunk_limit,
            filters=filters,
        )

    async def _siblings(
        self,
        query: str,
        seeds: Seeds,
        filters: SearchFilters | None,
        query_vector: list[float] | None,
    ) -> list[SearchResult]:
        """Find siblings of the seed entities when the question asks for a list."""
        if not is_enumeration_query(query):
            return []
        return await sibling_entities(
            self._graph,
            seeds.entity_ids,
            max_siblings=self._config.max_siblings,
            filters=filters,
        )

    async def _local_context(
        self,
        query: str,
        seeds: Seeds,
        filters: SearchFilters | None,
        query_vector: list[float] | None,
    ) -> list[SearchResult]:
        """Add the entities and neighbouring chunks around matched chunks."""
        return await local_context(
            self._graph,
            self._vector_store,
            self._config.chunk_collection,
            seeds.chunk_ids[: self._config.local_context_chunks],
        )

    # -- fusion -------------------------------------------------------
    async def _fuse(
        self,
        query: str,
        recipe: SearchConfig,
        raw_results: list[list[SearchResult]],
        query_vector: list[float] | None = None,
    ) -> list[SearchResult]:
        """Fuse results using the recipe's reranker strategy."""
        non_empty = [r for r in raw_results if r]
        if not non_empty:
            return []

        if recipe.reranker == "cross_encoder" and self._reranker:
            flat = [item for sublist in non_empty for item in sublist]
            fused = await cross_encoder_rerank(query, flat, self._reranker)
        elif recipe.reranker == "cosine" and query_vector is not None:
            fused = await self._cosine_fuse(
                query_vector, rrf(non_empty, k=self._config.rrf_k)
            )
        else:
            fused = rrf(non_empty, k=self._config.rrf_k)

        if self._config.reranker_min_score > 0:
            fused = [r for r in fused if r.score >= self._config.reranker_min_score]

        return fused[: recipe.limit]

    async def _cosine_fuse(
        self,
        query_vector: list[float],
        fused: list[SearchResult],
    ) -> list[SearchResult]:
        """Order chunks by cosine similarity within their fused positions.

        Each chunk slot in the fused list is refilled with the next chunk by cosine
        similarity, so the mix of result types that fusion chose is kept and
        truncating to the recipe limit cannot drop the non-chunk results.
        """
        vectors = await fetch_chunk_vectors(
            self._vector_store,
            self._config.chunk_collection,
            self._embedder,
            fused,
            min_coverage=self._config.cosine_vector_coverage,
        )
        chunks = [r for r in fused if isinstance(r.item, Chunk)]
        ranked = cosine_rerank(query_vector, chunks, vectors)
        ranked_ids = {r.item.id for r in ranked if isinstance(r.item, Chunk)}
        unranked = [
            r
            for r in chunks
            if isinstance(r.item, Chunk) and r.item.id not in ranked_ids
        ]
        chunk_order = iter([*ranked, *unranked])
        return [next(chunk_order) if isinstance(r.item, Chunk) else r for r in fused]


def _split_cypher(
    method_lists: list[list[SearchResult]],
) -> tuple[list[SearchResult], list[list[SearchResult]]]:
    """Separate Cypher rows from the lists that go into fusion."""
    rows: list[SearchResult] = []
    rest: list[list[SearchResult]] = []
    for results in method_lists:
        rows += [r for r in results if isinstance(r.item, CypherRow)]
        remaining = [r for r in results if not isinstance(r.item, CypherRow)]
        if remaining:
            rest.append(remaining)
    return rows, rest


def _resolve_recipe(name: str) -> SearchConfig:
    """Look up a recipe by name."""
    recipe_map = dict(recipes.ALL_RECIPES)
    recipe = recipe_map.get(name)
    if recipe is None:
        raise ValueError(f"Unknown recipe: {name!r}; options: {list(recipe_map)}")
    return recipe


__all__ = ["SearchEngine"]
