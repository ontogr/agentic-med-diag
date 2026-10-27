"""Keyword search over chunks: one BM25 query per keyword, merged by chunk id."""

from __future__ import annotations

from am_diag.common.data_models.search import Keywords, SearchResult
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.hybrid import hybrid_search
from am_diag.retrieval.retrievers.chunk import chunk_from_payload
from am_diag.vector.embedding.base import Embedder


async def chunk_keyword_search(
    store: VectorStoreBase,
    collection: str,
    embedder: Embedder,
    query: str,
    keywords: Keywords,
    *,
    keyword_queries: int,
    simple_queries: int,
    top_k: int,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Search chunks with the question and each keyword as a separate query.

    Each query is keyword-only (BM25). A chunk found by several queries keeps its
    highest score.

    Args:
        store: Vector store that holds the chunk collection.
        collection: Chunk collection name.
        embedder: Embedder required by the hybrid search signature.
        query: The full question.
        keywords: Keywords extracted from the question.
        keyword_queries: Maximum model keywords to query.
        simple_queries: Maximum simple keywords to query.
        top_k: Maximum chunks per query.
        filters: Optional search filters.

    Returns:
        Chunks ordered by descending score.
    """
    queries = dict.fromkeys(
        [query, *keywords.llm[:keyword_queries], *keywords.simple[:simple_queries]],
    )
    best: dict[str, SearchResult] = {}
    for text in queries:
        hits = await hybrid_search(
            store,
            collection,
            text,
            embedder,
            limit=top_k,
            filters=filters,
            alpha=0.0,
        )
        for hit in hits:
            key = str(hit.id)
            if key in best and best[key].score >= hit.score:
                continue
            best[key] = SearchResult(
                item=chunk_from_payload(hit.id, hit.payload),
                score=hit.score,
                source="chunk_keywords",
                matched_on="bm25",
            )
    return sorted(best.values(), key=lambda result: result.score, reverse=True)


__all__ = ["chunk_keyword_search"]
