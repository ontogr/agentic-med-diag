"""Retrieval pipeline configuration — env-overridable with RETRIEVAL_ prefix.

Collapsed from 3 relation collections to a single ``relation`` collection.
``SearchConfig`` is a data-only recipe type that selects which retriever
methods to run and how to fuse their results.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


GeneratorName = Literal[
    "entity",
    "relation",
    "chunk",
    "community",
    "text2cypher",
    "entity_lookup",
    "chunk_keywords",
]
ExpanderName = Literal[
    "relations_1hop",
    "relations_2hop",
    "chunks_mentions",
    "chunks_two_hop",
    "siblings",
    "local_context",
]


class RetrievalConfig(BaseSettings):
    """All fields env-overridable with RETRIEVAL_ prefix.

    Four vector collections (entity, relation, chunk, community) replace the
    previous 3-relation-collection scheme.
    """

    model_config = {"env_prefix": "RETRIEVAL_", "env_file": ".env", "extra": "ignore"}

    # Collection names — must match the embedding pipeline config
    entity_collection: str = "am_diag_entity"
    relation_collection: str = "am_diag_relation"
    chunk_collection: str = "am_diag_chunk"
    community_collection: str = "am_diag_community"

    # Entity search
    entity_top_k: int = 10
    entity_score_threshold: float | None = None

    # Relation search
    relation_top_k: int = 10
    relation_score_threshold: float | None = None

    # Chunk search
    chunk_top_k: int = 10
    chunk_score_threshold: float | None = None

    # Community report search
    community_top_k: int = 5
    community_report_level: int = 0

    # k-hop BFS traversal
    traversal_depth: int = 2
    traversal_limit: int = 50

    # Text-to-Cypher
    use_text_to_cypher: bool = True
    text2cypher_max_attempts: int = 2
    text2cypher_limit: int = 25
    text2cypher_max_limit: int = 100

    # Keyword-driven entity lookup
    keyword_limit: int = 10
    lookup_keywords: int = 8
    exact_per_keyword: int = 3
    contains_per_keyword: int = 5

    # Relation expansion around seed entities
    hop1_seeds: int = 15
    hop1_per_seed: int = 10
    hop1_limit: int = 150
    hop2_seeds: int = 5
    hop2_per_seed: int = 5
    hop2_limit: int = 25
    exclude_hierarchy_edges: bool = True
    hierarchy_edge_types: list[str] = ["PARENT", "IS_A"]

    # Fact filtering for relation facts
    fact_min_score: float = 0.25
    fact_max: int = 12
    fact_min_keep: int = 3

    # Sibling expansion for enumeration questions
    max_siblings: int = 20

    # Chunk keyword search and chunk expansion
    chunk_keyword_queries: int = 6
    chunk_simple_queries: int = 4
    chunk_keyword_top_k: int = 5
    mentions_seeds: int = 15
    mentions_per_entity: int = 3
    two_hop_chunk_seeds: int = 10
    two_hop_chunk_limit: int = 20
    local_context_chunks: int = 5

    # Minimum share of candidate chunks that must have stored vectors before
    # the cosine reranker stops embedding the missing ones
    cosine_vector_coverage: float = 0.9

    # Hybrid RRF
    rrf_k: int = 60
    hybrid_alpha: float = 0.5

    # Reranker
    reranker_min_score: float = 0.0


class SearchConfig(BaseModel):
    """Data-only recipe: which methods to run, which reranker, and result caps.

    A new retrieval strategy is simply a new constant of this type — no code
    changes are needed.

    ``methods`` run first and in parallel. ``expanders`` then run in parallel on
    the entities and chunks the methods found. ``facts`` makes the ``relation``
    method return :class:`~am_diag.common.data_models.search.Fact` items, filtered
    by relevance.
    """

    methods: list[GeneratorName]
    expanders: list[ExpanderName] = Field(default_factory=list)
    reranker: Literal["rrf", "cross_encoder", "cosine"] = "rrf"
    facts: bool = False
    bfs: bool = False
    bfs_depth: int = 2
    limit: int = 20
    filters: dict[str, Any] | None = None


__all__ = ["ExpanderName", "GeneratorName", "RetrievalConfig", "SearchConfig"]
