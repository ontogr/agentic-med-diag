"""Named search recipes — data-only ``SearchConfig`` constants.

Adding a new retrieval strategy means defining a new constant here;
no orchestration code needs to change.
"""

from __future__ import annotations

from am_diag.retrieval.config import SearchConfig


# Single-method recipes
ENTITY: SearchConfig = SearchConfig(
    methods=["entity"],
    reranker="rrf",
    limit=20,
)

RELATION: SearchConfig = SearchConfig(
    methods=["relation"],
    reranker="rrf",
    limit=20,
)

CHUNK: SearchConfig = SearchConfig(
    methods=["chunk"],
    reranker="rrf",
    limit=20,
)

COMMUNITY: SearchConfig = SearchConfig(
    methods=["community"],
    reranker="rrf",
    limit=20,
)

# Multi-method hybrid recipes
HYBRID_RRF: SearchConfig = SearchConfig(
    methods=["entity", "relation", "chunk", "community"],
    reranker="rrf",
    bfs=False,
    limit=20,
)

HYBRID_CROSS_ENCODER: SearchConfig = SearchConfig(
    methods=["entity", "relation", "chunk", "community"],
    reranker="cross_encoder",
    bfs=True,
    limit=20,
)

# Specialised recipes
BFS_EXPAND: SearchConfig = SearchConfig(
    methods=["entity"],
    reranker="rrf",
    bfs=True,
    bfs_depth=2,
    limit=50,
)

TEXT2CYPHER: SearchConfig = SearchConfig(
    methods=["text2cypher"],
    reranker="rrf",
    limit=20,
)

# Evidence-centred recipes
FACTS: SearchConfig = SearchConfig(
    methods=["relation"],
    facts=True,
    reranker="rrf",
    limit=12,
)

ENTITY_LOOKUP: SearchConfig = SearchConfig(
    methods=["entity_lookup", "entity"],
    reranker="rrf",
    limit=20,
)

RELATION_EXPAND: SearchConfig = SearchConfig(
    methods=["entity"],
    expanders=["relations_1hop", "relations_2hop"],
    reranker="rrf",
    limit=30,
)

ENUMERATE: SearchConfig = SearchConfig(
    methods=["entity_lookup", "entity"],
    expanders=["siblings"],
    reranker="rrf",
    limit=30,
)

CHUNK_MULTI_PATH: SearchConfig = SearchConfig(
    methods=["chunk", "chunk_keywords"],
    expanders=["chunks_mentions", "chunks_two_hop"],
    reranker="cosine",
    limit=15,
)

LOCAL: SearchConfig = SearchConfig(
    methods=["chunk"],
    expanders=["local_context"],
    reranker="rrf",
    limit=20,
)

MULTI_PATH: SearchConfig = SearchConfig(
    methods=[
        "relation",
        "entity_lookup",
        "entity",
        "chunk",
        "chunk_keywords",
        "community",
    ],
    expanders=[
        "relations_1hop",
        "relations_2hop",
        "chunks_mentions",
        "chunks_two_hop",
        "siblings",
    ],
    facts=True,
    reranker="cosine",
    limit=25,
)

MULTI_PATH_CYPHER: SearchConfig = MULTI_PATH.model_copy(
    update={"methods": [*MULTI_PATH.methods, "text2cypher"]},
)

# Convenience grouping for testing / iteration
ALL_RECIPES: list[tuple[str, SearchConfig]] = [
    ("entity", ENTITY),
    ("relation", RELATION),
    ("chunk", CHUNK),
    ("community", COMMUNITY),
    ("hybrid_rrf", HYBRID_RRF),
    ("hybrid_cross_encoder", HYBRID_CROSS_ENCODER),
    ("bfs_expand", BFS_EXPAND),
    ("text2cypher", TEXT2CYPHER),
    ("facts", FACTS),
    ("entity_lookup", ENTITY_LOOKUP),
    ("relation_expand", RELATION_EXPAND),
    ("enumerate", ENUMERATE),
    ("chunk_multi_path", CHUNK_MULTI_PATH),
    ("local", LOCAL),
    ("multi_path", MULTI_PATH),
    ("multi_path_cypher", MULTI_PATH_CYPHER),
]

__all__ = [
    "ALL_RECIPES",
    "BFS_EXPAND",
    "CHUNK",
    "CHUNK_MULTI_PATH",
    "COMMUNITY",
    "ENTITY",
    "ENTITY_LOOKUP",
    "ENUMERATE",
    "FACTS",
    "HYBRID_CROSS_ENCODER",
    "HYBRID_RRF",
    "LOCAL",
    "MULTI_PATH",
    "MULTI_PATH_CYPHER",
    "RELATION",
    "RELATION_EXPAND",
    "TEXT2CYPHER",
]
