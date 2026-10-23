"""The ``Graph`` object: one self-contained GraphRAG backend.

A ``Graph`` owns its live connections (Neo4j + vector store + embedder) and a
per-graph search engine built with its schema. Construct one directly with
:meth:`Graph.connect` (bring-your-own-backend) or obtain a named, pooled one from
a :class:`am_diag.Workspace`. It exposes single-graph retrieval, curation, schema,
a guarded raw query, and the durable write surface.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from am_diag.api.context import pack_context
from am_diag.api.errors import SchemaError
from am_diag.api.resources import (
    ChunkResource,
    CommunityResource,
    EntityResource,
    RelationResource,
)
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.job import JobConfig
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.data_models.search import SearchResult
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.factory import create_graph_store
from am_diag.db.graph.settings import FalkorDBSettings, Neo4jSettings
from am_diag.db.vector.base import VectorStoreBase
from am_diag.db.vector.qdrant import create_qdrant_store
from am_diag.db.vector.settings import QdrantSettings, WeaviateSettings
from am_diag.db.vector.weaviate import create_weaviate_store
from am_diag.ingestion.cache import IngestionCache
from am_diag.ingestion.jobs import IngestionJob, new_job_id
from am_diag.ingestion.models import IngestionReport
from am_diag.retrieval.config import RetrievalConfig, SearchConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.search import SearchEngine
from am_diag.vector.embedding.base import Embedder
from am_diag.vector.embedding.sentence_transformers import SentenceTransformersEmbedder
from am_diag.vector.reranking.base import Reranker


if TYPE_CHECKING:
    pass

_DEFAULT_STATE_DIR = Path(".am_diag")

_DEFAULT_COLLECTIONS: dict[str, str] = {
    "entity": "am_diag_entity",
    "relation": "am_diag_relation",
    "chunk": "am_diag_chunk",
    "community": "am_diag_community",
}

_WRITE_CLAUSE_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|FOREACH)\b", re.IGNORECASE
)


def _has_write_clause(cypher: str) -> bool:
    """Conservatively detect whether ``cypher`` contains a write clause."""
    return _WRITE_CLAUSE_RE.search(cypher) is not None


def _build_vector_store(
    vector: VectorStoreBase | QdrantSettings | WeaviateSettings,
) -> VectorStoreBase:
    """Return a live vector store from a live store or a settings object."""
    if isinstance(vector, VectorStoreBase):
        return vector
    if isinstance(vector, WeaviateSettings):
        return create_weaviate_store(vector)
    return create_qdrant_store(vector)


def _retrieval_config(collections: dict[str, str]) -> RetrievalConfig:
    """Build a RetrievalConfig whose collection names match this graph."""
    return RetrievalConfig(
        entity_collection=collections["entity"],
        relation_collection=collections["relation"],
        chunk_collection=collections["chunk"],
        community_collection=collections["community"],
    )


def _derive_graph_id(neo4j: GraphStore | Neo4jSettings | FalkorDBSettings) -> str:
    """Derive a stable graph id from the backend identity, else a uuid.

    Deriving from the Neo4j URI + database means reconnecting to the same backend
    reuses the same content-cache shard across runs (without a Workspace).
    """
    if isinstance(neo4j, Neo4jSettings):
        digest = hashlib.sha256(f"{neo4j.uri}|{neo4j.database}".encode()).hexdigest()[
            :12
        ]
        return f"auto-{digest}"
    if isinstance(neo4j, FalkorDBSettings):
        identity = f"falkordb|{neo4j.host}|{neo4j.port}|{neo4j.ssl}|{neo4j.graph}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
        return f"auto-{digest}"
    return uuid4().hex


class Graph:
    """One self-contained GraphRAG backend.

    Prefer :meth:`connect` or ``Workspace.graph(id)`` over calling this directly.

    Attributes:
        graph_id: Stable id; the content-cache shard and durable job dir key off
            it.
        schema: The graph's :class:`GraphSchema` (drives extraction + text2cypher).
    """

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        vector_store: VectorStoreBase,
        embedder: Embedder,
        search_engine: SearchEngine,
        schema: GraphSchema,
        collections: dict[str, str],
        graph_id: str,
        state_dir: Path,
        reranker: Reranker | None = None,
        cache: IngestionCache | None = None,
        job_registry: dict[str, IngestionJob] | None = None,
    ) -> None:
        """Store the live dependency set for one graph (see :meth:`connect`)."""
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._search_engine = search_engine
        self._schema = schema
        self._collections = collections
        self._graph_id = graph_id
        self._state_dir = state_dir
        self._reranker = reranker
        self._cache = cache
        self._job_registry = job_registry

    @classmethod
    def connect(
        cls,
        *,
        neo4j: GraphStore | Neo4jSettings | FalkorDBSettings,
        vector: VectorStoreBase | QdrantSettings | WeaviateSettings,
        schema: GraphSchema = MEDICAL_GRAPHRAG_SCHEMA,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        collections: dict[str, str] | None = None,
        graph_id: str | None = None,
        state_dir: Path | None = None,
        cache: IngestionCache | None = None,
        job_registry: dict[str, IngestionJob] | None = None,
    ) -> Graph:
        """Build a graph from live backends or settings (bring-your-own-backend).

        Connections are created eagerly but lazily validated (the first ingest
        installs schema constraints). Use as an async context manager so the
        connections close on exit.

        Args:
            neo4j: A live ``GraphStore``, ``Neo4jSettings`` or ``FalkorDBSettings``.
            vector: A live ``VectorStoreBase`` or a Qdrant/Weaviate settings.
            schema: The graph schema. Defaults to the medical GraphRAG schema.
            embedder: Query/ingest embedder. Defaults to the ST default model.
            reranker: Optional cross-encoder reranker.
            collections: Logical->physical vector collection names.
            graph_id: Stable id. Defaults to one derived from the backend, so
                reconnecting reuses the content cache.
            state_dir: Root for the content cache + durable jobs. Defaults to
                ``.am_diag``.
            cache: Shared content cache. Managed use injects the Workspace's
                per-graph shard; bring-your-own use leaves it ``None`` and a
                shard under ``state_dir`` is built on first ingest.
            job_registry: Shared in-session job registry. Managed use injects the
                Workspace's so ``jobs.resume`` can find live jobs.
        """
        graph_store = (
            neo4j if isinstance(neo4j, GraphStore) else create_graph_store(neo4j)
        )
        vector_store = _build_vector_store(vector)
        embedder = embedder or SentenceTransformersEmbedder()
        collections = collections or dict(_DEFAULT_COLLECTIONS)
        search_engine = SearchEngine(
            _retrieval_config(collections),
            vector_store,
            graph_store,
            embedder,
            schema=schema,
            reranker=reranker,
        )
        return cls(
            graph_store=graph_store,
            vector_store=vector_store,
            embedder=embedder,
            search_engine=search_engine,
            schema=schema,
            collections=collections,
            graph_id=graph_id or _derive_graph_id(neo4j),
            state_dir=state_dir or _DEFAULT_STATE_DIR,
            reranker=reranker,
            cache=cache,
            job_registry=job_registry,
        )

    @property
    def graph_id(self) -> str:
        """Stable id of this graph."""
        return self._graph_id

    @property
    def graph_store(self) -> GraphStore:
        """The live graph store (for the job engine and advanced callers)."""
        return self._graph_store

    @property
    def vector_store(self) -> VectorStoreBase:
        """The live vector store."""
        return self._vector_store

    @property
    def embedder(self) -> Embedder:
        """The embedder used for this graph's vectors."""
        return self._embedder

    @property
    def collections(self) -> dict[str, str]:
        """Logical->physical vector collection names for this graph."""
        return self._collections

    @property
    def search_engine(self) -> SearchEngine:
        """The per-graph search engine."""
        return self._search_engine

    def ensure_cache(self) -> IngestionCache:
        """Return (creating on first use) this graph's content-cache shard.

        Engine-facing seam used by durable ingestion/QA jobs.
        """
        if self._cache is None:
            self._cache = IngestionCache(
                self._state_dir / "cache" / f"{self._graph_id}.db"
            )
        return self._cache

    @property
    def state_dir(self) -> Path:
        """Root state directory for this graph's cache and durable jobs."""
        return self._state_dir

    async def search(
        self,
        query: str,
        *,
        recipe: SearchConfig | str = "hybrid_rrf",
        filters: SearchFilters | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Run a recipe-driven fused search, tagged with this graph's id."""
        results = await self._search_engine.search(
            query, recipe=recipe, filters=filters
        )
        for result in results:
            result.source_graph_id = self._graph_id
        return results[:limit] if limit is not None else results

    async def context(
        self,
        query: str,
        *,
        recipe: SearchConfig | str = "hybrid_rrf",
        filters: SearchFilters | None = None,
        max_characters: int = 2500,
    ) -> str:
        """Return a packed context block for ``query`` within a char budget."""
        results = await self.search(query, recipe=recipe, filters=filters)
        return pack_context(results, max_characters)

    @property
    def entities(self) -> EntityResource:
        """Entity curation resource."""
        return EntityResource(self._graph_store)

    @property
    def relations(self) -> RelationResource:
        """Relation curation resource."""
        return RelationResource(self._graph_store)

    @property
    def chunks(self) -> ChunkResource:
        """Read-only chunk resource."""
        return ChunkResource(self._graph_store)

    @property
    def communities(self) -> CommunityResource:
        """Read-only community resource."""
        return CommunityResource(self._graph_store)

    @property
    def schema(self) -> GraphSchema:
        """This graph's schema (drives extraction and text2cypher)."""
        return self._schema

    def set_schema(self, schema: GraphSchema) -> GraphSchema:
        """Replace this graph's schema in memory and rebuild the search engine.

        Applies to future ingestion/retrieval only. For a Workspace-managed graph,
        use ``Workspace.graphs.update(id, schema=...)`` to also persist it.
        """
        self._schema = schema
        self._search_engine = SearchEngine(
            _retrieval_config(self._collections),
            self._vector_store,
            self._graph_store,
            self._embedder,
            schema=schema,
            reranker=self._reranker,
        )
        return schema

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        read_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Run raw Cypher; reject write clauses unless ``read_only`` is False.

        Raises:
            SchemaError: If ``read_only`` and the query contains a write clause.
        """
        if read_only and _has_write_clause(cypher):
            raise SchemaError("Write Cypher rejected in read-only query()")
        return await self._graph_store.execute_query(
            cypher, params, read_only=read_only
        )

    async def ingest(
        self, sources: Any, *, config: JobConfig | None = None
    ) -> IngestionJob:
        """Create a durable ingestion job for one or more sources (does not run).

        Raises:
            ValueError: If ``sources`` is an empty list.
        """
        from am_diag.common.data_models.job import JobRecord  # noqa: PLC0415

        source_list = sources if isinstance(sources, list) else [sources]
        if not source_list:
            raise ValueError("ingest() requires at least one source")
        record = JobRecord(
            job_id=new_job_id(),
            kind="ingestion",
            graph_id=self._graph_id,
            config=config or JobConfig(),
        )
        job = IngestionJob(
            record,
            target=self,
            cache=self.ensure_cache(),
            state_dir=self._state_dir,
            sources=source_list,
        )
        if self._job_registry is not None:
            self._job_registry[job.id] = job
        return job

    async def add(
        self,
        *,
        text: str | None = None,
        documents: list[Document] | None = None,
        entities: list[Entity] | None = None,
        relations: list[Relation] | None = None,
        source_description: str | None = None,
    ) -> IngestionReport:
        """Ingest inline content now and return a summary report.

        Raises:
            ValueError: If no content is supplied, or if text or documents are
                mixed with entities or relations.
        """
        source = self._inline_source(
            text, documents, entities, relations, source_description
        )
        job = await self.ingest([source])
        record = await job.run()
        if record.progress.failed or not record.progress.succeeded:
            return IngestionReport(batch_id=record.job_id)
        return IngestionReport(
            batch_id=record.job_id,
            documents_ingested=len(documents or []) + (text is not None),
            entities_ingested=len(entities or []),
            relations_ingested=len(relations or []),
        )

    async def add_fact(
        self,
        *,
        head: str,
        relation: str,
        tail: str,
        head_label: str = "Entity",
        tail_label: str = "Entity",
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Relation:
        """Assert a single deterministic triple (curation), bypassing the LLM."""
        head_entity = Entity(name=head, label=head_label)
        tail_entity = Entity(name=tail, label=tail_label)
        rel = Relation(
            head_id=head_entity.id,
            tail_id=tail_entity.id,
            head_name=head,
            tail_name=tail,
            type=relation,
            description=description,
            schema_properties=attributes or {},
        )
        kg = KnowledgeGraph(
            entities=[head_entity, tail_entity],
            relations=[rel],
            graph_schema=self._schema,
        )
        await self._graph_store.write_knowledge_graph(kg)
        return rel

    async def finalize(self, *, config: Any | None = None) -> int:
        """Run GDS community detection over the whole graph; return the count."""
        from am_diag.ingestion.graph_ingestion import finalize_graph  # noqa: PLC0415

        return await finalize_graph(self._graph_store, config)

    def _inline_source(
        self,
        text: str | None,
        documents: list[Document] | None,
        entities: list[Entity] | None,
        relations: list[Relation] | None,
        source_description: str | None,
    ) -> Any:
        """Adapt raw inline content into a one-batch source for the job engine."""
        has_text = text is not None or bool(documents)
        has_graph = bool(entities or relations)
        if has_text and has_graph:
            raise ValueError(
                "add() takes text/documents or entities/relations, not both"
            )
        if has_text:
            docs = list(documents or [])
            if text is not None:
                docs.append(Document(text=text, source=source_description or "inline"))
            return _InlineCorpusSource(docs)
        if has_graph:
            kg = KnowledgeGraph(
                entities=list(entities or []),
                relations=list(relations or []),
                graph_schema=self._schema,
            )
            return _InlineGraphSource(kg)
        raise ValueError("add() requires text, documents, entities, or relations")

    async def close(self) -> None:
        """Close the graph store, vector store, embedder, and reranker."""
        await self._graph_store.close()
        await self._vector_store.close()
        await self._embedder.close()
        if self._reranker is not None:
            await self._reranker.close()

    async def __aenter__(self) -> Graph:
        """Enter the async context, returning self."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit the async context, closing connections."""
        await self.close()


class _InlineCorpusSource:
    """Adapter wrapping pre-built documents as a one-batch corpus source."""

    source_kind = "corpus"

    def __init__(self, documents: list[Document]) -> None:
        """Store the documents to stream as a single batch."""
        self._documents = documents

    async def astream(self, batch_size: int = 100):  # noqa: ARG002
        """Yield the documents as one batch."""
        yield self._documents


class _InlineGraphSource:
    """Adapter wrapping a pre-built KnowledgeGraph as a one-batch graph source."""

    source_kind = "graph"

    def __init__(self, kg: KnowledgeGraph) -> None:
        """Store the knowledge graph to stream as a single batch."""
        self._kg = kg

    async def astream(self, batch_size: int = 5000):  # noqa: ARG002
        """Yield the knowledge graph as one batch."""
        yield self._kg
