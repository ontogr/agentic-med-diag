"""Live-graph connection pool: the data-plane half of the facade.

Turns a serialized ``GraphRecord`` into a live, cache-wired, job-wired ``Graph``
and owns that graph's lifecycle (build once, cache, evict, close). Secrets are
resolved only here, in the managed layer. The control plane (records) lives in
``Registry``; this is the only place live backends are created.
"""

from __future__ import annotations

import os
from pathlib import Path

from am_diag.api.errors import ConnectionError as WorkspaceConnectionError
from am_diag.api.errors import RegistryError
from am_diag.api.graph import Graph
from am_diag.api.registry import Registry
from am_diag.common.data_models.connection import VectorConnection
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.db.graph.settings import FalkorDBSettings, Neo4jSettings
from am_diag.db.vector.settings import QdrantSettings, WeaviateSettings
from am_diag.ingestion.cache import CacheManager, IngestionCache
from am_diag.ingestion.jobs import IngestionJob
from am_diag.vector.embedding.sentence_transformers import SentenceTransformersEmbedder


def _require_env(name: str) -> str:
    """Return env var ``name`` or raise ConnectionError if it is unset."""
    value = os.environ.get(name)
    if value is None:
        raise WorkspaceConnectionError(
            f"Required environment variable is unset: {name}"
        )
    return value


def _vector_settings(vector: VectorConnection) -> QdrantSettings | WeaviateSettings:
    """Resolve a vector connection's env-ref secrets into store settings."""
    url = _require_env(vector.url_env)
    api_key = _require_env(vector.api_key_env) if vector.api_key_env is not None else ""
    if vector.backend == "weaviate":
        return WeaviateSettings(url=url, api_key=api_key)
    return QdrantSettings(url=url, api_key=api_key)


def build_graph(
    record: GraphRecord,
    *,
    state_dir: Path,
    cache: IngestionCache,
    job_registry: dict[str, IngestionJob],
) -> Graph:
    """Resolve a registry record's env-ref secrets and build a live ``Graph``.

    This is the managed construction path: it turns the serialized descriptor
    into settings and delegates to :meth:`Graph.connect`, the single construction
    path. Secrets are resolved only here, in the managed layer.

    Raises:
        ConnectionError: If any referenced environment variable is unset.
    """
    conn = record.neo4j
    credentials = {}
    if conn.user_env is not None:
        credentials["username"] = _require_env(conn.user_env)
    if conn.password_env is not None:
        credentials["password"] = _require_env(conn.password_env)
    neo4j: Neo4jSettings | FalkorDBSettings
    if conn.backend == "falkordb":
        neo4j = FalkorDBSettings.from_url(conn.uri, graph=conn.database, **credentials)
    else:
        neo4j = Neo4jSettings(
            uri=conn.uri,
            user=credentials["username"],
            password=credentials["password"],
            database=conn.database,
        )
    return Graph.connect(
        neo4j=neo4j,
        vector=_vector_settings(record.vector),
        schema=record.graph_schema,
        embedder=SentenceTransformersEmbedder(model_name=record.vector.embedding_model),
        collections={str(k): v for k, v in record.vector.collections.items()},
        graph_id=record.graph_id,
        state_dir=state_dir,
        cache=cache,
        job_registry=job_registry,
    )


class ConnectionPool:
    """Builds, caches, and closes live ``Graph`` instances from records.

    Args:
        registry: Source of graph records.
        state_dir: Root for content caches and durable jobs.
        cache_manager: Owns per-graph content-cache shards.
        live_jobs: Shared in-session job registry passed to each built graph.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        state_dir: Path,
        cache_manager: CacheManager,
        live_jobs: dict[str, IngestionJob],
    ) -> None:
        """Bind the pool to its record source and managed-layer collaborators."""
        self._registry = registry
        self._state_dir = state_dir
        self._cache = cache_manager
        self._live_jobs = live_jobs
        self._graphs: dict[str, Graph] = {}

    def graph(self, graph_id: str) -> Graph:
        """Return the named live graph, building it from the registry once.

        Building resolves the record's env-ref secrets eagerly, so a missing
        secret raises ``ConnectionError`` here rather than at first use.
        """
        graph = self._graphs.get(graph_id)
        if graph is None:
            record = self._registry.get_graph(graph_id)
            graph = build_graph(
                record,
                state_dir=self._state_dir,
                cache=self._cache.for_graph(graph_id),
                job_registry=self._live_jobs,
            )
            self._graphs[graph_id] = graph
        return graph

    def has_graph(self, graph_id: str) -> bool:
        """Return whether a live graph is cached under ``graph_id``."""
        return graph_id in self._graphs

    def attach(self, graph: Graph) -> None:
        """Register an already-built live graph under its id.

        Bring-your-own-backend path: makes a ``Graph`` built via ``Graph.connect``
        addressable by ``graph(id)`` without a registry record.

        Attaching transfers close responsibility to the pool: an attached graph
        is closed by ``close``/``evict_graph`` alongside pool-built graphs. This
        avoids owned-vs-borrowed bookkeeping; callers who want to retain control
        of a graph's lifecycle should not attach it.

        Raises:
            RegistryError: If a live graph is already cached under the same id.
        """
        if graph.graph_id in self._graphs:
            raise RegistryError(f"Graph already attached: {graph.graph_id!r}")
        self._graphs[graph.graph_id] = graph

    async def evict_graph(self, graph_id: str) -> None:
        """Close and drop the cached graph for ``graph_id`` if present.

        Used after a schema update or delete so the next ``graph`` rebuilds.
        """
        graph = self._graphs.pop(graph_id, None)
        if graph is not None:
            await graph.close()

    async def close(self) -> None:
        """Close every cached graph and clear the pool.

        A graph that fails to close does not stop the others from closing; the
        first failure is re-raised after cleanup.
        """
        graphs = list(self._graphs.values())
        self._graphs.clear()
        first_error: Exception | None = None
        for graph in graphs:
            try:
                await graph.close()
            except Exception as exc:  # noqa: BLE001
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
