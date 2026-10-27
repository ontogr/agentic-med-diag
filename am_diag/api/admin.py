"""Registry-backed admin namespaces for graphs and projects.

``client.graphs`` and ``client.projects`` expose CRUD over the registry. Creating
a graph eagerly builds its live connections to validate connectivity and install
schema constraints, so a returned record is known-reachable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from am_diag.api.errors import RegistryError
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.project import ProjectRecord
from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA


if TYPE_CHECKING:
    from am_diag.api.connection_pool import ConnectionPool
    from am_diag.api.registry import Registry


# The admin classes expose a ``.list()`` method that shadows the builtin ``list``
# inside the class namespace; this alias keeps ``list[...]`` annotations valid.
_List = list


class GraphAdmin:
    """CRUD namespace for graph deployments (``client.graphs``)."""

    def __init__(self, registry: Registry, pool: ConnectionPool) -> None:
        """Bind the admin namespace to the registry and connection pool."""
        self._registry = registry
        self._pool = pool

    async def create(
        self,
        graph_id: str,
        *,
        neo4j: GraphConnection,
        vector: VectorConnection,
        schema: GraphSchema = MEDICAL_GRAPHRAG_SCHEMA,
        name: str | None = None,
        description: str | None = None,
    ) -> GraphRecord:
        """Register a graph, validate connectivity, and install its schema.

        The record is persisted first, then its live graph is built eagerly to
        verify the backend is reachable, to create schema constraints, and to
        store the schema's ontology in the graph. A graph is tied to the ontology
        version it was created with. If validation fails the record is rolled
        back so no dangling entry remains.

        Args:
            graph_id: Caller-chosen unique id.
            neo4j: Neo4j connection metadata.
            vector: Internal vector-store connection metadata.
            schema: Graph schema, including its ontology version and mode.
                Defaults to the medical GraphRAG schema.
            name: Optional display name.
            description: Optional display description.

        Returns:
            The persisted graph record.

        Raises:
            RegistryError: If ``graph_id`` already exists, including as a live
                graph attached to the workspace.
            ConnectionError: If a referenced env var is unset or the backend is
                unreachable.
            OntologyVersionMismatch: If the backend already stores a different
                ontology for the graph, such as data left from an earlier graph
                that was deleted without ``drop_data``.
        """
        if self._pool.has_graph(graph_id):
            raise RegistryError(f"Graph already exists: {graph_id!r}")
        record = GraphRecord(
            graph_id=graph_id,
            name=name,
            description=description,
            neo4j=neo4j,
            vector=vector,
            graph_schema=schema,
        )
        self._registry.add_graph(record)
        try:
            graph = self._pool.graph(graph_id)
            await graph.graph_store.health_check()
            await graph.graph_store.setup_schema(schema)
            await graph.graph_store.register_ontology(schema)
        except Exception:
            await self._pool.evict_graph(graph_id)
            self._registry.remove_graph(graph_id)
            raise
        return record

    async def get(self, graph_id: str) -> GraphRecord:
        """Return the persisted record for ``graph_id``."""
        return self._registry.get_graph(graph_id)

    async def list(self) -> _List[GraphRecord]:
        """Return all graph records."""
        return self._registry.list_graphs()

    async def update(
        self,
        graph_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: GraphSchema | None = None,
    ) -> GraphRecord:
        """Update display metadata or schema, reconnecting the graph if needed."""
        updated = self._registry.update_graph(
            graph_id,
            name=name,
            description=description,
            graph_schema=schema,
        )
        if schema is not None:
            await self._pool.evict_graph(graph_id)
        return updated

    async def delete(self, graph_id: str, *, drop_data: bool = False) -> None:
        """Remove a graph from the registry and (optionally) wipe its data.

        Args:
            graph_id: Graph to remove.
            drop_data: When ``True``, also delete all Neo4j nodes and the
                graph's vector collections. Destructive and irreversible.

        Raises:
            GraphNotFoundError: If the graph is not registered.
            RegistryError: If ``drop_data`` is set and another registered graph
                uses one of the same vector collections.
        """
        record = self._registry.get_graph(graph_id)
        if drop_data:
            self._reject_shared_collections(record)
            graph = self._pool.graph(graph_id)
            await graph.graph_store.wipe()
            for collection in graph.collections.values():
                await graph.vector_store.delete_collection(collection)
        self._registry.remove_graph(graph_id)
        await self._pool.evict_graph(graph_id)

    def _reject_shared_collections(self, record: GraphRecord) -> None:
        """Refuse to drop vector collections that another graph also uses."""
        owned = set(record.vector.collections.values())
        for other in self._registry.list_graphs():
            if other.graph_id == record.graph_id:
                continue
            same_store = (
                other.vector.backend == record.vector.backend
                and other.vector.url_env == record.vector.url_env
            )
            if same_store and owned & set(other.vector.collections.values()):
                raise RegistryError(
                    f"Cannot drop data for {record.graph_id!r}: vector collections "
                    f"are shared with graph {other.graph_id!r}"
                )


class ProjectAdmin:
    """CRUD namespace for projects (``client.projects``)."""

    def __init__(self, registry: Registry) -> None:
        """Bind the admin namespace to the registry."""
        self._registry = registry

    async def create(
        self,
        project_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        graph_ids: _List[str] | None = None,
        agent_settings: dict[str, Any] | None = None,
    ) -> ProjectRecord:
        """Register a new project.

        Raises:
            RegistryError: If ``project_id`` already exists or a listed graph
                does not exist.
        """
        for gid in graph_ids or []:
            if gid not in {g.graph_id for g in self._registry.list_graphs()}:
                raise RegistryError(f"Unknown graph in project: {gid!r}")
        record = ProjectRecord(
            project_id=project_id,
            name=name,
            description=description,
            graph_ids=list(graph_ids or []),
            agent_settings=agent_settings,
        )
        self._registry.add_project(record)
        return record

    async def get(self, project_id: str) -> ProjectRecord:
        """Return the persisted record for ``project_id``."""
        return self._registry.get_project(project_id)

    async def list(self) -> _List[ProjectRecord]:
        """Return all project records."""
        return self._registry.list_projects()

    async def update(
        self,
        project_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        agent_settings: dict[str, Any] | None = None,
    ) -> ProjectRecord:
        """Update project display metadata or agent settings."""
        return self._registry.update_project(
            project_id,
            name=name,
            description=description,
            agent_settings=agent_settings,
        )

    async def delete(self, project_id: str) -> None:
        """Remove a project from the registry."""
        self._registry.remove_project(project_id)
