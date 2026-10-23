"""Leiden community detection via graspologic-native."""

from __future__ import annotations

from uuid import UUID

from am_diag.common.cypher import load
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.relation import Relation
from am_diag.db.graph.base import GraphStore
from am_diag.graph_construction.community.base import CommunityDetector, build_edge_list
from am_diag.graph_construction.config import CommunityDetectionConfig


class LeidenCommunityDetector(CommunityDetector):
    """Leiden clustering via graspologic-native.

    Returns hierarchical ``Community`` DataPoints from relations or a graph store.
    """

    def __init__(self, config: CommunityDetectionConfig | None = None) -> None:
        """Initialize detector with optional config."""
        self._config = config or CommunityDetectionConfig()

    async def detect(self, relations: list[Relation]) -> list[Community]:
        """Run hierarchical Leiden on the relation graph.

        Args:
            relations: Resolved ``Relation`` DataPoints.

        Returns:
            List of ``Community`` DataPoints with ``entity_ids`` populated.
        """
        return self._detect_edges(build_edge_list(relations))

    async def detect_all(self, graph_store: GraphStore) -> list[Community]:
        """Detect communities over all persisted Entity RELATES_TO edges.

        Pages reads but holds the full edge list in memory, suitable for up to a
        few million edges. Entities without a non-self RELATES_TO edge are omitted.
        Level zero contains the coarsest partitions; higher levels refine them.

        Args:
            graph_store: Store containing the entity graph.

        Returns:
            Communities whose parent IDs reference returned communities.
        """
        edges: dict[tuple[str, str], float] = {}
        offset = 0
        limit = 10_000
        query = load("community/all_entity_edges")
        while True:
            rows = await graph_store.execute_query(
                query, {"offset": offset, "limit": limit}, read_only=True
            )
            for row in rows:
                source, target = sorted((row["source"], row["target"]))
                if source != target:
                    key = (source, target)
                    edges[key] = edges.get(key, 0.0) + float(row["score"])
            if len(rows) < limit:
                break
            offset += len(rows)
        return self._detect_edges([(a, b, score) for (a, b), score in edges.items()])

    def _detect_edges(self, edges: list[tuple[str, str, float]]) -> list[Community]:
        """Cluster weighted edges and resolve parent IDs after building all nodes."""
        if not edges:
            return []
        try:
            from graspologic_native import (  # noqa: PLC0415
                hierarchical_leiden,  # type: ignore
            )
        except ImportError:
            raise ImportError(
                "Leiden community detection requires:"
                " pip install 'am_diag[community_detection]'",
            ) from None

        clusters = hierarchical_leiden(
            edges,
            max_cluster_size=self._config.max_cluster_size,
            resolution=self._config.resolution,
            randomness=self._config.randomness,
            use_modularity=self._config.use_modularity,
            iterations=self._config.iterations,
            seed=self._config.seed,
            starting_communities={},
        )

        communities: dict[tuple[int, int], Community] = {}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        for cluster in clusters:
            key = (cluster.level, cluster.cluster)
            if key not in communities:
                communities[key] = Community(
                    community_id=f"{cluster.level}:{cluster.cluster}",
                    level=cluster.level,
                )
            if cluster.parent_cluster is not None:
                parents[key] = (cluster.level - 1, cluster.parent_cluster)
            communities[key].entity_ids.append(UUID(cluster.node))

        for key, parent_key in parents.items():
            communities[key].parent = communities[parent_key].id
        return list(communities.values())
