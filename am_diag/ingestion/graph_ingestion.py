"""Whole-graph finalization with backend-aware community detection.

Structured-graph and corpus ingestion now run through the durable job engine in
``am_diag.ingestion.jobs``. ``finalize_graph`` remains the single place that runs
community detection once over the whole persisted graph, after all chosen
sources have been ingested. It is surfaced via ``IngestionJob.run(finalize=True)``
and ``Graph.finalize``.
"""

from __future__ import annotations

import logging
from typing import Any

from am_diag.common.cypher import load
from am_diag.db.graph.base import GraphStore


logger = logging.getLogger(__name__)


async def finalize_graph(graph_store: GraphStore, config: Any | None = None) -> int:
    """Run community detection once over the whole persisted graph; persist + link.

    Communities from an earlier finalization are replaced, so re-running never
    leaves duplicate nodes or stale links.

    Detection only: community **report** generation (LLM summarization) is deferred
    to a later slice because it is one LLM call per community over a graph that can
    hold thousands of them. Run this after all desired sources are ingested so the
    union gets correct community structure.

    Args:
        graph_store: Graph database client with declared algorithm capabilities.
        config: Optional ``CommunityDetectionConfig`` (seed, etc.).

    Returns:
        The number of communities detected.
    """
    from am_diag.graph_construction.community.gds import (  # noqa: PLC0415
        GdsCommunityDetector,
    )
    from am_diag.graph_construction.community.leiden import (  # noqa: PLC0415
        LeidenCommunityDetector,
    )

    if graph_store.capabilities.gds_leiden:
        communities = await GdsCommunityDetector(graph_store, config).detect_all()
    else:
        communities = await LeidenCommunityDetector(config).detect_all(graph_store)
    await graph_store.execute_query(load("community/delete_communities"))
    await graph_store.upsert_communities(communities)
    await graph_store.link_in_community(communities)
    await graph_store.link_parent_community(communities)
    logger.info("Finalized graph: %d communities", len(communities))
    return len(communities)


__all__ = ["finalize_graph"]
