"""Tests for whole-graph finalization (community detection).

``finalize_graph`` must run whole-graph detection once and upsert + link the
result. Structured-graph and corpus ingestion are now covered by the durable
job engine tests. External boundaries (graph store, detector) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from am_diag.common.data_models import Community
from am_diag.db.graph.base import GraphCapabilities
from am_diag.ingestion import graph_ingestion as gi


class TestFinalizeGraph:
    """Graph-global community detection, persistence, and linking."""

    @pytest.mark.parametrize("gds_leiden", [True, False])
    async def test_detects_and_links_communities(self, monkeypatch, gds_leiden) -> None:
        """finalize_graph runs detect_all and upserts + links the communities."""
        graph_store = AsyncMock()
        graph_store.capabilities = GraphCapabilities(gds_leiden=gds_leiden)
        communities = [Community(community_id="0:1", entity_ids=[])]
        detector = AsyncMock()
        detector.detect_all = AsyncMock(return_value=communities)
        detector_path = (
            "am_diag.graph_construction.community.gds.GdsCommunityDetector"
            if gds_leiden
            else "am_diag.graph_construction.community.leiden.LeidenCommunityDetector"
        )
        monkeypatch.setattr(detector_path, lambda *args: detector)

        count = await gi.finalize_graph(graph_store)

        assert count == 1
        if gds_leiden:
            detector.detect_all.assert_awaited_once_with()
        else:
            detector.detect_all.assert_awaited_once_with(graph_store)
        graph_store.upsert_communities.assert_awaited_once_with(communities)
        graph_store.link_in_community.assert_awaited_once_with(communities)
        graph_store.link_parent_community.assert_awaited_once_with(communities)

    async def test_replaces_prior_communities_before_upserting(
        self, monkeypatch
    ) -> None:
        """Old Community nodes are deleted after detection and before the upsert."""
        calls: list[str] = []
        graph_store = AsyncMock()
        graph_store.capabilities = GraphCapabilities(gds_leiden=True)
        graph_store.execute_query.side_effect = lambda *_a, **_k: calls.append("delete")
        graph_store.upsert_communities.side_effect = lambda *_a: calls.append("upsert")
        detector = AsyncMock()
        detector.detect_all = AsyncMock(return_value=[Community(community_id="0:1")])
        monkeypatch.setattr(
            "am_diag.graph_construction.community.gds.GdsCommunityDetector",
            lambda graph, config: detector,
        )

        await gi.finalize_graph(graph_store)

        assert calls == ["delete", "upsert"]
