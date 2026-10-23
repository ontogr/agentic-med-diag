"""Check persisted-edge loading and native Leiden hierarchy membership."""

from collections import Counter, defaultdict
from itertools import combinations
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from am_diag.graph_construction.community.leiden import LeidenCommunityDetector
from am_diag.graph_construction.config import CommunityDetectionConfig


class TestLeidenDetectAll:
    """Weighted graph loading, parent links, and entity coverage."""

    async def test_seeded_graph_preserves_hierarchy_and_leaf_coverage(self) -> None:
        """A ring of cliques has valid parents and disjoint terminal membership."""
        ids = [str(UUID(int=i + 1)) for i in range(64)]
        pairs = [
            (a, b)
            for group in range(16)
            for a, b in combinations(ids[group * 4 : (group + 1) * 4], 2)
        ]
        pairs += [(ids[g * 4], ids[((g + 1) % 16) * 4]) for g in range(16)]
        graph = AsyncMock()
        graph.execute_query.return_value = [
            {"source": a, "target": b, "score": 1.0} for a, b in pairs
        ]
        detector = LeidenCommunityDetector(
            CommunityDetectionConfig(max_cluster_size=4, seed=0xDEADBEEF)
        )

        communities = await detector.detect_all(graph)

        by_id = {c.id: c for c in communities}
        assert len(by_id) == len(communities)
        assert {c.level for c in communities} == {0, 1}
        roots = [c for c in communities if c.parent is None]
        assert Counter(e for c in roots for e in c.entity_ids) == Counter(
            UUID(e) for e in ids
        )
        children = defaultdict(list)
        for community in communities:
            assert community.community_id.count(":") == 1
            if community.parent is not None:
                parent = by_id[community.parent]
                assert parent.level == community.level - 1
                assert set(community.entity_ids) <= set(parent.entity_ids)
                children[parent.id].append(community)
        for parent_id, members in children.items():
            assert Counter(e for c in members for e in c.entity_ids) == Counter(
                by_id[parent_id].entity_ids
            )
        leaves = [c for c in communities if c.id not in children]
        assert Counter(e for c in leaves for e in c.entity_ids) == Counter(
            UUID(e) for e in ids
        )

    async def test_pages_and_combines_parallel_edges_with_config(self, monkeypatch):
        """Pagination retains scores and forwards every native Leiden setting."""
        a, b, c = [str(UUID(int=i)) for i in (1, 2, 3)]
        graph = AsyncMock()
        graph.execute_query.side_effect = [
            [{"source": a, "target": b, "score": 0.25}] * 10_000,
            [
                {"source": b, "target": a, "score": 0.5},
                {"source": b, "target": c, "score": 0.75},
                {"source": c, "target": c, "score": 1.0},
            ],
        ]
        received = {}

        def cluster(edges, **kwargs):
            received.update(edges=edges, **kwargs)
            return []

        monkeypatch.setattr("graspologic_native.hierarchical_leiden", cluster)
        config = CommunityDetectionConfig(
            max_cluster_size=7,
            resolution=0.8,
            randomness=0.2,
            use_modularity=False,
            iterations=3,
            seed=123,
        )

        await LeidenCommunityDetector(config).detect_all(graph)

        assert received == {
            "edges": [(a, b, 2500.5), (b, c, 0.75)],
            "max_cluster_size": 7,
            "resolution": 0.8,
            "randomness": 0.2,
            "use_modularity": False,
            "iterations": 3,
            "seed": 123,
            "starting_communities": {},
        }
        assert [call.args[1] for call in graph.execute_query.await_args_list] == [
            {"offset": 0, "limit": 10_000},
            {"offset": 10_000, "limit": 10_000},
        ]
        assert all(
            call.kwargs["read_only"] for call in graph.execute_query.await_args_list
        )

    @pytest.mark.parametrize("self_only", [False, True])
    async def test_empty_or_self_only_graph_has_no_communities(self, self_only):
        """Entities without non-self edges are excluded from clustering."""
        graph = AsyncMock()
        graph.execute_query.return_value = (
            [{"source": str(UUID(int=1)), "target": str(UUID(int=1)), "score": 1.0}]
            if self_only
            else []
        )

        assert await LeidenCommunityDetector().detect_all(graph) == []
