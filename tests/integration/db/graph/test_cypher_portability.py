"""Exercise every shared Cypher template against the same nonempty graph.

Write templates run through the public store contract. Read templates have
explicit cardinalities and column contracts, including optional-match maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.db.graph.base import GraphStore


_READ_CASES = {
    "admin/get_entity": ("entity", 1, {"id", "labels", "props"}),
    "admin/list_entities": ("page", 3, {"id", "labels", "props"}),
    "admin/entity_neighbors": ("entity", 2, {"id", "labels", "props"}),
    "admin/entity_relations": ("entity", 1, {"id", "props", "head_id", "tail_id"}),
    "admin/entity_chunks": ("entity", 1, {"id", "props"}),
    "admin/get_relation": ("relation", 1, {"id", "props", "head_id", "tail_id"}),
    "admin/list_relations": ("page", 2, {"id", "props", "head_id", "tail_id"}),
    "admin/get_chunk": ("chunk", 1, {"id", "props"}),
    "admin/list_chunks": ("page", 3, {"id", "props"}),
    "admin/get_community": ("community", 1, {"id", "props"}),
    "admin/list_communities": ("page", 2, {"id", "props"}),
    "retrieval/hydrate_entities": ("entity", 1, {"id", "labels", "props"}),
    "retrieval/hydrate_relations": (
        "relation",
        1,
        {"id", "props", "head_id", "tail_id"},
    ),
    "retrieval/bfs_expand": ("entity", 2, {"id", "label", "name", "hop", "rels"}),
    "retrieval/chunk_seed_entities": ("chunk", 2, {"id"}),
    "retrieval/community_seed_entities": ("community", 2, {"id"}),
    "retrieval/community_members": (
        "community",
        2,
        {"entity_id", "entity_name", "labels", "chunk_ids"},
    ),
    "retrieval/reports_at_level": (
        "community",
        1,
        {"id", "community_id", "title", "summary"},
    ),
    "retrieval/chunk_neighbors": (
        "chunk",
        1,
        {
            "chunk_id",
            "chunk_text",
            "entities",
            "relations",
            "communities",
            "next_chunk_id",
            "prev_chunk_id",
        },
    ),
    "community/member_subgraph": (
        "community",
        2,
        {"entity_id", "entity_name", "entity_labels", "edges"},
    ),
    "community/all_entity_edges": ("page", 2, {"source", "target", "score"}),
    "ontology/load_ontology": ("page", 0, {"version", "schema_json"}),
}
_WRITE_CASES = {
    "ingest/upsert_documents",
    "ingest/upsert_chunks",
    "ingest/upsert_entities",
    "ingest/upsert_relations",
    "ingest/link_part_of",
    "ingest/link_next_chunk",
    "ingest/link_has_entity",
    "community/upsert_communities",
    "community/upsert_reports",
    "community/link_in_community",
    "community/link_parent_community",
    "community/link_has_report",
    "community/delete_communities",
    "admin/update_entity",
    "admin/update_relation",
    "admin/delete_relation",
    "admin/delete_entity",
    "ontology/upsert_entity_types",
    "ontology/upsert_pairs",
    "ontology/upsert_meta",
}
_NEO4J_ONLY = {
    "schema/constraints",
    "schema/indexes",
    "schema/ontology_constraints",
    "community/run_gds_leiden",
    "admin/wipe_graph",
}


class TestCypherPortability:
    """Shared templates retain their column contracts on both servers."""

    def test_every_shared_template_has_a_case(self) -> None:
        """Adding a query requires extending the executable portability contract."""
        root = Path(__file__).parents[4] / "am_diag" / "common" / "cypher"
        shared = {
            p.relative_to(root).with_suffix("").as_posix()
            for p in root.rglob("*.cypher")
            if "falkordb" not in p.relative_to(root).parts
        }
        assert shared - _NEO4J_ONLY == set(_READ_CASES) | _WRITE_CASES

    @pytest.mark.parametrize("name", sorted(_READ_CASES))
    async def test_shared_read(
        self, name: str, graph_store: GraphStore, knowledge_graph: KnowledgeGraph
    ) -> None:
        """Read seeded records, not an empty graph that masks Cypher differences."""
        kg = knowledge_graph
        await graph_store.write_knowledge_graph(kg)
        kind, count, columns = _READ_CASES[name]
        ids = {
            "entity": str(kg.entities[0].id),
            "relation": str(kg.relations[0].id),
            "chunk": str(kg.chunks[0].id),
            "community": str(kg.communities[1].id),
            "page": None,
        }
        selected = ids[kind]
        params: dict[str, Any] = {
            "id": selected,
            "ids": [selected],
            "seed_ids": [ids["entity"]],
            "chunk_id": ids["chunk"],
            "community_id": ids["community"],
            "cursor": None,
            "label": None,
            "type": None,
            "depth": 2,
            "level": 0,
            "offset": 0,
            "limit": 100,
        }
        query = load(name).replace("${depth}", "2")
        rows = await graph_store.execute_query(query, params, read_only=True)
        assert len(rows) == count
        assert all(set(row) == columns for row in rows)
        if name == "retrieval/chunk_neighbors":
            row = rows[0]
            assert row["next_chunk_id"] == str(kg.chunks[1].id)
            assert row["prev_chunk_id"] is None
            assert {e["name"] for e in row["entities"]} == {"Alpha", "Beta"}
        elif name == "community/all_entity_edges":
            assert {row["score"] for row in rows} == {0.7, 0.9}
        elif name == "retrieval/community_members":
            assert {row["entity_name"] for row in rows} == {"Alpha", "Beta"}
        elif name == "retrieval/reports_at_level":
            assert rows[0]["title"] == "Contract report"

    async def test_structural_links_and_community_deletion(
        self, graph_store: GraphStore, knowledge_graph: KnowledgeGraph
    ) -> None:
        """One-chunk documents have no NEXT_CHUNK edge or null-endpoint error."""
        kg = knowledge_graph
        await graph_store.write_knowledge_graph(kg)
        assert await graph_store.execute_query(
            "RETURN range(0, -1) AS items", read_only=True
        ) == [{"items": []}]
        rows = await graph_store.execute_query(
            "MATCH (:Chunk)-[r:NEXT_CHUNK]->(:Chunk) RETURN count(r) AS count",
            read_only=True,
        )
        assert rows == [{"count": 1}]
        rows = await graph_store.execute_query(
            "MATCH (:Community)-[r:PARENT_COMMUNITY]->(:Community) "
            "RETURN count(r) AS count",
            read_only=True,
        )
        assert rows == [{"count": 1}]
        await graph_store.execute_query(load("community/delete_communities"))
        assert await graph_store.list_communities() == []
