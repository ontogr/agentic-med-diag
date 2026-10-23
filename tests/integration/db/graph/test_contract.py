"""Run the same persistence and safety contract on both graph backends.

The asymmetric directed chain detects missing endpoints, reversed edges, and
duplicate writes. The fixture requires a disposable Neo4j test instance.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.errors import GraphQueryRejected, GraphWriteRejected
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.bfs import bfs_expand


class TestGraphContract:
    """Both backends expose identical CRUD, traversal, and read-only semantics."""

    async def test_ingest_admin_retrieval_and_rewrite(
        self, graph_store: GraphStore, knowledge_graph: KnowledgeGraph
    ) -> None:
        """Ingestion is repeatable and CRUD preserves identities and endpoints."""
        kg = knowledge_graph
        first, second, third = (str(e.id) for e in kg.entities)
        relation_id = str(kg.relations[0].id)
        await graph_store.write_knowledge_graph(kg)
        await graph_store.write_knowledge_graph(kg)
        rows = await graph_store.list_entities()
        assert {r["id"] for r in rows} == {first, second, third}
        assert len(await graph_store.list_relations()) == 2
        assert len(await graph_store.list_chunks()) == 3
        assert len(await graph_store.list_communities()) == 2
        entity = await graph_store.get_entity(first)
        assert entity is not None
        assert entity["props"]["name"] == "Alpha"
        assert set(entity["labels"]) == {"Entity", "ContractEntity"}
        relation = await graph_store.get_relation(relation_id)
        assert relation is not None
        assert (relation["head_id"], relation["tail_id"]) == (first, second)
        assert {r["id"] for r in await graph_store.entity_neighbors(first, 2)} == {
            second,
            third,
        }
        assert len(await graph_store.entity_relations(second)) == 2
        assert [r["id"] for r in await graph_store.entity_chunks(first)] == [
            str(kg.chunks[0].id),
        ]
        results = await bfs_expand(graph_store, [first], depth=2)
        assert [(r.item.name, r.score) for r in results] == [
            ("Beta", 0.5),
            ("Gamma", 1 / 3),
        ]
        filtered = await bfs_expand(
            graph_store, [first], filters=SearchFilters(labels=["ContractEntity"])
        )
        assert {r.item.name for r in filtered} == {"Beta", "Gamma"}
        changed = await graph_store.update_entity(first, {"description": "changed"})
        assert changed is not None and changed["props"]["description"] == "changed"
        changed_relation = await graph_store.update_relation(
            relation_id, {"score": 0.4}
        )
        assert changed_relation is not None
        assert changed_relation["props"]["score"] == 0.4
        for key in ["id", "label", "labels"]:
            with pytest.raises(ValueError):
                await graph_store.update_entity(first, {key: "invalid"})
            with pytest.raises(ValueError):
                await graph_store.update_relation(relation_id, {key: "invalid"})
        await graph_store.delete_relation(relation_id)
        assert await graph_store.get_relation(relation_id) is None
        assert await graph_store.get_entity(first) is not None
        await graph_store.delete_entity(second)
        assert await graph_store.get_entity(second) is None
        assert await graph_store.list_relations() == []
        await graph_store.wipe()
        assert await graph_store.list_entities() == []
        await graph_store.write_knowledge_graph(kg)
        assert len(await graph_store.list_entities()) == 3

    async def test_server_rejects_writes_and_invalid_queries(
        self, graph_store: GraphStore
    ) -> None:
        """Read-only enforcement and EXPLAIN must be real server checks."""
        assert await graph_store.health_check()
        await graph_store.explain("RETURN 1 AS value")
        with pytest.raises(GraphQueryRejected):
            await graph_store.explain("MATCH (n RETURN n")
        with pytest.raises(GraphWriteRejected):
            await graph_store.execute_query("CREATE (:Forbidden)", read_only=True)
        assert await graph_store.execute_query(
            "MATCH (n:Forbidden) RETURN count(n) AS count", read_only=True
        ) == [{"count": 0}]
        await graph_store.wipe()
        await graph_store.wipe()
