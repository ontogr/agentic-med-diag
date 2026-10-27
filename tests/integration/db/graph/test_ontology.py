"""Ontology mirror behavior on real Neo4j and FalkorDB graphs.

The mirror uses reserved ``__Onto`` labels. These tests check that registration
round-trips, is idempotent, rejects a different version, and never leaks into
entity, relation or community reads.
"""

from __future__ import annotations

import pytest

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.errors import OntologyVersionMismatch


async def _entity_count(graph_store: GraphStore) -> int:
    """Return the number of nodes carrying the ``Entity`` label."""
    rows = await graph_store.execute_query(
        "MATCH (n:Entity) RETURN count(n) AS c", read_only=True
    )
    return rows[0]["c"]


class TestOntologyMirror:
    """Round-trip and isolation of the ontology mirror on a live backend."""

    async def test_register_then_load_returns_the_same_definition(
        self, graph_store: GraphStore, knowledge_graph: KnowledgeGraph
    ) -> None:
        """Load the registered definition with an equal fingerprint."""
        schema = knowledge_graph.graph_schema
        assert schema is not None
        await graph_store.register_ontology(schema)
        loaded = await graph_store.load_ontology()
        assert loaded is not None
        assert loaded.fingerprint(include_mode=False) == schema.fingerprint(
            include_mode=False
        )

    async def test_registering_the_same_schema_again_is_a_no_op(
        self, graph_store: GraphStore, knowledge_graph: KnowledgeGraph
    ) -> None:
        """Accept a repeated registration of an equal definition."""
        schema = knowledge_graph.graph_schema
        assert schema is not None
        await graph_store.register_ontology(schema)
        await graph_store.register_ontology(schema)
        loaded = await graph_store.load_ontology()
        assert loaded is not None
        assert loaded.fingerprint(include_mode=False) == schema.fingerprint(
            include_mode=False
        )

    async def test_registering_a_different_version_raises(
        self, graph_store: GraphStore, knowledge_graph: KnowledgeGraph
    ) -> None:
        """Refuse a second ontology version on the same graph."""
        schema = knowledge_graph.graph_schema
        assert schema is not None
        await graph_store.register_ontology(schema)
        newer = schema.model_copy(update={"version": "2.0"})
        with pytest.raises(OntologyVersionMismatch) as excinfo:
            await graph_store.register_ontology(newer)
        assert str(excinfo.value) == (
            "graph holds ontology 'unversioned', requested '2.0'; "
            "build a new graph to change ontology"
        )

    async def test_mirror_is_invisible_to_entity_and_edge_reads(
        self, graph_store: GraphStore, knowledge_graph: KnowledgeGraph
    ) -> None:
        """Keep reserved ontology labels out of entity, relation and community reads."""
        schema = knowledge_graph.graph_schema
        assert schema is not None
        await graph_store.write_knowledge_graph(knowledge_graph)
        entities_before = await _entity_count(graph_store)
        assert entities_before == len(knowledge_graph.entities)

        await graph_store.register_ontology(schema)

        assert await _entity_count(graph_store) == entities_before
        mirrored = await graph_store.execute_query(
            "MATCH (n:Entity) WHERE any(l IN labels(n) WHERE l STARTS WITH '__Onto') "
            "RETURN n",
            read_only=True,
        )
        assert mirrored == []
        listed = await graph_store.list_entities(limit=100)
        assert all(
            not label.startswith("__Onto") for row in listed for label in row["labels"]
        )
        edges = await graph_store.execute_query(
            load("community/all_entity_edges"),
            {"offset": 0, "limit": 100},
            read_only=True,
        )
        assert len(edges) == len(knowledge_graph.relations)
        entity_ids = {str(entity.id) for entity in knowledge_graph.entities}
        endpoint_ids = {row["source"] for row in edges} | {
            row["target"] for row in edges
        }
        assert endpoint_ids <= entity_ids
