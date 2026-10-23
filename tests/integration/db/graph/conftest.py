"""Backend contracts use disposable graphs, never a configured production database.

NEO4J_TEST_URI must point to an empty, disposable Neo4j instance. Its contents
are deleted. FalkorDB tests create and delete a unique named graph per test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import EntityType, GraphSchema
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.factory import create_graph_store
from am_diag.db.graph.settings import FalkorDBSettings, Neo4jSettings
from tests._env import get_config, is_configured


@pytest.fixture(params=["neo4j", "falkordb"])
async def graph_store(request: pytest.FixtureRequest) -> AsyncIterator[GraphStore]:
    """Open an isolated graph and remove all test data on exit."""
    if request.param == "neo4j":
        if not is_configured("NEO4J_TEST_URI"):
            pytest.skip("NEO4J_TEST_URI must name a disposable Neo4j instance")
        settings = Neo4jSettings(
            uri=get_config("NEO4J_TEST_URI") or "",
            user=get_config("NEO4J_USER") or "neo4j",
            password=get_config("NEO4J_PASSWORD") or "password",
            database=get_config("NEO4J_TEST_DATABASE") or "neo4j",
        )
    else:
        if not is_configured("FALKORDB_HOST"):
            pytest.skip("FALKORDB_HOST is not configured")
        settings = FalkorDBSettings(graph=f"am_diag_contract_{uuid4().hex}")
    store = create_graph_store(settings)
    try:
        await store.wipe()
        yield store
    finally:
        await store.wipe()
        await store.close()


@pytest.fixture
def knowledge_graph() -> KnowledgeGraph:
    """Provide a directed chain, two linked chunks, and a one-chunk document."""
    doc = Document(text="Alpha Beta Gamma", source="contract", external_id="one")
    single = Document(text="Delta", source="contract", external_id="two")
    chunks = [
        Chunk(
            text=text,
            document_id=document.id,
            document_source="contract",
            chunk_index=index,
            chunk_size=len(text),
            cut_type="sentence",
        )
        for document, index, text in [
            (doc, 0, "Alpha Beta"),
            (doc, 1, "Gamma"),
            (single, 0, "Delta"),
        ]
    ]
    entities = [
        Entity(
            name=name,
            label="ContractEntity",
            provenance=Provenance(chunk_ids=[chunks[index].id]),
        )
        for name, index in [("Alpha", 0), ("Beta", 0), ("Gamma", 1)]
    ]
    relations = [
        Relation(head_id=head.id, tail_id=tail.id, type="LINK", score=score)
        for head, tail, score in [
            (entities[0], entities[1], 0.7),
            (entities[1], entities[2], 0.9),
        ]
    ]
    parent = Community(
        community_id="contract-parent", level=1, entity_ids=[e.id for e in entities]
    )
    child = Community(
        community_id="contract-child",
        parent=parent.id,
        entity_ids=[e.id for e in entities[:2]],
    )
    report = CommunityReport(
        community_id=child.id, title="Contract report", summary="Alpha and Beta"
    )
    return KnowledgeGraph(
        documents=[doc, single],
        chunks=chunks,
        entities=entities,
        relations=relations,
        communities=[parent, child],
        reports=[report],
        graph_schema=GraphSchema(
            entity_types=[
                EntityType(
                    label="ContractEntity",
                    gliner_label="entity",
                    description="Test type",
                )
            ]
        ),
    )
