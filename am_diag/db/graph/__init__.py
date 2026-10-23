"""Neo4j graph database client for graph population and querying."""

from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.factory import create_graph_store
from am_diag.db.graph.neo4j import (
    Neo4jClient,
    Neo4jTransaction,
    create_neo4j_client,
)
from am_diag.db.graph.serialize import (
    chunk_record,
    community_record,
    document_record,
    entity_record,
    has_entity_records,
    has_report_records,
    in_community_records,
    parent_community_records,
    relation_record,
    report_record,
    to_params,
)
from am_diag.db.graph.settings import FalkorDBSettings, Neo4jSettings


__all__ = [
    "GraphStore",
    "FalkorDBSettings",
    "create_graph_store",
    "Neo4jClient",
    "Neo4jTransaction",
    "Neo4jSettings",
    "create_neo4j_client",
    "to_params",
    "document_record",
    "chunk_record",
    "entity_record",
    "relation_record",
    "community_record",
    "report_record",
    "has_entity_records",
    "in_community_records",
    "parent_community_records",
    "has_report_records",
]
