"""Shared Cypher graph operations for database transport adapters.

Loads all Cypher from ``am_diag.common.cypher.loader`` (``load`` for static
templates, ``render`` for templates with dynamic identifier placeholders such
as entity type labels). No Cypher string literals appear in this file.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from am_diag.common.cypher.loader import load, render
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.errors import OntologyVersionMismatch
from am_diag.db.graph.serialize import (
    chunk_record,
    community_record,
    document_record,
    entity_record,
    has_entity_records,
    has_report_records,
    in_community_records,
    ontology_entity_records,
    ontology_pair_records,
    parent_community_records,
    relation_record,
    report_record,
)


MAX_NEIGHBOR_DEPTH = 3
_NEIGHBOR_LIMIT = 1000
_RESERVED_PROPERTY_KEYS = frozenset({"id", "label", "labels"})


def _reject_reserved_keys(props: dict[str, Any]) -> None:
    """Raise ValueError if a patch touches structural properties."""
    reserved = _RESERVED_PROPERTY_KEYS & props.keys()
    if reserved:
        raise ValueError(f"Cannot update reserved properties: {sorted(reserved)}")


class CypherGraphStore(GraphStore):
    """Shared ingest and administration operations over a Cypher transport."""

    def __init__(self) -> None:
        """Create the lock that serializes ontology registration."""
        self._ontology_lock = asyncio.Lock()

    async def execute_batch(
        self,
        query: str,
        records: Sequence[Mapping[str, Any]],
        batch_size: int = 500,
    ) -> None:
        """Execute ``query`` for each ``batch_size`` slice of ``records``.

        Each slice runs as one auto-commit call, atomic on its own. Batches are
        independent; re-run safety comes from the query's idempotent merge, not
        cross-batch atomicity.

        Args:
            query: Cypher with an unwind clause and a ``$records`` parameter.
            records: Row data to bind to ``$records``.
            batch_size: Maximum records per driver call.
        """
        if not records:
            return
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            await self.execute_query(query, {"records": batch})

    async def upsert_documents(self, docs: list[Document]) -> None:
        """Insert or update Document nodes via upsert_documents.cypher.

        Args:
            docs: Documents to write; empty list is a no-op.
        """
        if not docs:
            return
        records = [document_record(d) for d in docs if d.id]
        await self.execute_batch(load("ingest/upsert_documents"), records)

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or update Chunk nodes via upsert_chunks.cypher.

        Args:
            chunks: Chunks to write; empty list is a no-op.
        """
        if not chunks:
            return
        records = [chunk_record(c) for c in chunks if c.id]
        await self.execute_batch(load("ingest/upsert_chunks"), records)

    async def upsert_entities(self, entities: list[Entity]) -> None:
        """Group entities by label and upsert each group via a rendered template.

        Each entity label requires its own rendered Cypher because the secondary
        Neo4j label (e.g. ``:Drug``) is a Cypher identifier, not a parameter.
        Grouping before rendering avoids N round-trips to the loader cache.

        Args:
            entities: Entities to write; empty list is a no-op.
        """
        if not entities:
            return
        by_label: dict[str, list[dict[str, Any]]] = {}
        for e in entities:
            if not e.id:
                continue
            by_label.setdefault(e.label, []).append(entity_record(e))
        for label, records in by_label.items():
            cypher = render("ingest/upsert_entities", Label=label)
            await self.execute_batch(cypher, records)

    async def upsert_relations(self, relations: list[Relation]) -> None:
        """Insert or update Relation edges for resolved head/tail pairs.

        Relations without ``head_id`` or ``tail_id`` are skipped because the
        template's node lookup would find no nodes and produce no edges.

        Args:
            relations: Relations to write; unresolved ones are silently skipped.
        """
        resolved = [r for r in relations if r.id and r.head_id and r.tail_id]
        if not resolved:
            return
        records = [relation_record(r) for r in resolved]
        await self.execute_batch(load("ingest/upsert_relations"), records)

    async def link_part_of(self, chunks: list[Chunk]) -> None:
        """Create PART_OF edges from each Chunk to its parent Document.

        Args:
            chunks: Chunks to link; empty list is a no-op.
        """
        if not chunks:
            return
        records = [
            {"id": str(c.id), "document_id": str(c.document_id)}
            for c in chunks
            if c.id and c.document_id
        ]
        await self.execute_batch(load("ingest/link_part_of"), records)

    async def link_next_chunk(self, document_ids: list[str]) -> None:
        """Create NEXT_CHUNK edges between consecutive chunks in each document.

        One query per document is issued; the Cypher template orders by
        chunk_index and merges edges between adjacent nodes.

        Args:
            document_ids: Document UUID strings to sequence chunks for.
        """
        for doc_id in document_ids:
            if not doc_id:
                continue
            await self.execute_query(
                load("ingest/link_next_chunk"),
                {"document_id": doc_id},
            )

    async def link_has_entity(self, entities: list[Entity]) -> None:
        """Create HAS_ENTITY edges from Chunk nodes to Entity nodes.

        Args:
            entities: Entities with populated ``provenance.chunk_ids``.
        """
        records = has_entity_records(entities)
        if records:
            await self.execute_batch(load("ingest/link_has_entity"), records)

    async def upsert_communities(self, communities: list[Community]) -> None:
        """Insert or update Community nodes via upsert_communities.cypher.

        Args:
            communities: Communities to write; empty list is a no-op.
        """
        if not communities:
            return
        records = [community_record(c) for c in communities if c.id]
        await self.execute_batch(load("community/upsert_communities"), records)

    async def link_in_community(self, communities: list[Community]) -> None:
        """Create IN_COMMUNITY edges from Entity nodes to Community nodes.

        Args:
            communities: Communities whose ``entity_ids`` should be linked.
        """
        records = in_community_records(communities)
        if records:
            await self.execute_batch(load("community/link_in_community"), records)

    async def link_parent_community(self, communities: list[Community]) -> None:
        """Create PARENT_COMMUNITY edges between nested Community nodes.

        Communities with ``parent is None`` produce no records and the call
        becomes a no-op.

        Args:
            communities: Communities to inspect for parent links.
        """
        records = parent_community_records(communities)
        if records:
            await self.execute_batch(load("community/link_parent_community"), records)

    async def upsert_reports(self, reports: list[CommunityReport]) -> None:
        """Insert or update CommunityReport nodes via upsert_reports.cypher.

        Args:
            reports: Reports to write; empty list is a no-op.
        """
        if not reports:
            return
        records = [report_record(r) for r in reports if r.id]
        await self.execute_batch(load("community/upsert_reports"), records)

    async def link_has_report(self, reports: list[CommunityReport]) -> None:
        """Create HAS_REPORT edges from Community nodes to CommunityReport nodes.

        Args:
            reports: Reports whose ``community_id`` links should be materialized.
        """
        records = has_report_records(reports)
        if records:
            await self.execute_batch(load("community/link_has_report"), records)

    async def write_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """Orchestrate a full KnowledgeGraph write in dependency order.

        The ordering guarantee — constraints → nodes → relationships — ensures
        that upsert statements on relationships always find their endpoint nodes
        and that uniqueness constraints are in place before any node is written.

        Args:
            kg: The assembled KnowledgeGraph to persist.
        """
        if kg.graph_schema:
            await self.setup_schema(kg.graph_schema)
        await self.upsert_documents(kg.documents)
        await self.upsert_chunks(kg.chunks)
        await self.upsert_entities(kg.entities)
        await self.upsert_communities(kg.communities)
        await self.upsert_reports(kg.reports)
        # Relationships after all nodes so node lookups always succeed.
        await self.link_part_of(kg.chunks)
        doc_ids = list({str(c.document_id) for c in kg.chunks})
        await self.link_next_chunk(doc_ids)
        await self.link_has_entity(kg.entities)
        await self.upsert_relations(kg.relations)
        await self.link_in_community(kg.communities)
        await self.link_parent_community(kg.communities)
        await self.link_has_report(kg.reports)

    async def load_ontology(self) -> GraphSchema | None:
        """Return the ontology mirror stored in the graph, or ``None``."""
        rows = await self.execute_query(
            load("ontology/load_ontology"), {}, read_only=True
        )
        if not rows:
            return None
        return GraphSchema.model_validate_json(rows[0]["schema_json"])

    async def register_ontology(self, schema: GraphSchema) -> None:
        """Store ``schema`` as the graph's ontology, or confirm it is already stored.

        A definition equal to the stored one is accepted. If only its ``mode``
        differs, the stored mode is updated. Registrations on one store instance
        run one at a time; concurrent registrations from separate processes are
        not coordinated.

        Args:
            schema: Ontology to mirror into the graph.

        Raises:
            OntologyVersionMismatch: If the graph holds a different ontology.
        """
        async with self._ontology_lock:
            stored = await self.load_ontology()
            if stored is not None:
                stored_fingerprint = stored.fingerprint(include_mode=False)
                if stored_fingerprint != schema.fingerprint(include_mode=False):
                    raise OntologyVersionMismatch(stored.version, schema.version)
                if stored.mode != schema.mode:
                    await self._write_ontology_meta(schema)
                return
            # Without a meta node, any entity or pair records come from an
            # interrupted registration and may describe another schema.
            await self.execute_query(load("ontology/clear_mirror"), {})
            await self.execute_batch(
                load("ontology/upsert_entity_types"), ontology_entity_records(schema)
            )
            await self.execute_batch(
                load("ontology/upsert_pairs"), ontology_pair_records(schema)
            )
            # FalkorDB has no client transactions. Writing the meta node last means
            # a partial registration leaves nothing to match on load, so the next
            # call clears it and starts again.
            await self._write_ontology_meta(schema)

    async def _write_ontology_meta(self, schema: GraphSchema) -> None:
        """Write the version, mode and full definition to the ontology meta node."""
        await self.execute_query(
            load("ontology/upsert_meta"),
            {
                "version": schema.version,
                "mode": schema.mode,
                "schema_json": schema.model_dump_json(),
            },
        )

    async def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Return one entity row ``{id, labels, props}`` or ``None`` if absent."""
        rows = await self.execute_query(
            load("admin/get_entity"), {"id": entity_id}, read_only=True
        )
        return rows[0] if rows else None

    async def list_entities(
        self,
        *,
        cursor: str | None = None,
        limit: int = 100,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a keyset page of entity rows ordered by ascending id."""
        return await self.execute_query(
            load("admin/list_entities"),
            {"cursor": cursor, "limit": limit, "label": label},
            read_only=True,
        )

    async def entity_neighbors(
        self, entity_id: str, depth: int = 1, *, limit: int = _NEIGHBOR_LIMIT
    ) -> list[dict[str, Any]]:
        """Return entity rows within ``depth`` RELATES_TO hops of ``entity_id``.

        ``depth`` is clamped to ``1..MAX_NEIGHBOR_DEPTH`` and at most ``limit``
        rows are returned.
        """
        bounded_depth = max(1, min(int(depth), MAX_NEIGHBOR_DEPTH))
        # Variable-length bounds cannot be Cypher parameters, so the clamped
        # integer is substituted into the template.
        query = load("admin/entity_neighbors").replace("${depth}", str(bounded_depth))
        return await self.execute_query(
            query, {"id": entity_id, "limit": limit}, read_only=True
        )

    async def entity_relations(self, entity_id: str) -> list[dict[str, Any]]:
        """Return relation rows incident to ``entity_id`` (either direction)."""
        return await self.execute_query(
            load("admin/entity_relations"), {"id": entity_id}, read_only=True
        )

    async def entity_chunks(self, entity_id: str) -> list[dict[str, Any]]:
        """Return chunk rows that mention ``entity_id`` via HAS_ENTITY."""
        return await self.execute_query(
            load("admin/entity_chunks"), {"id": entity_id}, read_only=True
        )

    async def update_entity(
        self, entity_id: str, props: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Patch an entity's semantic properties and return the updated row."""
        _reject_reserved_keys(props)
        rows = await self.execute_query(
            load("admin/update_entity"), {"id": entity_id, "props": props}
        )
        return rows[0] if rows else None

    async def delete_entity(self, entity_id: str) -> None:
        """Delete an entity and detach all incident relations (cascade)."""
        await self.execute_query(load("admin/delete_entity"), {"id": entity_id})

    async def get_relation(self, relation_id: str) -> dict[str, Any] | None:
        """Return one relation row or ``None`` if absent."""
        rows = await self.execute_query(
            load("admin/get_relation"), {"id": relation_id}, read_only=True
        )
        return rows[0] if rows else None

    async def list_relations(
        self,
        *,
        cursor: str | None = None,
        limit: int = 100,
        relation_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a keyset page of relation rows ordered by ascending id."""
        return await self.execute_query(
            load("admin/list_relations"),
            {"cursor": cursor, "limit": limit, "type": relation_type},
            read_only=True,
        )

    async def update_relation(
        self, relation_id: str, props: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Patch a relation's semantic properties and return the updated row."""
        _reject_reserved_keys(props)
        rows = await self.execute_query(
            load("admin/update_relation"), {"id": relation_id, "props": props}
        )
        return rows[0] if rows else None

    async def delete_relation(self, relation_id: str) -> None:
        """Delete a single relation edge (leaf; endpoints are untouched)."""
        await self.execute_query(load("admin/delete_relation"), {"id": relation_id})

    async def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Return one chunk row or ``None`` if absent."""
        rows = await self.execute_query(
            load("admin/get_chunk"), {"id": chunk_id}, read_only=True
        )
        return rows[0] if rows else None

    async def list_chunks(
        self, *, cursor: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return a keyset page of chunk rows ordered by ascending id."""
        return await self.execute_query(
            load("admin/list_chunks"),
            {"cursor": cursor, "limit": limit},
            read_only=True,
        )

    async def get_community(self, community_id: str) -> dict[str, Any] | None:
        """Return one community row or ``None`` if absent."""
        rows = await self.execute_query(
            load("admin/get_community"), {"id": community_id}, read_only=True
        )
        return rows[0] if rows else None

    async def list_communities(
        self, *, cursor: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return a keyset page of community rows ordered by ascending id."""
        return await self.execute_query(
            load("admin/list_communities"),
            {"cursor": cursor, "limit": limit},
            read_only=True,
        )
