"""Tests for am_diag/db/graph/base.py — GraphStore abstract interface.

These tests verify serialize.py helpers that replace the old embedded-cypher
record helpers, and confirm that Neo4jClient's granular methods call
execute_batch with the correct record shapes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from am_diag.common.cypher.loader import render
from am_diag.common.data_models import Entity, Relation
from am_diag.common.data_models.provenance import Provenance
from am_diag.db.graph.neo4j import Neo4jClient
from am_diag.db.graph.serialize import entity_record, relation_record
from am_diag.db.graph.settings import Neo4jSettings


def _make_entity(name: str, label: str) -> Entity:
    return Entity(
        name=name,
        label=label,
        provenance=Provenance(),
    )


def _make_relation(rel_type: str) -> Relation:
    return Relation(
        head_name="a",
        tail_name="b",
        type=rel_type,
        provenance=Provenance(),
    )


class TestEntityRecord:
    """Tests for entity_record helper."""

    def test_has_required_keys(self):
        """Verify entity_record output contains id, type_label, and props keys."""
        ent = _make_entity("metformin", "Drug")
        rec = entity_record(ent)
        assert "id" in rec
        assert "type_label" in rec
        assert "props" in rec

    def test_type_label_matches_label(self):
        """Verify type_label in record matches the entity label."""
        ent = _make_entity("aspirin", "Drug")
        rec = entity_record(ent)
        assert rec["type_label"] == "Drug"

    def test_id_is_string(self):
        """Verify entity id in record is a valid UUID string."""
        ent = _make_entity("x", "Drug")
        rec = entity_record(ent)
        UUID(rec["id"])


class TestRelationRecord:
    """Tests for relation_record helper."""

    def test_has_required_keys(self):
        """Verify relation_record output contains id, head_id, tail_id, and props."""
        rel = _make_relation("TREATED_BY")
        rec = relation_record(rel)
        assert "id" in rec
        assert "head_id" in rec
        assert "tail_id" in rec
        assert "props" in rec

    def test_unresolved_ids_are_none(self):
        """Verify unresolved relation endpoints produce None in the record."""
        rel = _make_relation("X")
        rec = relation_record(rel)
        assert rec["head_id"] is None
        assert rec["tail_id"] is None

    def test_resolved_ids_are_strings(self):
        """Verify resolved relation endpoints produce UUID strings in the record."""
        h, t = uuid4(), uuid4()
        rel = Relation(
            head_name="a",
            tail_name="b",
            type="X",
            head_id=h,
            tail_id=t,
            provenance=Provenance(),
        )
        rec = relation_record(rel)
        assert rec["head_id"] == str(h)
        assert rec["tail_id"] == str(t)


class TestUpsertEntitiesGroupsByLabel:
    """Verify Neo4jClient.upsert_entities groups by label and renders per-group."""

    _PATCH_DRIVER = "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver"

    @pytest.fixture
    def client(self):
        """Return a (Neo4jClient, mock_driver) tuple for testing."""
        mock_driver = MagicMock()
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query = AsyncMock(return_value=mock_result)
        mock_driver.close = AsyncMock()

        settings = Neo4jSettings(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="pass",
        )
        with patch(self._PATCH_DRIVER, return_value=mock_driver):
            return Neo4jClient(settings), mock_driver

    async def test_two_labels_yield_two_distinct_cypher_calls(self, client):
        """Verify entities with two labels produce two distinct Cypher queries."""
        client_obj, mock_driver = client

        entities = [
            _make_entity("metformin", "Drug"),
            _make_entity("diabetes", "Disease"),
        ]

        await client_obj.upsert_entities(entities)

        cyphers = [call.args[0] for call in mock_driver.execute_query.call_args_list]
        drug_cypher = render("ingest/upsert_entities", Label="Drug")
        disease_cypher = render("ingest/upsert_entities", Label="Disease")
        assert drug_cypher in cyphers
        assert disease_cypher in cyphers

    async def test_empty_entities_makes_no_calls(self, client):
        """Verify upserting an empty entity list makes no Cypher calls."""
        client_obj, mock_driver = client
        await client_obj.upsert_entities([])
        mock_driver.execute_query.assert_not_called()

    @pytest.mark.parametrize("empty_id", [None, ""])
    async def test_entities_without_ids_are_skipped(self, client, empty_id):
        """Unvalidated records cannot produce null-keyed MERGE operations."""
        client_obj, mock_driver = client
        entity = _make_entity("aspirin", "Drug").model_copy(update={"id": empty_id})

        await client_obj.upsert_entities([entity])

        mock_driver.execute_query.assert_not_called()


class TestUpsertRelationsSkipsUnresolved:
    """Tests for Neo4jClient.upsert_relations — unresolved endpoints are skipped."""

    _PATCH_DRIVER = "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver"

    @pytest.fixture
    def client(self):
        """Return a (Neo4jClient, mock_driver) tuple for testing."""
        mock_driver = MagicMock()
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query = AsyncMock(return_value=mock_result)
        mock_driver.close = AsyncMock()

        settings = Neo4jSettings(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="pass",
        )
        with patch(self._PATCH_DRIVER, return_value=mock_driver):
            return Neo4jClient(settings), mock_driver

    async def test_unresolved_relations_skipped(self, client):
        """Verify relations without resolved head/tail IDs are not written."""
        client_obj, mock_driver = client
        rel = _make_relation("TREATS")  # head_id and tail_id are None
        await client_obj.upsert_relations([rel])
        mock_driver.execute_query.assert_not_called()

    async def test_resolved_relations_written(self, client):
        """Verify relations with resolved head/tail IDs are written to the database."""
        client_obj, mock_driver = client
        h, t = uuid4(), uuid4()
        rel = Relation(
            head_name="a",
            tail_name="b",
            type="TREATS",
            head_id=h,
            tail_id=t,
            provenance=Provenance(),
        )
        await client_obj.upsert_relations([rel])
        mock_driver.execute_query.assert_called_once()
