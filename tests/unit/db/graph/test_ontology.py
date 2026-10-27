"""Ontology mirror registration on the shared Cypher store, without a server.

A recording transport replaces the database so each test can assert which
writes run and in what order. Schemas built by separate calls share a
definition but not entity ids, so equality checks compare fingerprints.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import pytest

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.schema import (
    EntityType,
    GraphSchema,
    OntologyMode,
    RelationType,
)
from am_diag.db.graph.cypher_store import CypherGraphStore
from am_diag.db.graph.errors import OntologyVersionMismatch


def _schema(
    *,
    version: str = "1.0",
    mode: OntologyMode = "open",
    drug_description: str = "A compound given to treat a condition.",
    treats_extractable: bool = True,
) -> GraphSchema:
    """Build a Drug and Disease ontology with one TREATS relation."""
    return GraphSchema(
        entity_types=[
            EntityType(label="Drug", gliner_label="drug", description=drug_description),
            EntityType(
                label="Disease",
                gliner_label="disease",
                description="A condition that affects a patient.",
            ),
        ],
        relation_types=[
            RelationType(
                type="TREATS",
                glirel_label="treats",
                head_labels=["Drug"],
                tail_labels=["Disease"],
                extractable=treats_extractable,
                description="A drug is used to treat a disease.",
            )
        ],
        version=version,
        mode=mode,
    )


def _stored_row(schema: GraphSchema) -> dict[str, Any]:
    """Return the row the ontology load query yields for a registered schema."""
    return {"version": schema.version, "schema_json": schema.model_dump_json()}


class _RecordingGraphStore(CypherGraphStore):
    """Cypher store that records each query and keeps one stored mirror row.

    A meta write replaces the stored row, so a later load sees it. With
    ``interleave`` each query yields to the event loop first, which lets
    concurrent registrations interleave at every database call.
    """

    def __init__(
        self,
        stored_rows: list[dict[str, Any]] | None = None,
        *,
        interleave: bool = False,
    ) -> None:
        """Start with ``stored_rows`` as the mirror, yielding per query if asked."""
        super().__init__()
        self.stored_rows = stored_rows or []
        self.executed: list[tuple[str, dict[str, Any] | None]] = []
        self._interleave = interleave

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        read_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Record the query, then serve or update the stored mirror row."""
        self.executed.append((query, params))
        if self._interleave:
            await asyncio.sleep(0)
        if query == load("ontology/upsert_meta") and params is not None:
            self.stored_rows = [
                {"version": params["version"], "schema_json": params["schema_json"]}
            ]
        if query == load("ontology/load_ontology"):
            return self.stored_rows
        return []

    async def explain(self, cypher: str) -> None:
        """Not used by ontology registration."""
        raise NotImplementedError

    async def wipe(self) -> None:
        """Not used by ontology registration."""
        raise NotImplementedError

    async def close(self) -> None:
        """Not used by ontology registration."""
        raise NotImplementedError

    def transaction(self) -> Any:
        """Not used by ontology registration."""
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Not used by ontology registration."""
        raise NotImplementedError

    async def setup_schema(self, schema: GraphSchema) -> None:
        """Not used by ontology registration."""
        raise NotImplementedError


class TestOntologyRegistration:
    """Register and load the ontology mirror through the Cypher store contract."""

    async def test_load_returns_none_when_no_meta_row_is_stored(self) -> None:
        """Report no ontology for a graph that never registered one."""
        store = _RecordingGraphStore(stored_rows=[])
        assert await store.load_ontology() is None

    async def test_load_rebuilds_the_stored_definition(self) -> None:
        """Rebuild the registered schema from the stored JSON definition."""
        stored = _schema()
        store = _RecordingGraphStore(stored_rows=[_stored_row(stored)])
        loaded = await store.load_ontology()
        assert loaded is not None
        assert loaded.version == "1.0"
        assert loaded.fingerprint() == stored.fingerprint()

    async def test_clears_then_writes_entity_types_then_pairs_then_meta(self) -> None:
        """Write the meta node last so an interrupted registration leaves none."""
        store = _RecordingGraphStore()
        await store.register_ontology(_schema())
        assert [query for query, _ in store.executed] == [
            load("ontology/load_ontology"),
            load("ontology/clear_mirror"),
            load("ontology/upsert_entity_types"),
            load("ontology/upsert_pairs"),
            load("ontology/upsert_meta"),
        ]

    async def test_mirrors_entity_types_and_pairs_as_records(self) -> None:
        """Send one record per entity type and one per allowed endpoint triple."""
        store = _RecordingGraphStore()
        await store.register_ontology(_schema())
        writes = dict(store.executed)
        assert writes[load("ontology/upsert_entity_types")] == {
            "records": [
                {
                    "label": "Drug",
                    "description": "A compound given to treat a condition.",
                    "gliner_label": "drug",
                    "extractable": True,
                },
                {
                    "label": "Disease",
                    "description": "A condition that affects a patient.",
                    "gliner_label": "disease",
                    "extractable": True,
                },
            ]
        }
        assert writes[load("ontology/upsert_pairs")] == {
            "records": [
                {
                    "type": "TREATS",
                    "head": "Drug",
                    "tail": "Disease",
                    "glirel_label": "treats",
                    "extractable": True,
                }
            ]
        }

    async def test_pair_records_skip_undeclared_endpoints_and_keep_first_duplicate(
        self,
    ) -> None:
        """Keep the first declaration of a triple and drop undeclared endpoints."""
        schema = GraphSchema(
            entity_types=[
                EntityType(label="Drug", gliner_label="drug", description="A drug."),
                EntityType(
                    label="Disease", gliner_label="disease", description="A disease."
                ),
            ],
            relation_types=[
                RelationType(
                    type="TREATS",
                    glirel_label="treats",
                    head_labels=["Drug"],
                    tail_labels=["Disease"],
                ),
                RelationType(
                    type="TREATS",
                    glirel_label="cures",
                    head_labels=["Drug"],
                    tail_labels=["Disease"],
                    extractable=False,
                ),
                RelationType(
                    type="TREATS",
                    glirel_label="treats",
                    head_labels=["Drug"],
                    tail_labels=["Symptom"],
                ),
            ],
        )
        store = _RecordingGraphStore()
        await store.register_ontology(schema)
        writes = dict(store.executed)
        assert writes[load("ontology/upsert_pairs")] == {
            "records": [
                {
                    "type": "TREATS",
                    "head": "Drug",
                    "tail": "Disease",
                    "glirel_label": "treats",
                    "extractable": True,
                }
            ]
        }

    async def test_meta_stores_version_mode_and_definition(self) -> None:
        """Record the version and mode on the meta node with the full definition."""
        schema = _schema(mode="warn")
        store = _RecordingGraphStore()
        await store.register_ontology(schema)
        meta = dict(store.executed)[load("ontology/upsert_meta")]
        assert meta["version"] == "1.0"
        assert meta["mode"] == "warn"
        stored = GraphSchema.model_validate_json(meta["schema_json"])
        assert stored.fingerprint() == schema.fingerprint()

    async def test_equal_stored_definition_writes_nothing(self) -> None:
        """Accept a re-registration of an equal definition without any write."""
        store = _RecordingGraphStore(stored_rows=[_stored_row(_schema())])
        await store.register_ontology(_schema())
        assert [query for query, _ in store.executed] == [
            load("ontology/load_ontology"),
        ]

    async def test_different_version_raises_and_writes_nothing(self) -> None:
        """Refuse a second ontology version and leave the stored mirror untouched."""
        store = _RecordingGraphStore(stored_rows=[_stored_row(_schema(version="1.0"))])
        with pytest.raises(OntologyVersionMismatch) as excinfo:
            await store.register_ontology(_schema(version="2.0"))
        assert str(excinfo.value) == (
            "graph holds ontology '1.0', requested '2.0'; "
            "build a new graph to change ontology"
        )
        assert [query for query, _ in store.executed] == [
            load("ontology/load_ontology"),
        ]

    @pytest.mark.parametrize(
        "changed",
        [
            {"drug_description": "A changed description."},
            {"treats_extractable": False},
        ],
        ids=["entity_description", "relation_extractable"],
    )
    async def test_same_version_with_other_definition_raises(
        self, changed: dict[str, Any]
    ) -> None:
        """Refuse a different definition even when the version matches."""
        store = _RecordingGraphStore(stored_rows=[_stored_row(_schema())])
        with pytest.raises(OntologyVersionMismatch) as excinfo:
            await store.register_ontology(_schema(**changed))
        assert str(excinfo.value) == (
            "graph holds a different definition of ontology '1.0'; "
            "build a new graph to change ontology"
        )
        assert [query for query, _ in store.executed] == [
            load("ontology/load_ontology"),
        ]

    async def test_mode_change_alone_rewrites_only_the_meta_node(self) -> None:
        """Accept a new mode and update only the meta node, with no raise."""
        store = _RecordingGraphStore(stored_rows=[_stored_row(_schema(mode="open"))])
        await store.register_ontology(_schema(mode="strict"))
        assert [query for query, _ in store.executed] == [
            load("ontology/load_ontology"),
            load("ontology/upsert_meta"),
        ]
        meta = dict(store.executed)[load("ontology/upsert_meta")]
        assert meta["mode"] == "strict"
        stored = GraphSchema.model_validate_json(meta["schema_json"])
        assert stored.mode == "strict"

    async def test_concurrent_registrations_write_one_schema(self) -> None:
        """Let exactly one of two racing versions write the mirror."""
        store = _RecordingGraphStore(interleave=True)
        results = await asyncio.gather(
            store.register_ontology(_schema(version="1.0")),
            store.register_ontology(_schema(version="2.0")),
            return_exceptions=True,
        )
        assert sum(result is None for result in results) == 1
        assert (
            sum(isinstance(result, OntologyVersionMismatch) for result in results) == 1
        )
        winning_version = ("1.0", "2.0")[results.index(None)]
        loaded = await store.load_ontology()
        assert loaded is not None
        assert loaded.version == winning_version
        writes = Counter(query for query, _ in store.executed)
        assert writes[load("ontology/clear_mirror")] == 1
        assert writes[load("ontology/upsert_entity_types")] == 1
        assert writes[load("ontology/upsert_pairs")] == 1
        assert writes[load("ontology/upsert_meta")] == 1
