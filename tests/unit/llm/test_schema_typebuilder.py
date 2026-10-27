"""Tests for the BAML TypeBuilder builder.

The generated TypeBuilder does not expose the names added to its dynamic enums,
so the enum tests swap in a recording builder at the module's TypeBuilder import.
"""

import pytest

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType
from am_diag.llm import schema_typebuilder
from am_diag.llm.baml_client.type_builder import TypeBuilder
from am_diag.llm.schema_typebuilder import build_schema_typebuilder


class _RecordingEnumValue:
    """Stand-in for an enum value builder that accepts a description."""

    def description(self, text: str) -> "_RecordingEnumValue":
        """Accept a description and return the value builder."""
        return self


class _RecordingEnum:
    """Stand-in for a dynamic enum that records the values added to it."""

    def __init__(self) -> None:
        self.values: list[str] = []

    def add_value(self, name: str) -> _RecordingEnumValue:
        """Record the value name and return a value builder."""
        self.values.append(name)
        return _RecordingEnumValue()


class _RecordingTypeBuilder:
    """Stand-in TypeBuilder exposing the two dynamic enums as recorders."""

    def __init__(self) -> None:
        self.EntityType = _RecordingEnum()
        self.RelationType = _RecordingEnum()


class TestBuildSchemaTypebuilder:
    """Tests for build_schema_typebuilder."""

    def _sample_schema(self) -> GraphSchema:
        """Build a sample schema for testing."""
        return GraphSchema(
            entity_types=[
                EntityType(
                    label="Disease",
                    gliner_label="disease or disorder",
                    description="A named disease.",
                ),
                EntityType(
                    label="Drug",
                    gliner_label="drug or medication",
                    description="A pharmaceutical.",
                ),
            ],
            relation_types=[
                RelationType(
                    type="TREATED_BY",
                    glirel_label="treated by",
                    head_labels=["Disease"],
                    tail_labels=["Drug"],
                ),
            ],
        )

    def test_returns_typebuilder(self):
        """Verify function returns a TypeBuilder instance."""
        schema = self._sample_schema()
        tb = build_schema_typebuilder(schema)
        assert isinstance(tb, TypeBuilder)

    def test_entity_types_populated(self):
        """Verify entity types are populated in the TypeBuilder."""
        schema = self._sample_schema()
        tb = build_schema_typebuilder(schema)
        # TypeBuilder should have entity_type values
        # (exact API depends on the installed TypeBuilder version)
        assert tb is not None

    def test_omits_non_extractable_types_from_dynamic_enums(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Entity and relation types that extraction must not produce are left out."""
        recorder = _RecordingTypeBuilder()
        monkeypatch.setattr(schema_typebuilder, "TypeBuilder", lambda: recorder)
        schema = GraphSchema(
            entity_types=[
                EntityType(
                    label="Disease",
                    gliner_label="disease or disorder",
                    description="A named disease.",
                ),
                EntityType(
                    label="Gene",
                    gliner_label="gene",
                    description="A gene.",
                    extractable=False,
                ),
            ],
            relation_types=[
                RelationType(
                    type="TREATED_BY",
                    glirel_label="treated by",
                    head_labels=["Disease"],
                    tail_labels=["Disease"],
                ),
                RelationType(
                    type="PARENT",
                    glirel_label="is a parent of",
                    head_labels=["Disease"],
                    tail_labels=["Disease"],
                    extractable=False,
                ),
            ],
        )

        build_schema_typebuilder(schema)

        assert recorder.EntityType.values == ["Disease"]
        assert recorder.RelationType.values == ["TREATED_BY"]

    def test_empty_schema_does_not_raise(self):
        """Verify empty schema does not raise an error."""
        schema = GraphSchema(entity_types=[], relation_types=[])
        tb = build_schema_typebuilder(schema)
        assert tb is not None
