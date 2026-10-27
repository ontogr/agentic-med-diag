"""Checks every GraphSchema constant exported by ``am_diag.common.schema``.

The schemas are discovered from ``__all__``, so a schema added to the package is
covered without editing this module. Each built-in schema must survive a dump and
reload with the same fingerprint, and its relation endpoints must name entity types
that the schema declares.
"""

from __future__ import annotations

import re

import pytest

import am_diag.common.schema as builtin_schemas
from am_diag.common.data_models.schema import GraphSchema


def _exported_graph_schemas() -> dict[str, GraphSchema]:
    """Return the GraphSchema constants exported by ``am_diag.common.schema``."""
    exported = {
        name: getattr(builtin_schemas, name) for name in builtin_schemas.__all__
    }
    return {
        name: schema
        for name, schema in exported.items()
        if isinstance(schema, GraphSchema)
    }


_BUILTIN_SCHEMAS = _exported_graph_schemas()


class TestBuiltinGraphSchemas:
    """Round-trip, fingerprint, and endpoint checks for the built-in schemas."""

    def test_exports_the_medical_and_optimuskg_schemas(self) -> None:
        """The exported constants include the medical and OptimusKG schemas."""
        assert {"MEDICAL_GRAPHRAG_SCHEMA", "OPTIMUSKG_SCHEMA"} <= set(_BUILTIN_SCHEMAS)

    @pytest.mark.parametrize("name", list(_BUILTIN_SCHEMAS))
    def test_survives_dump_and_reload_with_same_fingerprint(self, name: str) -> None:
        """A stored schema record reloads into a schema with the same fingerprint."""
        schema = _BUILTIN_SCHEMAS[name]

        restored = GraphSchema.model_validate(schema.model_dump())

        assert restored.fingerprint() == schema.fingerprint()

    @pytest.mark.parametrize("name", list(_BUILTIN_SCHEMAS))
    def test_fingerprint_is_sixteen_hex_characters(self, name: str) -> None:
        """The fingerprint is a 16-character lowercase hexadecimal string."""
        assert re.fullmatch(r"[0-9a-f]{16}", _BUILTIN_SCHEMAS[name].fingerprint())

    @pytest.mark.parametrize("name", list(_BUILTIN_SCHEMAS))
    def test_relation_endpoints_name_declared_entity_types(self, name: str) -> None:
        """Every relation head and tail is an entity type the schema declares."""
        schema = _BUILTIN_SCHEMAS[name]
        declared = {et.label for et in schema.entity_types}
        endpoints = {
            label
            for rt in schema.relation_types
            for label in (*rt.head_labels, *rt.tail_labels)
        }

        assert endpoints <= declared
