"""BAML TypeBuilder builder from GraphSchema.

Constructs a BAML TypeBuilder with dynamic EntityType and RelationType
enums populated from a GraphSchema instance. This enables BAML to constrain
extraction to schema-valid types at call time rather than via prompt text.

Usage::

    tb = build_schema_typebuilder(schema)
    result = await b.ExtractKnowledgeGraph(text, source, baml_options={"tb": tb})
"""

from __future__ import annotations

from am_diag.common.data_models.schema import GraphSchema
from am_diag.llm.baml_client.type_builder import TypeBuilder


def build_schema_typebuilder(schema: GraphSchema) -> TypeBuilder:
    """Build a BAML TypeBuilder with dynamic enums from a GraphSchema.

    The returned TypeBuilder should be passed to BAML functions that use
    @@dynamic enums (EntityType, RelationType) via ``baml_options={"tb": tb}``.

    Only types that text extractors may produce are offered.

    Each EntityType value is annotated with the entity type description so the
    LLM sees human-readable labels in the output schema, not just identifiers.
    RelationType values are annotated with the GLiREL label for the same reason.

    Args:
        schema: The graph schema defining valid entity and relation types.

    Returns:
        A TypeBuilder with EntityType and RelationType enums populated.
    """
    tb = TypeBuilder()
    for et in schema.extractable_entity_types():
        tb.EntityType.add_value(et.label).description(et.description)
    for rt in schema.extractable_relation_types():
        tb.RelationType.add_value(rt.type).description(rt.glirel_label)
    return tb
