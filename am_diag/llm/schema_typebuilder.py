"""BAML TypeBuilder builder from GraphSchema.

Constructs a BAML TypeBuilder with dynamic EntityType and RelationType
enums populated from a GraphSchema instance. This enables BAML to constrain
extraction to schema-valid types at call time. Relation endpoint pairs are also
passed to the extraction prompt.

Usage::

    tb = build_schema_typebuilder(schema)
    guide = schema.relation_guide()
    result = await b.ExtractKnowledgeGraph(text, source, guide, baml_options={"tb": tb})
"""

from __future__ import annotations

from am_diag.common.data_models.schema import GraphSchema
from am_diag.llm.baml_client.type_builder import TypeBuilder


def build_schema_typebuilder(schema: GraphSchema) -> TypeBuilder:
    """Build a BAML TypeBuilder with dynamic enums from a GraphSchema.

    The returned TypeBuilder should be passed to BAML functions that use
    @@dynamic enums (EntityType, RelationType) via ``baml_options={"tb": tb}``.

    Only types that text extractors may produce are offered.

    Entity values include their descriptions. Relation values include their
    descriptions when available.

    Args:
        schema: The graph schema defining valid entity and relation types.

    Returns:
        A TypeBuilder with EntityType and RelationType enums populated.
    """
    tb = TypeBuilder()
    for et in schema.extractable_entity_types():
        tb.EntityType.add_value(et.label).description(et.description)
    extractable_relations = {
        relation.type: relation for relation in schema.extractable_relation_types()
    }

    for group in schema.relation_groups():
        relation = extractable_relations.get(group.type)
        if relation is None:
            continue
        description = group.description or relation.glirel_label
        if description:
            tb.RelationType.add_value(group.type).description(description)
        else:
            tb.RelationType.add_value(group.type)
    return tb
