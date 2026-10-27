"""Unit tests for LLMEntityRelationExtractor and its helper functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from am_diag.common.data_models import Chunk
from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.extract.llm import (
    LLMExtractor as LLMEntityRelationExtractor,
)
from am_diag.graph_construction.extract.llm import _flatten_kg


def _make_chunk(
    text: str = "Metformin treats T2DM.",
) -> Chunk:
    return Chunk(
        text=text,
        document_id=uuid4(),
        document_source="textbooks",
        chunk_index=0,
        chunk_size=len(text),
        cut_type="sentence",
    )


def _mock_entity(
    *,
    entity_type: str,
    name: str,
    properties: dict | None = None,
) -> object:
    from unittest.mock import MagicMock

    ent = MagicMock()
    # Use snake_case attributes to match _flatten_kg access patterns
    ent.entity_type = entity_type
    ent.name = name
    ent.score = 1.0
    ent.properties = properties or {}
    return ent


def _mock_relation(*, relation_type: str, head_name: str, tail_name: str) -> object:
    from unittest.mock import MagicMock

    rel = MagicMock()
    # Use snake_case attributes to match _flatten_kg access patterns
    rel.relation_type = relation_type
    rel.head_name = head_name
    rel.tail_name = tail_name
    rel.score = 1.0
    rel.properties = {}
    return rel


def _mock_kg(entities=(), relations=()) -> object:
    from unittest.mock import MagicMock

    kg = MagicMock()
    kg.entities = list(entities)
    kg.relations = list(relations)
    return kg


class TestLLMEntityRelationExtractor:
    """Tests for l l m entity relation extractor."""

    def test_flatten_kg_drops_invalid_entity_type(self):
        """Entities whose entityType is absent from the schema are silently dropped."""
        chunk = _make_chunk()
        kg = _mock_kg(
            entities=[_mock_entity(entity_type="InvalidType", name="mystery_drug")],
        )
        entities, relations = _flatten_kg(
            kg,
            str(chunk.id),
            MEDICAL_GRAPHRAG_SCHEMA,
        )
        assert len(entities) == 0

    def test_flatten_kg_drops_invalid_relation_type(self):
        """Relations with unknown relationType dropped, even if endpoints valid."""
        chunk = _make_chunk()
        valid_entity = _mock_entity(entity_type="Drug", name="metformin")
        bad_relation = _mock_relation(
            relation_type="INVALID_REL",
            head_name="metformin",
            tail_name="metformin",
        )
        kg = _mock_kg(entities=[valid_entity], relations=[bad_relation])
        entities, relations = _flatten_kg(
            kg,
            str(chunk.id),
            MEDICAL_GRAPHRAG_SCHEMA,
        )
        assert len(relations) == 0

    def test_flatten_kg_drops_non_extractable_entity_and_relation_types(self):
        """Entities and relations of non-extractable types are dropped."""
        chunk = _make_chunk()
        schema = GraphSchema(
            entity_types=[
                EntityType(label="Drug", gliner_label="drug", description="A drug."),
                EntityType(
                    label="Disease",
                    gliner_label="disease",
                    description="A disease.",
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
                    type="TREATS",
                    glirel_label="treats",
                    head_labels=["Drug"],
                    tail_labels=["Disease"],
                ),
                RelationType(
                    type="PARENT",
                    glirel_label="is a parent of",
                    head_labels=["Drug"],
                    tail_labels=["Drug"],
                    extractable=False,
                ),
            ],
        )
        kg = _mock_kg(
            entities=[
                _mock_entity(entity_type="Drug", name="metformin"),
                _mock_entity(entity_type="Disease", name="T2DM"),
                _mock_entity(entity_type="Gene", name="TCF7L2"),
            ],
            relations=[
                _mock_relation(
                    relation_type="TREATS",
                    head_name="metformin",
                    tail_name="T2DM",
                ),
                _mock_relation(
                    relation_type="PARENT",
                    head_name="metformin",
                    tail_name="metformin",
                ),
            ],
        )
        entities, relations = _flatten_kg(kg, str(chunk.id), schema)
        assert [entity.name for entity in entities] == ["metformin", "T2DM"]
        assert [relation.type for relation in relations] == ["TREATS"]

    def test_flatten_kg_drops_dangling_relation(self):
        """Relations whose headName/tailName not in extracted entities are dropped."""
        chunk = _make_chunk()
        valid_entity = _mock_entity(entity_type="Drug", name="metformin")
        dangling_relation = _mock_relation(
            relation_type="TREATED_BY",
            head_name="metformin",
            tail_name="T2DM",
        )
        kg = _mock_kg(entities=[valid_entity], relations=[dangling_relation])
        entities, relations = _flatten_kg(
            kg,
            str(chunk.id),
            MEDICAL_GRAPHRAG_SCHEMA,
        )
        assert len(relations) == 0

    def test_flatten_kg_properties_to_schema_properties(self):
        """Entity properties dict is propagated verbatim as schema_properties."""
        chunk = _make_chunk()
        ent = _mock_entity(
            entity_type="Drug",
            name="metformin",
            properties={"mechanism": "biguanide"},
        )
        kg = _mock_kg(entities=[ent])
        entities, relations = _flatten_kg(
            kg,
            str(chunk.id),
            MEDICAL_GRAPHRAG_SCHEMA,
        )
        assert len(entities) == 1
        assert entities[0].schema_properties == {"mechanism": "biguanide"}

    async def test_llm_disabled_returns_empty(self):
        """When use_llm=False the extractor returns empty lists immediately."""
        cfg = ExtractionPipelineConfig(
            use_gliner=False,
            use_glirel=False,
            use_llm=False,
        )
        extractor = LLMEntityRelationExtractor(cfg, MEDICAL_GRAPHRAG_SCHEMA)
        entities, relations = await extractor.extract([])
        assert entities == []
        assert relations == []

    async def test_call_llm_receives_schema_prompt(self):
        """With use_llm=True, _call_llm is invoked once per chunk."""
        cfg = ExtractionPipelineConfig(use_gliner=False, use_glirel=False, use_llm=True)
        extractor = LLMEntityRelationExtractor(cfg, MEDICAL_GRAPHRAG_SCHEMA)
        chunk = _make_chunk()

        mock_kg = _mock_kg()
        with patch(
            "am_diag.graph_construction.extract.llm._call_llm",
            new_callable=AsyncMock,
            return_value=mock_kg,
        ) as mock_call:
            entities, relations = await extractor.extract([chunk])

        mock_call.assert_awaited_once()
        assert len(entities) == 0
        assert len(relations) == 0
