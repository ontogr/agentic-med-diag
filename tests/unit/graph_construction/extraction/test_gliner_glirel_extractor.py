"""Tests for GLiNER/GLiREL extractor internal helpers."""

from __future__ import annotations

from uuid import uuid4

from am_diag.common.data_models import Chunk, Entity
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.schema.optimuskg import OPTIMUSKG_SCHEMA
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.extract.gliner_entities import GLiNEREntityExtractor
from am_diag.graph_construction.extract.glirel_relations import (
    GLiRELRelationExtractor,
    _entities_for_glirel,
)


def _chunk(text: str) -> Chunk:
    return Chunk(
        text=text,
        document_id=uuid4(),
        document_source="inline",
        chunk_index=0,
        chunk_size=len(text),
        cut_type="sentence",
    )


def _entity_span(name: str, label: str, text: str) -> Entity:
    start = text.index(name)
    return Entity(
        name=name,
        label=label,
        provenance=Provenance(start_char=start, end_char=start + len(name)),
    )


class _FakeGLiNER:
    def __init__(self, spans: list[dict]) -> None:
        self.spans = spans
        self.labels: list[str] = []

    def batch_predict_entities(self, texts, labels, **kwargs):
        self.labels = labels
        return [self.spans]


class _FakeGLiREL:
    def __init__(self, relations: list[dict]) -> None:
        self.relations = relations
        self.labels: list[str] = []
        self.ner: list[list] = []

    def predict_relations(self, tokens, labels, *, threshold, ner):
        self.labels = labels
        self.ner = ner
        return self.relations


class TestEntitiesForGlirel:
    """Tests for entities for glirel."""

    def test_converts_entity_to_glirel_format(self):
        """Single entity with char offsets produces correct GLiREL token indices."""
        entity = Entity(
            name="Metformin",
            label="Drug",
            provenance=Provenance(
                start_char=0,
                end_char=9,
            ),
        )
        # token positions: M-e-t-f-o-r-m-i-n(0-9) treats(10-16) diabetes(17-25)
        char_to_token = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        result = _entities_for_glirel([entity], char_to_token)
        assert len(result) == 1
        assert result[0] == [0, 0, "Drug", "Metformin"]

    def test_skips_entity_without_char_offsets(self):
        """Entities without start_char/end_char are skipped."""
        entity = Entity(
            name="Metformin",
            label="Drug",
            provenance=Provenance(),
        )
        result = _entities_for_glirel([entity], {})
        assert result == []

    def test_handles_multiple_entities(self):
        """Multiple entities are all converted."""
        entity1 = Entity(
            name="Metformin",
            label="Drug",
            provenance=Provenance(
                start_char=0,
                end_char=9,
            ),
        )
        entity2 = Entity(
            name="diabetes",
            label="Disease",
            provenance=Provenance(
                start_char=17,
                end_char=25,
            ),
        )
        entity3 = Entity(
            name="insulin",
            label="Drug",
            provenance=Provenance(
                start_char=32,
                end_char=38,
            ),
        )
        char_to_token = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            17: 2,
            18: 2,
            19: 2,
            20: 2,
            21: 2,
            22: 2,
            23: 2,
            24: 2,
            32: 4,
            33: 4,
            34: 4,
            35: 4,
            36: 4,
            37: 4,
        }
        result = _entities_for_glirel([entity1, entity2, entity3], char_to_token)
        assert len(result) == 3


class TestGLiNEREntityExtractor:
    """Tests for OptimusKG GLiNER inputs and returned labels."""

    async def test_uses_descriptive_labels_and_drops_unknown_model_labels(self):
        """GLiNER receives descriptive labels and unknown outputs are ignored."""
        text = "metformin treats diabetes"
        chunk = _chunk(text)
        model = _FakeGLiNER(
            [
                {"text": "mystery", "label": "unknown label"},
                {
                    "text": "diabetes",
                    "label": "disease, disorder, syndrome, condition, or neoplasm",
                },
            ]
        )
        extractor = GLiNEREntityExtractor(
            ExtractionPipelineConfig(
                use_gliner=True,
                use_glirel=False,
                use_llm=False,
            ),
            OPTIMUSKG_SCHEMA,
        )
        extractor._model = model

        entities, _ = await extractor.extract([chunk])

        assert model.labels == [
            "anatomical structure, organ, tissue, or body part",
            "disease, disorder, syndrome, condition, or neoplasm",
            "drug, medication, or pharmaceutical agent",
            "environmental exposure, toxin, or chemical exposure",
            "gene, gene product, or protein",
            "symptom, clinical sign, or phenotypic abnormality",
            "biochemical or signaling pathway",
        ]
        assert [(entity.name, entity.label) for entity in entities] == [
            ("diabetes", "Disease")
        ]


class TestGLiRELRelationExtractor:
    """Tests for OptimusKG GLiREL endpoint and relation-label handling."""

    async def test_filters_spans_and_drops_unknown_relation_labels(self):
        """GLiREL receives useful spans and unknown output labels are ignored."""
        text = "metformin treats diabetes glycolysis"
        chunk = _chunk(text)
        indication_label = (
            "drug is indicated for (approved to treat) a disease or phenotype"
        )
        model = _FakeGLiREL(
            [
                {
                    "head_text": ["metformin"],
                    "tail_text": ["diabetes"],
                    "label": "unrecognized relation",
                },
                {
                    "head_text": ["metformin"],
                    "tail_text": ["diabetes"],
                    "label": indication_label,
                },
            ]
        )
        extractor = GLiRELRelationExtractor(
            ExtractionPipelineConfig(
                use_gliner=True,
                use_glirel=True,
                use_llm=False,
            ),
            OPTIMUSKG_SCHEMA,
        )
        extractor._model = model
        entities = [
            _entity_span("metformin", "Drug", text),
            _entity_span("diabetes", "Disease", text),
            _entity_span("glycolysis", "GoBiologicalProcess", text),
        ]

        relations = await extractor.extract_from_entities([chunk], [entities])

        assert [entry[2] for entry in model.ner] == ["Drug", "Disease"]
        assert indication_label in model.labels
        assert "INHIBITOR" not in model.labels
        assert [
            (relation.type, relation.head_name, relation.tail_name)
            for relation in relations
        ] == [("INDICATION", "metformin", "diabetes")]
