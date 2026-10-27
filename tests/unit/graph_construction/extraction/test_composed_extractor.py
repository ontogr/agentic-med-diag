"""Tests for the composed GLiNER+GLiREL+LLM extractor.

The three sub-extractors are replaced with fakes so no models load. These tests
cover the wiring the ``ComposedExtractor`` owns: GLiREL is driven via
``extract_from_entities`` with GLiNER spans grouped per chunk, combine+normalize
collapse duplicates across extractors, and endpoint-linking sets
``head_id``/``tail_id`` (dropping relations whose endpoints are unknown).
"""

from __future__ import annotations

import logging
from uuid import uuid4

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.common.schema.optimuskg import OPTIMUSKG_SCHEMA
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.extract.composed import ComposedExtractor


class _FakeGliner:
    """Return a fixed entity set, mirroring ``GLiNEREntityExtractor.extract``."""

    def __init__(self, entities: list[Entity]) -> None:
        self._entities = entities

    async def extract(self, chunks):
        return list(self._entities), []


class _FakeGlirel:
    """Record the per-chunk grouping it receives and return fixed relations."""

    def __init__(self, relations: list[Relation]) -> None:
        self._relations = relations
        self.received_per_chunk: list[list[Entity]] | None = None

    async def extract_from_entities(self, chunks, per_chunk_entities):
        self.received_per_chunk = per_chunk_entities
        return list(self._relations)


class _FakeLlm:
    """Return a fixed ``(entities, relations)`` pair like ``LLMExtractor``."""

    def __init__(self, entities: list[Entity], relations: list[Relation]) -> None:
        self._entities = entities
        self._relations = relations

    async def extract(self, chunks):
        return list(self._entities), list(self._relations)


def _chunk(text: str) -> Chunk:
    return Chunk(
        text=text,
        document_id=uuid4(),
        document_source="inline",
        chunk_index=0,
        chunk_size=len(text),
        cut_type="recursive",
    )


def _entity(name: str, label: str, chunk_ids, extractor=ExtractorName.GLINER) -> Entity:
    return Entity(
        name=name,
        label=label,
        provenance=Provenance(extractor=extractor, chunk_ids=list(chunk_ids)),
    )


def _composed(
    gliner=None, glirel=None, llm=None, schema=MEDICAL_GRAPHRAG_SCHEMA
) -> ComposedExtractor:
    ext = ComposedExtractor(
        ExtractionPipelineConfig(
            merge_strategy="union", min_entity_length=3, min_entity_score=0.0
        ),
        schema,
    )
    ext._gliner = gliner
    ext._glirel = glirel
    ext._llm = llm
    return ext


class TestComposedExtractor:
    """Wiring of grouping, combine/normalize, and endpoint-linking."""

    async def test_groups_gliner_entities_per_chunk_for_glirel(self):
        """GLiREL receives GLiNER entities grouped aligned to the chunk order."""
        c1 = _chunk("diabetes is treated by metformin")
        c2 = _chunk("aspirin treats headache pain")
        e1 = _entity("diabetes", "Disease", [c1.id])
        e2 = _entity("metformin", "Drug", [c1.id])
        e3 = _entity("aspirin", "Drug", [c2.id])
        glirel = _FakeGlirel([])
        ext = _composed(gliner=_FakeGliner([e1, e2, e3]), glirel=glirel, llm=None)

        await ext.extract([c1, c2])

        assert glirel.received_per_chunk == [[e1, e2], [e3]]

    async def test_combines_and_normalizes_duplicate_entities(self):
        """A duplicate entity across GLiNER and LLM collapses to one node."""
        c1 = _chunk("diabetes is a disease")
        gliner_ent = _entity("diabetes", "Disease", [c1.id])
        llm_ent = _entity("diabetes", "Disease", [c1.id], extractor=ExtractorName.LLM)
        ext = _composed(
            gliner=_FakeGliner([gliner_ent]),
            glirel=_FakeGlirel([]),
            llm=_FakeLlm([llm_ent], []),
        )

        entities, _ = await ext.extract([c1])

        assert len(entities) == 1
        assert entities[0].name == "diabetes"

    async def test_link_endpoints_sets_head_and_tail_ids(self):
        """Endpoint-linking resolves head/tail names to entity ids."""
        head = _entity("diabetes", "Disease", [])
        tail = _entity("metformin", "Drug", [])
        rel = Relation(head_name="diabetes", tail_name="metformin", type="TREATED_BY")

        linked = ComposedExtractor._link_endpoints([head, tail], [rel])

        assert len(linked) == 1
        assert linked[0].head_id == head.id
        assert linked[0].tail_id == tail.id
        assert linked[0].head_name == "diabetes"
        assert linked[0].tail_name == "metformin"

    def test_link_endpoints_drops_unresolved_relation(self, caplog):
        """A relation whose endpoint is not an entity is dropped and logged."""
        head = _entity("diabetes", "Disease", [])
        rel = Relation(head_name="diabetes", tail_name="unknown", type="TREATED_BY")

        with caplog.at_level(
            logging.DEBUG, logger="am_diag.graph_construction.extract.composed"
        ):
            linked = ComposedExtractor._link_endpoints([head], [rel])

        assert linked == []
        assert any("endpoint-link dropped" in r.message for r in caplog.records)

    def test_link_endpoints_yields_deterministic_id(self):
        """Reconstruction gives the linked relation the endpoint-derived uuid5."""
        head = _entity("diabetes", "Disease", [])
        tail = _entity("metformin", "Drug", [])
        rel = Relation(head_name="diabetes", tail_name="metformin", type="TREATED_BY")

        linked = ComposedExtractor._link_endpoints([head, tail], [rel])

        expected = Relation(head_id=head.id, tail_id=tail.id, type="TREATED_BY")
        assert linked[0].id == expected.id

    def test_link_endpoints_resolves_duplicate_names_from_allowed_pairs(self):
        """A unique allowed label pair selects the correct duplicate-name entity."""
        drug = _entity("insulin", "Drug", [])
        gene = _entity("insulin", "Gene", [])
        disease = _entity("diabetes", "Disease", [])
        relation = Relation(
            head_name="insulin",
            tail_name="diabetes",
            type="INDICATION",
        )
        allowed = frozenset(OPTIMUSKG_SCHEMA.allowed_pairs(extractable_only=True))

        linked = ComposedExtractor._link_endpoints(
            [drug, gene, disease], [relation], allowed
        )

        assert len(linked) == 1
        assert linked[0].head_id == drug.id
        assert linked[0].tail_id == disease.id

    def test_link_endpoints_drops_duplicate_names_when_multiple_pairs_fit(self):
        """Ambiguous matching stays unresolved when more than one pair is valid."""
        gene = _entity("insulin", "Gene", [])
        pathway = _entity("insulin", "Pathway", [])
        target = _entity("glucose", "Gene", [])
        relation = Relation(
            head_name="insulin",
            tail_name="glucose",
            type="INTERACTS_WITH",
        )
        allowed = frozenset(OPTIMUSKG_SCHEMA.allowed_pairs(extractable_only=True))

        linked = ComposedExtractor._link_endpoints(
            [gene, pathway, target], [relation], allowed
        )

        assert linked == []

    def test_link_endpoints_does_not_reverse_allowed_endpoint_pairs(self):
        """An allowed pair in reverse cannot resolve duplicate endpoint names."""
        disease = _entity("insulin", "Disease", [])
        gene = _entity("insulin", "Gene", [])
        drug = _entity("metformin", "Drug", [])
        relation = Relation(
            head_name="insulin",
            tail_name="metformin",
            type="INDICATION",
        )
        allowed = frozenset(OPTIMUSKG_SCHEMA.allowed_pairs(extractable_only=True))

        linked = ComposedExtractor._link_endpoints(
            [disease, gene, drug], [relation], allowed
        )

        assert linked == []

    async def test_extraction_links_duplicate_names_using_schema_pairs(self):
        """The pipeline keeps relations with resolvable duplicate endpoint names."""
        chunk = _chunk("insulin is indicated for diabetes")
        drug = _entity("insulin", "Drug", [chunk.id])
        gene = _entity("insulin", "Gene", [chunk.id])
        disease = _entity("diabetes", "Disease", [chunk.id])
        relation = Relation(
            head_name="insulin",
            tail_name="diabetes",
            type="INDICATION",
        )
        schema = OPTIMUSKG_SCHEMA.model_copy(update={"mode": "strict"})
        extractor = _composed(
            gliner=_FakeGliner([drug, gene, disease]),
            glirel=_FakeGlirel([relation]),
            schema=schema,
        )

        _, relations = await extractor.extract([chunk])

        assert len(relations) == 1
        assert relations[0].head_id == drug.id
        assert relations[0].tail_id == disease.id

    async def test_drops_relations_that_a_strict_schema_does_not_allow(self):
        """A strict schema removes extracted relations that break it."""
        c1 = _chunk("metformin treats type 2 diabetes and can cause nausea")
        metformin = _entity("metformin", "Drug", [c1.id])
        diabetes = _entity("type 2 diabetes", "Disease", [c1.id])
        nausea = _entity("nausea", "Symptom", [c1.id])
        allowed = Relation(
            head_name="type 2 diabetes", tail_name="metformin", type="TREATED_BY"
        )
        disallowed = Relation(
            head_name="metformin", tail_name="nausea", type="HAS_SYMPTOM"
        )
        strict_schema = MEDICAL_GRAPHRAG_SCHEMA.model_copy(update={"mode": "strict"})
        ext = _composed(
            gliner=_FakeGliner([metformin, diabetes, nausea]),
            glirel=_FakeGlirel([allowed, disallowed]),
            llm=None,
            schema=strict_schema,
        )

        _, relations = await ext.extract([c1])

        assert [r.type for r in relations] == ["TREATED_BY"]
