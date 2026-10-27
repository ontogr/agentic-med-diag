"""Tests for OntologyPruner, which applies a GraphSchema to extracted facts.

The schema has three entity types (Drug, Disease, Gene) and two relation types:
TREATS from Drug to Disease, and INTERACTS_WITH declared for Gene to Gene and for
Drug to Gene. A second schema adds a non-extractable Biomarker entity type and
non-extractable ASSOCIATED_WITH relation type, plus an extractable MEASURED_BY
relation that points at Biomarker and so is excluded from extraction. Each test
builds the schema with the mode it exercises. Everything runs in memory.
"""

from __future__ import annotations

import logging
from collections import Counter
from uuid import UUID

import pytest

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import (
    EntityType,
    GraphSchema,
    OntologyMode,
    RelationType,
)
from am_diag.graph_construction.prune import OntologyPruner


def _schema(mode: OntologyMode) -> GraphSchema:
    return GraphSchema(
        entity_types=[
            EntityType(label="Drug", gliner_label="drug", description="A medication."),
            EntityType(
                label="Disease", gliner_label="disease", description="A condition."
            ),
            EntityType(label="Gene", gliner_label="gene", description="A gene."),
        ],
        relation_types=[
            RelationType(
                type="TREATS",
                glirel_label="drug treats disease",
                head_labels=["Drug"],
                tail_labels=["Disease"],
            ),
            RelationType(
                type="INTERACTS_WITH",
                glirel_label="gene interacts with gene",
                head_labels=["Gene"],
                tail_labels=["Gene"],
            ),
            RelationType(
                type="INTERACTS_WITH",
                glirel_label="drug interacts with gene",
                head_labels=["Drug"],
                tail_labels=["Gene"],
            ),
        ],
        mode=mode,
    )


def _schema_with_non_extractable(mode: OntologyMode) -> GraphSchema:
    return GraphSchema(
        entity_types=[
            EntityType(label="Drug", gliner_label="drug", description="A medication."),
            EntityType(
                label="Disease", gliner_label="disease", description="A condition."
            ),
            EntityType(label="Gene", gliner_label="gene", description="A gene."),
            EntityType(
                label="Biomarker",
                gliner_label="biomarker",
                description="A measured lab value.",
                extractable=False,
            ),
        ],
        relation_types=[
            RelationType(
                type="TREATS",
                glirel_label="drug treats disease",
                head_labels=["Drug"],
                tail_labels=["Disease"],
            ),
            RelationType(
                type="ASSOCIATED_WITH",
                glirel_label="gene associated with disease",
                head_labels=["Gene"],
                tail_labels=["Disease"],
                extractable=False,
            ),
            RelationType(
                type="MEASURED_BY",
                glirel_label="disease measured by biomarker",
                head_labels=["Disease"],
                tail_labels=["Biomarker"],
            ),
        ],
        mode=mode,
    )


def _pair(head_label: str, tail_label: str) -> tuple[Entity, Entity]:
    return Entity(name="head", label=head_label), Entity(name="tail", label=tail_label)


def _relation(head: Entity, tail: Entity, relation_type: str) -> Relation:
    return Relation(
        head_id=head.id,
        tail_id=tail.id,
        head_name=head.name,
        tail_name=tail.name,
        type=relation_type,
    )


class TestOntologyPruner:
    """Schema enforcement by mode, endpoint swaps, and reporting."""

    @pytest.mark.parametrize(
        (
            "mode",
            "head_label",
            "tail_label",
            "relation_type",
            "expected_relations",
            "expected_violations",
        ),
        [
            ("strict", "Drug", "Disease", "TREATS", [("Drug", "Disease")], 0),
            ("strict", "Disease", "Drug", "TREATS", [("Drug", "Disease")], 0),
            ("warn", "Disease", "Drug", "TREATS", [("Drug", "Disease")], 0),
            ("strict", "Gene", "Disease", "TREATS", [], 1),
            ("warn", "Gene", "Disease", "TREATS", [("Gene", "Disease")], 1),
            ("open", "Gene", "Disease", "TREATS", [("Gene", "Disease")], 0),
            ("strict", "Gene", "Gene", "INTERACTS_WITH", [("Gene", "Gene")], 0),
            ("strict", "Drug", "Gene", "INTERACTS_WITH", [("Drug", "Gene")], 0),
            ("strict", "Gene", "Drug", "INTERACTS_WITH", [("Drug", "Gene")], 0),
            ("strict", "Drug", "Disease", "FOO", [], 1),
        ],
    )
    def test_applies_schema_to_relation_by_mode(
        self,
        mode: OntologyMode,
        head_label: str,
        tail_label: str,
        relation_type: str,
        expected_relations: list[tuple[str, str]],
        expected_violations: int,
    ):
        """Keep, swap, or drop a relation according to its endpoint labels."""
        head, tail = _pair(head_label, tail_label)
        pruner = OntologyPruner(_schema(mode))

        _, relations, report = pruner.prune(
            [head, tail], [_relation(head, tail, relation_type)]
        )

        labels_by_id: dict[UUID | None, str] = {
            head.id: head.label,
            tail.id: tail.label,
        }
        kept = [(labels_by_id[r.head_id], labels_by_id[r.tail_id]) for r in relations]
        assert kept == expected_relations
        assert report.relations_violating[relation_type] == expected_violations

    @pytest.mark.parametrize(
        ("mode", "expected_entity_labels", "expected_relation_count"),
        [
            ("strict", ["Disease"], 0),
            ("warn", ["Procedure", "Disease"], 1),
        ],
    )
    def test_drops_unknown_label_entity_and_its_relations_only_in_strict(
        self,
        mode: OntologyMode,
        expected_entity_labels: list[str],
        expected_relation_count: int,
    ):
        """An entity with a label outside the schema is dropped or kept by mode."""
        procedure = Entity(name="biopsy", label="Procedure")
        disease = Entity(name="headache", label="Disease")
        pruner = OntologyPruner(_schema(mode))

        entities, relations, report = pruner.prune(
            [procedure, disease], [_relation(procedure, disease, "TREATS")]
        )

        assert [entity.label for entity in entities] == expected_entity_labels
        assert len(relations) == expected_relation_count
        assert report.entities_kept == Counter({"Disease": 1})
        assert report.entities_violating == Counter({"Procedure": 1})

    @pytest.mark.parametrize("mode", ["strict", "warn"])
    def test_reverses_relation_into_schema_direction(self, mode: OntologyMode):
        """A reversed relation gets its head and tail, ids and names, exchanged."""
        drug = Entity(name="aspirin", label="Drug")
        disease = Entity(name="headache", label="Disease")
        reversed_relation = _relation(disease, drug, "TREATS")
        pruner = OntologyPruner(_schema(mode))

        _, relations, report = pruner.prune([drug, disease], [reversed_relation])

        (swapped,) = relations
        assert swapped.head_id == drug.id
        assert swapped.tail_id == disease.id
        assert swapped.head_name == "aspirin"
        assert swapped.tail_name == "headache"
        assert report.relations_swapped == Counter({"TREATS": 1})

    @pytest.mark.parametrize("reverse_input", [False, True])
    def test_never_swaps_relation_allowed_in_both_directions(self, reverse_input: bool):
        """A relation type declared Gene to Gene is kept in the extracted order."""
        gene_a = Entity(name="COX1", label="Gene")
        gene_b = Entity(name="COX2", label="Gene")
        head, tail = (gene_b, gene_a) if reverse_input else (gene_a, gene_b)
        relation = _relation(head, tail, "INTERACTS_WITH")
        pruner = OntologyPruner(_schema("strict"))

        _, relations, report = pruner.prune([gene_a, gene_b], [relation])

        assert relations == [relation]
        assert report.relations_swapped == Counter()

    def test_swapped_relation_keeps_description_score_and_provenance(self):
        """Swapping endpoints leaves the fact's own fields unchanged."""
        drug = Entity(name="aspirin", label="Drug")
        disease = Entity(name="headache", label="Disease")
        provenance = Provenance(
            extractor=ExtractorName.LLM,
            surface_forms=["aspirin"],
            start_char=0,
            end_char=7,
        )
        reversed_relation = Relation(
            head_id=disease.id,
            tail_id=drug.id,
            head_name="headache",
            tail_name="aspirin",
            type="TREATS",
            description="Aspirin relieves headache.",
            score=0.8,
            provenance=provenance,
        )
        pruner = OntologyPruner(_schema("strict"))

        _, relations, _ = pruner.prune([drug, disease], [reversed_relation])

        (swapped,) = relations
        assert swapped.description == "Aspirin relieves headache."
        assert swapped.score == 0.8
        assert swapped.provenance == provenance

    def test_swapped_relation_id_matches_relation_written_in_schema_direction(self):
        """The swapped relation has the id of the same fact written directly."""
        drug = Entity(name="aspirin", label="Drug")
        disease = Entity(name="headache", label="Disease")
        reversed_relation = _relation(disease, drug, "TREATS")
        written_directly = _relation(drug, disease, "TREATS")
        pruner = OntologyPruner(_schema("strict"))

        _, relations, _ = pruner.prune([drug, disease], [reversed_relation])

        assert relations[0].id == written_directly.id

    def test_same_input_produces_equal_reports(self):
        """Pruning the same input twice gives reports with equal counts."""
        drug = Entity(name="aspirin", label="Drug")
        disease = Entity(name="headache", label="Disease")
        gene = Entity(name="COX1", label="Gene")
        procedure = Entity(name="biopsy", label="Procedure")
        entities = [drug, disease, gene, procedure]
        relations = [
            _relation(drug, disease, "TREATS"),
            _relation(disease, drug, "TREATS"),
            _relation(gene, disease, "TREATS"),
            _relation(procedure, drug, "TREATS"),
        ]
        pruner = OntologyPruner(_schema("strict"))

        first = pruner.prune(entities, relations)[2]
        second = pruner.prune(entities, relations)[2]

        assert second == first

    def test_conformance_is_half_for_one_clean_and_one_swapped_relation(self):
        """One relation in schema direction and one reversed gives 0.5."""
        drug = Entity(name="aspirin", label="Drug")
        disease = Entity(name="headache", label="Disease")
        relations = [
            _relation(drug, disease, "TREATS"),
            _relation(disease, drug, "TREATS"),
        ]
        pruner = OntologyPruner(_schema("strict"))

        _, _, report = pruner.prune([drug, disease], relations)

        assert report.conformance == 0.5

    def test_conformance_is_one_when_no_relations_are_counted(self):
        """Conformance defaults to 1.0 when the input has no relations."""
        drug = Entity(name="aspirin", label="Drug")
        pruner = OntologyPruner(_schema("strict"))

        _, _, report = pruner.prune([drug], [])

        assert report.conformance == 1.0

    def test_samples_keep_first_three_distinct_violations_per_type(self):
        """Repeated violation patterns are sampled once, up to three per type."""
        drug = Entity(name="aspirin", label="Drug")
        disease = Entity(name="headache", label="Disease")
        gene = Entity(name="COX1", label="Gene")
        relations = [
            _relation(gene, disease, "TREATS"),
            _relation(gene, disease, "TREATS"),
            _relation(disease, gene, "TREATS"),
            _relation(gene, drug, "TREATS"),
            _relation(drug, gene, "TREATS"),
        ]
        pruner = OntologyPruner(_schema("warn"))

        _, _, report = pruner.prune([drug, disease, gene], relations)

        assert report.relations_violating == Counter({"TREATS": 5})
        assert report.samples == {
            "TREATS": [
                "Gene-[TREATS]->Disease",
                "Disease-[TREATS]->Gene",
                "Gene-[TREATS]->Drug",
            ]
        }

    def test_logs_violating_relation_type_with_its_examples(self, caplog):
        """A violating relation type is logged as a warning with its example."""
        disease = Entity(name="headache", label="Disease")
        gene = Entity(name="COX1", label="Gene")
        pruner = OntologyPruner(_schema("strict"))

        with caplog.at_level(
            logging.WARNING, logger="am_diag.graph_construction.prune"
        ):
            pruner.prune([gene, disease], [_relation(gene, disease, "TREATS")])

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert any(
            "TREATS" in message and "Gene-[TREATS]->Disease" in message
            for message in messages
        )

    def test_keeps_relation_with_unlinked_endpoint_unchanged(self):
        """A relation without an endpoint id is kept as extracted and counted."""
        drug = Entity(name="aspirin", label="Drug")
        unlinked = Relation(
            head_id=None,
            tail_id=drug.id,
            head_name="headache",
            tail_name="aspirin",
            type="TREATS",
        )
        pruner = OntologyPruner(_schema("strict"))

        _, relations, report = pruner.prune([drug], [unlinked])

        assert relations == [unlinked]
        assert report.relations_kept == Counter({"TREATS": 1})

    def test_open_mode_returns_inputs_and_counts_everything_as_kept(self):
        """Open mode returns the inputs and counts every fact as kept."""
        procedure = Entity(name="biopsy", label="Procedure")
        drug = Entity(name="aspirin", label="Drug")
        entities = [procedure, drug]
        relations = [_relation(procedure, drug, "TREATS")]
        pruner = OntologyPruner(_schema("open"))

        pruned_entities, pruned_relations, report = pruner.prune(entities, relations)

        assert pruned_entities == entities
        assert pruned_relations == relations
        assert report.entities_kept == Counter({"Procedure": 1, "Drug": 1})
        assert report.entities_violating == Counter()
        assert report.relations_kept == Counter({"TREATS": 1})

    @pytest.mark.parametrize(
        ("mode", "expected_kept", "expected_violations"),
        [("strict", False, 1), ("warn", True, 1), ("open", True, 0)],
    )
    def test_non_extractable_relation_type_is_dropped_only_in_strict(
        self,
        mode: OntologyMode,
        expected_kept: bool,
        expected_violations: int,
    ):
        """A relation of a non-extractable type is dropped in strict mode only."""
        gene = Entity(name="COX1", label="Gene")
        disease = Entity(name="headache", label="Disease")
        relation = _relation(gene, disease, "ASSOCIATED_WITH")
        pruner = OntologyPruner(_schema_with_non_extractable(mode))

        _, relations, report = pruner.prune([gene, disease], [relation])

        assert relations == ([relation] if expected_kept else [])
        assert report.relations_violating["ASSOCIATED_WITH"] == expected_violations

    @pytest.mark.parametrize(
        ("mode", "expected_head_names"),
        [("strict", []), ("warn", ["headache"])],
    )
    def test_does_not_swap_reversed_non_extractable_relation(
        self, mode: OntologyMode, expected_head_names: list[str]
    ):
        """A reversed relation of a non-extractable type is never swapped."""
        gene = Entity(name="COX1", label="Gene")
        disease = Entity(name="headache", label="Disease")
        reversed_relation = _relation(disease, gene, "ASSOCIATED_WITH")
        pruner = OntologyPruner(_schema_with_non_extractable(mode))

        _, relations, report = pruner.prune([gene, disease], [reversed_relation])

        assert [relation.head_name for relation in relations] == expected_head_names
        assert report.relations_swapped == Counter()
        assert report.relations_violating == Counter({"ASSOCIATED_WITH": 1})

    @pytest.mark.parametrize(
        ("mode", "expected_labels", "expected_relation_count", "expected_violations"),
        [
            ("strict", ["Disease"], 0, Counter({"Biomarker": 1})),
            ("warn", ["Biomarker", "Disease"], 1, Counter({"Biomarker": 1})),
            ("open", ["Biomarker", "Disease"], 1, Counter()),
        ],
    )
    def test_non_extractable_entity_is_dropped_with_its_relations_only_in_strict(
        self,
        mode: OntologyMode,
        expected_labels: list[str],
        expected_relation_count: int,
        expected_violations: Counter[str],
    ):
        """An entity of a non-extractable label is dropped in strict mode only."""
        biomarker = Entity(name="troponin", label="Biomarker")
        disease = Entity(name="infarction", label="Disease")
        pruner = OntologyPruner(_schema_with_non_extractable(mode))

        entities, relations, report = pruner.prune(
            [biomarker, disease], [_relation(disease, biomarker, "MEASURED_BY")]
        )

        assert [entity.label for entity in entities] == expected_labels
        assert len(relations) == expected_relation_count
        assert report.entities_violating == expected_violations

    def test_keeps_extractable_relation_between_extractable_entities_in_strict(self):
        """An extractable relation between extractable entities survives strict mode."""
        drug = Entity(name="aspirin", label="Drug")
        disease = Entity(name="headache", label="Disease")
        relation = _relation(drug, disease, "TREATS")
        pruner = OntologyPruner(_schema_with_non_extractable("strict"))

        _, relations, report = pruner.prune([drug, disease], [relation])

        assert relations == [relation]
        assert report.relations_violating == Counter()
