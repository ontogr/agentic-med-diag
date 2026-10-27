"""Tests for the generated OptimusKG union schema.

The schema is derived mechanically from the abbreviation map, the 27 structural
edge labels, and the semantic relation values enumerated from the released data;
these assertions pin the label set, the endpoint derivation, the relation types,
the release version, and which types text extraction is offered.
"""

from __future__ import annotations

import pytest

from am_diag.common.schema.optimuskg import (
    ABBREV_TO_LABEL,
    OPTIMUSKG_SCHEMA,
    SEMANTIC_RELATIONS,
    STRUCTURAL_LABELS,
    _endpoints,
)


class TestOptimusKGSchema:
    """Entity-label coverage, endpoint mapping, and relation generation."""

    def test_has_ten_entity_types(self) -> None:
        """One EntityType per unique PascalCase label (10 OptimusKG node types)."""
        assert len(OPTIMUSKG_SCHEMA.entity_types) == 10

    def test_entity_labels_are_the_pascalcase_set(self) -> None:
        """Entity labels are exactly the abbreviation map's target labels."""
        labels = {et.label for et in OPTIMUSKG_SCHEMA.entity_types}
        assert labels == set(ABBREV_TO_LABEL.values())

    def test_entity_identity_keyed_on_curie(self) -> None:
        """Structured entity types resolve by CURIE, not name."""
        gene = next(et for et in OPTIMUSKG_SCHEMA.entity_types if et.label == "Gene")
        assert gene.key_property == "curie"

    def test_endpoints_map_structural_label_to_pascalcase(self) -> None:
        """A HEAD-TAIL label resolves to its PascalCase endpoint labels."""
        assert _endpoints("ANA-GEN") == ("AnatomicalStructure", "Gene")
        assert _endpoints("DRG-DIS") == ("Drug", "Disease")

    def test_relation_endpoints_are_populated(self) -> None:
        """Every relation type carries head/tail labels from its structural pair."""
        for rt in OPTIMUSKG_SCHEMA.relation_types:
            assert rt.head_labels and rt.tail_labels

    def test_every_structural_label_has_semantic_relations(self) -> None:
        """No structural label falls back to the placeholder relation type."""
        assert set(SEMANTIC_RELATIONS) == set(STRUCTURAL_LABELS)
        assert all(SEMANTIC_RELATIONS[label] for label in STRUCTURAL_LABELS)

    def test_relation_types_are_the_released_semantic_values(self) -> None:
        """Relation types are exactly the released values, scoped to their endpoints."""
        interacts = ["INTERACTS_WITH"]
        expected = {
            "ANA-ANA": ["PARENT"],
            "ANA-GEN": ["EXPRESSION_ABSENT", "EXPRESSION_PRESENT"],
            "BPO-BPO": ["IS_A"],
            "BPO-GEN": interacts,
            "CCO-CCO": ["IS_A"],
            "CCO-GEN": interacts,
            "DIS-DIS": ["PARENT"],
            "DIS-GEN": ["ASSOCIATED_WITH"],
            "DIS-PHE": ["PHENOTYPE_PRESENT"],
            "DRG-BPO": ["INDICATION"],
            "DRG-DIS": ["CONTRAINDICATION", "INDICATION", "OFF_LABEL_USE"],
            "DRG-DRG": ["PARENT", "SYNERGISTIC_INTERACTION"],
            "DRG-GEN": [
                "ACTIVATOR",
                "AGONIST",
                "ALLOSTERIC_ANTAGONIST",
                "ANTAGONIST",
                "ANTISENSE_INHIBITOR",
                "BINDING_AGENT",
                "BLOCKER",
                "CARRIER",
                "CROSS_LINKING_AGENT",
                "DEGRADER",
                "DISRUPTING_AGENT",
                "ENZYME",
                "EXOGENOUS_GENE",
                "EXOGENOUS_PROTEIN",
                "HYDROLYTIC_ENZYME",
                "INHIBITOR",
                "INVERSE_AGONIST",
                "MODULATOR",
                "NEGATIVE_ALLOSTERIC_MODULATOR",
                "NEGATIVE_MODULATOR",
                "OPENER",
                "OTHER",
                "PARTIAL_AGONIST",
                "POSITIVE_ALLOSTERIC_MODULATOR",
                "POSITIVE_MODULATOR",
                "PROTEOLYTIC_ENZYME",
                "RELEASING_AGENT",
                "RNAI_INHIBITOR",
                "STABILISER",
                "SUBSTRATE",
                "TARGET",
                "TRANSPORTER",
                "VACCINE_ANTIGEN",
            ],
            "DRG-PHE": [
                "ADVERSE_DRUG_REACTION",
                "ASSOCIATED_WITH",
                "CONTRAINDICATION",
                "INDICATION",
                "OFF_LABEL_USE",
            ],
            "EXP-BPO": interacts,
            "EXP-CCO": interacts,
            "EXP-DIS": ["LINKED_TO"],
            "EXP-EXP": ["PARENT"],
            "EXP-GEN": interacts,
            "EXP-MFN": interacts,
            "GEN-GEN": interacts,
            "MFN-GEN": interacts,
            "MFN-MFN": ["IS_A"],
            "PHE-GEN": ["ASSOCIATED_WITH"],
            "PHE-PHE": ["PARENT"],
            "PWY-GEN": interacts,
            "PWY-PWY": ["PARENT"],
        }
        expected_types = {
            (relation, *_endpoints(structural))
            for structural, relations in expected.items()
            for relation in relations
        }
        actual_types = {
            (rt.type, rt.head_labels[0], rt.tail_labels[0])
            for rt in OPTIMUSKG_SCHEMA.relation_types
        }
        assert actual_types == expected_types
        assert len(OPTIMUSKG_SCHEMA.relation_types) == len(expected_types)

    def test_relation_properties_include_loaded_edge_properties(self) -> None:
        """Relation types advertise the edge properties the loader persists."""
        for rt in OPTIMUSKG_SCHEMA.relation_types:
            assert {"undirected", "sources", "indirect_sources"} <= set(rt.properties)
            assert "asserted_relations" in rt.properties

    def test_version_is_the_released_optimuskg_release(self) -> None:
        """The schema version names the Dataverse release its relations came from."""
        assert OPTIMUSKG_SCHEMA.version == "3.2"

    def test_relation_groups_one_per_released_relation_value(self) -> None:
        """Relation groups number the 46 distinct released relation values."""
        assert len(OPTIMUSKG_SCHEMA.relation_groups()) == 46

    def test_every_relation_endpoint_is_a_declared_entity_type(self) -> None:
        """No relation type names an endpoint label the schema does not declare."""
        declared = {et.label for et in OPTIMUSKG_SCHEMA.entity_types}
        endpoints = {
            label
            for rt in OPTIMUSKG_SCHEMA.relation_types
            for label in (*rt.head_labels, *rt.tail_labels)
        }
        assert endpoints <= declared

    @pytest.mark.parametrize(
        "label",
        ["GoBiologicalProcess", "GoCellularComponent", "GoMolecularFunction"],
    )
    def test_gene_ontology_entity_types_are_not_extractable(self, label: str) -> None:
        """Gene Ontology terms come from structured data, not from text extraction."""
        entity = next(et for et in OPTIMUSKG_SCHEMA.entity_types if et.label == label)
        assert entity.extractable is False

    @pytest.mark.parametrize("relation", ["PARENT", "IS_A"])
    def test_hierarchy_relation_types_are_not_extractable(self, relation: str) -> None:
        """Term-hierarchy relations are structure that extraction does not offer."""
        matching = [rt for rt in OPTIMUSKG_SCHEMA.relation_types if rt.type == relation]
        assert matching
        assert all(rt.extractable is False for rt in matching)

    def test_disease_and_interaction_types_stay_extractable(self) -> None:
        """Disease entities and interaction relations remain available to extraction."""
        disease = next(
            et for et in OPTIMUSKG_SCHEMA.entity_types if et.label == "Disease"
        )
        interactions = [
            rt for rt in OPTIMUSKG_SCHEMA.relation_types if rt.type == "INTERACTS_WITH"
        ]
        assert disease.extractable is True
        assert interactions
        assert all(rt.extractable for rt in interactions)

    def test_gliner_labels_are_the_seven_non_gene_ontology_labels(self) -> None:
        """GLiNER receives descriptive labels for the seven extractable types."""
        assert set(OPTIMUSKG_SCHEMA.gliner_labels()) == {
            "anatomical structure, organ, tissue, or body part",
            "disease, disorder, syndrome, condition, or neoplasm",
            "drug, medication, or pharmaceutical agent",
            "environmental exposure, toxin, or chemical exposure",
            "gene, gene product, or protein",
            "symptom, clinical sign, or phenotypic abnormality",
            "biochemical or signaling pathway",
        }

    def test_entity_text_labels_are_unique(self) -> None:
        """Every OptimusKG entity type has a distinct text-extraction label."""
        labels = [entity.gliner_label for entity in OPTIMUSKG_SCHEMA.entity_types]
        assert len(labels) == 10
        assert len(labels) == len(set(labels))

    @pytest.mark.parametrize("relation", ["PARENT", "IS_A"])
    def test_glirel_labels_exclude_hierarchy_relations(self, relation: str) -> None:
        """GLiREL is not offered term-hierarchy relations."""
        assert relation not in OPTIMUSKG_SCHEMA.glirel_to_type.values()

    def test_glirel_labels_include_interaction_relations(self) -> None:
        """GLiREL receives described relations but not fine drug-gene types."""
        labels = OPTIMUSKG_SCHEMA.glirel_labels()
        assert (
            "two genes or gene products interact, or a pathway involves a gene"
            in labels
        )
        assert len(labels) == len(set(labels))
        assert "INHIBITOR" not in OPTIMUSKG_SCHEMA.glirel_to_type.values()

    def test_relation_guide_groups_allowed_endpoint_pairs(self) -> None:
        """Prompt guidance groups pairs and excludes non-extractable types."""
        guide = OPTIMUSKG_SCHEMA.relation_guide()

        assert (
            "- PHENOTYPE_PRESENT [Disease -> Phenotype]: "
            "disease presents with a phenotype (symptom or sign)"
        ) in guide
        assert (
            "- INDICATION [Drug -> Disease, Drug -> Phenotype]: "
            "drug is indicated for (approved to treat) a disease or phenotype"
        ) in guide
        assert "PARENT" not in guide
        assert "IS_A" not in guide
