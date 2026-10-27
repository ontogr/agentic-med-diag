"""OptimusKG biomedical knowledge-graph schema (union of all source subgraphs).

OptimusKG nodes use 10 ``input_label`` abbreviations and edges 27 structural
``HEAD-TAIL`` labels. This module maps those onto am_diag PascalCase entity labels
and generates the ``GraphSchema`` used by the released-OptimusKG loader (and, as a
subset, by future per-source subgraph loaders).

The per-edge **semantic** relation (``Relation.type``, e.g. ``EXPRESSION_PRESENT``,
``INHIBITOR``) lives in the gold ``relation`` column, which the loader maps to
``type``. ``SEMANTIC_RELATIONS`` lists the distinct values per structural label,
enumerated from the Dataverse release 3.2 ``edges.parquet``. A structural label
missing from the map contributes a single placeholder type (the structural label
with ``-`` replaced by ``_``).
Relation types drive text2cypher, ``allowed_pairs`` and ontology pruning of text
extraction; they do not affect structured ingestion (``setup_schema`` keys
constraints on entity labels).

To move to a new OptimusKG release, refresh ``SEMANTIC_RELATIONS`` from the new
``edges.parquet``, update ``OPTIMUSKG_RELEASE``, and build a new graph. A graph is
tied to the release it was built with.
"""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


# Dataverse release that ``SEMANTIC_RELATIONS`` was enumerated from.
OPTIMUSKG_RELEASE = "3.2"

# Term-hierarchy relations and the Gene Ontology node types carry structure
# rather than facts that text states, so text extractors do not offer them.
_HIERARCHY_RELATIONS = frozenset({"PARENT", "IS_A"})
_GO_LABELS = frozenset(
    {"GoBiologicalProcess", "GoCellularComponent", "GoMolecularFunction"}
)

# OptimusKG node ``input_label`` abbreviation -> am_diag PascalCase entity label.
ABBREV_TO_LABEL: dict[str, str] = {
    "ANA": "AnatomicalStructure",
    "BPO": "GoBiologicalProcess",
    "CCO": "GoCellularComponent",
    "DIS": "Disease",
    "DRG": "Drug",
    "EXP": "Exposure",
    "GEN": "Gene",
    "MFN": "GoMolecularFunction",
    "PHE": "Phenotype",
    "PWY": "Pathway",
}

_LABEL_DESCRIPTIONS: dict[str, str] = {
    "AnatomicalStructure": "Anatomical structure or tissue (UBERON).",
    "GoBiologicalProcess": "Gene Ontology biological process term.",
    "GoCellularComponent": "Gene Ontology cellular component term.",
    "Disease": "Disease or disorder (MONDO / UMLS).",
    "Drug": "Drug or chemical compound (ChEMBL / DrugBank).",
    "Exposure": "Environmental exposure (CTD).",
    "Gene": "Human gene or genetic locus (Ensembl / HGNC).",
    "GoMolecularFunction": "Gene Ontology molecular function term.",
    "Phenotype": "Phenotypic feature (HPO).",
    "Pathway": "Biological pathway (Reactome).",
}

_TEXT_ENTITY_LABELS: dict[str, str] = {
    "AnatomicalStructure": "anatomical structure, organ, tissue, or body part",
    "GoBiologicalProcess": "biological process",
    "GoCellularComponent": "cellular component or subcellular location",
    "Disease": "disease, disorder, syndrome, condition, or neoplasm",
    "Drug": "drug, medication, or pharmaceutical agent",
    "Exposure": "environmental exposure, toxin, or chemical exposure",
    "Gene": "gene, gene product, or protein",
    "GoMolecularFunction": "molecular function",
    "Phenotype": "symptom, clinical sign, or phenotypic abnormality",
    "Pathway": "biochemical or signaling pathway",
}

_RELATION_DESCRIPTIONS: dict[str, str] = {
    "INDICATION": "drug is indicated for (approved to treat) a disease or phenotype",
    "CONTRAINDICATION": "drug is contraindicated in a disease or phenotype",
    "OFF_LABEL_USE": "drug is used off label for a disease or phenotype",
    "ADVERSE_DRUG_REACTION": "drug causes an adverse reaction (a phenotype)",
    "PHENOTYPE_PRESENT": "disease presents with a phenotype (symptom or sign)",
    "ASSOCIATED_WITH": (
        "disease or phenotype is associated with a gene, or a drug with a phenotype"
    ),
    "EXPRESSION_PRESENT": "gene is expressed in an anatomical structure",
    "EXPRESSION_ABSENT": "gene is not expressed in an anatomical structure",
    "SYNERGISTIC_INTERACTION": "two drugs act synergistically",
    "LINKED_TO": "environmental exposure is linked to a disease",
    "INTERACTS_WITH": (
        "two genes or gene products interact, or a pathway involves a gene"
    ),
    "PARENT": "term is a child of a broader term of the same kind",
    "IS_A": "term is a subtype of a broader term",
}

_GLIREL_TYPES = frozenset(
    {
        "INDICATION",
        "CONTRAINDICATION",
        "ADVERSE_DRUG_REACTION",
        "PHENOTYPE_PRESENT",
        "ASSOCIATED_WITH",
        "EXPRESSION_PRESENT",
        "SYNERGISTIC_INTERACTION",
        "LINKED_TO",
        "INTERACTS_WITH",
    }
)


def _relation_description(relation_type: str) -> str:
    """Return the plain-language meaning of an OptimusKG relation type."""
    if relation_type in _RELATION_DESCRIPTIONS:
        return _RELATION_DESCRIPTIONS[relation_type]
    return f"drug acts on a gene product as {relation_type.lower().replace('_', ' ')}"


# 27 OptimusKG edge ``input_label`` values (structural ``HEAD-TAIL``).
STRUCTURAL_LABELS: list[str] = [
    "ANA-GEN",
    "ANA-ANA",
    "EXP-DIS",
    "EXP-EXP",
    "EXP-GEN",
    "DIS-GEN",
    "DIS-DIS",
    "PHE-GEN",
    "DRG-GEN",
    "DRG-DRG",
    "DRG-DIS",
    "DRG-BPO",
    "DRG-PHE",
    "BPO-GEN",
    "CCO-GEN",
    "MFN-GEN",
    "PWY-PWY",
    "PWY-GEN",
    "EXP-BPO",
    "EXP-MFN",
    "EXP-CCO",
    "CCO-CCO",
    "BPO-BPO",
    "MFN-MFN",
    "DIS-PHE",
    "PHE-PHE",
    "GEN-GEN",
]

# Distinct semantic ``relation`` values per structural label, enumerated from the
# ``edges.parquet`` of Dataverse release 3.2. Refresh this map when a release adds
# relation values.
SEMANTIC_RELATIONS: dict[str, list[str]] = {
    "ANA-ANA": ["PARENT"],
    "ANA-GEN": ["EXPRESSION_ABSENT", "EXPRESSION_PRESENT"],
    "BPO-BPO": ["IS_A"],
    "BPO-GEN": ["INTERACTS_WITH"],
    "CCO-CCO": ["IS_A"],
    "CCO-GEN": ["INTERACTS_WITH"],
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
    "EXP-BPO": ["INTERACTS_WITH"],
    "EXP-CCO": ["INTERACTS_WITH"],
    "EXP-DIS": ["LINKED_TO"],
    "EXP-EXP": ["PARENT"],
    "EXP-GEN": ["INTERACTS_WITH"],
    "EXP-MFN": ["INTERACTS_WITH"],
    "GEN-GEN": ["INTERACTS_WITH"],
    "MFN-GEN": ["INTERACTS_WITH"],
    "MFN-MFN": ["IS_A"],
    "PHE-GEN": ["ASSOCIATED_WITH"],
    "PHE-PHE": ["PARENT"],
    "PWY-GEN": ["INTERACTS_WITH"],
    "PWY-PWY": ["PARENT"],
}

# Properties every structured relation may carry (queryable on the edge).
_RELATION_PROPERTIES = [
    "undirected",
    "sources",
    "indirect_sources",
    "asserted_relations",
    "structural_label",
]


def _endpoints(structural_label: str) -> tuple[str, str]:
    """Return the ``(head_label, tail_label)`` for a structural ``HEAD-TAIL`` label."""
    head, tail = structural_label.split("-")
    return ABBREV_TO_LABEL[head], ABBREV_TO_LABEL[tail]


def _build_entity_types() -> list[EntityType]:
    """Build one EntityType per unique am_diag label (identity keyed on curie)."""
    labels = list(dict.fromkeys(ABBREV_TO_LABEL.values()))
    return [
        EntityType(
            label=label,
            gliner_label=_TEXT_ENTITY_LABELS[label],
            description=_LABEL_DESCRIPTIONS[label],
            key_property="curie",
            extractable=label not in _GO_LABELS,
        )
        for label in labels
    ]


def _build_relation_types() -> list[RelationType]:
    """Build RelationTypes from structural labels and their semantic relations."""
    relation_types: list[RelationType] = []
    for structural in STRUCTURAL_LABELS:
        head_label, tail_label = _endpoints(structural)
        semantics = SEMANTIC_RELATIONS.get(structural) or [structural.replace("-", "_")]
        for semantic in semantics:
            relation_types.append(
                RelationType(
                    type=semantic,
                    glirel_label=(
                        _relation_description(semantic)
                        if semantic in _GLIREL_TYPES
                        else ""
                    ),
                    head_labels=[head_label],
                    tail_labels=[tail_label],
                    properties=_RELATION_PROPERTIES,
                    extractable=semantic not in _HIERARCHY_RELATIONS,
                    description=_relation_description(semantic),
                )
            )
    return relation_types


ENTITY_TYPES: list[EntityType] = _build_entity_types()
RELATION_TYPES: list[RelationType] = _build_relation_types()

OPTIMUSKG_SCHEMA = GraphSchema(
    version=OPTIMUSKG_RELEASE,
    entity_types=ENTITY_TYPES,
    relation_types=RELATION_TYPES,
)

__all__ = [
    "OPTIMUSKG_RELEASE",
    "ABBREV_TO_LABEL",
    "STRUCTURAL_LABELS",
    "SEMANTIC_RELATIONS",
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "OPTIMUSKG_SCHEMA",
]
