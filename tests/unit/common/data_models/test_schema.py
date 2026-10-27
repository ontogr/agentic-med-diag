"""Tests for GraphSchema validation, relation grouping, and extractor label views.

Text extractors read the label views (``gliner_labels``, ``glirel_labels`` and the
matching lookup maps) and the prompt blocks, which leave out types that
extraction must not produce. Graph-facing output (``as_text``, ``allowed_pairs``)
keeps every declared type. The fingerprint covers definitions and ignores ids and
timestamps, so a schema rebuilt from stored records keeps its fingerprint.
"""

from __future__ import annotations

import logging
import re
from uuid import uuid4

import pytest

from am_diag.common.data_models.schema import (
    EntityType,
    GraphSchema,
    OntologyMode,
    RelationType,
)


def _entity(
    label: str,
    gliner_label: str,
    description: str,
    *,
    extractable: bool = True,
) -> EntityType:
    """Build an entity type with the given labels and description."""
    return EntityType(
        label=label,
        gliner_label=gliner_label,
        description=description,
        extractable=extractable,
    )


def _relation(
    name: str,
    glirel_label: str,
    head_labels: list[str],
    tail_labels: list[str],
    *,
    extractable: bool = True,
    description: str = "",
) -> RelationType:
    """Build a relation type with the given endpoints."""
    return RelationType(
        type=name,
        glirel_label=glirel_label,
        head_labels=head_labels,
        tail_labels=tail_labels,
        extractable=extractable,
        description=description,
    )


def _fingerprint_schema(
    *,
    version: str = "unversioned",
    mode: OntologyMode = "open",
    drug_description: str = "A drug.",
    drug_extractable: bool = True,
    relation_extractable: bool = True,
    timestamp: int = 0,
) -> GraphSchema:
    """Build a small schema with fresh ids and the given creation timestamp."""
    return GraphSchema(
        version=version,
        mode=mode,
        entity_types=[
            EntityType(
                id=uuid4(),
                created_at=timestamp,
                updated_at=timestamp,
                label="Drug",
                gliner_label="drug",
                description=drug_description,
                extractable=drug_extractable,
            ),
            EntityType(
                id=uuid4(),
                created_at=timestamp,
                updated_at=timestamp,
                label="Disease",
                gliner_label="disease",
                description="A disease.",
            ),
        ],
        relation_types=[
            RelationType(
                id=uuid4(),
                created_at=timestamp,
                updated_at=timestamp,
                type="INDICATION",
                glirel_label="drug is indicated for disease",
                head_labels=["Drug"],
                tail_labels=["Disease"],
                extractable=relation_extractable,
            ),
        ],
    )


def _mixed_extractability_schema() -> GraphSchema:
    """Build a schema with one non-extractable entity type and mixed relations.

    ``ASSOCIATED_WITH`` has one extractable head and one non-extractable head, so
    it stays extractable while its Gene pairs are hidden. ``EXPRESSED_IN`` has only
    non-extractable heads and is left out entirely. ``PARENT`` is not extractable.
    """
    return GraphSchema(
        entity_types=[
            _entity("Drug", "drug", "A drug."),
            _entity("Disease", "disease", "A disease."),
            _entity("Gene", "gene", "A gene.", extractable=False),
        ],
        relation_types=[
            _relation("TREATS", "treats", ["Drug"], ["Disease"]),
            _relation(
                "ASSOCIATED_WITH",
                "associated with",
                ["Drug", "Gene"],
                ["Disease"],
            ),
            _relation("EXPRESSED_IN", "is expressed in", ["Gene"], ["Disease"]),
            _relation(
                "PARENT",
                "is a parent of",
                ["Disease"],
                ["Disease"],
                extractable=False,
            ),
        ],
    )


class TestGraphSchema:
    """Validation, defaults, relation grouping, and extractor-facing views."""

    def test_rejects_duplicate_entity_labels(self) -> None:
        """Two entity types with the same label are rejected."""
        with pytest.raises(ValueError, match=r"duplicate entity labels: \['Drug'\]"):
            GraphSchema(
                entity_types=[
                    _entity("Drug", "drug", "A drug."),
                    _entity("Drug", "medication", "Another drug."),
                ]
            )

    def test_rejects_extractable_types_sharing_a_gliner_label(self) -> None:
        """Two extractable entity types may not offer the same GLiNER label."""
        with pytest.raises(ValueError, match=r"share a gliner_label: \['drug'\]"):
            GraphSchema(
                entity_types=[
                    _entity("Drug", "drug", "A drug."),
                    _entity("Medication", "drug", "A medication."),
                ]
            )

    @pytest.mark.parametrize(
        ("drug_extractable", "medication_extractable", "expected_gliner_labels"),
        [
            pytest.param(True, False, ["drug"], id="second_not_extractable"),
            pytest.param(False, True, ["drug"], id="first_not_extractable"),
            pytest.param(False, False, [], id="neither_extractable"),
        ],
    )
    def test_allows_shared_gliner_label_when_a_type_is_not_extractable(
        self,
        drug_extractable: bool,
        medication_extractable: bool,
        expected_gliner_labels: list[str],
    ) -> None:
        """A shared GLiNER label is allowed when at most one type is extractable."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug.", extractable=drug_extractable),
                _entity(
                    "Medication",
                    "drug",
                    "A medication.",
                    extractable=medication_extractable,
                ),
            ]
        )
        assert schema.gliner_labels() == expected_gliner_labels

    def test_logs_warning_for_undeclared_relation_endpoint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An endpoint label without an entity type logs a warning and still builds."""
        with caplog.at_level(
            logging.WARNING, logger="am_diag.common.data_models.schema"
        ):
            schema = GraphSchema(
                entity_types=[_entity("Drug", "drug", "A drug.")],
                relation_types=[
                    _relation("TREATS", "drug treats disease", ["Drug"], ["Disease"]),
                ],
            )

        assert schema.relation_type_names() == {"TREATS"}
        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "['Disease']" in caplog.records[0].getMessage()

    def test_declared_relation_endpoints_log_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Relation endpoints that name declared entity types log no warning."""
        with caplog.at_level(
            logging.WARNING, logger="am_diag.common.data_models.schema"
        ):
            GraphSchema(
                entity_types=[
                    _entity("Drug", "drug", "A drug."),
                    _entity("Disease", "disease", "A disease."),
                ],
                relation_types=[
                    _relation("TREATS", "drug treats disease", ["Drug"], ["Disease"]),
                ],
            )

        assert caplog.records == []

    def test_version_and_mode_default_for_new_schema(self) -> None:
        """A schema built without a release id is unversioned and open."""
        schema = GraphSchema()
        assert schema.version == "unversioned"
        assert schema.mode == "open"

    def test_version_and_mode_default_for_record_without_them(self) -> None:
        """A stored schema record that predates version and mode loads with defaults."""
        schema = GraphSchema.model_validate({"entity_types": [], "relation_types": []})
        assert schema.version == "unversioned"
        assert schema.mode == "open"

    def test_relation_groups_merge_pairs_of_a_repeated_type(self) -> None:
        """A relation type declared twice yields one group with both endpoint pairs."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Gene", "gene", "A gene."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation("USES", "uses", ["Drug"], ["Gene"]),
                _relation("USES", "uses", ["Drug"], ["Disease"]),
            ],
        )

        groups = schema.relation_groups()

        assert [group.type for group in groups] == ["USES"]
        assert set(groups[0].pairs) == {("Drug", "Gene"), ("Drug", "Disease")}

    def test_relation_groups_keep_the_first_description(self) -> None:
        """The group description comes from the first declaration of the type."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Gene", "gene", "A gene."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "USES",
                    "uses",
                    ["Drug"],
                    ["Gene"],
                    description="The drug acts on the gene.",
                ),
                _relation(
                    "USES",
                    "uses",
                    ["Drug"],
                    ["Disease"],
                    description="The drug is used for the disease.",
                ),
            ],
        )

        assert schema.relation_groups()[0].description == "The drug acts on the gene."

    def test_relation_groups_follow_declaration_order(self) -> None:
        """Groups appear in the order their relation types are first declared."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation("USES", "uses", ["Drug"], ["Disease"]),
                _relation("TREATS", "treats", ["Drug"], ["Disease"]),
                _relation("USES", "uses", ["Drug"], ["Disease"]),
            ],
        )

        assert [group.type for group in schema.relation_groups()] == [
            "USES",
            "TREATS",
        ]

    def test_as_text_renders_nodes_and_grouped_relations(self) -> None:
        """as_text lists each node type and one line per relation type."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "INDICATION",
                    "drug is indicated for disease",
                    ["Drug"],
                    ["Disease"],
                ),
            ],
        )

        assert schema.as_text() == (
            "Graph schema:\n"
            "Nodes (each is :Entity plus its label):\n"
            "  :Drug - A drug.\n"
            "  :Disease - A disease.\n"
            "Relationships (all are :RELATES_TO with a `type` property; "
            "head -> tail):\n"
            "  type 'INDICATION': Drug->Disease\n"
            "Valid type values: ['INDICATION']"
        )

    def test_as_text_keeps_non_extractable_types(self) -> None:
        """Graph-facing text still describes types that extraction does not offer."""
        schema = GraphSchema(
            entity_types=[_entity("Gene", "gene", "A gene.", extractable=False)],
            relation_types=[
                _relation(
                    "PARENT",
                    "is a parent of",
                    ["Gene"],
                    ["Gene"],
                    extractable=False,
                ),
            ],
        )

        text = schema.as_text()

        assert "  :Gene - A gene." in text
        assert "  type 'PARENT': Gene->Gene" in text

    def test_as_text_includes_relation_description_when_set(self) -> None:
        """A relation description is appended to its as_text group line."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "INDICATION",
                    "drug is indicated for disease",
                    ["Drug"],
                    ["Disease"],
                    description="The drug treats the disease.",
                ),
            ],
        )

        assert schema.as_text() == (
            "Graph schema:\n"
            "Nodes (each is :Entity plus its label):\n"
            "  :Drug - A drug.\n"
            "  :Disease - A disease.\n"
            "Relationships (all are :RELATES_TO with a `type` property; "
            "head -> tail):\n"
            "  type 'INDICATION': Drug->Disease - The drug treats the disease.\n"
            "Valid type values: ['INDICATION']"
        )

    def test_allowed_pairs_keep_every_declared_triple(self) -> None:
        """allowed_pairs lists each head and tail combination, extractable or not."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Compound", "compound", "A compound."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "TREATS",
                    "treats",
                    ["Drug", "Compound"],
                    ["Disease"],
                ),
                _relation(
                    "PARENT",
                    "is a parent of",
                    ["Disease"],
                    ["Disease"],
                    extractable=False,
                ),
            ],
        )

        assert schema.allowed_pairs() == {
            ("Drug", "TREATS", "Disease"),
            ("Compound", "TREATS", "Disease"),
            ("Disease", "PARENT", "Disease"),
        }

    def test_gliner_labels_omit_non_extractable_entity_types(self) -> None:
        """Entity types that are not extractable are not offered to GLiNER."""
        schema = GraphSchema(
            entity_types=[
                _entity("Disease", "disease or condition", "A disease."),
                _entity("Gene", "gene or genetic locus", "A gene.", extractable=False),
            ]
        )

        assert schema.gliner_labels() == ["disease or condition"]

    def test_gliner_to_label_omits_non_extractable_entity_types(self) -> None:
        """Only extractable entity types map from a GLiNER label."""
        schema = GraphSchema(
            entity_types=[
                _entity("Disease", "disease or condition", "A disease."),
                _entity("Gene", "gene or genetic locus", "A gene.", extractable=False),
            ]
        )

        assert schema.gliner_to_label == {"disease or condition": "Disease"}

    @pytest.mark.parametrize(
        ("extractable", "glirel_label"),
        [
            pytest.param(
                False, "drug is contraindicated for disease", id="not_extractable"
            ),
            pytest.param(True, "", id="empty_glirel_label"),
        ],
    )
    def test_glirel_labels_omit_relation_types_not_offered_to_glirel(
        self, extractable: bool, glirel_label: str
    ) -> None:
        """A relation type with no GLiREL label or not extractable is not offered."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "INDICATION",
                    "drug is indicated for disease",
                    ["Drug"],
                    ["Disease"],
                ),
                _relation(
                    "CONTRAINDICATION",
                    glirel_label,
                    ["Drug"],
                    ["Disease"],
                    extractable=extractable,
                ),
            ],
        )

        assert schema.glirel_labels() == ["drug is indicated for disease"]

    @pytest.mark.parametrize(
        ("extractable", "glirel_label"),
        [
            pytest.param(
                False, "drug is contraindicated for disease", id="not_extractable"
            ),
            pytest.param(True, "", id="empty_glirel_label"),
        ],
    )
    def test_glirel_to_type_omits_relation_types_not_offered_to_glirel(
        self, extractable: bool, glirel_label: str
    ) -> None:
        """Only relation types offered to GLiREL map from a GLiREL label."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "INDICATION",
                    "drug is indicated for disease",
                    ["Drug"],
                    ["Disease"],
                ),
                _relation(
                    "CONTRAINDICATION",
                    glirel_label,
                    ["Drug"],
                    ["Disease"],
                    extractable=extractable,
                ),
            ],
        )

        assert schema.glirel_to_type == {"drug is indicated for disease": "INDICATION"}

    @pytest.mark.parametrize(
        ("head_labels", "tail_labels"),
        [
            pytest.param(["Gene"], ["Disease"], id="all_heads_not_extractable"),
            pytest.param(["Drug"], ["Gene"], id="all_tails_not_extractable"),
        ],
    )
    def test_glirel_labels_omit_relation_type_with_only_non_extractable_endpoints(
        self, head_labels: list[str], tail_labels: list[str]
    ) -> None:
        """A relation type whose heads or tails are all non-extractable is dropped."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
                _entity("Gene", "gene", "A gene.", extractable=False),
            ],
            relation_types=[
                _relation(
                    "INDICATION",
                    "drug is indicated for disease",
                    ["Drug"],
                    ["Disease"],
                ),
                _relation(
                    "ASSOCIATED_WITH",
                    "associated with",
                    head_labels,
                    tail_labels,
                ),
            ],
        )

        assert schema.glirel_labels() == ["drug is indicated for disease"]

    def test_glirel_labels_keep_relation_type_with_some_extractable_heads(
        self,
    ) -> None:
        """A relation type with at least one extractable head label is still offered."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
                _entity("Gene", "gene", "A gene.", extractable=False),
            ],
            relation_types=[
                _relation(
                    "ASSOCIATED_WITH",
                    "associated with",
                    ["Gene", "Drug"],
                    ["Disease"],
                ),
            ],
        )

        assert schema.glirel_labels() == ["associated with"]

    def test_glirel_labels_dedupe_a_type_declared_for_several_pairs(self) -> None:
        """A relation type declared for several endpoint pairs yields one label."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Gene", "gene", "A gene."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation("USES", "uses", ["Drug"], ["Gene"]),
                _relation("TREATS", "treats", ["Drug"], ["Disease"]),
                _relation("USES", "uses", ["Drug"], ["Disease"]),
            ],
        )

        assert schema.glirel_labels() == ["uses", "treats"]

    def test_rejects_extractable_relation_types_sharing_a_glirel_label(self) -> None:
        """Two extractable relation types of different names may not share a label."""
        with pytest.raises(ValueError, match=r"share a glirel_label: \['treats'\]"):
            GraphSchema(
                entity_types=[
                    _entity("Drug", "drug", "A drug."),
                    _entity("Disease", "disease", "A disease."),
                ],
                relation_types=[
                    _relation("TREATS", "treats", ["Drug"], ["Disease"]),
                    _relation("CURES", "treats", ["Drug"], ["Disease"]),
                ],
            )

    @pytest.mark.parametrize(
        (
            "second_name",
            "first_label",
            "second_label",
            "second_extractable",
            "expected",
        ),
        [
            pytest.param(
                "CURES",
                "treats",
                "treats",
                False,
                ["treats"],
                id="second_not_extractable",
            ),
            pytest.param("CURES", "", "", True, [], id="empty_glirel_label"),
            pytest.param(
                "TREATS",
                "treats",
                "treats",
                True,
                ["treats"],
                id="same_type_name",
            ),
        ],
    )
    def test_allows_shared_glirel_label_when_it_does_not_conflict(
        self,
        second_name: str,
        first_label: str,
        second_label: str,
        second_extractable: bool,
        expected: list[str],
    ) -> None:
        """A shared GLiREL label is allowed unless two extractable names claim it."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation("TREATS", first_label, ["Drug"], ["Disease"]),
                _relation(
                    second_name,
                    second_label,
                    ["Drug"],
                    ["Disease"],
                    extractable=second_extractable,
                ),
            ],
        )

        assert schema.glirel_labels() == expected

    def test_extractable_type_names_exclude_non_extractable_types(self) -> None:
        """Name views keep only extractable entity and relation types."""
        schema = _mixed_extractability_schema()

        assert schema.extractable_entity_type_names() == {"Drug", "Disease"}
        assert schema.extractable_relation_type_names() == {
            "TREATS",
            "ASSOCIATED_WITH",
        }

    def test_allowed_pairs_extractable_only_drops_pairs_touching_hidden_labels(
        self,
    ) -> None:
        """Extractable-only pairs drop any triple with a non-extractable endpoint."""
        schema = _mixed_extractability_schema()

        assert schema.allowed_pairs(extractable_only=True) == {
            ("Drug", "TREATS", "Disease"),
            ("Drug", "ASSOCIATED_WITH", "Disease"),
        }
        assert schema.allowed_pairs() == {
            ("Drug", "TREATS", "Disease"),
            ("Drug", "ASSOCIATED_WITH", "Disease"),
            ("Gene", "ASSOCIATED_WITH", "Disease"),
            ("Gene", "EXPRESSED_IN", "Disease"),
            ("Disease", "PARENT", "Disease"),
        }

    def test_entity_prompt_block_omits_non_extractable_entity_types(self) -> None:
        """The entity prompt lists only extractable entity types."""
        schema = GraphSchema(
            entity_types=[
                _entity("Disease", "disease", "A disease."),
                _entity("Gene", "gene", "A gene.", extractable=False),
            ]
        )

        assert schema.entity_prompt_block() == "- Disease: A disease."

    def test_relation_prompt_block_omits_non_extractable_relation_types(self) -> None:
        """The relation prompt lists only extractable relation types."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "INDICATION",
                    "drug is indicated for disease",
                    ["Drug"],
                    ["Disease"],
                ),
                _relation(
                    "PARENT",
                    "is a parent of",
                    ["Disease"],
                    ["Disease"],
                    extractable=False,
                ),
            ],
        )

        assert schema.relation_prompt_block() == (
            "- INDICATION [Drug → Disease]: drug is indicated for disease"
        )

    def test_relation_prompt_block_includes_relation_description_when_set(
        self,
    ) -> None:
        """A relation description follows its GLiREL label in the prompt line."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "INDICATION",
                    "drug is indicated for disease",
                    ["Drug"],
                    ["Disease"],
                    description="The drug treats the disease.",
                ),
            ],
        )

        assert schema.relation_prompt_block() == (
            "- INDICATION [Drug → Disease]: drug is indicated for disease"
            " - The drug treats the disease."
        )

    def test_relation_prompt_block_does_not_repeat_matching_descriptions(self) -> None:
        """A relation meaning shared by both fields appears only once."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation(
                    "INDICATION",
                    "drug is indicated for disease",
                    ["Drug"],
                    ["Disease"],
                    description="drug is indicated for disease",
                ),
            ],
        )

        assert schema.relation_prompt_block() == (
            "- INDICATION [Drug → Disease]: drug is indicated for disease"
        )

    def test_relation_prompt_block_keeps_relation_type_without_glirel_label(
        self,
    ) -> None:
        """A relation with no GLiREL label stays in the LLM prompt."""
        schema = GraphSchema(
            entity_types=[
                _entity("Drug", "drug", "A drug."),
                _entity("Disease", "disease", "A disease."),
            ],
            relation_types=[
                _relation("CONTRAINDICATION", "", ["Drug"], ["Disease"]),
            ],
        )

        assert schema.relation_prompt_block() == (
            "- CONTRAINDICATION [Drug → Disease]: "
        )

    def test_fingerprint_ignores_ids_and_timestamps(self) -> None:
        """Schemas with equal definitions match even when ids and timestamps differ."""
        first = _fingerprint_schema(timestamp=1_700_000_000_000)
        second = _fingerprint_schema(timestamp=1_800_000_000_000)

        assert first.fingerprint() == second.fingerprint()

    def test_fingerprint_is_sixteen_hex_characters(self) -> None:
        """The fingerprint is a 16-character lowercase hexadecimal string."""
        assert re.fullmatch(r"[0-9a-f]{16}", _fingerprint_schema().fingerprint())

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"version": "3.3"}, id="version"),
            pytest.param(
                {"drug_description": "A changed drug."}, id="entity_description"
            ),
            pytest.param({"drug_extractable": False}, id="entity_extractable"),
            pytest.param({"relation_extractable": False}, id="relation_extractable"),
            pytest.param({"mode": "strict"}, id="mode"),
        ],
    )
    def test_fingerprint_changes_when_a_definition_or_mode_changes(
        self, change: dict[str, object]
    ) -> None:
        """A new version, description, extractable flag, or mode changes the hash."""
        baseline = _fingerprint_schema().fingerprint()

        assert _fingerprint_schema(**change).fingerprint() != baseline

    def test_fingerprint_without_mode_ignores_enforcement_mode(self) -> None:
        """With include_mode=False, a strict and an open schema share a fingerprint."""
        strict = _fingerprint_schema(mode="strict")
        open_ = _fingerprint_schema(mode="open")

        assert strict.fingerprint(include_mode=False) == open_.fingerprint(
            include_mode=False
        )
