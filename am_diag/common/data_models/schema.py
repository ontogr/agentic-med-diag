"""GraphSchema: entity and relation type definitions for the medical knowledge graph."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from am_diag.common.data_models.annotations import Dedup
from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.metadata import MetaData


logger = logging.getLogger(__name__)

OntologyMode = Literal["open", "warn", "strict"]


class EntityType(DataPoint):
    """A type definition for medical entities (config node, not extracted entity).

    Identity = (label,).

    Attributes:
        label: Entity label, such as ``Disease``.
        gliner_label: Natural-language label offered to the GLiNER extractor.
        description: Plain-language meaning, shown to LLMs and text-to-Cypher.
        key_property: Node property that identifies an entity of this type.
        extractable: Whether text extractors may produce this type. Set it to
            ``False`` for types that only structured graph data should supply.
    """

    label: Annotated[str, Dedup()]
    gliner_label: str
    description: str
    key_property: str = "name"
    extractable: bool = True
    metadata: MetaData = {
        "index_fields": ["gliner_label"],
        "identity_fields": ["label"],
    }


class RelationType(DataPoint):
    """A type definition for medical relations (config node, not extracted relation).

    Identity = (type,).

    Attributes:
        type: Relation type name, such as ``TREATS``.
        glirel_label: Natural-language label offered to the GLiREL extractor.
            An empty string keeps the type away from GLiREL; the LLM extractor
            can still produce it.
        head_labels: Entity labels allowed as the relation head.
        tail_labels: Entity labels allowed as the relation tail.
        properties: Names of properties an edge of this type may carry.
        extractable: Whether text extractors may produce this type.
        description: Plain-language meaning, shown to LLMs and text-to-Cypher.
    """

    type: Annotated[str, Dedup()]
    glirel_label: str
    head_labels: list[str] = Field(default_factory=list)
    tail_labels: list[str] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)
    extractable: bool = True
    description: str = ""
    metadata: MetaData = {"index_fields": ["glirel_label"], "identity_fields": ["type"]}


class OntologyEntityRecord(TypedDict):
    """Row that mirrors one entity type of a schema into a graph."""

    label: str
    description: str
    gliner_label: str
    extractable: bool


class OntologyPairRecord(TypedDict):
    """Row that mirrors one allowed ``(head, type, tail)`` triple into a graph."""

    type: str
    head: str
    tail: str
    glirel_label: str
    extractable: bool


@dataclass(frozen=True)
class RelationGroup:
    """One relation type with every endpoint pair the schema allows for it.

    Attributes:
        type: Relation type name.
        pairs: Allowed ``(head_label, tail_label)`` pairs, merged across every
            ``RelationType`` that shares ``type``.
        description: Description of the first ``RelationType`` declared for
            ``type``.
    """

    type: str
    pairs: frozenset[tuple[str, str]]
    description: str


class GraphSchema(BaseModel):
    """Runtime-injectable KG schema — entity and relation type definitions.

    Used by extractors, resolvers, community summarizers, text2cypher, and agents.

    Attributes:
        entity_types: Entity type definitions. Labels must be unique.
        relation_types: Relation type definitions. A type may appear more than
            once to allow several endpoint pairs.
        version: Release identifier of the ontology, such as a data release id.
            Schemas that do not track a release use ``"unversioned"``.
        mode: How strictly extracted facts must follow this schema. ``open``
            keeps everything, ``warn`` keeps and reports violations, and
            ``strict`` drops them.
    """

    entity_types: list[EntityType] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)
    version: str = "unversioned"
    mode: OntologyMode = "open"

    @model_validator(mode="after")
    def _validate_labels(self) -> GraphSchema:
        """Reject duplicate labels and warn about undeclared relation endpoints."""
        labels = [et.label for et in self.entity_types]
        duplicate_labels = sorted({x for x in labels if labels.count(x) > 1})
        if duplicate_labels:
            raise ValueError(f"duplicate entity labels: {duplicate_labels}")
        text_labels = [et.gliner_label for et in self.entity_types if et.extractable]
        shared = sorted({x for x in text_labels if text_labels.count(x) > 1})
        if shared:
            raise ValueError(f"extractable entity types share a gliner_label: {shared}")
        types_by_glirel_label: dict[str, set[str]] = {}
        for rt in self.extractable_relation_types():
            if rt.glirel_label:
                types_by_glirel_label.setdefault(rt.glirel_label, set()).add(rt.type)
        conflicting = sorted(
            label for label, types in types_by_glirel_label.items() if len(types) > 1
        )
        if conflicting:
            raise ValueError(
                f"extractable relation types share a glirel_label: {conflicting}"
            )
        endpoints = {
            label
            for rt in self.relation_types
            for label in (*rt.head_labels, *rt.tail_labels)
        }
        undeclared = sorted(endpoints - set(labels))
        if undeclared:
            logger.warning(
                "relation endpoints name undeclared entity labels: %s", undeclared
            )
        return self

    def gliner_labels(self) -> list[str]:
        """GLiNER label strings for all extractable entity types."""
        return [et.gliner_label for et in self.extractable_entity_types()]

    def glirel_labels(self) -> list[str]:
        """GLiREL label strings for extractable relation types with a label.

        A label shared by several relation types, such as one type declared per
        endpoint pair, appears once, at its first position.
        """
        labels = (
            rt.glirel_label
            for rt in self.extractable_relation_types()
            if rt.glirel_label
        )
        return list(dict.fromkeys(labels))

    @property
    def gliner_to_label(self) -> dict[str, str]:
        """Map GLiNER label → entity type label (for post-extraction normalization)."""
        return {et.gliner_label: et.label for et in self.extractable_entity_types()}

    @property
    def glirel_to_type(self) -> dict[str, str]:
        """Map GLiREL label → relation type (for post-extraction normalization)."""
        return {
            rt.glirel_label: rt.type
            for rt in self.extractable_relation_types()
            if rt.glirel_label
        }

    def entity_type_names(self) -> set[str]:
        """Return the set of all entity type labels."""
        return {et.label for et in self.entity_types}

    def relation_type_names(self) -> set[str]:
        """Return the set of all relation type names."""
        return {rt.type for rt in self.relation_types}

    def extractable_entity_types(self) -> list[EntityType]:
        """Entity types that text extractors may produce."""
        return [et for et in self.entity_types if et.extractable]

    def extractable_relation_types(self) -> list[RelationType]:
        """Relation types that text extractors may produce.

        A type is left out when it is not extractable or when every head label or
        every tail label belongs to a non-extractable entity type.
        """
        hidden = self._non_extractable_labels()

        def all_hidden(labels: list[str]) -> bool:
            return bool(labels) and set(labels) <= hidden

        return [
            rt
            for rt in self.relation_types
            if rt.extractable
            and not all_hidden(rt.head_labels)
            and not all_hidden(rt.tail_labels)
        ]

    def extractable_entity_type_names(self) -> set[str]:
        """Return the labels of the extractable entity types."""
        return {et.label for et in self.extractable_entity_types()}

    def extractable_relation_type_names(self) -> set[str]:
        """Return the names of the extractable relation types."""
        return {rt.type for rt in self.extractable_relation_types()}

    def relation_groups(self) -> list[RelationGroup]:
        """Return one entry per distinct relation type, in declaration order."""
        pairs: dict[str, set[tuple[str, str]]] = {}
        descriptions: dict[str, str] = {}
        for rt in self.relation_types:
            descriptions.setdefault(rt.type, rt.description)
            pairs.setdefault(rt.type, set()).update(
                (head, tail) for head in rt.head_labels for tail in rt.tail_labels
            )
        return [
            RelationGroup(name, frozenset(group_pairs), descriptions[name])
            for name, group_pairs in pairs.items()
        ]

    def entity_prompt_block(self) -> str:
        """Serialized extractable entity type list for LLM/BAML prompt injection."""
        return "\n".join(
            f"- {et.label}: {et.description}" for et in self.extractable_entity_types()
        )

    def relation_prompt_block(self) -> str:
        """Serialized extractable relation type list for LLM/BAML prompt injection."""
        lines = []
        for rt in self.extractable_relation_types():
            heads = ", ".join(rt.head_labels)
            tails = ", ".join(rt.tail_labels)
            line = f"- {rt.type} [{heads} → {tails}]: {rt.glirel_label}"
            if rt.description and rt.description != rt.glirel_label:
                line += f" - {rt.description}"
            lines.append(line)
        return "\n".join(lines)

    def relation_guide(self) -> str:
        """Return each extractable relation type with its allowed endpoint pairs."""
        allowed_pairs: dict[str, set[tuple[str, str]]] = {}
        for head, relation_type, tail in self.allowed_pairs(extractable_only=True):
            allowed_pairs.setdefault(relation_type, set()).add((head, tail))

        lines = []
        for group in self.relation_groups():
            pairs = sorted(allowed_pairs.get(group.type, set()))
            if not pairs:
                continue
            pair_text = ", ".join(f"{head} -> {tail}" for head, tail in pairs)
            description = f": {group.description}" if group.description else ""
            lines.append(f"- {group.type} [{pair_text}]{description}")
        return "\n".join(lines)

    def as_text(self) -> str:
        """Full schema string for text2cypher / BAML dynamic enum injection.

        Describes the generic ``:RELATES_TO {type}`` pattern with one line per
        distinct relation type and its allowed endpoint pairs, so text2cypher can
        generate accurate Cypher for the generic-edge model.
        """
        lines = ["Graph schema:", "Nodes (each is :Entity plus its label):"]
        for et in self.entity_types:
            lines.append(f"  :{et.label} - {et.description}")
        lines.append(
            "Relationships (all are :RELATES_TO with a `type` property; head -> tail):"
        )
        for group in self.relation_groups():
            pairs = ", ".join(f"{head}->{tail}" for head, tail in sorted(group.pairs))
            line = f"  type '{group.type}': {pairs}"
            if group.description:
                line += f" - {group.description}"
            lines.append(line)
        lines.append(f"Valid type values: {sorted(self.relation_type_names())}")
        return "\n".join(lines)

    def allowed_pairs(
        self, *, extractable_only: bool = False
    ) -> set[tuple[str, str, str]]:
        """All ``(head_label, relation_type, tail_label)`` triples this schema permits.

        Args:
            extractable_only: Keep only triples that text extractors may produce:
                extractable relation types between extractable entity types.
        """
        hidden = self._non_extractable_labels() if extractable_only else set()
        relation_types = (
            self.extractable_relation_types()
            if extractable_only
            else self.relation_types
        )
        return {
            (h, rt.type, t)
            for rt in relation_types
            for h in rt.head_labels
            for t in rt.tail_labels
            if h not in hidden and t not in hidden
        }

    def fingerprint(self, *, include_mode: bool = True) -> str:
        """Return a short stable hash of the schema definition.

        The hash covers the release version and every entity and relation
        definition, and ignores per-instance fields such as ids and timestamps.
        Two schemas with the same definition have the same fingerprint.

        Args:
            include_mode: Whether ``mode`` is part of the hash. Pass ``False`` to
                compare definitions regardless of enforcement mode.
        """
        definition = {
            "version": self.version,
            "entities": sorted(
                (
                    {
                        "label": et.label,
                        "gliner_label": et.gliner_label,
                        "description": et.description,
                        "key_property": et.key_property,
                        "extractable": et.extractable,
                    }
                    for et in self.entity_types
                ),
                key=lambda d: d["label"],
            ),
            "relations": sorted(
                (
                    {
                        "type": rt.type,
                        "glirel_label": rt.glirel_label,
                        "head_labels": sorted(rt.head_labels),
                        "tail_labels": sorted(rt.tail_labels),
                        "properties": sorted(rt.properties),
                        "extractable": rt.extractable,
                        "description": rt.description,
                    }
                    for rt in self.relation_types
                ),
                key=lambda d: (d["type"], d["head_labels"], d["tail_labels"]),
            ),
        }
        if include_mode:
            definition["mode"] = self.mode
        payload = json.dumps(definition, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _non_extractable_labels(self) -> set[str]:
        """Labels of entity types that text extractors must not produce."""
        return {et.label for et in self.entity_types if not et.extractable}
