"""Enforce a GraphSchema on entities and relations extracted from text."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema, OntologyMode


logger = logging.getLogger(__name__)

_MAX_SAMPLES_PER_TYPE = 3


@dataclass
class OntologyReport:
    """Counts of how extracted facts relate to the schema.

    Attributes:
        entities_kept: Entities that stay in the output, counted by label.
        entities_violating: Entities whose label is not in the schema, counted by
            label. They are dropped in ``strict`` mode and kept in ``warn`` mode.
        relations_kept: Relations that stay in the output, counted by type.
            Relations with swapped endpoints are included.
        relations_swapped: Relations kept with head and tail exchanged, counted by
            type.
        relations_violating: Relations that break the schema, counted by type.
            This includes relations dropped because an endpoint entity was dropped.
        samples: Up to three distinct examples per relation type of violating
            relations, formatted as ``Head-[TYPE]->Tail``.
    """

    entities_kept: Counter[str] = field(default_factory=Counter)
    entities_violating: Counter[str] = field(default_factory=Counter)
    relations_kept: Counter[str] = field(default_factory=Counter)
    relations_swapped: Counter[str] = field(default_factory=Counter)
    relations_violating: Counter[str] = field(default_factory=Counter)
    samples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def conformance(self) -> float:
        """Share of counted relations that match the schema without a swap.

        Returns ``1.0`` when no relations were counted.
        """
        total = self.relations_kept.total() + self.relations_violating.total()
        if total == 0:
            return 1.0
        clean = self.relations_kept.total() - self.relations_swapped.total()
        return clean / total


def _endpoint_labels(
    relation: Relation, label_by_id: dict[UUID, str]
) -> tuple[str, str] | None:
    """Return the head and tail labels, or ``None`` if either endpoint is unknown."""
    if relation.head_id is None or relation.tail_id is None:
        return None
    head_label = label_by_id.get(relation.head_id)
    tail_label = label_by_id.get(relation.tail_id)
    if head_label is None or tail_label is None:
        return None
    return head_label, tail_label


def _swap_endpoints(relation: Relation) -> Relation:
    """Return a copy of *relation* with head and tail exchanged.

    The copy is rebuilt the same way ``ComposedExtractor`` links endpoints, so its
    id is derived from the new head, tail, and type.
    """
    data = relation.model_dump(exclude={"id"})
    data["head_id"], data["tail_id"] = data["tail_id"], data["head_id"]
    data["head_name"], data["tail_name"] = data["tail_name"], data["head_name"]
    return Relation(**data)


def _add_sample(report: OntologyReport, relation_type: str, sample: str) -> None:
    """Record *sample* unless it is already present or the type has enough samples."""
    samples = report.samples.setdefault(relation_type, [])
    if sample not in samples and len(samples) < _MAX_SAMPLES_PER_TYPE:
        samples.append(sample)


def _log_report(report: OntologyReport, mode: OntologyMode) -> None:
    """Log the violations and endpoint swaps recorded in *report*."""
    if report.entities_violating:
        logger.warning(
            "entities with labels outside the %s ontology: %s",
            mode,
            dict(sorted(report.entities_violating.items())),
        )
    for relation_type, count in sorted(report.relations_violating.items()):
        logger.warning(
            "%d %s relations do not match the %s ontology; examples: %s",
            count,
            relation_type,
            mode,
            report.samples.get(relation_type, []),
        )
    if report.relations_swapped:
        logger.info(
            "swapped endpoints of relations to match the schema: %s",
            dict(sorted(report.relations_swapped.items())),
        )


class OntologyPruner:
    """Apply a GraphSchema to extracted entities and relations.

    The schema ``mode`` sets what happens to facts that break the schema. ``open``
    keeps everything, ``warn`` keeps and reports violations, and ``strict`` drops
    them. A relation allowed only with its endpoints reversed is swapped into the
    schema direction in ``warn`` and ``strict`` modes.
    """

    def __init__(self, schema: GraphSchema) -> None:
        """Capture the mode and the entity and relation definitions of *schema*."""
        self._mode = schema.mode
        self._entity_labels = schema.entity_type_names()
        self._allowed_pairs = schema.allowed_pairs()

    def prune(
        self, entities: list[Entity], relations: list[Relation]
    ) -> tuple[list[Entity], list[Relation], OntologyReport]:
        """Apply the schema to *entities* and *relations*.

        A relation is dropped when it touches a dropped entity, or when its
        endpoint labels are allowed in neither direction. A relation whose
        endpoint id is ``None`` or matches no entity in *entities* is kept
        unchanged.

        Args:
            entities: Entities extracted from text.
            relations: Relations extracted from text. Endpoints are matched to
                *entities* by ``id``.

        Returns:
            The kept entities, the kept relations, and a report of the counts.
            In ``open`` mode the input lists are returned as they are.
        """
        report = OntologyReport()
        if self._mode == "open":
            report.entities_kept.update(entity.label for entity in entities)
            report.relations_kept.update(relation.type for relation in relations)
            return entities, relations, report

        kept_entities: list[Entity] = []
        dropped_entity_ids: set[UUID] = set()
        for entity in entities:
            if self._check_entity(entity, report):
                kept_entities.append(entity)
            else:
                dropped_entity_ids.add(entity.id)
        label_by_id = {entity.id: entity.label for entity in kept_entities}

        kept_relations: list[Relation] = []
        for relation in relations:
            kept = self._check_relation(
                relation, label_by_id, dropped_entity_ids, report
            )
            if kept is not None:
                kept_relations.append(kept)

        _log_report(report, self._mode)
        return kept_entities, kept_relations, report

    def _check_entity(self, entity: Entity, report: OntologyReport) -> bool:
        """Count *entity* and return whether the schema mode keeps it."""
        if entity.label in self._entity_labels:
            report.entities_kept[entity.label] += 1
            return True
        report.entities_violating[entity.label] += 1
        return self._mode == "warn"

    def _check_relation(
        self,
        relation: Relation,
        label_by_id: dict[UUID, str],
        dropped_entity_ids: set[UUID],
        report: OntologyReport,
    ) -> Relation | None:
        """Count *relation* and return the relation to keep, or ``None``."""
        touches_dropped = (
            relation.head_id in dropped_entity_ids
            or relation.tail_id in dropped_entity_ids
        )
        if touches_dropped:
            report.relations_violating[relation.type] += 1
            return None
        labels = _endpoint_labels(relation, label_by_id)
        if labels is None:
            report.relations_kept[relation.type] += 1
            return relation
        head_label, tail_label = labels
        if (head_label, relation.type, tail_label) in self._allowed_pairs:
            report.relations_kept[relation.type] += 1
            return relation
        if (tail_label, relation.type, head_label) in self._allowed_pairs:
            report.relations_kept[relation.type] += 1
            report.relations_swapped[relation.type] += 1
            return _swap_endpoints(relation)
        report.relations_violating[relation.type] += 1
        sample = f"{head_label}-[{relation.type}]->{tail_label}"
        _add_sample(report, relation.type, sample)
        if self._mode == "warn":
            return relation
        return None
