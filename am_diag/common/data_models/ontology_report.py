"""OntologyReport: counts of how extracted facts relate to a graph schema."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class OntologyReport:
    """Counts of how extracted facts relate to the schema.

    Attributes:
        entities_kept: Entities that stay in the output, counted by label.
        entities_violating: Entities whose label is not an extractable entity type of
            the schema, counted by
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
