"""Unified retrieval result model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation


class CypherRow(BaseModel):
    """One raw row returned by a generated Cypher query.

    Generated queries return arbitrary columns, so a row cannot be mapped onto
    ``Entity``, ``Relation``, ``Chunk`` or ``CommunityReport``.
    """

    values: dict[str, Any]

    def __str__(self) -> str:
        """Render the row as ``column: value`` pairs."""
        return ", ".join(f"{key}: {value}" for key, value in self.values.items())


class Fact(BaseModel):
    """One edge, or a two-edge path, with the evidence behind it.

    Edges extracted from text carry a ``description`` and ``chunk_ids``. Edges
    loaded from a structured source have no sentence, so they carry the source
    ``structural_label`` and ``sources`` instead. For a two-edge path, ``type`` is
    ``"FIRST>SECOND"`` and ``via_name`` and ``via_id`` identify the middle entity.
    """

    head_id: str
    head_name: str
    type: str
    tail_id: str
    tail_name: str
    description: str | None = None
    structural_label: str | None = None
    sources: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    hops: int = 1
    via_name: str | None = None
    via_id: str | None = None

    def text(self) -> str:
        """Render the fact as one line the model can read."""
        if self.hops == 2 and self.via_name:
            first, _, second = self.type.partition(">")
            return (
                f"{self.head_name} \u2014[{first}]\u2192 {self.via_name} "
                f"\u2014[{second}]\u2192 {self.tail_name}"
            )
        line = f"{self.head_name} \u2014[{self.type}]\u2192 {self.tail_name}"
        if self.description:
            line += f": {self.description}"
        if self.sources:
            line += f" (sources: {', '.join(self.sources)})"
        return line

    def __str__(self) -> str:
        """Render the fact with :meth:`text`."""
        return self.text()


class Keywords(BaseModel):
    """Search keywords for one question.

    Attributes:
        llm: Clinical entity names and terms from the model, best first.
        simple: Content words from the question, in question order.
    """

    model_config = ConfigDict(frozen=True)

    llm: list[str] = Field(default_factory=list)
    simple: list[str] = Field(default_factory=list)


class GuardedCypher(BaseModel):
    """Generated Cypher after checking.

    Attributes:
        cypher: The query, with a ``LIMIT`` added or lowered when needed.
        errors: Problems the model should fix. Empty when the query may run.
    """

    model_config = ConfigDict(frozen=True)

    cypher: str
    errors: list[str]


class Seeds(BaseModel):
    """Ranked, de-duplicated ids that expanders start from."""

    model_config = ConfigDict(frozen=True)

    entity_ids: list[str]
    chunk_ids: list[str]


class SearchResult(BaseModel):
    """A single result from the retrieval layer, after hydration.

    Unifies the five stray result classes previously in retrieval/models.py.
    The `item` discriminated union allows consumers to inspect the concrete type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: Entity | Relation | Chunk | CommunityReport | CypherRow | Fact
    score: float
    source: str
    matched_on: str | None = None

    # Provenance for federated (cross-graph) search. None for single-graph
    # results; set to the originating graph_id when a Project fuses graphs.
    source_graph_id: str | None = None
