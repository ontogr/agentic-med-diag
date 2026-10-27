"""Tests for expander seeds (am_diag/retrieval/methods/base.py)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.search import CypherRow, Fact, SearchResult
from am_diag.retrieval.methods.base import collect_seeds


def _result(item: Any) -> SearchResult:
    return SearchResult(item=item, score=1.0, source="test")


class TestCollectSeeds:
    """Tests for collect_seeds."""

    def test_collects_entities_from_every_item_type_in_rank_order(self) -> None:
        """Entities, relation endpoints and fact endpoints all seed entities."""
        entity = Entity(id=uuid4(), name="a", label="Drug")
        head, tail = uuid4(), uuid4()
        relation = Relation(head_id=head, tail_id=tail, type="T")
        fact = Fact(
            head_id="d", head_name="d", type="T", tail_id=str(entity.id), tail_name="a"
        )

        seeds = collect_seeds([_result(entity), _result(relation), _result(fact)])

        assert seeds.entity_ids == [str(entity.id), str(head), str(tail), "d"]

    def test_collects_chunk_ids_without_repeats(self) -> None:
        """A chunk found by two methods seeds once."""
        chunk = Chunk(
            id=uuid4(),
            text="t",
            document_id=uuid4(),
            document_source="s",
            chunk_index=0,
            chunk_size=1,
            cut_type="x",
        )

        seeds = collect_seeds([_result(chunk), _result(chunk)])

        assert seeds.chunk_ids == [str(chunk.id)]

    def test_cypher_row_seeds_the_ids_it_contains(self) -> None:
        """UUID strings in a Cypher row seed entities; other values do not."""
        entity_id = str(uuid4())
        row = CypherRow(values={"id": entity_id, "name": "asthma", "n": 3})

        seeds = collect_seeds([_result(row)])

        assert seeds.entity_ids == [entity_id]

    def test_limits_each_list(self) -> None:
        """Only the first `limit` ids are kept."""
        entities = [Entity(id=uuid4(), name=f"e{i}", label="Drug") for i in range(5)]

        seeds = collect_seeds([_result(e) for e in entities], limit=2)

        assert seeds.entity_ids == [str(e.id) for e in entities[:2]]
