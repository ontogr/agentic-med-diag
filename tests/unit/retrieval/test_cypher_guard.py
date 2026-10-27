"""Tests for generated-Cypher checks (am_diag/retrieval/cypher_guard.py)."""

from __future__ import annotations

import pytest

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType
from am_diag.retrieval.cypher_guard import guard_cypher


_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(label="Drug", gliner_label="Drug", description="A drug"),
        EntityType(label="Disease", gliner_label="Disease", description="A disease"),
    ],
    relation_types=[
        RelationType(
            type="INDICATION",
            glirel_label="INDICATION",
            head_labels=["Drug"],
            tail_labels=["Disease"],
        ),
    ],
)


def _guard(cypher: str, dialect: str = "neo4j"):
    return guard_cypher(
        cypher, _SCHEMA, dialect=dialect, default_limit=25, max_limit=100
    )


class TestGuardCypher:
    """Tests for guard_cypher."""

    def test_adds_limit_when_missing(self) -> None:
        """A query with no LIMIT gets the default one."""
        guarded = _guard("MATCH (n:Entity) RETURN n")

        assert guarded.errors == []
        assert guarded.cypher == "MATCH (n:Entity) RETURN n\nLIMIT 25"

    def test_lowers_limit_above_the_maximum(self) -> None:
        """A LIMIT above the maximum is lowered to it."""
        guarded = _guard("MATCH (n:Entity) RETURN n LIMIT 5000")

        assert guarded.cypher == "MATCH (n:Entity) RETURN n LIMIT 100"

    def test_keeps_limit_below_the_maximum(self) -> None:
        """A LIMIT within bounds is left alone."""
        guarded = _guard("MATCH (n:Entity) RETURN n LIMIT 5")

        assert guarded.cypher == "MATCH (n:Entity) RETURN n LIMIT 5"

    def test_limit_inside_a_subquery_does_not_bound_the_outer_query(self) -> None:
        """A subquery LIMIT stays as written and the outer query still gets one."""
        cypher = (
            "MATCH (n:Entity) CALL { WITH n MATCH (n)-[:RELATES_TO]-(m:Entity) "
            "RETURN m LIMIT 3 } RETURN n.name AS name"
        )

        guarded = _guard(cypher)

        assert guarded.errors == []
        assert guarded.cypher == cypher + "\nLIMIT 25"

    def test_limit_text_inside_a_string_is_neither_counted_nor_changed(self) -> None:
        """A quoted LIMIT is data: it neither satisfies nor is rewritten."""
        cypher = "MATCH (n:Entity) WHERE n.name = 'LIMIT 5000' RETURN n"

        guarded = _guard(cypher)

        assert guarded.cypher == cypher + "\nLIMIT 25"

    def test_lowers_only_the_limit_that_bounds_the_outer_query(self) -> None:
        """The outer LIMIT is capped even when a subquery has its own."""
        guarded = _guard(
            "MATCH (n:Entity) CALL { WITH n MATCH (n)--(m:Entity) RETURN m LIMIT 3 } "
            "RETURN n LIMIT 5000"
        )

        assert guarded.cypher.endswith("RETURN n LIMIT 100")
        assert "RETURN m LIMIT 3 }" in guarded.cypher

    def test_accepts_schema_labels_and_relation_type(self) -> None:
        """Typed labels and a valid relation type produce no errors."""
        guarded = _guard(
            "MATCH (d:Entity:Drug)-[r:RELATES_TO {type: 'INDICATION'}]->"
            "(x:Entity:Disease) RETURN d.name AS drug LIMIT 5"
        )

        assert guarded.errors == []

    @pytest.mark.parametrize(
        ("cypher", "expected"),
        [
            ("MATCH (n:Entity) SET n.x = 1 RETURN n LIMIT 5", "write clause"),
            ("MATCH (n:Entity) DETACH DELETE n", "write clause"),
            ("MATCH (n:Zzz) RETURN n LIMIT 5", "Unknown node labels ['Zzz']"),
            (
                "MATCH (a)-[:TREATS_X]->(b) RETURN a LIMIT 5",
                "Unknown relationship types ['TREATS_X']",
            ),
            (
                "MATCH (a)-[r:RELATES_TO {type: 'ZZZ_REL'}]->(b) RETURN a LIMIT 5",
                "Unknown relation type values ['ZZZ_REL']",
            ),
            (
                "MATCH (a)-[r:RELATES_TO]->(b) "
                "WHERE r.type IN ['INDICATION', 'ZZZ_REL'] RETURN a LIMIT 5",
                "Unknown relation type values ['ZZZ_REL']",
            ),
        ],
    )
    def test_reports_errors_for_the_model_to_fix(
        self, cypher: str, expected: str
    ) -> None:
        """Each problem is named in an error the model can retry on."""
        guarded = _guard(cypher)

        assert any(expected in error for error in guarded.errors)

    @pytest.mark.parametrize(
        "limit", ["$n", "toInteger('5000')", "5 + 5000", "size([1, 2])"]
    )
    def test_rejects_limits_that_are_not_whole_numbers(self, limit: str) -> None:
        """An expression or parameter could return more rows than the cap allows."""
        guarded = _guard(f"MATCH (n:Entity) RETURN n LIMIT {limit}")

        assert any("LIMIT must be a whole number" in e for e in guarded.errors)

    def test_property_named_limit_is_not_a_limit_clause(self) -> None:
        """A `limit` property does not confuse the whole-number check."""
        guarded = _guard("MATCH (n:Entity) RETURN n.limit AS cap LIMIT 5")

        assert guarded.errors == []

    def test_ignores_keywords_inside_string_literals(self) -> None:
        """A value such as 'SET' is data, not a write clause."""
        guarded = _guard("MATCH (n:Entity) WHERE n.name = 'SET' RETURN n LIMIT 5")

        assert guarded.errors == []

    def test_path_functions_are_an_error_on_falkordb_only(self) -> None:
        """FalkorDB gets feedback about path syntax; Neo4j accepts it."""
        cypher = "MATCH p = shortestPath((a:Entity)-[*]-(b:Entity)) RETURN p LIMIT 5"

        assert _guard(cypher, "neo4j").errors == []
        assert any("shortestPath" in e for e in _guard(cypher, "falkordb").errors)
