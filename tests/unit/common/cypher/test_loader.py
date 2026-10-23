"""Cypher loader — safe identifier validation, file loading, and render substitution."""

from __future__ import annotations

import pytest

from am_diag.common.cypher.loader import load, render, safe_identifier


class TestCypherLoader:
    """Safe identifier validation, Cypher file loading, and render substitution."""

    def test_valid_identifier_returned_with_backticks(self):
        """Valid identifier is wrapped in backticks."""
        assert safe_identifier("Disease") == "`Disease`"

    def test_valid_with_underscore(self):
        """Identifier with underscore is accepted."""
        assert safe_identifier("Drug_Class") == "`Drug_Class`"

    def test_rejects_space(self):
        """Identifier with space is rejected."""
        with pytest.raises(ValueError, match="Unsafe"):
            safe_identifier("Bad Label")

    def test_rejects_leading_digit(self):
        """Identifier starting with digit is rejected."""
        with pytest.raises(ValueError, match="Unsafe"):
            safe_identifier("1BadLabel")

    def test_rejects_semicolon_injection(self):
        """Identifier with semicolon is rejected as injection risk."""
        with pytest.raises(ValueError, match="Unsafe"):
            safe_identifier("Drug; DROP DATABASE")

    def test_rejects_backtick_injection(self):
        """Identifier containing backtick is rejected."""
        with pytest.raises(ValueError, match="Unsafe"):
            safe_identifier("`Drug`")

    def test_load_ingest_upsert_entities(self):
        """upsert_entities.cypher loads correctly."""
        text = load("ingest/upsert_entities")
        assert "MERGE (n:Entity {id: rec.id})" in text
        assert "${Label}" in text

    def test_load_ingest_upsert_relations(self):
        """upsert_relations.cypher loads correctly."""
        text = load("ingest/upsert_relations")
        assert "RELATES_TO" in text
        assert "MERGE (h)-[r:RELATES_TO {id: rec.id}]->(t)" in text

    def test_load_link_next_chunk(self):
        """link_next_chunk.cypher loads correctly."""
        text = load("ingest/link_next_chunk")
        assert "NEXT_CHUNK" in text
        assert "chunk_index" in text

    def test_load_bfs_expand(self):
        """bfs_expand.cypher loads correctly."""
        text = load("retrieval/bfs_expand")
        assert "RELATES_TO*1..${depth}" in text
        assert "$seed_ids" in text

    def test_load_caches_result(self):
        """lru_cache returns same object identity on second call."""
        t1 = load("ingest/upsert_documents")
        t2 = load("ingest/upsert_documents")
        assert t1 is t2

    def test_load_missing_file_raises(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load("nonexistent/query")

    def test_all_expected_cypher_files_exist(self):
        """Every file referenced by the client must exist in the package."""
        names = [
            "ingest/upsert_documents",
            "ingest/upsert_chunks",
            "ingest/upsert_entities",
            "ingest/upsert_relations",
            "ingest/link_part_of",
            "ingest/link_next_chunk",
            "ingest/link_has_entity",
            "community/upsert_communities",
            "community/link_in_community",
            "community/link_parent_community",
            "community/upsert_reports",
            "community/link_has_report",
            "community/member_subgraph",
            "retrieval/bfs_expand",
            "retrieval/hydrate_entities",
            "retrieval/hydrate_relations",
            "retrieval/community_members",
            "retrieval/chunk_neighbors",
            "schema/constraints",
            "schema/indexes",
        ]
        for name in names:
            text = load(name)
            assert text.strip(), f"Empty .cypher file: {name}"

    def test_render_substitutes_label(self):
        """Render substitutes ${Label} placeholder."""
        query = render("ingest/upsert_entities", Label="Disease")
        assert "`Disease`" in query
        assert "${Label}" not in query

    def test_render_rejects_unsafe_label(self):
        """Render rejects unsafe label values."""
        with pytest.raises(ValueError):
            render("ingest/upsert_entities", Label="Bad Label")

    def test_render_leaves_params_intact(self):
        """Cypher $params are not replaced by render."""
        query = render("ingest/upsert_entities", Label="Drug")
        # $records is a Cypher parameter, not a render-time identifier — must stay
        assert "$records" in query
