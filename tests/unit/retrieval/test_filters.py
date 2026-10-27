"""Tests for SearchFilters — typed metadata filtering for vector and graph searches."""

from __future__ import annotations

import pytest

from am_diag.retrieval.filters import SearchFilters


class TestSearchFiltersToPayloadFilter:
    """Tests for SearchFilters.to_payload_filter."""

    def test_none_fields_returns_none(self) -> None:
        """Verify no filters set returns None from payload filter conversion."""
        filters = SearchFilters()
        assert filters.to_payload_filter() is None

    def test_single_label_produces_scalar(self) -> None:
        """Verify a single label produces a scalar match value."""
        filters = SearchFilters(labels=["Drug"])
        result = filters.to_payload_filter()
        assert result == {"label": "Drug"}

    def test_multiple_labels_produces_list(self) -> None:
        """Verify multiple labels produce a list match value."""
        filters = SearchFilters(labels=["Drug", "Disease"])
        result = filters.to_payload_filter()
        assert result == {"label": ["Drug", "Disease"]}

    def test_relation_types(self) -> None:
        """Verify relation type filters produce correct payload filter."""
        filters = SearchFilters(relation_types=["TREATED_BY"])
        result = filters.to_payload_filter()
        assert result == {"type": "TREATED_BY"}

    def test_all_fields_combined(self) -> None:
        """Verify all filter fields combine into a single payload dict."""
        filters = SearchFilters(
            labels=["Drug"],
            relation_types=["TREATED_BY", "CAUSES"],
            document_sources=["pubmed"],
            community_level=1,
            properties={"verified": True},
        )
        result = filters.to_payload_filter()
        assert result is not None
        assert result["label"] == "Drug"
        assert result["type"] == ["TREATED_BY", "CAUSES"]
        assert result["document_source"] == "pubmed"
        assert result["level"] == 1
        assert result["verified"] is True


class TestSearchFiltersToCypherWhere:
    """Tests for SearchFilters.to_cypher_where."""

    def test_no_filters_returns_empty(self) -> None:
        """Verify no filters produce an empty Cypher WHERE clause."""
        filters = SearchFilters()
        clause, params = filters.to_cypher_where()
        assert clause == ""
        assert params == {}

    def test_labels_use_label_predicate(self) -> None:
        """Verify labels use the n:Label Cypher predicate syntax."""
        filters = SearchFilters(labels=["Drug", "Disease"])
        clause, params = filters.to_cypher_where()
        assert "n:Drug" in clause
        assert "n:Disease" in clause
        assert params == {}

    def test_relation_types(self) -> None:
        """Verify relation types become a parameterised type(n) filter."""
        filters = SearchFilters(relation_types=["TREATED_BY"])
        clause, params = filters.to_cypher_where()
        assert "type(n)" in clause
        assert "$relation_types" in clause
        assert params["relation_types"] == ["TREATED_BY"]

    def test_document_sources(self) -> None:
        """Verify document sources become a parameterised filter clause."""
        filters = SearchFilters(document_sources=["pubmed"])
        clause, params = filters.to_cypher_where()
        assert "document_source" in clause
        assert "$document_sources" in clause
        assert params["document_sources"] == ["pubmed"]

    def test_community_level(self) -> None:
        """Verify community level becomes a parameterised filter."""
        filters = SearchFilters(community_level=2)
        clause, params = filters.to_cypher_where()
        assert "level" in clause
        assert "$community_level" in clause
        assert params["community_level"] == 2

    def test_custom_var_name(self) -> None:
        """Verify a custom Cypher variable name is used in the clause."""
        filters = SearchFilters(community_level=1)
        clause, params = filters.to_cypher_where(var="entity")
        assert "entity.level" in clause
        assert "$community_level" in clause

    def test_properties(self) -> None:
        """Verify arbitrary property filters are included in the WHERE clause."""
        filters = SearchFilters(properties={"verified": True})
        clause, params = filters.to_cypher_where()
        assert "n.verified" in clause
        assert params["verified"] is True


class TestCypherIdentifierValidation:
    """Labels and property keys are written into Cypher, so unsafe ones fail."""

    @pytest.mark.parametrize("label", ["Drug) OR (n", "drug", "Dru g", "Drug`"])
    def test_rejects_unsafe_label(self, label: str) -> None:
        """A label that is not PascalCase never reaches the query text."""
        with pytest.raises(ValueError, match="label"):
            SearchFilters(labels=[label]).to_cypher_where()

    @pytest.mark.parametrize("key", ["x = 1 OR n.y", "a.b", "1abc", "k`"])
    def test_rejects_unsafe_property_key(self, key: str) -> None:
        """A property key that is not a plain identifier is refused."""
        with pytest.raises(ValueError, match="identifier"):
            SearchFilters(properties={key: 1}).to_cypher_where()
