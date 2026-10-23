"""Tests for the public-API data models.

Covers JSON round-trips for the registry/job records, JobProgress arithmetic,
the AgentEvent discriminated union, and connection defaults. These models are
pure (no I/O) so the tests need no mocking.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from am_diag.common.data_models import (
    AgentEvent,
    ClinicalAnswer,
    GraphConnection,
    GraphRecord,
    JobConfig,
    JobItem,
    JobProgress,
    JobRecord,
    Message,
    ProjectRecord,
    VectorConnection,
)
from am_diag.common.data_models.agent_events import FinalEvent, PlanEvent, TokenEvent
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA


class TestConnections:
    """Tests for GraphConnection and VectorConnection descriptors."""

    def test_graph_connection_defaults_database(self):
        """Verify database defaults to neo4j."""
        conn = GraphConnection(uri="bolt://h:7687", user_env="U", password_env="P")
        assert conn.database == "neo4j"
        assert conn.backend == "neo4j"

    @pytest.mark.parametrize("scheme", ["redis", "rediss"])
    def test_falkordb_allows_no_credentials(self, scheme):
        """Allow unauthenticated Redis connections in persisted records."""
        conn = GraphConnection(backend="falkordb", uri=f"{scheme}://h:6379")
        assert conn.user_env is None
        assert conn.password_env is None

    @pytest.mark.parametrize(
        ("backend", "uri"), [("neo4j", "redis://h"), ("falkordb", "bolt://h")]
    )
    def test_backend_rejects_wrong_uri(self, backend, uri):
        """Reject a connection whose URI belongs to the other backend."""
        with pytest.raises(ValueError, match="backend cannot use"):
            GraphConnection(backend=backend, uri=uri, user_env="U", password_env="P")

    def test_neo4j_requires_credential_refs(self):
        """Keep credential env references mandatory for Neo4j."""
        with pytest.raises(ValueError, match="requires user_env and password_env"):
            GraphConnection(uri="bolt://h")

    @pytest.mark.parametrize("userinfo", ["user:secret", "user", ":secret", ""])
    def test_falkordb_rejects_persisted_inline_credentials(self, userinfo):
        """Reject userinfo when loading a serialized connection descriptor."""
        payload = (
            '{"backend":"falkordb","uri":"rediss://'
            + userinfo
            + '@host:6379","user_env":"U","password_env":"P"}'
        )
        with pytest.raises(ValueError, match="use user_env and password_env instead"):
            GraphConnection.model_validate_json(payload)

    def test_vector_connection_default_collections(self):
        """Verify the four per-kind collection names default correctly."""
        conn = VectorConnection(url_env="URL")
        assert conn.backend == "qdrant"
        assert set(conn.collections) == {"entity", "relation", "chunk", "community"}
        assert conn.collections["entity"] == "am_diag_entity"
        assert conn.dim == 384


class TestGraphRecord:
    """Tests for the persisted GraphRecord descriptor."""

    def _make(self, **overrides) -> GraphRecord:
        kwargs = {
            "graph_id": "med",
            "neo4j": GraphConnection(
                uri="bolt://h:7687", user_env="U", password_env="P"
            ),
            "vector": VectorConnection(url_env="URL"),
        }
        kwargs.update(overrides)
        return GraphRecord(**kwargs)

    def test_default_schema_is_medical(self):
        """Verify the default graph schema is the medical GraphRAG schema."""
        rec = self._make()
        assert rec.graph_schema == MEDICAL_GRAPHRAG_SCHEMA

    def test_json_round_trip(self):
        """Verify a GraphRecord survives a JSON dump/load unchanged."""
        rec = self._make(name="Medical", description="d")
        loaded = GraphRecord.model_validate_json(rec.model_dump_json())
        assert loaded.graph_id == "med"
        assert loaded.neo4j.user_env == "U"
        assert loaded.vector.collections == rec.vector.collections
        assert loaded.graph_schema.entity_type_names() == (
            rec.graph_schema.entity_type_names()
        )

    def test_secrets_not_in_serialized_record(self):
        """Verify only env-var names are persisted, never secret values."""
        rec = GraphRecord.model_validate(
            {
                "graph_id": "med",
                "neo4j": {
                    "uri": "bolt://h:7687",
                    "user_env": "U",
                    "password_env": "P",
                    "password": "sentinel-secret",
                },
                "vector": {"url_env": "URL", "api_key": "sentinel-secret"},
            }
        )
        dumped = rec.model_dump_json()
        assert "sentinel-secret" not in dumped
        assert "user_env" in dumped
        assert "password_env" in dumped

    @pytest.mark.parametrize("graph_id", ["../escape", "a/b", ".hidden", "", "a b"])
    def test_rejects_unsafe_graph_id(self, graph_id):
        """Verify ids that are unsafe as cache file names are rejected."""
        with pytest.raises(ValueError, match="graph_id"):
            self._make(graph_id=graph_id)

    def test_partial_collections_filled_with_defaults(self):
        """Verify omitted collection kinds fall back to their default names."""
        conn = VectorConnection(url_env="URL", collections={"entity": "custom"})
        assert conn.collections["entity"] == "custom"
        assert conn.collections["chunk"] == "am_diag_chunk"


class TestProjectRecord:
    """Tests for the persisted ProjectRecord descriptor."""

    def test_json_round_trip(self):
        """Verify ProjectRecord round-trips with graph membership."""
        rec = ProjectRecord(project_id="clinical", graph_ids=["med", "drug"])
        loaded = ProjectRecord.model_validate_json(rec.model_dump_json())
        assert loaded.project_id == "clinical"
        assert loaded.graph_ids == ["med", "drug"]
        assert loaded.members == []

    def test_rejects_duplicate_graph_ids(self):
        """Verify a project cannot list the same graph twice."""
        with pytest.raises(ValueError, match="duplicates"):
            ProjectRecord(project_id="clinical", graph_ids=["med", "med"])

    def test_rejects_credentials_in_agent_settings(self):
        """Verify agent settings cannot carry an API key into the registry."""
        with pytest.raises(ValueError, match="api_key"):
            ProjectRecord(project_id="clinical", agent_settings={"api_key": "x"})

    def test_rejects_non_json_agent_settings(self):
        """Verify agent settings must be JSON-serializable."""
        with pytest.raises(ValueError, match="JSON"):
            ProjectRecord(project_id="clinical", agent_settings={"model": object()})


class TestJobModels:
    """Tests for JobProgress, JobItem, and JobRecord."""

    def test_progress_percent_zero_total(self):
        """Verify percent_complete is 0 when there are no items."""
        assert JobProgress().percent_complete == 0.0

    def test_progress_percent_counts_resolved(self):
        """Verify percent_complete counts succeeded, failed, and skipped."""
        prog = JobProgress(total=10, succeeded=6, failed=2, skipped=1, processing=1)
        assert prog.percent_complete == pytest.approx(90.0)

    def test_job_record_round_trip(self):
        """Verify a JobRecord with nested config/progress round-trips."""
        rec = JobRecord(
            job_id="j1",
            kind="ingestion",
            graph_id="med",
            config=JobConfig(extract_concurrency=4),
            progress=JobProgress(total=3, succeeded=3),
        )
        loaded = JobRecord.model_validate_json(rec.model_dump_json())
        assert loaded.kind == "ingestion"
        assert loaded.config.extract_concurrency == 4
        assert loaded.progress.percent_complete == pytest.approx(100.0)

    def test_job_item_defaults_pending(self):
        """Verify a new JobItem starts pending with no error."""
        item = JobItem(item_id="a", sequence_index=0, kind="chunk")
        assert item.status == "pending"
        assert item.error is None


class TestAgentEvents:
    """Tests for the AgentEvent discriminated union."""

    def test_discriminator_parses_token_event(self):
        """Verify the union resolves a token event from its type tag."""
        adapter = TypeAdapter(AgentEvent)
        ev = adapter.validate_python({"type": "token", "text": "hi"})
        assert isinstance(ev, TokenEvent)
        assert ev.text == "hi"

    def test_discriminator_parses_final_event(self):
        """Verify the union resolves a final event carrying a ClinicalAnswer."""
        adapter = TypeAdapter(AgentEvent)
        ev = adapter.validate_python(
            {"type": "final", "answer": ClinicalAnswer(answer="x").model_dump()}
        )
        assert isinstance(ev, FinalEvent)
        assert ev.answer.answer == "x"

    def test_plan_event_default_type(self):
        """Verify the plan event tag defaults correctly."""
        ev = PlanEvent(plan={"sub_questions": [], "notes": ""})
        assert ev.type == "plan"


class TestMessage:
    """Tests for the multi-turn Message model."""

    def test_defaults_to_user_role(self):
        """Verify role defaults to user."""
        assert Message(content="hi").role == "user"
