"""Tests for the project-scoped agent answer surface.

The DeepAgents graph is replaced with a fake agent (mocked ainvoke), so answer()
is verified for both string and multi-turn message inputs without building the
real agent or touching any backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from am_diag.agents.settings import AgentSettings
from am_diag.api.graph import Graph
from am_diag.api.project import Project
from am_diag.api.registry import Registry
from am_diag.api.workspace import Workspace
from am_diag.common.data_models.agent import ClinicalAnswer
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.message import Message
from am_diag.common.data_models.project import ProjectRecord
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA


def _fake_graph(tmp_path) -> Graph:
    """Build a real Graph over fake dependencies, attachable under id 'med'."""
    return Graph(
        graph_store=SimpleNamespace(close=AsyncMock(), register_ontology=AsyncMock()),
        vector_store=SimpleNamespace(close=AsyncMock()),
        embedder=SimpleNamespace(close=AsyncMock()),
        search_engine=SimpleNamespace(),
        schema=MEDICAL_GRAPHRAG_SCHEMA,
        collections={"entity": "e", "relation": "r", "chunk": "c", "community": "k"},
        graph_id="med",
        state_dir=tmp_path / "state",
    )


def _setup(tmp_path, monkeypatch, agent_out, *, patch_agent=True):
    """Register a project; patch Project._agent to return a fake agent."""
    registry = Registry.open(tmp_path / "registry.json")
    client = Workspace(registry, state_dir=tmp_path / "state")
    registry.add_graph(
        GraphRecord(
            graph_id="med",
            neo4j=GraphConnection(uri="bolt://h", user_env="U", password_env="P"),
            vector=VectorConnection(url_env="URL"),
        )
    )
    registry.add_project(ProjectRecord(project_id="p", graph_ids=["med"]))
    client.attach_graph(_fake_graph(tmp_path))
    fake_agent = SimpleNamespace(ainvoke=AsyncMock(return_value=agent_out))

    async def fake_build(self, graphs, settings):
        return fake_agent

    if patch_agent:
        monkeypatch.setattr(Project, "_agent", fake_build)
    return client, fake_agent


class TestAgentAnswer:
    """Tests for Project.answer with string and message inputs."""

    async def test_answer_string_returns_clinical_answer(self, tmp_path, monkeypatch):
        """Verify a string question yields a ClinicalAnswer."""
        client, _ = _setup(
            tmp_path, monkeypatch, {"structured_response": ClinicalAnswer(answer="x")}
        )
        ans = await client.project("p").answer("Why is lactate elevated?")
        assert isinstance(ans, ClinicalAnswer)
        assert ans.answer == "x"

    async def test_answer_messages_accepted(self, tmp_path, monkeypatch):
        """Verify a list of Message turns is accepted and forwarded."""
        client, agent = _setup(
            tmp_path, monkeypatch, {"structured_response": ClinicalAnswer(answer="y")}
        )
        ans = await client.project("p").answer(
            [Message(role="user", content="hi"), Message(role="assistant", content="?")]
        )
        assert ans.answer == "y"
        sent = agent.ainvoke.call_args.args[0]["messages"]
        assert sent == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "?"},
        ]


class TestAgentSettingsPropagation:
    """Per-call agent settings reach the agent for answer, stream and batches."""

    async def test_stream_forwards_recursion_limit(self, tmp_path, monkeypatch):
        """Verify stream() passes the configured recursion limit to the agent."""
        client, agent = _setup(tmp_path, monkeypatch, {})

        async def no_events(*args, **kwargs):
            return
            yield

        agent.astream_events = MagicMock(side_effect=no_events)
        settings = AgentSettings(recursion_limit=7)
        async for _ in client.project("p").stream("q", settings=settings):
            pass
        assert agent.astream_events.call_args.kwargs["config"] == {"recursion_limit": 7}

    async def test_answer_batch_answers_with_override_settings(
        self, tmp_path, monkeypatch
    ):
        """Verify answer_batch uses its settings override when answering."""
        client, agent = _setup(
            tmp_path, monkeypatch, {"structured_response": ClinicalAnswer(answer="x")}
        )
        settings = AgentSettings(recursion_limit=7)
        job = await client.project("p").answer_batch(["q1"], settings=settings)
        await job.run()
        assert agent.ainvoke.call_args.kwargs["config"] == {"recursion_limit": 7}


def _fake_build() -> AsyncMock:
    """Return a build function whose agents answer with a fixed ClinicalAnswer."""
    out = {"structured_response": ClinicalAnswer(answer="x")}
    return AsyncMock(
        side_effect=lambda fed, settings: SimpleNamespace(
            ainvoke=AsyncMock(return_value=out)
        )
    )


class TestAgentCache:
    """Cached agents follow the project's graphs and settings."""

    async def test_agent_rebuilt_when_graph_instance_replaced(
        self, tmp_path, monkeypatch
    ):
        """Verify a graph evicted and rebuilt by the pool gets a fresh agent."""
        client, _ = _setup(tmp_path, monkeypatch, {}, patch_agent=False)
        build = _fake_build()
        monkeypatch.setattr("am_diag.agents.agent.build_clinical_agent", build)
        project = client.project("p")

        await project.answer("q")
        await project.answer("q")
        assert build.await_count == 1

        await client.graphs.update("med", schema=MEDICAL_GRAPHRAG_SCHEMA)
        client.attach_graph(_fake_graph(tmp_path))
        await project.answer("q")
        assert build.await_count == 2

    async def test_agent_rebuilt_when_settings_change(self, tmp_path, monkeypatch):
        """Verify different agent settings do not share a cached agent."""
        client, _ = _setup(tmp_path, monkeypatch, {}, patch_agent=False)
        build = _fake_build()
        monkeypatch.setattr("am_diag.agents.agent.build_clinical_agent", build)
        project = client.project("p")

        await project.answer("q", settings=AgentSettings(recursion_limit=7))
        await project.answer("q", settings=AgentSettings(recursion_limit=8))
        assert build.await_count == 2
