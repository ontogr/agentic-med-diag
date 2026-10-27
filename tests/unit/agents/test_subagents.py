"""Subagent builders — planner, verifier, and researcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from am_diag.agents.subagents import (
    build_planner_subagent,
    build_researcher_subagent,
    build_verifier_subagent,
)
from am_diag.retrieval.search import SearchEngine


class TestBuildSubagents:
    """Subagent builders — planner/verifier formats and researcher config."""

    def test_planner_has_response_format(self) -> None:
        """Planner subagent has AgentPlan response format."""
        with patch("am_diag.agents.subagents.build_chat_model") as mock_build:
            mock_build.return_value = MagicMock()
            config = build_planner_subagent()
            assert config["name"] == "planner"
            assert "response_format" in config
            assert "AgentPlan" in str(config["response_format"])

    def test_verifier_has_response_format(self) -> None:
        """Verifier subagent has SufficientContext response format."""
        with patch("am_diag.agents.subagents.build_chat_model") as mock_build:
            mock_build.return_value = MagicMock()
            config = build_verifier_subagent()
            assert config["name"] == "sufficient_context"
            assert "response_format" in config
            assert "SufficientContext" in str(config["response_format"])

    def test_researcher_has_tools(self) -> None:
        """Researcher subagent is configured with 11 tools."""
        with patch("am_diag.agents.subagents.build_chat_model") as mock_build:
            mock_build.return_value = MagicMock()
            engine = MagicMock(spec=SearchEngine)
            engine.search = MagicMock()
            config = build_researcher_subagent(engine)
            assert config["name"] == "researcher"
            assert "tools" in config
            assert len(config["tools"]) == 11

    def test_researcher_tools_are_callable(self) -> None:
        """All researcher tools have valid names."""
        with patch("am_diag.agents.subagents.build_chat_model") as mock_build:
            mock_build.return_value = MagicMock()
            engine = MagicMock(spec=SearchEngine)
            engine.search = MagicMock()
            config = build_researcher_subagent(engine)
            for tool in config["tools"]:
                assert tool.name is not None and len(tool.name) > 0

    def test_planner_prompt_contains_decompose(self) -> None:
        """Planner system prompt mentions decompose."""
        with patch("am_diag.agents.subagents.build_chat_model") as mock_build:
            mock_build.return_value = MagicMock()
            config = build_planner_subagent()
            prompt = config["system_prompt"]
            assert "decompose" in prompt.lower()

    def test_verifier_prompt_contains_sufficiency(self) -> None:
        """Verifier system prompt mentions sufficiency."""
        with patch("am_diag.agents.subagents.build_chat_model") as mock_build:
            mock_build.return_value = MagicMock()
            config = build_verifier_subagent()
            prompt = config["system_prompt"]
            assert "sufficiency" in prompt.lower()
