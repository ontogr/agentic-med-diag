"""Shared environment helpers for test configuration and service-skip gating.

Used by both the top-level ``tests/conftest.py`` collection hook and the
``tests/integration/conftest.py`` soft-defaults fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


_DOTENV = dotenv_values(Path(__file__).resolve().parents[1] / ".env")

# Service-skip GATING vars: anything any integration path requires. The
# integration soft-default fixture must NOT pre-fill these (would mask skips).
# Kept disjoint from the import-safety placeholders by construction.
GATING_VARS: frozenset[str] = frozenset(
    {
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEO4J_TEST_URI",
        "FALKORDB_HOST",
        "QDRANT_URL",
        "WEAVIATE_URL",
    }
)


def get_config(name: str) -> str | None:
    """Value from process env or repo-root .env, without mutating env."""
    return os.environ.get(name) or _DOTENV.get(name)


def is_configured(name: str) -> bool:
    """True if set in process env or repo-root .env (without mutating env)."""
    return bool(get_config(name))


def required_config_for(path: str, name: str) -> list[str]:
    """Env vars that MUST be set for this integration test's service.

    Public HF datasets need none → ``loaders/`` returns []. Other services list
    their required creds; the hook skips cleanly when any is absent. Fails
    CLOSED for any unclassified integration path so a new service test cannot
    silently run network+no-config.  *path* is ``item.path.as_posix()``.

    Parameters
    ----------
    path:
        The test file's absolute path in POSIX form.
    name:
        The test file's basename (used for heuristic matching, e.g.
        ``"qdrant"`` in the filename).
    """
    if "/loaders/" in path:
        return []  # public HF datasets; HF_TOKEN optional, never required
    if "/db/graph/" in path and name in {
        "test_contract.py",
        "test_cypher_portability.py",
    }:
        # Each backend fixture gates its own configuration independently.
        return []
    if "/db/graph/" in path:
        return ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
    if "/db/vector/" in path and "qdrant" in name:
        return ["QDRANT_URL"]  # QDRANT_API_KEY only for Cloud/auth tests
    if "/db/vector/" in path and "weaviate" in name:
        return ["WEAVIATE_URL"]  # WEAVIATE_API_KEY only for hosted/auth tests
    if "/ingestion/" in path:
        # End-to-end ingestion writes to the graph store and the vector store.
        return ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "QDRANT_URL"]
    raise RuntimeError(  # surfaced as UsageError by the hook
        f"integration test path not classified for required config: {path}. "
        "Update required_config_for() in tests/_env.py (use [] if it needs none)."
    )
