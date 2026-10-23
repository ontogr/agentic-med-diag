"""Select a graph transport from its connection settings."""

from importlib import import_module

from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.settings import FalkorDBSettings, Neo4jSettings


def create_graph_store(settings: Neo4jSettings | FalkorDBSettings) -> GraphStore:
    """Return a client without importing the unused backend driver.

    Args:
        settings: Backend-specific connection settings.

    Raises:
        TypeError: If the settings type is unsupported.
    """
    if isinstance(settings, Neo4jSettings):
        return import_module("am_diag.db.graph.neo4j").Neo4jClient(settings)
    if isinstance(settings, FalkorDBSettings):
        return import_module("am_diag.db.graph.falkordb").FalkorDBClient(settings)
    raise TypeError(f"Unsupported graph settings: {type(settings).__name__}")
