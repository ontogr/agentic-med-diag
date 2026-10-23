"""Neo4j connection settings loaded from environment variables."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jSettings(BaseSettings):
    """Neo4j connection settings.

    Reads from environment variables with the `NEO4J_` prefix.
    All fields can also be supplied directly as constructor arguments.

    Attributes:
        uri: Bolt URI of the Neo4j instance, e.g. `bolt://localhost:7687`.
        user: Neo4j username.
        password: Neo4j password.
        database: Target database name. Defaults to `"neo4j"`.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        env_file=".env",
        extra="ignore",
    )

    uri: str = ""
    user: str = ""
    password: str = ""
    database: str = "neo4j"


class FalkorDBSettings(BaseSettings):
    """FalkorDB connection and query limits, read from ``FALKORDB_*`` variables.

    Attributes:
        host: Server hostname. Env: ``FALKORDB_HOST``.
        port: Redis port. Env: ``FALKORDB_PORT``.
        username: Optional ACL user. Env: ``FALKORDB_USERNAME``.
        password: Optional secret. Env: ``FALKORDB_PASSWORD``.
        graph: Graph key. Env: ``FALKORDB_GRAPH``.
        ssl: Require TLS. Env: ``FALKORDB_SSL``.
        query_timeout_ms: Server query timeout. Env: ``FALKORDB_QUERY_TIMEOUT_MS``.
        max_connections: Pool limit. Env: ``FALKORDB_MAX_CONNECTIONS``.
        batch_size: Records per write. Env: ``FALKORDB_BATCH_SIZE``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FALKORDB_", env_file=".env", extra="ignore"
    )

    host: str = Field(default="localhost", min_length=1)
    port: int = Field(default=6379, ge=1, le=65535)
    username: str = ""
    password: SecretStr = SecretStr("")
    graph: str = Field(default="am_diag", min_length=1)
    ssl: bool = False
    query_timeout_ms: int = Field(default=30_000, gt=0)
    max_connections: int = Field(default=16, gt=0)
    batch_size: int = Field(default=500, gt=0)

    @classmethod
    def from_url(cls, url: str, **overrides: Any) -> FalkorDBSettings:
        """Parse a Redis URL, using its path as the graph key.

        Args:
            url: ``redis://`` or ``rediss://`` URL with optional credentials.
            **overrides: Explicit settings that take precedence over URL fields.

        Raises:
            ValueError: If the URL has no host or uses unsupported components.
        """
        parsed = urlsplit(url)
        if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
            raise ValueError("FalkorDB URL must use redis:// or rediss:// with a host")
        if parsed.query or parsed.fragment:
            raise ValueError("FalkorDB URL cannot contain a query or fragment")
        values: dict[str, Any] = {
            "host": parsed.hostname,
            "port": parsed.port if parsed.port is not None else 6379,
            "username": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "ssl": parsed.scheme == "rediss",
        }
        if parsed.path.strip("/"):
            values["graph"] = unquote(parsed.path.lstrip("/"))
        values.update(overrides)
        return cls(**values)
