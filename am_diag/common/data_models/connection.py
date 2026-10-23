"""Connection descriptors for a Graph's backend deployments.

Secrets are stored as the *names* of environment variables, never the secret
values. They are resolved against the process environment at connect time by the
connection pool, so a serialized ``GraphRecord`` is safe to persist to disk.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


CollectionKind = Literal["entity", "relation", "chunk", "community"]


def _default_collections() -> dict[CollectionKind, str]:
    """Return the default logical-to-physical vector collection names."""
    return {
        "entity": "am_diag_entity",
        "relation": "am_diag_relation",
        "chunk": "am_diag_chunk",
        "community": "am_diag_community",
    }


class GraphConnection(BaseModel):
    """Graph backend connection metadata for one Graph.

    Secrets are env-var names, resolved at connect time — never the secret
    values themselves.

    Attributes:
        backend: Graph database backend. Defaults to Neo4j for old records.
        uri: Neo4j connection URI or a Redis URI for FalkorDB.
        user_env: Name of the env var holding the username.
        password_env: Name of the env var holding the password.
        database: Target database or FalkorDB graph name.
    """

    uri: str
    backend: Literal["neo4j", "falkordb"] = "neo4j"
    user_env: str | None = None
    password_env: str | None = None
    database: str = "neo4j"

    @model_validator(mode="after")
    def _validate_backend(self) -> GraphConnection:
        """Reject URI/backend mismatches and missing Neo4j credential refs."""
        parsed = urlsplit(self.uri)
        scheme = parsed.scheme
        schemes = (
            {"redis", "rediss"}
            if self.backend == "falkordb"
            else {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}
        )
        if scheme not in schemes:
            raise ValueError(f"{self.backend} backend cannot use a {scheme}:// uri")
        if self.backend == "falkordb" and "@" in parsed.netloc:
            raise ValueError(
                "FalkorDB connection URI cannot contain inline credentials; "
                "use user_env and password_env instead"
            )
        if self.backend == "neo4j" and (not self.user_env or not self.password_env):
            raise ValueError("neo4j backend requires user_env and password_env")
        return self


class VectorConnection(BaseModel):
    """Vector-store connection metadata for one Graph's internal vectors.

    Attributes:
        backend: ``"qdrant"`` or ``"weaviate"``.
        url_env: Env var holding the store URL.
        api_key_env: Env var holding the API key. ``None`` when the backend
            needs no key; an empty-string env value is also accepted.
        collections: Logical-to-physical collection names for this graph's four
            vector spaces. Kinds that are not given use the per-kind default.
        embedding_model: Model id used to build these vectors. Recorded so a
            query embeds with the matching model. Defaults to the ST default.
        dim: Embedding dimensionality.
    """

    backend: Literal["qdrant", "weaviate"] = "qdrant"
    url_env: str
    api_key_env: str | None = None
    collections: dict[CollectionKind, str] = Field(
        default_factory=_default_collections,
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384

    @field_validator("collections")
    @classmethod
    def _fill_default_collections(
        cls, collections: dict[CollectionKind, str]
    ) -> dict[CollectionKind, str]:
        """Fill collection kinds the caller omitted with their default names."""
        return {**_default_collections(), **collections}
