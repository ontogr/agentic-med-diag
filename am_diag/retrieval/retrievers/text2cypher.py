"""Text-to-Cypher retriever: BAML ``GenerateCypherQuery`` + EXPLAIN check + retry.

Accepts the new ``:RELATES_TO {type}`` generic-edge schema.
Schema context is provided by ``GraphSchema.as_text()``.
"""

from __future__ import annotations

import logging

from pydantic_core import to_jsonable_python

from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.data_models.search import CypherRow, SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.errors import GraphWriteRejected
from am_diag.retrieval.config import RetrievalConfig


logger = logging.getLogger(__name__)


class Text2CypherRetriever:
    """NL-driven Cypher retrieval with EXPLAIN validation and read-only execution.

    Flow:
    1. Serialise ``GraphSchema`` as ``:RELATES_TO {type}`` description.
    2. Call BAML ``GenerateCypherQuery(question, schema, examples)``.
    3. ``EXPLAIN`` validation — if it fails, retry up to ``max_attempts``.
    4. Execute the validated query in a read transaction and return one
       ``SearchResult`` per row. A query that tries to write returns no results.

    Args:
        config: Retrieval settings (``use_text_to_cypher``, ``max_attempts``).
        graph_store: Graph database client.
        schema: KG schema for prompt injection.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        graph_store: GraphStore,
        schema: GraphSchema,
    ) -> None:
        """Initialize retriever with config and dependencies."""
        self._config = config
        self._graph = graph_store
        self._schema = schema
        self._schema_desc: str | None = None

    async def retrieve(
        self,
        query: str,
        filters: dict | None = None,  # noqa: ARG002
    ) -> list[SearchResult]:
        """Generate and execute a Cypher query for ``query``.

        Args:
            query: Natural-language question for Cypher generation.
            filters: Ignored for text2cypher (schema-based filtering).

        Returns:
            One ``SearchResult`` per result row.
        """
        if not self._config.use_text_to_cypher:
            return []

        from am_diag.llm.baml_client import b  # noqa: PLC0415

        schema_desc = self._get_schema_description()

        last_error: str | None = None
        cypher_result = None

        for attempt in range(self._config.text2cypher_max_attempts):
            try:
                result = await b.GenerateCypherQuery(query, schema_desc, "")
            except Exception as exc:
                last_error = f"BAML generation failed (attempt {attempt + 1}): {exc}"
                logger.warning(last_error)
                continue

            cypher = result.cypher.strip()

            # EXPLAIN checks syntax and the query plan before anything runs.
            try:
                await self._graph.explain(cypher)
                cypher_result = cypher
                break
            except Exception as exc:
                last_error = f"EXPLAIN rejected (attempt {attempt + 1}): {exc}"
                logger.warning(last_error)

        if cypher_result is None:
            logger.error(
                "text2cypher: all %d attempts failed: %s",
                self._config.text2cypher_max_attempts,
                last_error,
            )
            return []

        # Generated Cypher is untrusted: a read transaction makes the server
        # reject any write clause the model produced.
        try:
            rows = await self._graph.execute_query(cypher_result, read_only=True)
        except GraphWriteRejected as exc:
            logger.error("text2cypher: write query rejected: %s", exc)
            return []
        return [
            SearchResult(
                item=CypherRow(values=to_jsonable_python(row, fallback=str)),
                score=1.0 - (i * 0.01),
                source="cypher",
                matched_on="text2cypher",
            )
            for i, row in enumerate(rows)
        ]

    def _get_schema_description(self) -> str:
        """Lazily build and cache the schema description string."""
        if self._schema_desc is None:
            self._schema_desc = self._schema.as_text()
        return self._schema_desc


__all__ = ["Text2CypherRetriever"]
