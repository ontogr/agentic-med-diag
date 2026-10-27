"""Text-to-Cypher retriever: BAML ``GenerateCypherQuery`` + EXPLAIN check + retry.

Accepts the new ``:RELATES_TO {type}`` generic-edge schema.
Schema context is provided by ``GraphSchema.as_text()``. Generated queries pass
through :func:`~am_diag.retrieval.cypher_guard.guard_cypher` before ``EXPLAIN``.
"""

from __future__ import annotations

import logging

from pydantic_core import to_jsonable_python

from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.data_models.search import CypherRow, SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.errors import GraphWriteRejected
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.cypher_guard import guard_cypher
from am_diag.retrieval.filters import SearchFilters


logger = logging.getLogger(__name__)


class Text2CypherRetriever:
    """NL-driven Cypher retrieval with EXPLAIN validation and read-only execution.

    Flow:
    1. Serialise ``GraphSchema`` as ``:RELATES_TO {type}`` description.
    2. Call BAML ``GenerateCypherQuery``, passing the reason for the last
       rejection as feedback.
    3. Check the query against the schema, add or lower its ``LIMIT``, then
       ``EXPLAIN`` it. A rejected query is retried up to ``max_attempts``.
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
        filters: SearchFilters | None = None,  # noqa: ARG002
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
        feedback = ""
        cypher_result = None

        for attempt in range(self._config.text2cypher_max_attempts):
            try:
                result = await b.GenerateCypherQuery(query, schema_desc, "", feedback)
            except Exception as exc:
                last_error = f"BAML generation failed (attempt {attempt + 1}): {exc}"
                logger.warning(last_error)
                continue

            guarded = guard_cypher(
                result.cypher.strip(),
                self._schema,
                dialect=self._graph.dialect,
                default_limit=self._config.text2cypher_limit,
                max_limit=self._config.text2cypher_max_limit,
            )
            if guarded.errors:
                feedback = " ".join(guarded.errors)
                last_error = f"guard rejected (attempt {attempt + 1}): {feedback}"
                logger.warning(last_error)
                continue

            # EXPLAIN checks syntax and the query plan before anything runs.
            try:
                await self._graph.explain(guarded.cypher)
                cypher_result = guarded.cypher
                break
            except Exception as exc:
                feedback = str(exc)
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
