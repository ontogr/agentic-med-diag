"""Retrieval recipe tools for the DeepAgents harness.

Each function wraps a Plan-D ``SearchEngine.search()`` call with a specific
recipe, serialising results for the LLM. The ``tool_retry_decorator`` from
the previous harness is retained.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Protocol

from langchain_core.tools import tool

from am_diag.common.data_models.search import Fact, SearchResult
from am_diag.retrieval import recipes
from am_diag.retrieval.assemble import assemble_context
from am_diag.retrieval.question_type import question_hint


logger = logging.getLogger(__name__)


class SearchProvider(Protocol):
    """Structural type for any object exposing ``search(query, recipe)``.

    Both a per-graph ``SearchEngine`` and a cross-graph ``FederatedSearch``
    satisfy this protocol, so the retrieval tools work over either unchanged.
    """

    async def search(
        self, query: str, recipe: Any = ..., filters: Any = ...
    ) -> list[SearchResult]:
        """Return ranked search results for ``query`` under ``recipe``."""
        ...


def tool_retry_decorator(max_retries: int = 3):
    """Tenacity-based retry for retrieval tools.

    Args:
        max_retries: Max attempts before giving up.

    Returns:
        A decorator that retries the wrapped async function.
    """
    from tenacity import (  # noqa: PLC0415
        AsyncRetrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            retrier = AsyncRetrying(
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            )
            async for attempt in retrier:
                with attempt:
                    return await func(*args, **kwargs)
            return None  # unreachable

        return wrapper

    return decorator


def _serialize_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Serialize ``SearchResult`` items to JSON-safe dicts."""
    out: list[dict[str, Any]] = []
    for r in results:
        item = r.item
        item_dict: dict[str, Any] = {}
        if hasattr(item, "model_dump"):
            try:
                item_dict = item.model_dump()
            except Exception:
                item_dict = {"id": str(getattr(item, "id", ""))}
        elif isinstance(item, dict):
            item_dict = item
        else:
            item_dict = {"value": str(item)}
        if isinstance(item, Fact):
            item_dict["text"] = item.text()
        out.append(
            {
                "item": item_dict,
                "score": r.score,
                "source": r.source,
                "source_graph_id": r.source_graph_id,
            }
        )
    return out


def make_retrieval_tools(search_engine: SearchProvider) -> list:
    """Create LangChain ``@tool`` functions wrapping a search provider.

    Args:
        search_engine: Any object implementing the :class:`SearchProvider`
            protocol (a single ``SearchEngine`` or a federated search adapter).

    Returns:
        A list of LangChain-compatible tool functions.
    """

    @tool
    @tool_retry_decorator()
    async def entity_search(query: str) -> list[dict]:
        """Search medical entities by semantic similarity, return matching entities."""
        results = await search_engine.search(query, recipe=recipes.ENTITY)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def relation_search(query: str) -> list[dict]:
        """Search medical relations (e.g. "treats", "causes") by semantic similarity."""
        results = await search_engine.search(query, recipe=recipes.RELATION)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def chunk_search(query: str) -> list[dict]:
        """Search text chunks by semantic similarity to find relevant passages."""
        results = await search_engine.search(query, recipe=recipes.CHUNK)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def community_search(query: str) -> list[dict]:
        """Search community report summaries for thematic answers."""
        results = await search_engine.search(query, recipe=recipes.COMMUNITY)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def hybrid_search(query: str) -> list[dict]:
        """Multi-strategy search over entities, relations, chunks, and communities."""
        results = await search_engine.search(query, recipe=recipes.HYBRID_RRF)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def cypher_search(query: str) -> list[dict]:
        """Generate and execute a Cypher query from natural language."""
        results = await search_engine.search(query, recipe=recipes.TEXT2CYPHER)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def entity_lookup(query: str) -> list[dict]:
        """Find entities by exact name, alias or abbreviation, then by substring.

        Use when the question names specific diseases, drugs or genes.
        """
        results = await search_engine.search(query, recipe=recipes.ENTITY_LOOKUP)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def relation_expand(query: str) -> list[dict]:
        """Return the facts one and two hops around the entities in the question.

        Use when the question asks how things are related or connected.
        """
        results = await search_engine.search(query, recipe=recipes.RELATION_EXPAND)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def enumerate_entities(query: str) -> list[dict]:
        """Find entities, and entities similar to them, for list-style questions.

        Use when the question asks to list, name or enumerate things.
        """
        results = await search_engine.search(query, recipe=recipes.ENUMERATE)
        return _serialize_results(results)

    @tool
    @tool_retry_decorator()
    async def multi_path(query: str) -> dict:
        """Broad search over entities, facts, passages and community reports.

        Use when the question needs facts about several things at once. Returns
        a sectioned `context` string with an answer-format hint, and the raw
        `results`.
        """
        results = await search_engine.search(query, recipe=recipes.MULTI_PATH)
        context = assemble_context(results, hint=question_hint(query))
        return {"context": context, "results": _serialize_results(results)}

    @tool
    @tool_retry_decorator()
    async def global_map_reduce(query: str, level: int = 0) -> dict:
        """Broad question answering over all community reports via map-reduce.

        This tool uses BAML internally and returns a synthesised answer.
        """
        # Falls back to community summary retrieval
        results = await search_engine.search(query, recipe=recipes.COMMUNITY)
        serialized = _serialize_results(results)
        return {"results": serialized, "level": level}

    return [
        entity_search,
        relation_search,
        chunk_search,
        community_search,
        hybrid_search,
        cypher_search,
        entity_lookup,
        relation_expand,
        enumerate_entities,
        multi_path,
        global_map_reduce,
    ]
