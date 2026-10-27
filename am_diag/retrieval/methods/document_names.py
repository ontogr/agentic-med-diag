"""Source document names for chunk results."""

from __future__ import annotations

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore


async def attach_document_names(
    graph_store: GraphStore,
    results: list[SearchResult],
) -> None:
    """Record each chunk's source document name in ``Chunk.properties``.

    The name is the document title, or its external id when it has no title. A
    chunk whose document is missing from the graph is left unchanged. One query
    covers all chunks, and no query runs when there are none.

    Args:
        graph_store: Graph database client.
        results: Results to update in place. Other item types are ignored.
    """
    chunks = {str(r.item.id): r.item for r in results if isinstance(r.item, Chunk)}
    if not chunks:
        return

    rows = await graph_store.execute_query(
        load("retrieval/chunk_documents"),
        {"ids": list(chunks)},
        read_only=True,
    )
    for row in rows:
        name = row.get("title") or row.get("external_id")
        chunk = chunks.get(str(row["chunk_id"]))
        if name and chunk is not None:
            chunk.properties["document_name"] = name


__all__ = ["attach_document_names"]
