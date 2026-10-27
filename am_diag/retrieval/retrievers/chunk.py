"""Chunk retriever: hybrid search on chunks, build results from payload.

Builds each ``Chunk`` straight from ``VectorHit.payload`` (no graph hydration).
When BFS is requested, seeds are the entities the chunks mention, fetched via
``chunk_seed_entities.cypher``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.vector_record import VectorHit
from am_diag.retrieval.retrievers.base import HybridRetriever


def chunk_from_payload(chunk_id: UUID, payload: dict[str, Any]) -> Chunk:
    """Build a ``Chunk`` from a vector-store payload."""
    text = payload.get("text", "")
    return Chunk(
        id=chunk_id,
        text=text,
        document_id=payload.get("document_id", ""),
        document_source=payload.get("document_source", ""),
        chunk_index=payload.get("chunk_index", 0),
        chunk_size=payload.get("chunk_size", len(text)),
        cut_type=payload.get("cut_type", "hybrid"),
    )


class ChunkRetriever(HybridRetriever):
    """Hybrid search on the chunk collection, payload-built results."""

    name = "chunk"
    hydrate_cypher = None  # build straight from payload

    def _build(self, hit: VectorHit, row: dict[str, Any]) -> Chunk:
        """Build a ``Chunk`` from the hit payload."""
        return chunk_from_payload(hit.id, hit.payload)

    async def _seed_ids(
        self,
        hits: list[VectorHit],
        rows: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Seed BFS from the entities the matched chunks mention."""
        ids = [str(h.id) for h in hits]
        out = await self._graph.execute_query(
            load("retrieval/chunk_seed_entities"),
            {"ids": ids},
        )
        return [r["id"] for r in out if r.get("id")]


__all__ = ["ChunkRetriever", "chunk_from_payload"]
