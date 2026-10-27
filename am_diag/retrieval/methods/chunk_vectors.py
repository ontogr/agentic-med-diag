"""Embeddings for candidate chunks, read from the vector store when stored."""

from __future__ import annotations

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.search import SearchResult
from am_diag.db.vector.base import VectorStoreBase
from am_diag.vector.embedding.base import Embedder


async def fetch_chunk_vectors(
    store: VectorStoreBase,
    collection: str,
    embedder: Embedder,
    results: list[SearchResult],
    *,
    min_coverage: float,
) -> dict[str, list[float]]:
    """Return an embedding for each chunk in ``results``, keyed by chunk id.

    Stored vectors are used first. When fewer than ``min_coverage`` of the chunks
    have one, the missing chunks are embedded from their text instead.

    Args:
        store: Vector store that holds the chunk collection.
        collection: Chunk collection name.
        embedder: Embedder for chunks without a stored vector.
        results: Results to read chunks from. Other item types are ignored.
        min_coverage: Share of chunks, from 0 to 1, that must have a stored vector
            before the rest are left out.

    Returns:
        Chunk id to vector. A chunk with no vector is absent.
    """
    chunks = {str(r.item.id): r.item for r in results if isinstance(r.item, Chunk)}
    if not chunks:
        return {}

    records = await store.retrieve(collection, list(chunks), with_vectors=True)
    vectors = {str(record.id): record.vector for record in records if record.vector}

    if len(vectors) / len(chunks) >= min_coverage:
        return vectors

    missing = [chunk_id for chunk_id in chunks if chunk_id not in vectors]
    embedded = await embedder.embed([chunks[chunk_id].text for chunk_id in missing])
    return {**vectors, **dict(zip(missing, embedded, strict=True))}


__all__ = ["fetch_chunk_vectors"]
