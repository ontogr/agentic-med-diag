"""The durable, resumable ``IngestionJob`` engine.

Streams one or more sources into a single graph, persisting per-item state (via a
:class:`JobLedger`) so a crashed run resumes without re-doing succeeded work.
Extraction and embedding are content-addressed cached so retries and overlapping
sources skip the expensive LLM/embedder calls. A poison item never sinks the job:
it is marked failed and the run continues, ending ``partial``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.job import JobItem, JobProgress, JobRecord
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.data_models.vector_record import VectorRecord
from am_diag.ingestion.cache import IngestionCache
from am_diag.ingestion.graph_ingestion import finalize_graph
from am_diag.ingestion.interfaces import IngestionTarget
from am_diag.ingestion.jobs.ledger import JobLedger
from am_diag.loaders.corpus.base import CorpusLoader
from am_diag.loaders.graph.base import GraphLoader


logger = logging.getLogger(__name__)


def _schema_version(graph_schema: GraphSchema) -> str:
    """Return a short stable hash of the extraction-relevant schema definitions."""
    return graph_schema.fingerprint()


def _rebind_provenance(chunk: Chunk, points: list[Entity] | list[Relation]) -> None:
    """Point cached extraction results at the chunk they are reused for."""
    for point in points:
        provenance = point.provenance
        if provenance is None:
            continue
        provenance.chunk_ids = [chunk.id]
        if provenance.chunk_texts:
            provenance.chunk_texts = [chunk.text]


def _source_kind(source: Any) -> str:
    """Classify a source as ``"corpus"`` or ``"graph"``.

    Raises:
        ValueError: If the source is neither a known loader nor an inline source.
    """
    if isinstance(source, GraphLoader):
        return "graph"
    if isinstance(source, CorpusLoader):
        return "corpus"
    kind = getattr(source, "source_kind", None)
    if kind in ("corpus", "graph"):
        return kind
    raise ValueError(f"Unsupported ingestion source: {type(source).__name__}")


class IngestionJob:
    """Durable, resumable ingestion of one or more sources into one graph.

    Args:
        record: The job manifest.
        target: Live dependencies of the graph being written into.
        cache: Content-addressed cache for this graph.
        state_dir: Root state directory (jobs live under ``jobs/{job_id}``).
        sources: Loaders or inline sources to ingest.
        extractor: Optional injected ``Extractor`` (testing). When ``None``, a
            ``ComposedExtractor`` is built lazily on first use.
        extractor_config: Optional ``ExtractionPipelineConfig`` for the lazily
            built extractor. Its values version the extraction cache.
        chunker: Optional chunker seam. When ``None``, the recursive-character
            chunker is built lazily.
    """

    def __init__(
        self,
        record: JobRecord,
        *,
        target: IngestionTarget,
        cache: IngestionCache,
        state_dir: Path,
        sources: list[Any],
        extractor: Any | None = None,
        extractor_config: Any | None = None,
        chunker: Any | None = None,
    ) -> None:
        """Initialize the job and its ledger, loading any persisted item state."""
        self._ledger = JobLedger(record, state_dir=state_dir)
        self._target = target
        self._cache = cache
        self._sources = sources
        self._extractor = extractor
        self._extractor_config = extractor_config
        self._chunker = chunker
        self._schema_version = _schema_version(target.schema)
        self._extractor_version: str | None = None

    @property
    def id(self) -> str:
        """Unique job id."""
        return self._ledger.job_id

    @property
    def record(self) -> JobRecord:
        """The current job manifest."""
        return self._ledger.record

    @property
    def status(self) -> str:
        """Overall job status."""
        return self._ledger.status

    @property
    def progress(self) -> JobProgress:
        """Aggregate item counters."""
        return self._ledger.progress

    def failures(self) -> list[JobItem]:
        """Return the items that failed after exhausting retries."""
        return self._ledger.failures()

    async def run(self, *, finalize: bool = False) -> JobRecord:
        """Run the job to completion, optionally finalizing communities.

        Returns:
            The final job manifest (``succeeded`` or ``partial``).
        """
        record = self._ledger.record
        if finalize:
            record.finalize = True
        self._ledger.begin()
        for source in self._sources:
            if _source_kind(source) == "graph":
                await self._ingest_graph(source)
            else:
                await self._ingest_corpus(source)
        if record.finalize and not record.finalized:
            await finalize_graph(self._target.graph_store)
            record.finalized = True
            self._ledger.persist()
        self._ledger.finish()
        return record

    async def resume(self) -> JobRecord:
        """Resume the job, skipping items already marked succeeded.

        Resume re-streams the in-memory sources; content-addressed item ids and
        the cache ensure succeeded chunks are skipped and unchanged extractions
        are not recomputed. Finalization requested by the original run is retried
        if it did not complete.
        """
        return await self.run()

    async def _ingest_corpus(self, source: Any) -> None:
        """Chunk, extract, embed, and write one corpus source.

        Each write batch is processed and checkpointed in slices of
        ``embed_batch_size`` chunks, so a crash loses at most one slice.
        """
        config = self._ledger.record.config
        async for doc_batch in source.astream(batch_size=config.write_batch_size):
            chunks: list[Chunk] = []
            for doc in doc_batch:
                chunks.extend(await self._chunk([doc]))
            pending = self._skip_succeeded(chunks, kind="chunk")
            self._ledger.persist()
            docs_by_id = {doc.id: doc for doc in doc_batch}
            for start in range(0, len(pending), config.embed_batch_size):
                piece = pending[start : start + config.embed_batch_size]
                piece_docs = [
                    docs_by_id[doc_id]
                    for doc_id in dict.fromkeys(c.document_id for c in piece)
                    if doc_id in docs_by_id
                ]
                await self._process_chunk_slice(piece_docs, piece)
                self._ledger.persist()

    async def _process_chunk_slice(
        self, doc_batch: list[Any], chunks: list[Chunk]
    ) -> None:
        """Extract (cached), embed (cached), and write one slice of chunks."""
        for chunk in chunks:
            self._item_for(chunk, kind="chunk").status = "processing"
        self._ledger.persist()
        ok_entities, ok_relations, ok_chunks, new_cache = await self._extract_slice(
            chunks
        )
        if not ok_chunks:
            return
        await self._write_slice(
            doc_batch, ok_entities, ok_relations, ok_chunks, new_cache
        )

    async def _extract_slice(
        self, chunks: list[Chunk]
    ) -> tuple[list[Entity], list[Relation], list[Chunk], list[tuple[str, dict]]]:
        """Extract a slice concurrently and mark per-chunk item status.

        Returns:
            ``(entities, relations, chunks, cache_writes)`` for the chunks that
            extracted cleanly. Failed chunks are marked failed in place.
        """
        sem = asyncio.Semaphore(self._ledger.record.config.extract_concurrency)
        outcomes = await asyncio.gather(*(self._extract_chunk(c, sem) for c in chunks))
        ok_entities: list[Entity] = []
        ok_relations: list[Relation] = []
        ok_chunks: list[Chunk] = []
        new_cache: list[tuple[str, dict]] = []
        for chunk, ents, rels, err, cache_payload in outcomes:
            item = self._item_for(chunk, kind="chunk")
            if err is not None:
                item.status = "failed"
                item.error = err
                continue
            ok_entities.extend(ents)
            ok_relations.extend(rels)
            ok_chunks.append(chunk)
            if cache_payload is not None:
                new_cache.append((self._extraction_key(chunk), cache_payload))
            item.status = "succeeded"
            item.error = None
        return ok_entities, ok_relations, ok_chunks, new_cache

    async def _write_slice(
        self,
        doc_batch: list[Any],
        ok_entities: list[Entity],
        ok_relations: list[Relation],
        ok_chunks: list[Chunk],
        new_cache: list[tuple[str, dict]],
    ) -> None:
        """Embed/upsert and write one slice; a write failure fails its chunks."""
        try:
            await self._embed_and_upsert(ok_entities, ok_relations, ok_chunks)
            kg = KnowledgeGraph(
                documents=doc_batch,
                chunks=ok_chunks,
                entities=ok_entities,
                relations=ok_relations,
                graph_schema=self._target.schema,
            )
            await self._target.graph_store.write_knowledge_graph(kg)
            await self._cache.put_extraction_many(new_cache)
        except Exception as exc:  # noqa: BLE001
            for chunk in ok_chunks:
                item = self._item_for(chunk, kind="chunk")
                item.status = "failed"
                item.error = {"type": type(exc).__name__, "message": str(exc)}

    async def _ingest_graph(self, source: Any) -> None:
        """Embed (cached) and write each structured-graph batch."""
        name = getattr(source, "dataset_name", type(source).__name__)
        index = 0
        write_batch_size = self._ledger.record.config.write_batch_size
        async for kg in source.astream(batch_size=write_batch_size):
            item_id = f"{name}:{index}"
            index += 1
            existing = self._ledger.get(item_id)
            if existing and existing.status in ("succeeded", "skipped"):
                existing.status = "skipped"
                continue
            item = self._ledger.add(
                JobItem(
                    item_id=item_id,
                    sequence_index=len(self._ledger.items),
                    kind="structured_batch",
                    ref={"loader": name, "batch": index - 1},
                )
            )
            item.status = "processing"
            self._ledger.persist()
            try:
                await self._embed_and_upsert(kg.entities, kg.relations, [])
                await self._target.graph_store.write_knowledge_graph(kg)
                item.status = "succeeded"
                item.error = None
            except Exception as exc:  # noqa: BLE001
                item.status = "failed"
                item.error = {"type": type(exc).__name__, "message": str(exc)}
            self._ledger.persist()

    async def _extract_chunk(
        self, chunk: Chunk, sem: asyncio.Semaphore
    ) -> tuple[Chunk, list[Entity], list[Relation], dict[str, Any] | None, Any]:
        """Extract one chunk, using the cache and isolating failures.

        Returns:
            ``(chunk, entities, relations, error, cache_payload)`` where
            ``cache_payload`` is non-None only on a fresh (uncached) extraction
            that should be written to the cache.
        """
        cached = await self._cache.get_extraction(self._extraction_key(chunk))
        if cached is not None:
            entities = [Entity.model_validate(e) for e in cached.get("entities", [])]
            relations = [
                Relation.model_validate(r) for r in cached.get("relations", [])
            ]
            _rebind_provenance(chunk, entities)
            _rebind_provenance(chunk, relations)
            return chunk, entities, relations, None, None
        async with sem:
            error = None
            for attempt in range(self._ledger.record.config.max_retries):
                try:
                    entities, relations = await self._extract(chunk)
                    payload = {
                        "entities": [e.model_dump(mode="json") for e in entities],
                        "relations": [r.model_dump(mode="json") for r in relations],
                    }
                    return chunk, entities, relations, None, payload
                except Exception as exc:  # noqa: BLE001
                    error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "attempt": attempt + 1,
                    }
        return chunk, [], [], error, None

    async def _embed_and_upsert(
        self,
        entities: list[Entity],
        relations: list[Relation],
        chunks: list[Chunk],
    ) -> None:
        """Embed entities/relations/chunks via the cache and upsert vectors."""
        coll = self._target.collections
        scope = self._ledger.record.config.embedding_scope
        if entities:
            vectors = await self._embed_texts(
                [e.canonical_name or e.name for e in entities]
            )
            records = [
                VectorRecord(
                    id=e.id,
                    vector=vec,
                    payload={
                        "id": str(e.id),
                        "name": e.name,
                        "canonical_name": e.canonical_name or e.name,
                        "label": e.label,
                    },
                )
                for e, vec in zip(entities, vectors, strict=False)
            ]
            await self._target.vector_store.upsert(coll["entity"], records)
        if relations and scope == "entities+relations":
            texts = [
                r.description or f"{r.head_name} {r.type} {r.tail_name}"
                for r in relations
            ]
            vectors = await self._embed_texts(texts)
            records = [
                VectorRecord(
                    id=r.id,
                    vector=vec,
                    payload={
                        "id": str(r.id),
                        "head_name": r.head_name,
                        "tail_name": r.tail_name,
                        "type": r.type,
                    },
                )
                for r, vec in zip(relations, vectors, strict=False)
            ]
            await self._target.vector_store.upsert(coll["relation"], records)
        if chunks:
            vectors = await self._embed_texts([c.text for c in chunks])
            records = [
                VectorRecord(
                    id=c.id,
                    vector=vec,
                    payload={
                        "id": str(c.id),
                        "text": c.text[:500],
                        "document_id": str(c.document_id),
                        "document_source": c.document_source,
                    },
                )
                for c, vec in zip(chunks, vectors, strict=False)
            ]
            await self._target.vector_store.upsert(coll["chunk"], records)

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, serving cache hits and embedding only the misses."""
        if not texts:
            return []
        model_id = self._target.embedder.model_id
        keys = [IngestionCache.embedding_key(t, model_id) for t in texts]
        cached = await self._cache.get_embeddings(keys)
        miss_indices = [i for i, k in enumerate(keys) if k not in cached]
        if miss_indices:
            batch_size = self._ledger.record.config.embed_batch_size
            miss_texts = [texts[i] for i in miss_indices]
            miss_vectors: list[list[float]] = []
            for start in range(0, len(miss_texts), batch_size):
                miss_vectors.extend(
                    await self._target.embedder.embed(
                        miss_texts[start : start + batch_size]
                    )
                )
            await self._cache.put_embeddings(
                [(keys[i], v) for i, v in zip(miss_indices, miss_vectors, strict=False)]
            )
            for i, vec in zip(miss_indices, miss_vectors, strict=False):
                cached[keys[i]] = vec
        return [cached[k] for k in keys]

    async def _extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        """Run the configured extractor over one chunk."""
        extractor = self._extractor or self._ensure_extractor()
        return await extractor.extract([chunk])

    def _ensure_extractor(self) -> Any:
        """Build the composed extractor lazily on first use."""
        if self._extractor is None:
            from am_diag.graph_construction import ComposedExtractor  # noqa: PLC0415
            from am_diag.graph_construction.config import (  # noqa: PLC0415
                ExtractionPipelineConfig,
            )

            self._extractor = ComposedExtractor(
                self._extractor_config or ExtractionPipelineConfig(),
                self._target.schema,
            )
        return self._extractor

    def _extractor_fingerprint(self) -> str:
        """Return a stable hash of the effective extractor, computed once.

        An injected extractor is identified by its class. Otherwise the hash
        covers the full pipeline config (models, thresholds, toggles).
        """
        if self._extractor_version is None:
            if self._extractor is not None and self._extractor_config is None:
                identity = type(self._extractor).__qualname__
            else:
                from am_diag.graph_construction.config import (  # noqa: PLC0415
                    ExtractionPipelineConfig,
                )

                config = self._extractor_config or ExtractionPipelineConfig()
                identity = config.model_dump_json()
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            self._extractor_version = digest[:16]
        return self._extractor_version

    async def _chunk(self, docs: list[Any]) -> list[Chunk]:
        """Chunk documents with the configured chunker, built lazily."""
        if self._chunker is None:
            from am_diag.chunking.recursive_character_text_splitter import (  # noqa: PLC0415
                RecursiveCharacterTextChunker,
            )

            self._chunker = RecursiveCharacterTextChunker()
        return await self._chunker.chunk(docs)

    def _extraction_key(self, chunk: Chunk) -> str:
        """Return the text-only cache key for a chunk's extraction result."""
        return IngestionCache.extraction_key(
            chunk.text, self._schema_version, self._extractor_fingerprint()
        )

    def _chunk_item_id(self, chunk: Chunk) -> str:
        """Return the ledger id for a chunk.

        Unlike the extraction key it includes the chunk identity, so distinct
        chunks with identical text are tracked separately.
        """
        parts = (
            str(chunk.id),
            self._extraction_key(chunk),
        )
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

    def _item_for(self, chunk: Chunk, *, kind: str) -> JobItem:
        """Return (creating if needed) the JobItem tracking a chunk."""
        item_id = self._chunk_item_id(chunk)
        existing = self._ledger.get(item_id)
        if existing is not None:
            return existing
        return self._ledger.add(
            JobItem(
                item_id=item_id,
                sequence_index=len(self._ledger.items),
                kind=kind,
                ref={"document_id": str(chunk.document_id), "index": chunk.chunk_index},
            )
        )

    def _skip_succeeded(self, chunks: list[Chunk], *, kind: str) -> list[Chunk]:
        """Mark already-succeeded chunks skipped and return the rest."""
        pending: list[Chunk] = []
        for chunk in chunks:
            existing = self._ledger.get(self._chunk_item_id(chunk))
            if existing and existing.status in ("succeeded", "skipped"):
                existing.status = "skipped"
                continue
            pending.append(chunk)
        return pending
