"""Tests for the durable IngestionJob engine.

Drives the corpus path with an injected fake extractor and chunker against a
real temp SQLite cache and a mocked graph/vector target. Verifies item
lifecycle, poison-item isolation (partial + failures), resume skipping, and that
extraction/embedding cache hits skip the extractor and embedder. The schema
version that keys cached extraction tracks the schema definition.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.job import JobConfig, JobRecord
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.schema import EntityType, GraphSchema
from am_diag.ingestion.cache import IngestionCache
from am_diag.ingestion.jobs import IngestionJob
from am_diag.ingestion.jobs.ingestion_job import _schema_version


class _FakeCorpusSource:
    """One-batch corpus source yielding fixed documents."""

    source_kind = "corpus"

    def __init__(self, documents):
        self._documents = documents

    async def astream(self, batch_size=100):
        yield self._documents


class _FakeExtractor:
    """Wrap a per-chunk async fn as an ``Extractor`` for the job seam."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    async def extract(self, chunks):
        entities, relations = [], []
        for chunk in chunks:
            self.calls += 1
            ents, rels = await self._fn(chunk)
            entities.extend(ents)
            relations.extend(rels)
        return entities, relations


class _FakeChunker:
    """Chunker that emits one Chunk per document using the document text."""

    async def chunk(self, docs):
        return [
            Chunk(
                text=d.text,
                document_id=d.id,
                document_source=d.source,
                chunk_index=0,
                chunk_size=len(d.text),
                cut_type="recursive",
            )
            for d in docs
        ]


def _record() -> GraphRecord:
    return GraphRecord(
        graph_id="med",
        neo4j=GraphConnection(uri="bolt://h", user_env="U", password_env="P"),
        vector=VectorConnection(url_env="URL"),
    )


def _target():
    """Return a fake IngestionTarget with mocked store/vector/embedder."""
    record = _record()
    embedder = SimpleNamespace(
        embed=AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts]),
        model_id="fake-model",
    )
    return SimpleNamespace(
        schema=record.graph_schema,
        collections=record.vector.collections,
        graph_store=SimpleNamespace(write_knowledge_graph=AsyncMock()),
        vector_store=SimpleNamespace(upsert=AsyncMock()),
        embedder=embedder,
    )


def _docs(n: int) -> list[Document]:
    return [
        Document(text=f"doc {i}", source="inline", external_id=str(i)) for i in range(n)
    ]


def _make_job(tmp_path, target, cache, sources, extractor, record=None, **kwargs):
    return IngestionJob(
        record or JobRecord(job_id=uuid4().hex, kind="ingestion", graph_id="med"),
        target=target,
        cache=cache,
        state_dir=tmp_path,
        sources=sources,
        extractor=extractor,
        chunker=_FakeChunker(),
        **kwargs,
    )


class TestIngestionRun:
    """Tests for run(), caching, and poison isolation."""

    async def test_run_marks_items_succeeded(self, tmp_path):
        """Verify a clean run marks every chunk succeeded."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        extractor = _FakeExtractor(fn)
        cache = IngestionCache(tmp_path / "med.db")
        source = _FakeCorpusSource(_docs(2))
        job = _make_job(tmp_path, _target(), cache, [source], extractor)
        record = await job.run()
        assert record.status == "succeeded"
        assert record.progress.succeeded == 2
        assert extractor.calls == 2

    async def test_extraction_cache_skips_extractor(self, tmp_path):
        """Verify a second job over the same content hits the extraction cache."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        extractor = _FakeExtractor(fn)
        cache = IngestionCache(tmp_path / "med.db")
        docs = _docs(2)
        await _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], extractor
        ).run()
        assert extractor.calls == 2
        await _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], extractor
        ).run()
        assert extractor.calls == 2

    async def test_embedding_cache_skips_embedder(self, tmp_path):
        """Verify a second job reuses cached embeddings (embedder not called)."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        cache = IngestionCache(tmp_path / "med.db")
        docs = _docs(2)
        target2 = _target()
        await _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], _FakeExtractor(fn)
        ).run()
        await _make_job(
            tmp_path, target2, cache, [_FakeCorpusSource(docs)], _FakeExtractor(fn)
        ).run()
        target2.embedder.embed.assert_not_called()

    async def test_poison_item_yields_partial(self, tmp_path):
        """Verify a chunk that always fails leaves the job partial with failures."""

        async def fn(chunk):
            if "poison" in chunk.text:
                raise RuntimeError("boom")
            return [Entity(name=chunk.text, label="Disease")], []

        cache = IngestionCache(tmp_path / "med.db")
        docs = [
            Document(text="good", source="inline", external_id="1"),
            Document(text="poison", source="inline", external_id="2"),
        ]
        job = _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], _FakeExtractor(fn)
        )
        record = await job.run()
        assert record.status == "partial"
        assert record.progress.failed == 1
        assert len(job.failures()) == 1

    async def test_resume_skips_succeeded(self, tmp_path):
        """Verify resuming a completed job skips already-succeeded chunks."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        extractor = _FakeExtractor(fn)
        cache = IngestionCache(tmp_path / "med.db")
        job = _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(_docs(2))], extractor
        )
        await job.run()
        assert extractor.calls == 2
        record = await job.resume()
        assert record.status == "succeeded"
        assert record.progress.skipped == 2
        assert extractor.calls == 2

    async def test_resume_from_persisted_state_in_new_job(self, tmp_path):
        """Verify a job rebuilt from disk skips chunks that succeeded earlier."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        extractor = _FakeExtractor(fn)
        cache = IngestionCache(tmp_path / "med.db")
        record = JobRecord(job_id=uuid4().hex, kind="ingestion", graph_id="med")
        docs = _docs(2)
        await _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], extractor, record
        ).run()
        rebuilt = _make_job(
            tmp_path,
            _target(),
            cache,
            [_FakeCorpusSource(docs)],
            extractor,
            record.model_copy(deep=True),
        )
        result = await rebuilt.resume()
        assert result.progress.skipped == 2
        assert extractor.calls == 2


class TestChunkIdentity:
    """Tests for ledger ids, cache keys, and provenance of reused extractions."""

    async def test_distinct_chunks_with_same_text_are_all_processed(self, tmp_path):
        """Verify identical text in two documents yields two ledger items."""

        async def fn(chunk):
            return [Entity(name="shared", label="Disease")], []

        docs = [
            Document(text="same", source="inline", external_id="1"),
            Document(text="same", source="inline", external_id="2"),
        ]
        cache = IngestionCache(tmp_path / "med.db")
        job = _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], _FakeExtractor(fn)
        )
        record = await job.run()
        assert record.progress.total == 2
        assert record.progress.succeeded == 2

    async def test_cached_extraction_is_rebound_to_the_current_chunk(self, tmp_path):
        """Verify a cache hit carries the new chunk's id, not the original's."""

        async def fn(chunk):
            entity = Entity(
                name="shared",
                label="Disease",
                provenance=Provenance(chunk_ids=[chunk.id]),
            )
            return [entity], []

        docs = [
            Document(text="same", source="inline", external_id="1"),
            Document(text="same", source="inline", external_id="2"),
        ]
        cache = IngestionCache(tmp_path / "med.db")
        first = _target()
        await _make_job(
            tmp_path, first, cache, [_FakeCorpusSource(docs[:1])], _FakeExtractor(fn)
        ).run()
        second = _target()
        extractor = _FakeExtractor(fn)
        await _make_job(
            tmp_path, second, cache, [_FakeCorpusSource(docs[1:])], extractor
        ).run()
        kg = second.graph_store.write_knowledge_graph.await_args.args[0]
        assert extractor.calls == 0
        assert kg.entities[0].provenance.chunk_ids == [kg.chunks[0].id]

    async def test_embedding_calls_respect_batch_size(self, tmp_path):
        """Verify embedding requests are split by embed_batch_size."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        target = _target()
        record = JobRecord(
            job_id=uuid4().hex,
            kind="ingestion",
            graph_id="med",
            config=JobConfig(embed_batch_size=2),
        )
        await _make_job(
            tmp_path,
            target,
            IngestionCache(tmp_path / "med.db"),
            [_FakeCorpusSource(_docs(5))],
            _FakeExtractor(fn),
            record,
        ).run()
        sizes = [len(call.args[0]) for call in target.embedder.embed.await_args_list]
        assert sizes
        assert max(sizes) <= 2


class TestFinalizationResume:
    """Tests that requested finalization survives an interrupted run."""

    async def test_resume_retries_unfinished_finalization(self, tmp_path, monkeypatch):
        """Verify a failed finalize is retried on resume and then not repeated."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        finalize = AsyncMock(side_effect=[RuntimeError("gds down"), 3, 3])
        monkeypatch.setattr(
            "am_diag.ingestion.jobs.ingestion_job.finalize_graph", finalize
        )
        cache = IngestionCache(tmp_path / "med.db")
        record = JobRecord(job_id=uuid4().hex, kind="ingestion", graph_id="med")
        docs = _docs(1)
        job = _make_job(
            tmp_path,
            _target(),
            cache,
            [_FakeCorpusSource(docs)],
            _FakeExtractor(fn),
            record,
        )
        with pytest.raises(RuntimeError):
            await job.run(finalize=True)

        rebuilt = _make_job(
            tmp_path,
            _target(),
            cache,
            [_FakeCorpusSource(docs)],
            _FakeExtractor(fn),
            JobRecord(job_id=record.job_id, kind="ingestion", graph_id="med"),
        )
        result = await rebuilt.resume()
        await rebuilt.resume()

        assert result.finalized is True
        assert finalize.await_count == 2


def _schema(
    *, mode: str = "open", version: str = "3.2", extractable: bool = True
) -> GraphSchema:
    """Build a one-type schema with the given mode, version, and extractable flag."""
    return GraphSchema(
        version=version,
        mode=mode,
        entity_types=[
            EntityType(
                label="Disease",
                gliner_label="disease",
                description="A disease.",
                extractable=extractable,
            ),
        ],
    )


class TestSchemaVersion:
    """Schema version that keys cached extraction results."""

    def test_matches_for_identically_built_schemas(self) -> None:
        """Two schemas with the same definition share a schema version."""
        assert _schema_version(_schema()) == _schema_version(_schema())

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"mode": "strict"}, id="mode"),
            pytest.param({"version": "3.3"}, id="version"),
            pytest.param({"extractable": False}, id="extractable"),
        ],
    )
    def test_differs_when_mode_version_or_extractable_flag_changes(
        self, change: dict[str, object]
    ) -> None:
        """A changed mode, version, or extractable flag yields a new schema version."""
        assert _schema_version(_schema(**change)) != _schema_version(_schema())
