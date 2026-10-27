"""Tests for the am-diag CLI argument parser and source resolution.

Verifies the subcommand grammar parses and that ``corpus:``/``loader:`` source
specs resolve against the relocated registries. No commands are executed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from am_diag.cli import main as cli_main
from am_diag.cli.main import _build_source, build_parser
from am_diag.cli.registries import (
    CORPUS_LOADER_REGISTRY,
    DATASET_REGISTRY,
    GRAPH_LOADER_REGISTRY,
)
from am_diag.common.data_models.job import JobItem, JobProgress, JobRecord
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA, OPTIMUSKG_SCHEMA
from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader
from am_diag.loaders.graph.drugbank import DrugBankGraphLoader
from am_diag.loaders.graph.gene_ontology import GeneOntologyGraphLoader


class TestParser:
    """Tests for the CLI parser grammar."""

    def test_graphs_create_parses(self):
        """Verify graphs create parses required arguments."""
        args = build_parser().parse_args(
            ["graphs", "create", "--graph-id", "med", "--neo4j-uri", "bolt://h:7687"]
        )
        assert args.command == "graphs"
        assert args.action == "create"
        assert args.graph_id == "med"

    def test_ingest_collects_multiple_sources(self):
        """Verify --source is repeatable and --finalize is a flag."""
        args = build_parser().parse_args(
            [
                "ingest",
                "--graph",
                "med",
                "--source",
                "corpus:pubmed",
                "--source",
                "loader:drugbank:/data/drugbank",
                "--reference-dir",
                "/data/reference",
                "--finalize",
            ]
        )
        assert args.source == ["corpus:pubmed", "loader:drugbank:/data/drugbank"]
        assert args.reference_dir == "/data/reference"
        assert args.finalize is True

    def test_search_takes_positional_query(self):
        """Verify search accepts a project and a positional query."""
        args = build_parser().parse_args(
            ["search", "--project", "clinical", "sepsis management"]
        )
        assert args.project == "clinical"
        assert args.query == "sepsis management"

    def test_jobs_resume_is_not_offered(self):
        """Verify jobs resume is rejected because jobs cannot resume across runs."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["jobs", "resume", "--job-id", "abc"])

    @pytest.mark.parametrize("limit", ["0", "-3"])
    def test_qa_limit_must_be_positive(self, limit):
        """Verify a non-positive --limit is rejected instead of meaning no limit."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["qa", "--project", "p", "--datasets", "medqa", "--limit", limit]
            )

    @pytest.mark.parametrize(
        "extra_args",
        [["--schema", "unknown"], ["--ontology-mode", "loose"]],
    )
    def test_graphs_create_rejects_unknown_schema_or_mode(self, extra_args):
        """Verify graphs create exits on a schema or ontology mode it does not offer."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "graphs",
                    "create",
                    "--graph-id",
                    "g",
                    "--neo4j-uri",
                    "bolt://h",
                    *extra_args,
                ]
            )

    def test_missing_subcommand_errors(self):
        """Verify a missing subcommand exits with an error."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestRegistries:
    """Tests for the relocated loader/dataset registries."""

    def test_registries_are_populated(self):
        """Verify the three registries expose the expected loaders."""
        assert "pubmed" in CORPUS_LOADER_REGISTRY
        assert "drugbank" in GRAPH_LOADER_REGISTRY
        assert "medqa" in DATASET_REGISTRY


class TestBuildSource:
    """Tests for resolving --source specs into loader instances."""

    def test_corpus_spec_builds_corpus_loader(self):
        """Verify a corpus spec resolves to its loader without extra arguments."""
        assert isinstance(_build_source("corpus:pubmed"), PubMedCorpusLoader)

    def test_graph_loader_spec_takes_a_source_dir(self):
        """Verify a graph loader that needs only a directory is built from the spec."""
        source = _build_source("loader:gene_ontology:/data/go")

        assert isinstance(source, GeneOntologyGraphLoader)

    def test_graph_loader_with_shared_services_uses_reference_dir(self):
        """Verify harmonizer and enricher are built from the reference directory."""
        source = _build_source(
            "loader:drugbank:/data/drugbank", reference_dir="/data/reference"
        )

        assert isinstance(source, DrugBankGraphLoader)
        assert str(source._harmonizer._ref.reference_dir) == "/data/reference"
        assert source._enricher._ref is source._harmonizer._ref

    def test_graph_loader_needing_reference_without_it_exits(self):
        """Verify a missing reference directory is reported, not a TypeError."""
        with pytest.raises(SystemExit, match="--reference-dir"):
            _build_source("loader:drugbank:/data/drugbank")

    def test_graph_loader_without_source_dir_exits(self):
        """Verify a loader spec with no directory is rejected with a clear message."""
        with pytest.raises(SystemExit, match="source directory"):
            _build_source("loader:gene_ontology")

    @pytest.mark.parametrize("spec", ["pubmed", "corpus:nope", "loader:nope:/x"])
    def test_unknown_spec_exits(self, spec):
        """Verify malformed or unknown specs exit with the offending spec."""
        with pytest.raises(SystemExit, match="Unknown source spec"):
            _build_source(spec)


class TestGraphsCreate:
    """Tests for the connection, vector, and schema built by ``graphs create``."""

    async def test_schema_defaults_to_medical_in_open_mode(self):
        """Verify graphs create passes the medical schema in open mode by default."""
        client = SimpleNamespace(
            graphs=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(graph_id="g"))
            )
        )
        args = build_parser().parse_args(
            ["graphs", "create", "--graph-id", "g", "--neo4j-uri", "bolt://h"]
        )

        await cli_main._handle_graphs(client, args)

        schema = client.graphs.create.await_args.kwargs["schema"]
        assert schema == MEDICAL_GRAPHRAG_SCHEMA
        assert schema.mode == "open"

    async def test_schema_and_ontology_mode_are_passed_to_create(self):
        """Verify --schema and --ontology-mode select the schema graph create gets."""
        client = SimpleNamespace(
            graphs=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(graph_id="g"))
            )
        )
        args = build_parser().parse_args(
            [
                "graphs",
                "create",
                "--graph-id",
                "g",
                "--neo4j-uri",
                "bolt://h",
                "--schema",
                "optimuskg",
                "--ontology-mode",
                "warn",
            ]
        )

        await cli_main._handle_graphs(client, args)

        schema = client.graphs.create.await_args.kwargs["schema"]
        assert schema.version == "3.2"
        assert schema.mode == "warn"

    async def test_ontology_mode_override_leaves_bundled_schema_unchanged(self):
        """Verify overriding the mode on the CLI does not mutate the bundled schema."""
        client = SimpleNamespace(
            graphs=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(graph_id="g"))
            )
        )
        args = build_parser().parse_args(
            [
                "graphs",
                "create",
                "--graph-id",
                "g",
                "--neo4j-uri",
                "bolt://h",
                "--schema",
                "optimuskg",
                "--ontology-mode",
                "strict",
            ]
        )

        await cli_main._handle_graphs(client, args)

        assert client.graphs.create.await_args.kwargs["schema"].mode == "strict"
        assert OPTIMUSKG_SCHEMA.mode == "open"

    async def test_falkordb_inline_credentials_are_not_registered(self):
        """Reject URL secrets before they reach the registry writer."""
        client = SimpleNamespace(graphs=SimpleNamespace(create=AsyncMock()))
        args = build_parser().parse_args(
            [
                "graphs",
                "create",
                "--graph-id",
                "g",
                "--graph-backend",
                "falkordb",
                "--falkordb-url",
                "rediss://user:secret@host:6379",
            ]
        )
        with pytest.raises(ValueError, match="use user_env and password_env instead"):
            await cli_main._handle_graphs(client, args)
        client.graphs.create.assert_not_awaited()

    async def test_falkordb_record_has_optional_credentials(self):
        """Register FalkorDB without requiring Neo4j or Redis credentials."""
        client = SimpleNamespace(
            graphs=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(graph_id="g"))
            )
        )
        args = build_parser().parse_args(
            [
                "graphs",
                "create",
                "--graph-id",
                "g",
                "--graph-backend",
                "falkordb",
                "--falkordb-url",
                "rediss://h:6379",
                "--falkordb-graph",
                "medical",
            ]
        )
        await cli_main._handle_graphs(client, args)
        conn = client.graphs.create.await_args.kwargs["neo4j"]
        assert conn.model_dump() == {
            "backend": "falkordb",
            "uri": "rediss://h:6379",
            "database": "medical",
            "user_env": None,
            "password_env": None,
        }

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            ([], "--neo4j-uri is required"),
            (["--graph-backend", "falkordb"], "--falkordb-url is required"),
            (
                ["--neo4j-uri", "bolt://h", "--falkordb-url", "redis://h"],
                "require --graph-backend falkordb",
            ),
        ],
    )
    async def test_missing_or_conflicting_backend_flags(self, argv, message):
        """Reject incomplete backend configuration before creating a record."""
        args = build_parser().parse_args(["graphs", "create", "--graph-id", "g", *argv])
        with pytest.raises(SystemExit, match=message):
            await cli_main._handle_graphs(SimpleNamespace(), args)

    @pytest.mark.parametrize(
        ("argv", "expected_env"),
        [
            ([], "QDRANT_URL"),
            (["--vector-backend", "weaviate"], "WEAVIATE_URL"),
            (["--vector-backend", "weaviate", "--vector-url-env", "MY_URL"], "MY_URL"),
        ],
    )
    async def test_vector_url_env_follows_backend(self, argv, expected_env):
        """Verify the URL env var defaults per backend unless overridden."""
        client = SimpleNamespace(
            graphs=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(graph_id="g"))
            )
        )
        args = build_parser().parse_args(
            ["graphs", "create", "--graph-id", "g", "--neo4j-uri", "bolt://h", *argv]
        )

        await cli_main._handle_graphs(client, args)

        vector = client.graphs.create.await_args.kwargs["vector"]
        assert vector.url_env == expected_env


class TestHandleQa:
    """Tests for the ``qa`` subcommand."""

    @staticmethod
    def _client(items):
        record = JobRecord(
            job_id="job1",
            kind="qa",
            project_id="p",
            status="partial",
            progress=JobProgress(total=len(items), succeeded=1, failed=1),
        )
        job = SimpleNamespace(run=AsyncMock(return_value=record), items=lambda: items)
        project = SimpleNamespace(answer_batch=AsyncMock(return_value=job))
        return SimpleNamespace(project=lambda _id: project), project

    async def test_strips_names_forwards_limit_and_writes_results(
        self, monkeypatch, tmp_path
    ):
        """Verify spaced dataset names load with the limit and results are saved."""
        loaded = {}

        def registry_entry(name):
            def factory():
                dataset = MagicMock()
                dataset.load.side_effect = lambda **kw: (
                    loaded.setdefault(name, kw),
                    [SimpleNamespace(question=f"{name}-q")],
                )[1]
                return dataset

            return factory

        monkeypatch.setattr(
            cli_main,
            "DATASET_REGISTRY",
            {"medqa": registry_entry("medqa"), "medmcqa": registry_entry("medmcqa")},
        )
        items = [
            JobItem(
                item_id="a",
                sequence_index=0,
                kind="question",
                ref={"question": "medqa-q"},
                status="succeeded",
                output={"answer": "yes"},
            ),
            JobItem(
                item_id="b",
                sequence_index=1,
                kind="question",
                ref={"question": "medmcqa-q"},
                status="failed",
                error={"type": "RuntimeError", "message": "boom"},
            ),
        ]
        client, project = self._client(items)
        output = tmp_path / "out.json"
        args = build_parser().parse_args(
            [
                "qa",
                "--project",
                "p",
                "--datasets",
                "medqa, medmcqa",
                "--limit",
                "5",
                "--output",
                str(output),
            ]
        )

        await cli_main._handle_qa(client, args)

        assert loaded == {"medqa": {"limit": 5}, "medmcqa": {"limit": 5}}
        assert project.answer_batch.await_args.args[0] == ["medqa-q", "medmcqa-q"]
        report = json.loads(output.read_text("utf-8"))
        assert report["status"] == "partial"
        assert report["items"][0]["output"] == {"answer": "yes"}
        assert report["items"][1]["error"]["message"] == "boom"

    async def test_unknown_dataset_exits(self):
        """Verify an unknown dataset name is reported instead of a KeyError."""
        args = build_parser().parse_args(["qa", "--project", "p", "--datasets", "nope"])

        with pytest.raises(SystemExit, match="Unknown dataset"):
            await cli_main._handle_qa(SimpleNamespace(), args)
