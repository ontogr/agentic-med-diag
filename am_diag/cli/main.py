"""Consolidated ``am-diag`` command-line interface over a Workspace.

Subcommands: graphs / projects (registry admin), ingest / finalize (durable
ingestion), search (federated retrieval), qa (durable batch evaluation), and
jobs (list/get). All real work is delegated to :class:`Workspace`.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
from typing import Any

from am_diag.api.workspace import Workspace
from am_diag.cli.registries import (
    CORPUS_LOADER_REGISTRY,
    DATASET_REGISTRY,
    GRAPH_LOADER_REGISTRY,
    SCHEMA_REGISTRY,
)
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.reference_data import ReferenceData


logger = logging.getLogger(__name__)


def _build_graph_loader(
    name: str, source_dir: str, reference_dir: str | None
) -> GraphLoader:
    """Build a registered graph loader over ``source_dir``.

    Loaders that take a ``reference``, ``harmonizer`` or ``enricher`` get them
    built from ``reference_dir``.

    Raises:
        SystemExit: If the loader needs reference data and ``reference_dir`` is
            not set.
    """
    loader_cls = GRAPH_LOADER_REGISTRY[name]
    params = inspect.signature(loader_cls).parameters
    shared = {"reference", "harmonizer", "enricher"} & params.keys()
    kwargs: dict[str, Any] = {}
    if shared:
        if reference_dir is None:
            raise SystemExit(
                f"Loader {name!r} needs reference data: pass --reference-dir"
            )
        reference = ReferenceData(reference_dir)
        builders = {
            "reference": lambda: reference,
            "harmonizer": lambda: Harmonizer(reference),
            "enricher": lambda: NodeEnricher(reference),
        }
        kwargs = {key: builders[key]() for key in shared}
    return loader_cls(source_dir, **kwargs)


def _build_source(spec: str, reference_dir: str | None = None) -> object:
    """Resolve a ``corpus:<name>`` or ``loader:<name>:<source_dir>`` spec.

    Args:
        spec: Source spec. Corpus loaders take no arguments. Graph loaders take
            the directory that holds their source files.
        reference_dir: Reference data directory for graph loaders that need it.

    Returns:
        A loader instance.

    Raises:
        SystemExit: If the spec is malformed, names an unknown loader, or a
            graph loader lacks its source directory or reference data.
    """
    kind, _, rest = spec.partition(":")
    if kind == "corpus" and rest in CORPUS_LOADER_REGISTRY:
        return CORPUS_LOADER_REGISTRY[rest]()
    if kind == "loader":
        name, _, source_dir = rest.partition(":")
        if name in GRAPH_LOADER_REGISTRY:
            if not source_dir:
                raise SystemExit(
                    f"Loader {name!r} needs a source directory: loader:{name}:<dir>"
                )
            return _build_graph_loader(name, source_dir, reference_dir)
    raise SystemExit(f"Unknown source spec: {spec!r}")


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value!r}")
    return number


def _add_graphs(sub: Any) -> None:
    """Register the ``graphs`` subcommands."""
    graphs = sub.add_parser("graphs", help="Manage graph deployments")
    g_sub = graphs.add_subparsers(dest="action", required=True)
    g_create = g_sub.add_parser("create", help="Register a graph")
    g_create.add_argument("--graph-id", required=True)
    g_create.add_argument(
        "--graph-backend", choices=["neo4j", "falkordb"], default="neo4j"
    )
    g_create.add_argument("--neo4j-uri")
    g_create.add_argument("--neo4j-user-env", default="NEO4J_USER")
    g_create.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD")
    g_create.add_argument("--neo4j-database", default="neo4j")
    g_create.add_argument("--falkordb-url")
    g_create.add_argument("--falkordb-user-env")
    g_create.add_argument("--falkordb-password-env")
    g_create.add_argument("--falkordb-graph", default="am_diag")
    g_create.add_argument(
        "--vector-backend", choices=["qdrant", "weaviate"], default="qdrant"
    )
    g_create.add_argument(
        "--vector-url-env",
        default=None,
        help="Env var holding the vector store URL (default: <BACKEND>_URL)",
    )
    g_create.add_argument("--vector-api-key-env", default=None)
    g_create.add_argument("--name", default=None)
    g_create.add_argument(
        "--schema",
        choices=list(SCHEMA_REGISTRY),
        default="medical",
        help=(
            "Bundled ontology the graph is tied to. The schema is fixed when the "
            "graph is created (default: medical)"
        ),
    )
    g_create.add_argument(
        "--ontology-mode",
        choices=["open", "warn", "strict"],
        default="open",
        help=(
            "Handling of text-extracted facts that break the ontology: open keeps "
            "them, warn keeps them with a warning, strict drops them (default: open)"
        ),
    )
    g_sub.add_parser("list", help="List graphs")
    g_delete = g_sub.add_parser("delete", help="Delete a graph")
    g_delete.add_argument("--graph-id", required=True)
    g_delete.add_argument("--drop-data", action="store_true")


def _add_projects(sub: Any) -> None:
    """Register the ``projects`` subcommands."""
    projects = sub.add_parser("projects", help="Manage projects")
    p_sub = projects.add_subparsers(dest="action", required=True)
    p_create = p_sub.add_parser("create", help="Create a project")
    p_create.add_argument("--project", required=True)
    p_create.add_argument("--graphs", default="")
    p_sub.add_parser("list", help="List projects")
    p_add = p_sub.add_parser("add-graph", help="Add a graph to a project")
    p_add.add_argument("--project", required=True)
    p_add.add_argument("--graph", required=True)
    p_rm = p_sub.add_parser("remove-graph", help="Remove a graph from a project")
    p_rm.add_argument("--project", required=True)
    p_rm.add_argument("--graph", required=True)


def _add_ops(sub: Any) -> None:
    """Register the ingest/finalize/search/qa/jobs subcommands."""
    ingest = sub.add_parser("ingest", help="Ingest sources into a graph")
    ingest.add_argument("--graph", required=True)
    ingest.add_argument(
        "--source",
        action="append",
        required=True,
        help="corpus:<name> or loader:<name>:<source_dir> (repeatable)",
    )
    ingest.add_argument(
        "--reference-dir",
        default=None,
        help="Reference data directory for graph loaders that need it",
    )
    ingest.add_argument("--finalize", action="store_true")

    finalize = sub.add_parser("finalize", help="Run community detection on a graph")
    finalize.add_argument("--graph", required=True)

    search = sub.add_parser("search", help="Federated search over a project")
    search.add_argument("--project", required=True)
    search.add_argument("--graphs", default=None)
    search.add_argument("--recipe", default="hybrid_rrf")
    search.add_argument("query")

    qa = sub.add_parser("qa", help="Batch QA evaluation over a project")
    qa.add_argument("--project", required=True)
    qa.add_argument("--datasets", required=True)
    qa.add_argument("--limit", type=_positive_int, default=None)
    qa.add_argument("--output", default="qa_results.json")

    jobs = sub.add_parser("jobs", help="Inspect durable jobs")
    j_sub = jobs.add_subparsers(dest="action", required=True)
    j_sub.add_parser("list", help="List jobs")
    j_get = j_sub.add_parser("get", help="Show one job")
    j_get.add_argument("--job-id", required=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="am-diag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_graphs(sub)
    _add_projects(sub)
    _add_ops(sub)
    return parser


async def _run(args: argparse.Namespace) -> None:
    """Dispatch a parsed command against a Workspace."""
    async with Workspace.from_env() as client:
        if args.command == "graphs":
            await _handle_graphs(client, args)
        elif args.command == "projects":
            await _handle_projects(client, args)
        elif args.command == "ingest":
            sources = [_build_source(s, args.reference_dir) for s in args.source]
            job = await client.graph(args.graph).ingest(sources)
            record = await job.run(finalize=args.finalize)
            pct = record.progress.percent_complete
            print(f"Job {record.job_id}: {record.status} ({pct:.0f}%)")
        elif args.command == "finalize":
            count = await client.graph(args.graph).finalize()
            print(f"Detected {count} communities")
        elif args.command == "search":
            graphs = args.graphs.split(",") if args.graphs else None
            results = await client.project(args.project).search(
                args.query, graphs=graphs, recipe=args.recipe
            )
            for r in results:
                print(f"[{r.source_graph_id}] {r.score:.3f} {r.source}")
        elif args.command == "qa":
            await _handle_qa(client, args)
        elif args.command == "jobs":
            await _handle_jobs(client, args)


async def _handle_graphs(client: Workspace, args: argparse.Namespace) -> None:
    """Handle the ``graphs`` subcommands."""
    if args.action == "create":
        if args.graph_backend == "falkordb":
            if not args.falkordb_url:
                raise SystemExit("--falkordb-url is required for the falkordb backend")
            if args.neo4j_uri:
                raise SystemExit("--neo4j-uri cannot be used with the falkordb backend")
            connection = GraphConnection(
                backend="falkordb",
                uri=args.falkordb_url,
                user_env=args.falkordb_user_env,
                password_env=args.falkordb_password_env,
                database=args.falkordb_graph,
            )
        else:
            if not args.neo4j_uri:
                raise SystemExit("--neo4j-uri is required for the neo4j backend")
            if (
                args.falkordb_url
                or args.falkordb_user_env
                or args.falkordb_password_env
            ):
                raise SystemExit(
                    "FalkorDB connection flags require --graph-backend falkordb"
                )
            connection = GraphConnection(
                uri=args.neo4j_uri,
                user_env=args.neo4j_user_env,
                password_env=args.neo4j_password_env,
                database=args.neo4j_database,
            )
        # Copy the bundled schema so the mode override never mutates the registry.
        schema = SCHEMA_REGISTRY[args.schema].model_copy(
            update={"mode": args.ontology_mode}, deep=True
        )
        rec = await client.graphs.create(
            args.graph_id,
            neo4j=connection,
            vector=VectorConnection(
                backend=args.vector_backend,
                url_env=args.vector_url_env or f"{args.vector_backend.upper()}_URL",
                api_key_env=args.vector_api_key_env,
            ),
            schema=schema,
            name=args.name,
        )
        print(f"Created graph {rec.graph_id}")
    elif args.action == "list":
        for rec in await client.graphs.list():
            print(rec.graph_id)
    elif args.action == "delete":
        await client.graphs.delete(args.graph_id, drop_data=args.drop_data)
        print(f"Deleted graph {args.graph_id}")


async def _handle_projects(client: Workspace, args: argparse.Namespace) -> None:
    """Handle the ``projects`` subcommands."""
    if args.action == "create":
        graph_ids = [g for g in args.graphs.split(",") if g]
        rec = await client.projects.create(args.project, graph_ids=graph_ids)
        print(f"Created project {rec.project_id}")
    elif args.action == "list":
        for rec in await client.projects.list():
            print(rec.project_id)
    elif args.action == "add-graph":
        await client.project(args.project).add_graph(args.graph)
        print(f"Added {args.graph} to {args.project}")
    elif args.action == "remove-graph":
        await client.project(args.project).remove_graph(args.graph)
        print(f"Removed {args.graph} from {args.project}")


async def _handle_qa(client: Workspace, args: argparse.Namespace) -> None:
    """Handle the ``qa`` subcommand: load datasets and run a durable QAJob."""
    questions: list[str] = []
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in names if name not in DATASET_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(DATASET_REGISTRY))}"
        )
    for name in names:
        samples = DATASET_REGISTRY[name]().load(limit=args.limit)
        questions.extend(s.question for s in samples)
    job = await client.project(args.project).answer_batch(questions)
    record = await job.run()
    print(f"QA job {record.job_id}: {record.status}, {len(questions)} questions")
    report = {
        "job_id": record.job_id,
        "status": record.status,
        "progress": record.progress.model_dump(),
        "items": [item.model_dump(mode="json") for item in job.items()],
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


async def _handle_jobs(client: Workspace, args: argparse.Namespace) -> None:
    """Handle the ``jobs`` subcommands."""
    if args.action == "list":
        for rec in client.jobs.list():
            print(f"{rec.job_id} {rec.kind} {rec.status}")
    elif args.action == "get":
        rec = client.jobs.get(args.job_id)
        print(rec.model_dump_json(indent=2))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (``am-diag``)."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
