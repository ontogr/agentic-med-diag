# Contributing to am_diag

Thank you for your interest in contributing. We welcome contributions of all kinds — new corpus loaders, evaluation datasets, extraction and retrieval components, bug fixes, documentation improvements, and experimental features.

This guide covers how to set up your environment, the project conventions, and how to submit changes.

## Development Environment

The project uses [uv](https://github.com/astral-sh/uv) for dependency management and Python ≥ 3.11.

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/agentic-med-diag.git
   cd agentic-med-diag
   ```

2. Install uv and sync all dependencies (including dev tooling):

   ```bash
   pip install uv
   make sync
   ```

3. Install the pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

4. Copy the example environment file and add your credentials. The ingestion and QA commands, and the integration tests for the graph, vector store, and LLM components, need a Neo4j instance, a vector store (Qdrant or Weaviate), and an OpenAI-compatible LLM endpoint. The corpus and dataset loader integration tests need only network access to Hugging Face:

   ```bash
   cp .env.example .env
   ```

Run all Python through `uv run` so commands use the project's environment.

## Project Structure

```text
am_diag/
├── common/
│   ├── data_models/        # all domain models (Entity, Relation, Community, Chunk, ...)
│   ├── cypher/             # externalised .cypher files + loader
│   └── schema/             # GraphSchema + MEDICAL_GRAPHRAG_SCHEMA
├── chunking/               # text chunkers
├── graph_construction/     # extraction, normalization, aggregation, resolution, communities
├── ingestion/              # durable ingestion job engine + cache
├── db/
│   ├── graph/              # Neo4j client + record serialization
│   └── vector/             # Qdrant / Weaviate stores
├── vector/                 # embedders + rerankers
├── retrieval/              # methods, retrievers, rerankers, recipes, SearchEngine
├── agents/                 # DeepAgents harness
├── evaluation/             # dataset metrics + judging
├── api/                    # public API: Workspace, Graph, Project
├── cli/                    # am-diag CLI + loader registries
├── llm/                    # BAML sources + generated client
└── loaders/                # corpus, dataset and graph loaders
tests/
├── _env.py                 # shared env helpers
├── conftest.py             # directory-based collection hook
├── unit/                   # unit tests — mock all external services
│   ├── conftest.py         # env defaults for unit tests
│   ├── ingestion/          # mirrors am_diag/ingestion
│   ├── loaders/            # mirrors am_diag/loaders
│   ├── db/                 # mirrors am_diag/db
│   ├── ...                 # other modules mirroring am_diag/
└── integration/            # integration tests — external services
    ├── conftest.py         # env defaults
    ├── ingestion/          # mirrors am_diag/ingestion
    └── loaders/            # mirrors am_diag/loaders
```

The package uses a flat layout (`am_diag/` at the root, no `src/`).

## Code Style

- **Formatting and linting:** [Ruff](https://docs.astral.sh/ruff/) for both. Run `make lint-fmt` to format and auto-fix.
- **Type checking:** [ty](https://github.com/astral-sh/ty) (not mypy). Run `make lint-typing`.
- **Docstrings:** Google-style on all public modules, classes, and functions. Add non-trivial inline comments where intent is not obvious; do not restate the code.
- **Async:** every public component API is async.
- **Configuration:** components take their config at construction and operands in a single public async method. Configuration objects subclass `pydantic_settings.BaseSettings` with an env prefix (`NEO4J_`, `QDRANT_`, `EMBEDDING_`, `EXTRACTION_`, `RETRIEVAL_`, `AGENT_`, etc.).

Two architecture rules are enforced by guard tests:

- All domain models live in `am_diag/common/data_models` — none in feature submodules.
- No multi-line Cypher string literals in Python — queries live in `.cypher` files under `am_diag/common/cypher/` and are loaded by the thin loader.

## Adding Components

- **Corpus loader** (`am_diag/loaders/corpus/`) — subclass `CorpusLoader`, set the `corpus_name` / `hf_repo` class variables, implement `_row_to_document`, and stream `Document` batches via `astream()`. HuggingFace iterators run in `asyncio.to_thread` so they do not block the event loop. Register it in the `CORPUS_LOADER_REGISTRY` in `am_diag/cli/registries.py`.
- **Dataset loader** (`am_diag/loaders/dataset/`) — subclass `QADataset`, set `dataset_name` / `hf_repo`, and implement `load()` returning `list[QASample]`. Register it in the `DATASET_REGISTRY` in `am_diag/cli/registries.py`.
- **Graph-construction / retrieval component** — keep it pure and storage-agnostic, accept the `GraphSchema` where relevant, and follow the per-family base class.
- **BAML function** — edit the `.baml` sources under `am_diag/llm/baml_src/`, then regenerate the typed client with `make baml-gen`. Never edit `am_diag/llm/baml_client/` by hand; it is fully generated.

## Testing

The test suite follows a directory-based split with two distinct conventions:

1. **Unit test** → put it under `tests/unit/…`, name `test_*.py`, no markers.
   - Sockets are blocked by default (`--disable-socket`); asyncio's AF_UNIX wakeup pipe is covered by `--allow-unix-socket`.
   - **Never** add `enable_socket` — if a unit test raises `SocketBlockedError`, fix the mock instead.
   - Use `@pytest.mark.allow_hosts([...])` only for intentional loopback-server tests, with an inline comment.

2. **Integration test** → put it under `tests/integration/…`, name `test_*.py` (no `_integration` suffix).
   - Markers `integration` + `enable_socket` + `flaky(reruns=3, reruns_delay=30, rerun_except=[AssertionError])` are **auto-applied** by `tests/conftest.py` based on the directory — do not add them manually.
   - The auto-marking hook gracefully skips integration tests that require missing external service configuration (checked against process env and repo-root `.env`). Real creds come from env/`.env`; the integration conftest only fills *absent* import-safety placeholders.

- `pytest-asyncio` runs in `auto` mode — no `@pytest.mark.asyncio` decorator is needed.
- `--strict-markers` and `--strict-config` are enabled — a typo'd marker or misused manual marker raises a hard error.
- Mock all external services (Neo4j, Qdrant, Weaviate, HuggingFace) in unit tests, patching at the driver level.
- Unit tests run in parallel (`-n auto --dist loadscope`); integration tests also run in parallel but can be adjusted per-service if shared-state races arise. Avoid shared mutable state between test classes.
- The `testpaths` default `["tests/unit"]` ensures a plain `pytest` only discovers unit tests. Use the `make` targets below for all common workflows.

```bash
make test                  # unit tests with coverage
make test-integration      # integration tests (requires external services) — runs tests/integration

# run a single file, class, or test
uv run pytest tests/unit/db/graph/test_client.py::TestNeo4jClient -v
uv run pytest tests/unit/db/vector/test_qdrant.py -k "test_search" -v
```

## Make Commands

```bash
make sync                  # install all dependencies
make test                  # run unit tests with coverage
make test-integration      # run integration tests
make lint-fmt              # format + auto-fix
make lint-check            # check only, no modifications
make lint-typing           # type check with ty
make lint-all              # format + lint + type check + typos
make baml-gen              # regenerate the BAML client from am_diag/llm/baml_src/
make clean                 # remove build artifacts and caches
```

## Submitting a Pull Request

1. Create a feature branch off `main` (for example, `feat/add-bioasq-loader` or `fix/relation-hydration`).
2. Make your change with tests and Google-style docstrings.
3. Run `make lint-all` and `make test` and make sure both pass. If you touched any `.baml` file, run `make baml-gen` and commit the regenerated client.
4. Update `README.md` if your change affects user-facing behaviour.
5. Open a pull request against `main` with a clear description of the change and the motivation. Link any related issue.

## Reporting Issues

Please use [GitHub Issues](https://github.com/ontogr/agentic-med-diag/issues) to report bugs or request features. For bugs, include a minimal reproduction, the expected and actual behaviour, and your environment (OS, Python version, relevant service versions).

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
