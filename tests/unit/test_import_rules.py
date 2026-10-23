"""Test that import direction rules are enforced.

Layer hierarchy (lower → higher):
1. common (types, version, schema)
2. base contracts (db.graph.base, db.vector.base, vector.embedding.base, llm.base)
3. provider implementations (db.graph.neo4j, db.vector.qdrant, etc.)
4. graph_construction
5. retrieval
6. ingestion
7. agents

Rules:
- A module can import from its own layer or any lower layer
- A module cannot import from a higher layer
- Cross imports between same-layer modules are allowed
"""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "am_diag"

# Map each module to its layer number (lower = more foundational)
LAYERS: dict[str, int] = {
    "am_diag.common": 1,
    # Base contracts and shared settings (config models are not pipeline logic)
    "am_diag.db.graph.base": 2,
    "am_diag.db.graph.errors": 2,
    "am_diag.db.graph.cypher_store": 2,
    "am_diag.db.graph.settings": 2,
    "am_diag.db.vector.base": 2,
    "am_diag.db.vector.settings": 2,
    "am_diag.vector.embedding.base": 2,
    "am_diag.vector.embedding.settings": 2,
    "am_diag.vector.reranking.base": 2,
    "am_diag.vector.reranking.settings": 2,
    "am_diag.llm.base": 2,
    "am_diag.graph_construction.config": 2,
    "am_diag.ingestion.embedding_config": 2,
    # Provider implementations
    "am_diag.db.graph.neo4j": 3,
    "am_diag.db.graph.falkordb": 3,
    "am_diag.db.graph.factory": 3,
    "am_diag.db.vector.qdrant": 3,
    "am_diag.db.vector.weaviate": 3,
    "am_diag.vector.embedding.openai": 3,
    "am_diag.vector.embedding.sentence_transformers": 3,
    "am_diag.vector.reranking.sentence_transformers": 3,
    # Graph construction
    "am_diag.graph_construction": 4,
    # Retrieval
    "am_diag.retrieval": 5,
    # Ingestion pipelines
    "am_diag.ingestion": 6,
    "am_diag.ingestion.models": 6,
    # Agents
    "am_diag.agents": 7,
}


def get_module_layer(module_path: str) -> int | None:
    """Return the layer number for a module path."""
    for prefix, layer in sorted(LAYERS.items(), key=lambda x: -len(x[0])):
        if module_path.startswith(prefix):
            return layer
    return None


def get_module_name_from_file(filepath: Path) -> str:
    """Convert a file path to its module name."""
    rel = filepath.relative_to(PACKAGE_ROOT.parent)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


class ImportRuleChecker(ast.NodeVisitor):
    """AST visitor that checks import direction rules."""

    def __init__(self, source_module: str) -> None:
        """Initialise the checker.

        Args:
            source_module: Fully qualified name of the module being checked.
        """
        self.source_module = source_module
        self.source_layer = get_module_layer(source_module) or 0
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Check all imports in an ``import x`` statement."""
        for alias in node.names:
            self._check_import(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check imports in a ``from x import y`` statement."""
        if node.module:
            self._check_import(node.module)

    def _check_import(self, imported_module: str) -> None:
        """Check if an import violates layer direction."""
        target_layer = get_module_layer(imported_module)
        if target_layer and target_layer > self.source_layer:
            self.violations.append(
                f"{self.source_module} (layer {self.source_layer}) "
                f"imports {imported_module} (layer {target_layer}) — "
                f"violates direction rule",
            )


def test_import_direction() -> None:
    """Verify no module imports from a higher layer."""
    violations: list[str] = []
    for pyfile in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "baml_client" in str(pyfile) or "baml_src" in str(pyfile):
            continue
        module_name = get_module_name_from_file(pyfile)
        source_layer = get_module_layer(module_name)
        if source_layer is None:
            continue
        try:
            tree = ast.parse(pyfile.read_text())
        except SyntaxError:
            continue
        checker = ImportRuleChecker(module_name)
        checker.visit(tree)
        violations.extend(checker.violations)

    assert not violations, (
        f"Import direction violations found ({len(violations)}):\n"
        + "\n".join(violations)
    )
