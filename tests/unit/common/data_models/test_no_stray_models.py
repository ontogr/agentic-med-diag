"""Guard test: enforce that no domain data models live outside common/data_models.

Project rule:
"All data structures live in am_diag/common/data_models; prefer extending over
adding; nothing in feature submodules."

This test greps for `class .*(BaseModel|DataPoint)`, `@dataclass`, and `TypedDict`
definitions under ``am_diag/`` outside of ``am_diag/common/data_models/`` and
fails if any are found (with configurable allowlist for BAML-generated code
and pydantic_settings subclasses).
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest


# Allowlisted patterns: files whose full paths match these patterns are exempt.
# BAML-generated code, settings subclasses, and TypedDict used as function
# return annotations are permitted.
_ALLOWED_PATTERNS: list[re.Pattern] = [
    re.compile(r"baml_client"),
    re.compile(r"types\.py$"),  # BAML generated types
    re.compile(r"stream_types\.py$"),  # BAML generated stream types
    re.compile(r"inlinedbaml\.py$"),  # BAML inline
    re.compile(r"__init__\.py$"),  # re-exports are fine
    # TypedDict pipeline states are not domain models
    re.compile(r"ingestion/search_pipeline\.py$"),
    # Ingestion report models (pipeline output types, not graph domain models)
    re.compile(r"ingestion/models\.py$"),
    # Agent capability model (startup helper, not a domain model)
    re.compile(r"agents/models\.py$"),
    # Reranker result types (framework-required transport types)
    re.compile(r"vector/reranking/base\.py$"),
    # Backend feature flags describe the storage contract, not domain data.
    re.compile(r"db/graph/base\.py$"),
    # Retrieval config and filter types (configuration, not domain models)
    re.compile(r"retrieval/filters\.py$"),
    re.compile(r"retrieval/config\.py$"),
    # Settings subclasses (framework-required, not domain models)
    re.compile(r"settings\.py$"),
]


# Root of the am_diag package (4 levels up from this test file).
# tests/am_diag/common/data_models/ → project root → am_diag/
_AM_DIAG_ROOT = Path(__file__).resolve().parents[4] / "am_diag"


def _iter_python_files(root: Path) -> list[Path]:
    """Recursively collect all ``.py`` files under *root*."""
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "__pycache__" in dirpath or ".mypy_cache" in dirpath:
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(Path(dirpath) / fn)
    return files


def _is_allowed(filepath: Path) -> bool:
    """Check if *filepath* matches any allowlist pattern."""
    path_str = str(filepath.as_posix())
    return any(p.search(path_str) for p in _ALLOWED_PATTERNS)


def _find_stray_domain_models(
    filepath: Path,
) -> list[tuple[int, str]]:
    """Return ``(lineno, line)`` tuples of suspected domain model definitions.

    Looks for:
    - ``class X(BaseModel)`` and ``class X(DataPoint)``
    - ``@dataclass`` class definitions
    - ``class X(TypedDict)`` definitions
    that are NOT in ``__init__.py`` and NOT in allowlisted paths.
    """
    if _is_allowed(filepath):
        return []

    violations: list[tuple[int, str]] = []
    try:
        tree = ast.parse(filepath.read_text("utf-8"), filename=str(filepath))
    except SyntaxError:
        return []  # skip files with syntax errors (e.g. generated stubs)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check bases
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            decorator_names = [
                d.id for d in node.decorator_list if isinstance(d, ast.Name)
            ]
            # Check for @dataclass
            if "dataclass" in decorator_names:
                violations.append((node.lineno, ast.unparse(node)))
            # Check for class X(BaseModel), class X(DataPoint), class X(TypedDict)
            for base in bases:
                if base in ("BaseModel", "DataPoint", "TypedDict"):
                    violations.append((node.lineno, ast.unparse(node)))
                    break

    return violations


# ── Test: no stray models in feature modules ──────────────────────────────


class TestNoStrayDomainModels:
    """Enforce that domain model classes live only in ``common/data_models``."""

    def test_no_stray_basemodel_outside_common(self) -> None:
        """Scan all ``am_diag/`` subpackages except ``common/data_models/``."""
        am_diag_root = _AM_DIAG_ROOT
        assert am_diag_root.is_dir(), f"am_diag not found at {am_diag_root}"

        data_models_dir = am_diag_root / "common" / "data_models"
        all_violations: list[tuple[str, int, str]] = []

        for pyfile in _iter_python_files(am_diag_root):
            # Skip the data_models directory itself
            if str(pyfile.resolve()).startswith(str(data_models_dir.resolve())):
                continue
            violations = _find_stray_domain_models(pyfile)
            for lineno, line in violations:
                rel_path = pyfile.relative_to(am_diag_root)
                all_violations.append((str(rel_path), lineno, line))

        if all_violations:
            msg_lines = [
                "Found domain model definitions outside am_diag/common/data_models/"
            ]
            for path, lineno, line in all_violations:
                msg_lines.append(f"  {path}:{lineno}  {line}")
            msg_lines.append(
                "\nRule 1: All domain models must be in common/data_models."
            )
            pytest.fail("\n".join(msg_lines))

    def test_no_dataclass_outside_common(self) -> None:
        """``@dataclass`` domain models must also live in ``common/data_models``."""
        am_diag_root = _AM_DIAG_ROOT
        assert am_diag_root.is_dir()

        data_models_dir = am_diag_root / "common" / "data_models"
        dataclass_pattern = re.compile(r"@dataclass")
        violations: list[str] = []

        for pyfile in _iter_python_files(am_diag_root):
            if str(pyfile.resolve()).startswith(str(data_models_dir.resolve())):
                continue
            if _is_allowed(pyfile):
                continue
            text = pyfile.read_text("utf-8")
            if dataclass_pattern.search(text):
                # Check if it's actually defining a dataclass, not importing it
                for match in dataclass_pattern.finditer(text):
                    # Look ahead for "class" within a few lines
                    subsequent = text[match.end() : match.end() + 200]
                    if "class " in subsequent:
                        rel_path = pyfile.relative_to(am_diag_root)
                        violations.append(
                            f"  {rel_path}:{text[: match.start()].count(chr(10)) + 1}"
                        )
                        break

        if violations:
            pytest.fail(
                "Found @dataclass definitions outside common/data_models:\n"
                + "\n".join(violations)
            )
