"""Public user-facing API for am_diag (Workspace, Project, Graph)."""

from __future__ import annotations

from am_diag.api.errors import (
    ConnectionError,  # noqa: A004
    EntityNotFoundError,
    GraphNotFoundError,
    JobError,
    OntologyMismatchError,
    ProjectNotFoundError,
    RegistryError,
    RelationNotFoundError,
    SchemaError,
    WorkspaceError,
)
from am_diag.api.graph import Graph
from am_diag.api.project import Project
from am_diag.api.registry import Registry
from am_diag.api.workspace import Workspace


__all__ = [
    "WorkspaceError",
    "ConnectionError",
    "EntityNotFoundError",
    "Graph",
    "GraphNotFoundError",
    "JobError",
    "OntologyMismatchError",
    "Project",
    "ProjectNotFoundError",
    "Registry",
    "RegistryError",
    "RelationNotFoundError",
    "SchemaError",
    "Workspace",
]
