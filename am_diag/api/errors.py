"""Exception hierarchy for the public ``am_diag`` API.

All public-facing errors derive from :class:`WorkspaceError` so callers can catch a
single base type. Subtypes name the failing resource or stage.
"""

from __future__ import annotations


class WorkspaceError(Exception):
    """Base class for all public ``am_diag`` API errors."""


class RegistryError(WorkspaceError):
    """The registry rejected an operation (for example a duplicate id)."""


class ProjectNotFoundError(WorkspaceError):
    """No project with the requested id exists in the registry."""


class GraphNotFoundError(WorkspaceError):
    """No graph with the requested id exists in the registry."""


class EntityNotFoundError(WorkspaceError):
    """No entity with the requested id exists in the graph."""


class RelationNotFoundError(WorkspaceError):
    """No relation with the requested id exists in the graph."""


class ConnectionError(WorkspaceError):  # noqa: A001
    """A backend connection could not be established.

    Raised when a referenced environment variable is unset or a configured
    backend is unreachable.
    """


class SchemaError(WorkspaceError):
    """A schema or query constraint was violated."""


class OntologyMismatchError(SchemaError):
    """A graph already holds a different ontology than the one requested.

    A graph is tied to the ontology it was created with. Use a new graph to
    change it.
    """


class JobError(WorkspaceError):
    """A durable job operation failed."""
