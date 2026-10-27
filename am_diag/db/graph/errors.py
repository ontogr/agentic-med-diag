"""Backend-independent graph failures for storage callers."""


class GraphStoreError(Exception):
    """Base class for graph store failures."""


class GraphUnavailableError(GraphStoreError):
    """The graph backend could not be reached."""


class GraphQueryRejected(GraphStoreError):  # noqa: N818
    """The server refused a query."""


class GraphWriteRejected(GraphQueryRejected):
    """A write was attempted through read-only execution."""


class OntologyVersionMismatch(GraphStoreError):  # noqa: N818
    """The graph already mirrors a different ontology than the one requested.

    A graph holds at most one ontology. Changing it requires a new graph.
    """

    def __init__(self, stored: str, requested: str) -> None:
        """Describe the conflict using the stored and requested versions."""
        if stored != requested:
            message = (
                f"graph holds ontology {stored!r}, requested {requested!r}; "
                "build a new graph to change ontology"
            )
        else:
            message = (
                f"graph holds a different definition of ontology {stored!r}; "
                "build a new graph to change ontology"
            )
        super().__init__(message)
