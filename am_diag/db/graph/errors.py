"""Backend-independent graph failures for storage callers."""


class GraphStoreError(Exception):
    """Base class for graph store failures."""


class GraphUnavailableError(GraphStoreError):
    """The graph backend could not be reached."""


class GraphQueryRejected(GraphStoreError):  # noqa: N818
    """The server refused a query."""


class GraphWriteRejected(GraphQueryRejected):
    """A write was attempted through read-only execution."""
