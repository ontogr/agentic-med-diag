"""Generic DataPoint → Neo4j parameter serializer.

Converts DataPoint subclasses to the flat dict shapes expected by the
.cypher templates in common/cypher/. UUID fields become strings; nested
models and dicts become JSON strings; all other values pass through.

Each public function corresponds to one .cypher template's expected
``{id, props}`` or ``{id, head_id, tail_id, props}`` record shape.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, TypeVar, get_args
from uuid import UUID

from pydantic import BaseModel

from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema


# Provenance lists promoted to top-level (queryable, union-preserving) node/edge
# properties. Kept out of ``schema_properties`` on readback so they do not leak
# into it — the authoritative copy round-trips via the ``provenance`` JSON blob.
_PROMOTED_PROVENANCE_LISTS = frozenset({"chunk_ids", "surface_forms"})


def _json_dumps(v: Any) -> str:
    """Serialize a value to JSON, handling UUID objects.

    Pydantic ``model_dump()`` in Python mode preserves UUID objects as
    ``uuid.UUID`` instances, which ``json.dumps`` cannot serialize natively.
    ``default=str`` coerces UUIDs (and any other non-serializable types) to
    their string representation.

    Args:
        v: The value to serialize (typically a dict or list).

    Returns:
        A JSON string with UUIDs converted to strings.
    """
    return json.dumps(v, default=str)


def _is_homogeneous_primitive_list(v: list[Any]) -> bool:
    """Return whether *v* can be stored as a native Neo4j list property."""
    if not v:
        return True
    kinds = {bool if isinstance(x, bool) else type(x) for x in v}
    if kinds <= {int, float}:
        return True
    return len(kinds) == 1 and next(iter(kinds)) in (str, bool)


def _serialize_value(v: Any) -> Any:
    """Convert a single value to a Neo4j-safe scalar, list, or JSON string.

    UUIDs become strings and Pydantic models / dicts become JSON strings. Lists
    of primitives are kept as native Neo4j list properties (so they stay
    queryable, e.g. ``x IN n.aliases``); a list containing nested structures
    (dicts, lists, or non-primitive scalars) is JSON-encoded instead, since
    Neo4j only stores homogeneous primitive lists.

    Args:
        v: The value to convert.

    Returns:
        A Neo4j-compatible scalar, primitive list, or JSON string.
    """
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, BaseModel):
        return v.model_dump_json()
    if isinstance(v, dict):
        return _json_dumps(v)
    if isinstance(v, list):
        if _is_homogeneous_primitive_list(v):
            return v
        return _json_dumps(v)
    return v


def _split_properties(dp: DataPoint, raw: dict[str, Any]) -> dict[str, Any]:
    """Spread ``schema_properties`` entries into top-level (queryable) keys.

    Scalar and primitive-list properties become individual Neo4j properties so
    they can be filtered in Cypher; genuinely nested values (dicts, or lists
    containing dicts) are kept JSON-encoded under a residual ``schema_properties``
    key. A property whose name collides with a core model field is rejected, so
    spreading can never shadow a structural field such as ``score`` or ``label``.

    Args:
        dp: The DataPoint being serialized (its ``model_fields`` define the
            reserved core names).
        raw: ``model_dump`` output for ``dp``.

    Returns:
        ``raw`` with ``schema_properties`` flattened; nested residue, if any,
        re-attached under ``schema_properties``.

    Raises:
        ValueError: If a ``schema_properties`` key collides with a core field.
    """
    core = set(type(dp).model_fields)
    props = dict(raw)
    extra = props.pop("schema_properties", None) or {}
    nested: dict[str, Any] = {}
    for key, value in extra.items():
        if key in core or key in _PROMOTED_PROVENANCE_LISTS:
            raise ValueError(
                f"schema_properties key {key!r} collides with a core or "
                f"reserved {type(dp).__name__} field"
            )
        if isinstance(value, dict) or (
            isinstance(value, list) and any(isinstance(x, (dict, list)) for x in value)
        ):
            nested[key] = value
        else:
            props[key] = value
    if nested:
        props["schema_properties"] = nested
    return props


def to_params(dp: DataPoint, *, drop: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Serialize a DataPoint to a flat dict for Neo4j Cypher parameters.

    UUID fields become strings; nested BaseModels and dicts/lists become
    JSON strings; primitives pass through. Fields in ``drop`` are excluded.

    Args:
        dp: The DataPoint to serialize.
        drop: Field names to exclude from the output dict.

    Returns:
        A flat dict safe to use as Neo4j Cypher parameters.
    """
    raw = dp.model_dump(exclude=set(drop))
    flat = _split_properties(dp, raw)
    return {k: _serialize_value(v) for k, v in flat.items()}


_T = TypeVar("_T", bound=DataPoint)


def reconstruct_schema_properties(
    cls: type[DataPoint], props: dict[str, Any]
) -> dict[str, Any]:
    """Re-collect spread + residual properties into a ``schema_properties`` mapping.

    Inverse of ``_split_properties``: every non-core property plus the residual
    JSON-encoded nested block recombine into the original ``schema_properties``.

    Args:
        cls: DataPoint subclass whose ``model_fields`` define the core names.
        props: Property map read back from the graph node/edge.

    Returns:
        The reconstructed ``schema_properties`` mapping.
    """
    core = set(cls.model_fields)
    residual_raw = props.get("schema_properties")
    if isinstance(residual_raw, str):
        residual = json.loads(residual_raw) if residual_raw else {}
    else:
        residual = residual_raw or {}
    extra = {
        k: v
        for k, v in props.items()
        if k not in core
        and k != "schema_properties"
        and k not in _PROMOTED_PROVENANCE_LISTS
    }
    return {**extra, **residual}


def _is_string_field(cls: type[DataPoint], name: str) -> bool:
    """Return whether *name* is a plain ``str`` (or ``str | None``) model field."""
    annotation = cls.model_fields[name].annotation
    return (
        annotation is str
        or set(get_args(annotation)) <= {str, type(None)}
        and bool(get_args(annotation))
    )


def from_props(cls: type[_T], props: dict[str, Any]) -> _T:
    """Rebuild a DataPoint from a Neo4j property map (inverse of ``to_params``).

    Re-collects spread + residual properties into ``schema_properties`` and
    JSON-decodes any core field that was stored as an encoded nested value (e.g.
    ``provenance``).

    Args:
        cls: Target DataPoint subclass.
        props: Property map read back from the graph node/edge.

    Returns:
        A validated instance of ``cls``.
    """
    core = set(cls.model_fields)
    data = {k: v for k, v in props.items() if k in core}
    # Core fields holding nested models/dicts were stored as JSON strings by
    # _serialize_value; decode the ones that look like JSON objects/arrays.
    for key, value in list(data.items()):
        if _is_string_field(cls, key):
            continue
        if isinstance(value, str) and value[:1] in "{[":
            with contextlib.suppress(ValueError, TypeError):
                data[key] = json.loads(value)
    data["schema_properties"] = reconstruct_schema_properties(cls, props)
    return cls.model_validate(data)


def document_record(doc: Document) -> dict[str, Any]:
    """Build the ``{id, props}`` record shape expected by upsert_documents.cypher.

    Args:
        doc: The Document DataPoint to serialize.

    Returns:
        Dict with ``id`` (str) and ``props`` (flat param dict).
    """
    params = to_params(doc)
    doc_id = params.pop("id")
    return {"id": doc_id, "props": params}


def chunk_record(chunk: Chunk) -> dict[str, Any]:
    """Build the ``{id, props}`` record shape expected by upsert_chunks.cypher.

    Args:
        chunk: The Chunk DataPoint to serialize.

    Returns:
        Dict with ``id`` (str) and ``props`` (flat param dict).
    """
    params = to_params(chunk)
    chunk_id = params.pop("id")
    return {"id": chunk_id, "props": params}


def entity_record(entity: Entity) -> dict[str, Any]:
    """Build the record shape expected by upsert_entities.cypher.

    ``type_label`` is extracted separately so the client can group entities by
    label and render the cypher template once per type group — the template
    receives it as a backtick-escaped identifier, not a Cypher parameter.
    ``chunk_ids`` and ``surface_forms`` are promoted to top-level lists so the
    template can union them across ingest runs instead of clobbering them (they
    otherwise live only inside the serialized ``provenance`` blob).

    Args:
        entity: The Entity DataPoint to serialize.

    Returns:
        Dict with ``id``, ``type_label`` (str), ``props`` (flat param dict), and
        the promoted ``chunk_ids`` / ``surface_forms`` lists.
    """
    params = to_params(entity)
    entity_id = params.pop("id")
    type_label = params.pop("label")  # rendered into Cypher, not a $parameter
    prov = entity.provenance
    return {
        "id": entity_id,
        "props": params,
        "type_label": type_label,
        "chunk_ids": list(dict.fromkeys(str(c) for c in prov.chunk_ids))
        if prov
        else [],
        "surface_forms": list(dict.fromkeys(prov.surface_forms)) if prov else [],
    }


def relation_record(rel: Relation) -> dict[str, Any]:
    """Build the record shape expected by upsert_relations.cypher.

    ``head_id`` and ``tail_id`` are pulled out at the top level so the Cypher
    template can MATCH the endpoint nodes by id in a single UNWIND pass.
    ``chunk_ids`` (from provenance) and the top-level ``sources`` list are
    promoted so the template can union them across ingest runs.

    Args:
        rel: The Relation DataPoint to serialize.

    Returns:
        Dict with ``id``, ``head_id``, ``tail_id`` (all str), ``props``, and the
        promoted ``chunk_ids`` / ``sources`` lists.
    """
    params = to_params(rel)
    rel_id = params.pop("id")
    head_id = params.pop("head_id")
    tail_id = params.pop("tail_id")
    sources = params.pop("sources", [])
    prov = rel.provenance
    return {
        "id": rel_id,
        "head_id": head_id,
        "tail_id": tail_id,
        "props": params,
        "chunk_ids": list(dict.fromkeys(str(c) for c in prov.chunk_ids))
        if prov
        else [],
        "sources": list(dict.fromkeys(sources)),
    }


def community_record(community: Community) -> dict[str, Any]:
    """Build the ``{id, props}`` record for upsert_communities.cypher.

    Args:
        community: The Community DataPoint to serialize.

    Returns:
        Dict with ``id`` (str) and ``props`` (flat param dict).
    """
    params = to_params(community)
    c_id = params.pop("id")
    return {"id": c_id, "props": params}


def report_record(report: CommunityReport) -> dict[str, Any]:
    """Build the ``{id, props}`` record for upsert_reports.cypher.

    Args:
        report: The CommunityReport DataPoint to serialize.

    Returns:
        Dict with ``id`` (str) and ``props`` (flat param dict).
    """
    params = to_params(report)
    r_id = params.pop("id")
    return {"id": r_id, "props": params}


def has_entity_records(entities: list[Entity]) -> list[dict[str, str]]:
    """Build the ``[{entity_id, chunk_id}]`` records for link_has_entity.cypher.

    One record is produced per (entity, chunk_id) pair because a single entity
    may appear in multiple chunks and the cypher template merges each edge
    independently.

    Args:
        entities: Entities whose provenance chunk_ids should be linked.

    Returns:
        List of dicts each containing ``entity_id`` and ``chunk_id`` as strings.
    """
    records = []
    for entity in entities:
        prov = entity.provenance
        if prov is None:
            continue
        for chunk_id in prov.chunk_ids:
            records.append(
                {
                    "entity_id": str(entity.id),
                    "chunk_id": str(chunk_id),
                }
            )
    return records


def in_community_records(communities: list[Community]) -> list[dict[str, str]]:
    """Build the ``[{entity_id, community_id}]`` records for link_in_community.cypher.

    One record per (entity, community) pair; the cypher template merges the
    IN_COMMUNITY edge for each pair independently.

    Args:
        communities: Communities whose entity membership should be linked.

    Returns:
        List of dicts with ``entity_id`` and ``community_id`` as strings.
    """
    records = []
    for community in communities:
        for entity_id in community.entity_ids:
            records.append(
                {
                    "entity_id": str(entity_id),
                    "community_id": str(community.id),
                }
            )
    return records


def parent_community_records(communities: list[Community]) -> list[dict[str, str]]:
    """Build the ``[{child_id, parent_id}]`` records for link_parent_community.cypher.

    Only communities with a non-None ``parent`` UUID are included; top-level
    communities have no parent edge and are silently skipped.

    Args:
        communities: Communities to inspect for parent links.

    Returns:
        List of dicts with ``child_id`` and ``parent_id`` as strings.
    """
    return [
        {"child_id": str(c.id), "parent_id": str(c.parent)}
        for c in communities
        if c.parent is not None
    ]


def has_report_records(reports: list[CommunityReport]) -> list[dict[str, str]]:
    """Build the ``[{community_id, report_id}]`` records for link_has_report.cypher.

    Args:
        reports: Community reports whose community links should be materialized.

    Returns:
        List of dicts with ``community_id`` and ``report_id`` as strings.
    """
    return [
        {"community_id": str(r.community_id), "report_id": str(r.id)} for r in reports
    ]


def ontology_entity_records(schema: GraphSchema) -> list[dict[str, Any]]:
    """Build the records for upsert_entity_types.cypher, one per entity type.

    Args:
        schema: Ontology whose entity types are mirrored into the graph.

    Returns:
        Dicts with ``label``, ``description``, ``gliner_label`` and
        ``extractable`` keys, in declaration order.
    """
    return [
        {
            "label": et.label,
            "description": et.description,
            "gliner_label": et.gliner_label,
            "extractable": et.extractable,
        }
        for et in schema.entity_types
    ]


def ontology_pair_records(schema: GraphSchema) -> list[dict[str, Any]]:
    """Build the records for upsert_pairs.cypher, one per allowed triple.

    Pairs whose head or tail is not a declared entity type are omitted because
    their endpoint match would find no node. When several relation types share
    a ``(type, head, tail)`` triple, the first one declared supplies the
    properties.

    Args:
        schema: Ontology whose endpoint pairs are mirrored into the graph.

    Returns:
        Dicts with ``type``, ``head``, ``tail``, ``glirel_label`` and
        ``extractable`` keys, sorted by ``(type, head, tail)``.
    """
    declared = schema.entity_type_names()
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rt in schema.relation_types:
        for head in rt.head_labels:
            for tail in rt.tail_labels:
                if head not in declared or tail not in declared:
                    continue
                records.setdefault(
                    (rt.type, head, tail),
                    {
                        "type": rt.type,
                        "head": head,
                        "tail": tail,
                        "glirel_label": rt.glirel_label,
                        "extractable": rt.extractable,
                    },
                )
    return [records[key] for key in sorted(records)]


__all__ = [
    "to_params",
    "from_props",
    "reconstruct_schema_properties",
    "document_record",
    "chunk_record",
    "entity_record",
    "relation_record",
    "community_record",
    "report_record",
    "has_entity_records",
    "in_community_records",
    "parent_community_records",
    "has_report_records",
    "ontology_entity_records",
    "ontology_pair_records",
]
