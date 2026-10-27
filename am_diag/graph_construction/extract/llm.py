"""LLM-based entity/relation extraction via BAML calls."""

from __future__ import annotations

import asyncio
import warnings
from uuid import UUID

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.extract.base import Extractor
from am_diag.llm.schema_typebuilder import build_schema_typebuilder


async def _call_llm(text: str, source: str, schema: GraphSchema) -> object:
    from am_diag.llm.baml_client import b  # noqa: PLC0415

    tb = build_schema_typebuilder(schema)
    return await b.ExtractKnowledgeGraph(text, source, baml_options={"tb": tb})


def _entity_type_str(entity_type: object) -> str:
    v = getattr(entity_type, "value", None)
    return str(v) if v is not None else str(entity_type)


def _relation_type_str(relation_type: object) -> str:
    v = getattr(relation_type, "value", None)
    return str(v) if v is not None else str(relation_type)


def _flatten_kg(
    kg: object,
    chunk_id: str,
    schema: GraphSchema,
) -> tuple[list[Entity], list[Relation]]:
    entities: list[Entity] = []
    relations: list[Relation] = []

    valid_labels = schema.extractable_entity_type_names()
    for ent in getattr(kg, "entities", []):
        label = _entity_type_str(ent.entity_type)
        if label not in valid_labels:
            continue
        entities.append(
            Entity(
                name=ent.name,
                label=label,
                score=float(getattr(ent, "score", 1.0)),
                schema_properties=dict(getattr(ent, "properties", {})),
                provenance=Provenance(
                    extractor=ExtractorName.LLM,
                    chunk_ids=[UUID(chunk_id)],
                ),
            ),
        )

    valid_types = schema.extractable_relation_type_names()
    valid_entity_names: set[str] = {e.name for e in entities}
    for rel in getattr(kg, "relations", []):
        rel_type = _relation_type_str(rel.relation_type)
        if rel_type not in valid_types:
            continue
        if (
            rel.head_name not in valid_entity_names
            or rel.tail_name not in valid_entity_names
        ):
            continue
        relations.append(
            Relation(
                head_name=rel.head_name,
                tail_name=rel.tail_name,
                type=rel_type,
                score=float(getattr(rel, "score", 1.0)),
                schema_properties=dict(getattr(rel, "properties", {})),
                provenance=Provenance(
                    extractor=ExtractorName.LLM,
                    chunk_ids=[UUID(chunk_id)],
                ),
            ),
        )

    return entities, relations


class LLMExtractor(Extractor):
    """LLM-based KG extraction via BAML dynamic enums and asyncio.gather."""

    def __init__(self, config: ExtractionPipelineConfig, schema: GraphSchema) -> None:
        """Initialize LLM extractor with config and schema."""
        self._config = config
        self._schema = schema

    async def extract(
        self,
        chunks: list[Chunk],
    ) -> tuple[list[Entity], list[Relation]]:
        """Run LLM-based extraction over *chunks*.

        Returns ``(entities, relations)`` in chunk order.
        Returns empty lists if ``config.use_llm`` is ``False``.
        """
        if not self._config.use_llm:
            return [], []

        semaphore = asyncio.Semaphore(self._config.llm_max_concurrent)

        async def extract_one(
            chunk: Chunk,
        ) -> tuple[list[Entity], list[Relation]]:
            async with semaphore:
                text = chunk.text[: self._config.max_text_chars]
                try:
                    kg = await _call_llm(text, chunk.document_source, self._schema)
                except Exception:
                    return [], []
                return _flatten_kg(kg, str(chunk.id), self._schema)

        outcomes = await asyncio.gather(
            *[extract_one(chunk) for chunk in chunks],
            return_exceptions=True,
        )

        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        error_count = 0
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                error_count += 1
            else:
                entities, relations = outcome
                all_entities.extend(entities)
                all_relations.extend(relations)

        if error_count == len(outcomes) and outcomes:
            warnings.warn(
                f"LLM extraction failed for all {error_count} chunks. "
                "Check OPENAI_GENERIC_BASE_URL and OPENAI_GENERIC_API_KEY.",
                stacklevel=2,
            )

        return all_entities, all_relations


__all__ = ["LLMExtractor"]
