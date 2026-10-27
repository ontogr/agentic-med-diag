"""GLiREL relation extraction — NER-then-RE over entity pairs."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from am_diag.common.data_models import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.graph_construction._provenance import ensure_provenance
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.extract.base import Extractor


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _char_to_token_map(text: str, tokens: list[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    pos = 0
    for token_idx, token in enumerate(tokens):
        start = text.find(token, pos)
        if start == -1:
            continue
        for c in range(start, start + len(token)):
            mapping[c] = token_idx
        pos = start + len(token)
    return mapping


def _entities_for_glirel(
    entities: list[Entity],
    char_to_token: dict[int, int],
) -> list[list]:
    result = []
    for ent in entities:
        ent_prov = ensure_provenance(ent)
        sc = ent_prov.start_char
        ec = ent_prov.end_char
        if sc is None or ec is None:
            continue
        start_tok = char_to_token.get(sc)
        end_tok = char_to_token.get(ec - 1)
        if start_tok is not None and end_tok is not None:
            result.append([start_tok, end_tok, ent.label, ent.name])
    return result


def _run_glirel_sequential(
    texts: list[str],
    gliner_entities: list[list[Entity]],
    threshold: float,
    labels: list[str],
    glirel_model: Any,
) -> list[list[dict]]:
    all_relations = []
    for text, entities in zip(texts, gliner_entities, strict=False):
        tokens = _tokenize(text)
        char_to_token = _char_to_token_map(text, tokens)
        ner = _entities_for_glirel(entities, char_to_token)
        if len(ner) < 2:
            all_relations.append([])
            continue
        rels = glirel_model.predict_relations(
            tokens,
            labels,
            threshold=threshold,
            ner=ner,
        )
        all_relations.append(rels)
    return all_relations


class GLiRELRelationExtractor(Extractor):
    """GLiREL relation extraction — sequential per chunk, requires GLiNER spans.

    Takes pre-extracted GLiNER entities and predicts relations between them.
    """

    def __init__(self, config: ExtractionPipelineConfig, schema: GraphSchema) -> None:
        """Initialize extractor with config and schema; model loads lazily."""
        self._config = config
        self._schema = schema
        self._model: object | None = None

    @property
    def model(self) -> Any:
        """Lazy-loaded GLiREL model instance."""
        if self._model is None:
            from glirel import GLiREL  # noqa: PLC0415

            self._model = GLiREL.from_pretrained(self._config.glirel_model)
        return self._model

    async def extract(
        self,
        chunks: list[Chunk],
    ) -> tuple[list[Entity], list[Relation]]:
        """GLiREL extraction requires GLiNER entities — returns empty from this path.

        Use ``extract_from_entities(chunks, gliner_entities)`` instead.
        """
        return [], []

    async def extract_from_entities(
        self,
        chunks: list[Chunk],
        per_chunk_entities: list[list[Entity]],
    ) -> list[Relation]:
        """Run GLiREL over chunks given pre-extracted GLiNER entities.

        Args:
            chunks: Source chunks (aligned with ``per_chunk_entities``).
            per_chunk_entities: GLiNER entities per chunk.

        Returns:
            Extracted relations.
        """
        cfg = self._config
        schema = self._schema
        texts = [c.text[: cfg.max_text_chars] for c in chunks]

        if not cfg.use_glirel:
            return []

        labels = schema.glirel_labels()
        if not labels:
            return []
        relation_types = set(schema.glirel_to_type.values())
        endpoint_labels = {
            label
            for head, relation_type, tail in schema.allowed_pairs(extractable_only=True)
            if relation_type in relation_types
            for label in (head, tail)
        }
        per_chunk_entities = [
            [entity for entity in entities if entity.label in endpoint_labels]
            for entities in per_chunk_entities
        ]

        all_rels = await asyncio.to_thread(
            _run_glirel_sequential,
            texts,
            per_chunk_entities,
            cfg.glirel_threshold,
            labels,
            self.model,
        )

        relations: list[Relation] = []
        for chunk, rels in zip(chunks, all_rels, strict=False):
            for r in rels:
                relation_type = schema.glirel_to_type.get(r.get("label", ""))
                if relation_type is None:
                    continue
                relations.append(
                    Relation(
                        head_name=" ".join(r["head_text"]),
                        tail_name=" ".join(r["tail_text"]),
                        type=relation_type,
                        score=float(r.get("score", 1.0)),
                        provenance=Provenance(
                            extractor=ExtractorName.GLIREL,
                            chunk_ids=[chunk.id],
                        ),
                    ),
                )

        return relations


__all__ = [
    "GLiRELRelationExtractor",
    "_char_to_token_map",
    "_entities_for_glirel",
    "_tokenize",
]
