"""GLiNER NER extraction — batch-predict entities only."""

from __future__ import annotations

import asyncio
from typing import Any

from am_diag.common.data_models import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.extract.base import Extractor


def _run_gliner_batch(
    texts: list[str],
    batch_size: int,
    threshold: float,
    labels: list[str],
    model: Any,
) -> list[list[dict]]:
    return model.batch_predict_entities(
        texts,
        labels,
        threshold=threshold,
        batch_size=batch_size,
        flat_ner=True,
    )


class GLiNEREntityExtractor(Extractor):
    """GLiNER NER extraction — batch-predict entities only.

    Returns ``Entity`` DataPoints with ``ExtractorName.GLINER``
    provenance and character offsets.
    """

    def __init__(self, config: ExtractionPipelineConfig, schema: GraphSchema) -> None:
        """Initialize extractor with config and schema; model loads lazily."""
        self._config = config
        self._schema = schema
        self._model: object | None = None

    @property
    def model(self) -> Any:
        """Lazy-loaded GLiNER model instance."""
        if self._model is None:
            from gliner import GLiNER  # noqa: PLC0415

            self._model = GLiNER.from_pretrained(self._config.gliner_model)
        return self._model

    async def extract(
        self,
        chunks: list[Chunk],
    ) -> tuple[list[Entity], list[Relation]]:
        """Run GLiNER entity extraction over *chunks*.

        Only returns entities; relations are empty (use ``GLiRELRelationExtractor``).
        """
        cfg = self._config
        schema = self._schema
        texts = [c.text[: cfg.max_text_chars] for c in chunks]

        if not cfg.use_gliner:
            return [], []

        all_spans = await asyncio.to_thread(
            _run_gliner_batch,
            texts,
            cfg.gliner_batch_size,
            cfg.gliner_threshold,
            schema.gliner_labels(),
            self.model,
        )

        entities: list[Entity] = []
        for chunk, spans in zip(chunks, all_spans, strict=False):
            for s in spans:
                entity_label = schema.gliner_to_label.get(s["label"])
                if entity_label is None:
                    continue
                ent = Entity(
                    name=s["text"],
                    label=entity_label,
                    score=float(s.get("score", 1.0)),
                    provenance=Provenance(
                        extractor=ExtractorName.GLINER,
                        start_char=s.get("start"),
                        end_char=s.get("end"),
                        chunk_ids=[chunk.id],
                    ),
                )
                entities.append(ent)

        return entities, []


__all__ = ["GLiNEREntityExtractor"]
