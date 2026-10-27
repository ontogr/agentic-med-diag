"""Graph construction components — extraction, normalization, aggregation, resolution.

Components are pure, storage-agnostic building blocks composed by pipelines in
``am_diag.ingestion``. Each component exposes a plain async method over typed
arguments (no shared mutable state, no graph framework coupling), which makes
them directly testable in isolation.
"""

from __future__ import annotations

from am_diag.graph_construction.aggregate import KGAggregator
from am_diag.graph_construction.community.context import build_community_context
from am_diag.graph_construction.community.gds import GdsCommunityDetector
from am_diag.graph_construction.community.leiden import LeidenCommunityDetector
from am_diag.graph_construction.community.summarize import CommunitySummarizer
from am_diag.graph_construction.config import (
    CommunityDetectionConfig,
    ExtractionPipelineConfig,
)
from am_diag.graph_construction.extract.base import Extractor
from am_diag.graph_construction.extract.combine import ExtractionCombiner
from am_diag.graph_construction.extract.composed import ComposedExtractor
from am_diag.graph_construction.extract.gliner_entities import GLiNEREntityExtractor
from am_diag.graph_construction.extract.glirel_relations import GLiRELRelationExtractor
from am_diag.graph_construction.extract.llm import LLMExtractor
from am_diag.graph_construction.normalize import EntityRelationNormalizer
from am_diag.graph_construction.prune import OntologyPruner, OntologyReport
from am_diag.graph_construction.resolve.apply import AliasApplier
from am_diag.graph_construction.resolve.candidates import build_bm25, top_k
from am_diag.graph_construction.resolve.cluster import cluster_items
from am_diag.graph_construction.resolve.deterministic import DeterministicDeduplicator
from am_diag.graph_construction.resolve.llm_dedupe import LLMDeduplicator
from am_diag.graph_construction.resolve.resolver import EntityEdgeResolver
from am_diag.graph_construction.resolve.text import normalize_text, singularize


__all__ = [
    "AliasApplier",
    "CommunityDetectionConfig",
    "CommunitySummarizer",
    "ComposedExtractor",
    "DeterministicDeduplicator",
    "EntityEdgeResolver",
    "EntityRelationNormalizer",
    "ExtractionCombiner",
    "ExtractionPipelineConfig",
    "Extractor",
    "GdsCommunityDetector",
    "GLiNEREntityExtractor",
    "GLiRELRelationExtractor",
    "KGAggregator",
    "LeidenCommunityDetector",
    "LLMDeduplicator",
    "LLMExtractor",
    "OntologyPruner",
    "OntologyReport",
    "build_bm25",
    "build_community_context",
    "cluster_items",
    "normalize_text",
    "singularize",
    "top_k",
]
