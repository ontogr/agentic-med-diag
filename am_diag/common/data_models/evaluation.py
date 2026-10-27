"""Evaluation result data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, computed_field


@dataclass(frozen=True)
class RetrievalQuestion:
    """A question with the ids of the items that answer it."""

    question_id: str
    text: str
    gold_ids: frozenset[str]


@dataclass(frozen=True)
class QuestionResult:
    """Outcome of one recipe run on one question."""

    recipe: str
    question_id: str
    repeat: int
    hit_at_k: bool
    reciprocal_rank: float
    latency_ms: float


@dataclass(frozen=True)
class RecipeSummary:
    """Scores for one recipe, averaged over its questions and repeats."""

    recipe: str
    recall_at_k: float
    mean_reciprocal_rank: float
    latency_p50_ms: float
    latency_p95_ms: float


class MetricResult(BaseModel):
    """Single metric output for one sample."""

    metric_name: str
    score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SampleEvaluationResult(BaseModel):
    """Evaluation output for a single sample across metrics."""

    sample_id: str
    dataset: str
    metrics: list[MetricResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def aggregate_score(self) -> float:
        """Return the mean metric score for the sample."""
        if not self.metrics:
            return 0.0
        return sum(metric.score for metric in self.metrics) / len(self.metrics)


class DatasetEvaluationSummary(BaseModel):
    """Aggregate metrics for a dataset run."""

    dataset: str
    samples_evaluated: int
    metric_averages: dict[str, float]
    errors: int = 0


@dataclass(frozen=True)
class MCQAccuracyResult:
    """Parsed MCQ grading outcome."""

    correct: bool
    method: str
    matched_answer: str | None
    correct_answer: str
