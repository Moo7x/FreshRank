"""Impression-level binary ranking metrics for M1."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, log2
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ImpressionMetrics:
    ndcg_at_10: float | None
    mrr: float | None


@dataclass(frozen=True)
class MetricSummary:
    total_impressions: int
    evaluated_impressions: int
    no_positive_impressions: int
    mean_ndcg_at_10: float | None
    mean_mrr: float | None


def score_impression(
    article_ids: Sequence[int], scores: Sequence[float], labels: Sequence[int]
) -> ImpressionMetrics:
    """Score one candidate group; ties favor the smaller article ID.

    No-positive groups return None for both metrics so the caller can report
    their count separately instead of pretending they have a ranking target.
    MRR uses the full candidate list; NDCG uses at most the first 10 ranks.
    """
    if not article_ids or len(article_ids) != len(scores) or len(scores) != len(labels):
        raise ValueError("Candidate IDs, scores and labels must be nonempty and aligned")
    if len(set(article_ids)) != len(article_ids):
        raise ValueError("Candidate article IDs must be unique within an impression")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("M1 metrics require binary click labels")
    if any(not isfinite(score) for score in scores):
        raise ValueError("Scores must be finite")
    positives = sum(labels)
    if positives == 0:
        return ImpressionMetrics(None, None)

    ranked = sorted(range(len(article_ids)), key=lambda i: (-scores[i], article_ids[i]))
    cutoff = min(10, len(ranked))
    dcg = fsum(labels[i] / log2(rank + 2) for rank, i in enumerate(ranked[:cutoff]))
    ideal_dcg = fsum(1 / log2(rank + 2) for rank in range(min(positives, cutoff)))
    first_positive_rank = next(rank for rank, i in enumerate(ranked, start=1) if labels[i] == 1)
    return ImpressionMetrics(dcg / ideal_dcg, 1 / first_positive_rank)


def summarize_metrics(results: Iterable[ImpressionMetrics]) -> MetricSummary:
    """Macro-average valid impression scores and count no-positive groups."""
    values = list(results)
    valid = [result for result in values if result.ndcg_at_10 is not None]
    if any((result.ndcg_at_10 is None) != (result.mrr is None) for result in values):
        raise ValueError("Each impression must have both metrics or neither")
    return MetricSummary(
        total_impressions=len(values),
        evaluated_impressions=len(valid),
        no_positive_impressions=len(values) - len(valid),
        mean_ndcg_at_10=fsum(result.ndcg_at_10 for result in valid) / len(valid) if valid else None,
        mean_mrr=fsum(result.mrr for result in valid) / len(valid) if valid else None,
    )
