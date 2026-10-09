"""Leakage-conscious popularity and recency baselines for M1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from freshrank.metrics import MetricSummary, score_impression, summarize_metrics


@dataclass(frozen=True)
class BaselineComparison:
    article_id_control: MetricSummary
    popularity: MetricSummary
    recency: MetricSummary
    history_click_events: int
    history_last_click: datetime
    evaluated_candidate_appearances: int
    candidates_without_history_count: int
    clicked_candidate_appearances: int
    clicked_candidates_without_history_count: int
    all_zero_popularity_impressions: int
    recency_better_impressions: int
    popularity_better_impressions: int
    equal_ndcg_impressions: int
    popularity_changes_id_order: int
    popularity_better_than_id_impressions: int
    id_better_than_popularity_impressions: int
    equal_ndcg_popularity_vs_id_impressions: int


def historical_popularity(history: pl.DataFrame, period_start: datetime) -> tuple[dict[int, int], datetime, int]:
    """Count only fixed-history clicks recorded before the behavior period."""
    if history.is_empty() or history["article_id_fixed"].null_count() or history["impression_time_fixed"].null_count():
        raise ValueError("Fixed history is empty or contains null lists")
    if history.filter(
        pl.col("article_id_fixed").list.len() != pl.col("impression_time_fixed").list.len()
    ).height:
        raise ValueError("History article IDs and timestamps are misaligned")
    events = history.select("article_id_fixed", "impression_time_fixed").explode(
        "article_id_fixed", "impression_time_fixed", empty_as_null=False
    )
    if events.is_empty() or events["article_id_fixed"].null_count() or events["impression_time_fixed"].null_count():
        raise ValueError("Fixed history contains no complete click events")
    if events.filter(pl.col("impression_time_fixed") >= period_start).height:
        raise ValueError("History contains a click at or after the behavior period starts")
    counts = events.group_by("article_id_fixed").len()
    return dict(counts.select("article_id_fixed", "len").iter_rows()), events["impression_time_fixed"].max(), events.height


def popularity_scores(article_ids: list[int], counts: dict[int, int]) -> list[float]:
    """An unseen article receives zero, not a future-derived total."""
    return [float(counts.get(article_id, 0)) for article_id in article_ids]


def recency_scores(
    impression_time: datetime, article_ids: list[int], publication_times: dict[int, datetime]
) -> list[float]:
    """Higher means newer; fail if an article did not yet exist."""
    scores = []
    for article_id in article_ids:
        published = publication_times.get(article_id)
        if published is None:
            raise ValueError(f"Missing publication time for candidate {article_id}")
        age_seconds = (impression_time - published).total_seconds()
        if age_seconds < 0:
            raise ValueError(f"Candidate {article_id} was published after the impression")
        scores.append(-age_seconds)
    return scores


def compare_baselines(
    tuning: pl.DataFrame, articles: pl.DataFrame, history: pl.DataFrame, period_start: datetime
) -> BaselineComparison:
    """Evaluate both fixed rules on the same grouped tuning impressions."""
    if tuning.is_empty():
        raise ValueError("Tuning impressions are empty")
    if "impression_id" in tuning.columns and tuning["impression_id"].n_unique() != tuning.height:
        raise ValueError("Tuning impression IDs must be unique")
    if articles["article_id"].n_unique() != articles.height:
        raise ValueError("Article IDs must be unique")
    counts, last_history_click, history_events = historical_popularity(history, period_start)
    publication_times = dict(articles.select("article_id", "published_time").iter_rows())
    popularity_results = []
    recency_results = []
    id_control_results = []
    candidate_appearances = 0
    unseen_appearances = 0
    clicked_appearances = 0
    clicked_unseen = 0
    all_zero_popularity = 0
    recency_better = 0
    popularity_better = 0
    equal_ndcg = 0
    popularity_changes_id_order = 0
    popularity_better_than_id = 0
    id_better_than_popularity = 0
    equal_popularity_id = 0
    for row in tuning.select("impression_time", "article_ids_inview", "article_ids_clicked").iter_rows(named=True):
        impression_time = row["impression_time"]
        if impression_time < period_start:
            raise ValueError("Tuning impression precedes the behavior period")
        article_ids = row["article_ids_inview"]
        clicked_ids = row["article_ids_clicked"]
        if clicked_ids is None or len(set(clicked_ids)) != len(clicked_ids) or not set(clicked_ids).issubset(article_ids):
            raise ValueError("Tuning labels are malformed; prepare the data first")
        labels = [int(article_id in clicked_ids) for article_id in article_ids]
        popularity_result = score_impression(article_ids, popularity_scores(article_ids, counts), labels)
        id_control_result = score_impression(article_ids, [0.0] * len(article_ids), labels)
        recency_result = score_impression(
            article_ids, recency_scores(impression_time, article_ids, publication_times), labels
        )
        popularity_results.append(popularity_result)
        recency_results.append(recency_result)
        id_control_results.append(id_control_result)
        popularity_changes_id_order += int(
            sorted(article_ids, key=lambda article_id: (-counts.get(article_id, 0), article_id))
            != sorted(article_ids)
        )
        if popularity_result.ndcg_at_10 is not None:
            if recency_result.ndcg_at_10 > popularity_result.ndcg_at_10:
                recency_better += 1
            elif recency_result.ndcg_at_10 < popularity_result.ndcg_at_10:
                popularity_better += 1
            else:
                equal_ndcg += 1
            if popularity_result.ndcg_at_10 > id_control_result.ndcg_at_10:
                popularity_better_than_id += 1
            elif popularity_result.ndcg_at_10 < id_control_result.ndcg_at_10:
                id_better_than_popularity += 1
            else:
                equal_popularity_id += 1
        candidate_appearances += len(article_ids)
        unseen_appearances += sum(article_id not in counts for article_id in article_ids)
        clicked_appearances += len(clicked_ids)
        clicked_unseen += sum(article_id not in counts for article_id in clicked_ids)
        all_zero_popularity += int(not any(article_id in counts for article_id in article_ids))
    return BaselineComparison(
        article_id_control=summarize_metrics(id_control_results),
        popularity=summarize_metrics(popularity_results),
        recency=summarize_metrics(recency_results),
        history_click_events=history_events,
        history_last_click=last_history_click,
        evaluated_candidate_appearances=candidate_appearances,
        candidates_without_history_count=unseen_appearances,
        clicked_candidate_appearances=clicked_appearances,
        clicked_candidates_without_history_count=clicked_unseen,
        all_zero_popularity_impressions=all_zero_popularity,
        recency_better_impressions=recency_better,
        popularity_better_impressions=popularity_better,
        equal_ndcg_impressions=equal_ndcg,
        popularity_changes_id_order=popularity_changes_id_order,
        popularity_better_than_id_impressions=popularity_better_than_id,
        id_better_than_popularity_impressions=id_better_than_popularity,
        equal_ndcg_popularity_vs_id_impressions=equal_popularity_id,
    )
