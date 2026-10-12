"""Attribute static-popularity coverage gaps without changing the ranking task."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from freshrank.baselines import historical_popularity
from freshrank.ranking_data import candidate_rows


GROUPS = (
    "history_count_present",
    "no_count_published_after_history_end",
    "no_count_published_by_history_end",
    "history_count_with_later_publication",
)


def audit_popularity_coverage(
    tuning: pl.DataFrame,
    articles: pl.DataFrame,
    history: pl.DataFrame,
    period_start: datetime,
) -> dict[str, object]:
    """Return aggregate evidence; later labels affect only outcome summaries.

    The boundary is the last observed fixed-history event, not a chosen age
    threshold. Recorded publication time is an assumption, not proof of why an
    older article has no clicks. No candidate or impression is removed.
    """
    if tuning.is_empty():
        raise ValueError("Tuning impressions are empty")
    if articles["article_id"].null_count() or articles["article_id"].n_unique() != articles.height:
        raise ValueError("Article IDs must be non-null and unique")
    if tuning["impression_time"].null_count() or tuning["impression_time"].min() < period_start:
        raise ValueError("Tuning timestamps must be present and at or after the period start")
    counts, history_end, history_events = historical_popularity(history, period_start)
    count_table = pl.DataFrame(
        {"article_id": list(counts), "history_click_count": list(counts.values())}
    )
    candidates = candidate_rows(tuning)
    rows = (
        candidates.join(articles.select("article_id", "published_time"), on="article_id", how="left")
        .join(count_table, on="article_id", how="left")
        .with_columns(pl.col("history_click_count").fill_null(0))
    )
    if rows.height != candidates.height:
        raise AssertionError("Coverage joins changed the candidate population")
    if rows["published_time"].null_count():
        raise ValueError("Candidate publication time or article reference is missing")
    if rows.filter(pl.col("published_time") > pl.col("impression_time")).height:
        raise ValueError("Candidate was published after its impression; prepare the data first")
    rows = rows.with_columns(
        pl.when((pl.col("history_click_count") > 0) & (pl.col("published_time") > history_end))
        .then(pl.lit(GROUPS[3]))
        .when(pl.col("history_click_count") > 0)
        .then(pl.lit(GROUPS[0]))
        .when(pl.col("published_time") > history_end)
        .then(pl.lit(GROUPS[1]))
        .otherwise(pl.lit(GROUPS[2]))
        .alias("coverage_group")
    )
    summary = rows.group_by("coverage_group").agg(
        pl.len().alias("candidate_appearances"),
        pl.col("article_id").n_unique().alias("distinct_candidate_articles"),
        pl.col("clicked").sum().alias("clicked_candidate_appearances"),
    )
    observed = {row["coverage_group"]: row for row in summary.iter_rows(named=True)}
    groups = [
        observed.get(
            name,
            {
                "coverage_group": name,
                "candidate_appearances": 0,
                "distinct_candidate_articles": 0,
                "clicked_candidate_appearances": 0,
            },
        )
        for name in GROUPS
    ]
    if sum(group["candidate_appearances"] for group in groups) != candidates.height:
        raise AssertionError("Coverage groups do not reconcile with the candidate population")
    return {
        "period_start": period_start,
        "history_last_click": history_end,
        "history_click_events": history_events,
        "evaluated_impressions": tuning.height,
        "candidate_appearances": rows.height,
        "distinct_candidate_articles": rows["article_id"].n_unique(),
        "clicked_candidate_appearances": int(rows["clicked"].sum()),
        "candidates_without_history_count": rows.filter(pl.col("history_click_count") == 0).height,
        "groups": groups,
    }
