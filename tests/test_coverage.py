"""Test timing attribution, unchanged candidates, and join integrity."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from freshrank.coverage import audit_popularity_coverage


START = datetime(2023, 5, 18, 7)
END = START - timedelta(minutes=1)


def _inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    tuning = pl.DataFrame(
        {
            "impression_id": [1, 2],
            "impression_time": [START + timedelta(days=1)] * 2,
            "article_ids_inview": [[10, 20, 30], [20, 40, 50]],
            "article_ids_clicked": [[20], [40]],
        }
    )
    articles = pl.DataFrame(
        {
            "article_id": [10, 20, 30, 40, 50],
            "published_time": [
                END - timedelta(days=1),
                END + timedelta(hours=1),
                END - timedelta(hours=1),
                END,  # Exactly the observed boundary belongs to "by", not "after".
                END + timedelta(hours=2),  # Count + later publication is a contradiction category.
            ],
        }
    )
    history = pl.DataFrame(
        {"article_id_fixed": [[10, 50]], "impression_time_fixed": [[END, END]]}
    )
    return tuning, articles, history


def test_timing_groups_preserve_repeated_candidate_appearances_and_boundary() -> None:
    result = audit_popularity_coverage(*_inputs(), START)
    groups = {row["coverage_group"]: row for row in result["groups"]}
    assert result["candidate_appearances"] == 6
    assert result["distinct_candidate_articles"] == 5
    assert result["candidates_without_history_count"] == 4
    assert groups["history_count_present"]["candidate_appearances"] == 1
    assert groups["no_count_published_after_history_end"]["candidate_appearances"] == 2
    assert groups["no_count_published_after_history_end"]["distinct_candidate_articles"] == 1
    assert groups["no_count_published_by_history_end"]["candidate_appearances"] == 2
    assert groups["history_count_with_later_publication"]["candidate_appearances"] == 1
    assert groups["no_count_published_by_history_end"]["clicked_candidate_appearances"] == 1


def test_later_labels_change_only_outcome_counts_not_coverage_attribution() -> None:
    tuning, articles, history = _inputs()
    before = audit_popularity_coverage(tuning, articles, history, START)
    changed = tuning.with_columns(pl.Series("article_ids_clicked", [[10], [50]]))
    after = audit_popularity_coverage(changed, articles, history, START)
    for left, right in zip(before["groups"], after["groups"]):
        assert left["coverage_group"] == right["coverage_group"]
        assert left["candidate_appearances"] == right["candidate_appearances"]
        assert left["distinct_candidate_articles"] == right["distinct_candidate_articles"]
    assert before["groups"][0]["clicked_candidate_appearances"] != after["groups"][0]["clicked_candidate_appearances"]


def test_duplicate_article_keys_cannot_silently_multiply_rows() -> None:
    tuning, articles, history = _inputs()
    duplicates = pl.concat([articles, articles.head(1)])
    with pytest.raises(ValueError, match="unique"):
        audit_popularity_coverage(tuning, duplicates, history, START)


def test_missing_article_reference_is_not_a_zero_popularity_observation() -> None:
    tuning, articles, history = _inputs()
    missing = articles.filter(pl.col("article_id") != 20)
    with pytest.raises(ValueError, match="reference is missing"):
        audit_popularity_coverage(tuning, missing, history, START)


def test_future_history_rejection_is_preserved() -> None:
    tuning, articles, _ = _inputs()
    future = pl.DataFrame({"article_id_fixed": [[10]], "impression_time_fixed": [[START]]})
    with pytest.raises(ValueError, match="at or after"):
        audit_popularity_coverage(tuning, articles, future, START)


def test_candidate_published_after_impression_requires_preparation() -> None:
    tuning, articles, history = _inputs()
    future = articles.with_columns(
        pl.when(pl.col("article_id") == 20)
        .then(pl.lit(START + timedelta(days=2)))
        .otherwise(pl.col("published_time"))
        .alias("published_time")
    )
    with pytest.raises(ValueError, match="published after"):
        audit_popularity_coverage(tuning, future, history, START)
