"""Tests for feature availability and first baseline comparison."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from freshrank.baselines import (
    compare_baselines,
    historical_popularity,
    popularity_scores,
    recency_scores,
)


START = datetime(2023, 5, 18, 7)


def _history() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 2],
            "article_id_fixed": [[10, 10], [20]],
            "impression_time_fixed": [
                [START - timedelta(days=2), START - timedelta(days=1)],
                [START - timedelta(seconds=1)],
            ],
        }
    )


def test_fixed_history_counts_and_future_event_guard() -> None:
    counts, latest, events = historical_popularity(_history(), START)
    assert counts == {10: 2, 20: 1}
    assert latest == START - timedelta(seconds=1)
    assert events == 3
    assert popularity_scores([10, 20, 30], counts) == [2.0, 1.0, 0.0]

    future_history = _history().with_columns(
        pl.when(pl.col("user_id") == 2)
        .then(pl.lit([START]))
        .otherwise(pl.col("impression_time_fixed"))
        .alias("impression_time_fixed")
    )
    with pytest.raises(ValueError, match="at or after"):
        historical_popularity(future_history, START)


def test_recency_uses_impression_time_and_rejects_future_article() -> None:
    publication = {10: START - timedelta(days=2), 20: START - timedelta(hours=1)}
    assert recency_scores(START, [10, 20], publication) == [-172800.0, -3600.0]
    with pytest.raises(ValueError, match="published after"):
        recency_scores(START, [30], {30: START + timedelta(seconds=1)})
    with pytest.raises(ValueError, match="Missing publication"):
        recency_scores(START, [30], publication)


def test_comparison_keeps_future_labels_out_of_scores() -> None:
    tuning = pl.DataFrame(
        {
            "impression_time": [START + timedelta(days=4)],
            "article_ids_inview": [[10, 20, 30]],
            "article_ids_clicked": [[10]],
        }
    )
    articles = pl.DataFrame(
        {
            "article_id": [10, 20, 30],
            "published_time": [START - timedelta(days=2), START - timedelta(hours=2), START - timedelta(hours=1)],
        }
    )
    result = compare_baselines(tuning, articles, _history(), START)
    assert result.popularity.mean_ndcg_at_10 == pytest.approx(1.0)
    assert result.article_id_control.mean_ndcg_at_10 == pytest.approx(1.0)
    assert result.recency.mean_ndcg_at_10 == pytest.approx(0.5)
    assert result.recency.mean_mrr == pytest.approx(1 / 3)
    assert result.candidates_without_history_count == 1
    assert result.clicked_candidate_appearances == 1
    assert result.clicked_candidates_without_history_count == 0
    assert result.all_zero_popularity_impressions == 0
    assert (result.recency_better_impressions, result.popularity_better_impressions, result.equal_ndcg_impressions) == (0, 1, 0)
    assert (result.popularity_better_than_id_impressions, result.id_better_than_popularity_impressions, result.equal_ndcg_popularity_vs_id_impressions) == (0, 0, 1)

    # Changing a *later outcome* changes evaluation, not either scoring rule.
    altered_labels = tuning.with_columns(pl.Series("article_ids_clicked", [[30]]))
    assert popularity_scores([10, 20, 30], {10: 2, 20: 1}) == [2.0, 1.0, 0.0]
    assert recency_scores(START + timedelta(days=4), [10, 20, 30], dict(articles.iter_rows())) == [
        -(START + timedelta(days=4) - published).total_seconds() for published in articles["published_time"]
    ]
    assert compare_baselines(altered_labels, articles, _history(), START).popularity.mean_ndcg_at_10 != result.popularity.mean_ndcg_at_10


def test_all_unseen_candidates_are_reported_not_silently_replaced() -> None:
    tuning = pl.DataFrame(
        {
            "impression_time": [START + timedelta(days=1)],
            "article_ids_inview": [[30, 40]],
            "article_ids_clicked": [[40]],
        }
    )
    articles = pl.DataFrame(
        {
            "article_id": [30, 40],
            "published_time": [START - timedelta(hours=2), START - timedelta(hours=1)],
        }
    )
    result = compare_baselines(tuning, articles, _history(), START)
    assert result.all_zero_popularity_impressions == 1
    assert result.candidates_without_history_count == 2
    assert result.clicked_candidates_without_history_count == 1
    assert result.popularity.mean_mrr == pytest.approx(0.5)  # article-ID tie break
    assert result.popularity.mean_ndcg_at_10 == result.article_id_control.mean_ndcg_at_10


def test_popularity_is_distinct_from_the_id_only_control_when_history_helps() -> None:
    tuning = pl.DataFrame(
        {
            "impression_time": [START + timedelta(days=1)],
            "article_ids_inview": [[20, 10]],
            "article_ids_clicked": [[20]],
        }
    )
    articles = pl.DataFrame(
        {"article_id": [10, 20], "published_time": [START - timedelta(days=2)] * 2}
    )
    history = pl.DataFrame(
        {
            "article_id_fixed": [[20, 20, 20, 10]],
            "impression_time_fixed": [[START - timedelta(hours=4), START - timedelta(hours=3),
                                       START - timedelta(hours=2), START - timedelta(hours=1)]],
        }
    )
    result = compare_baselines(tuning, articles, history, START)
    assert result.popularity.mean_ndcg_at_10 == pytest.approx(1.0)
    assert result.article_id_control.mean_ndcg_at_10 == pytest.approx(1 / 1.584962500721156)
    assert result.popularity_changes_id_order == 1
    assert (result.popularity_better_than_id_impressions, result.id_better_than_popularity_impressions) == (1, 0)
