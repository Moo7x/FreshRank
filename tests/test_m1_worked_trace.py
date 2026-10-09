"""Small, executable M1 lesson: one impression through data, scores, and metrics."""

from datetime import datetime, timedelta
from math import log2

import polars as pl
import pytest

from freshrank.baselines import compare_baselines, historical_popularity, popularity_scores
from freshrank.ranking_data import candidate_rows


START = datetime(2023, 5, 18, 7)


def test_one_impression_from_nested_lists_to_three_rankings() -> None:
    # These two events happened before the behavior period. The later click on
    # article 10 belongs to the tuning impression, not to the scoring history.
    history = pl.DataFrame(
        {
            "article_id_fixed": [[20, 20]],
            "impression_time_fixed": [[START - timedelta(hours=3), START - timedelta(hours=2)]],
        }
    )
    tuning = pl.DataFrame(
        {
            "impression_id": [7],
            "impression_time": [START + timedelta(days=1)],
            "article_ids_inview": [[10, 20, 30]],
            "article_ids_clicked": [[10]],
        }
    )
    articles = pl.DataFrame(
        {
            "article_id": [10, 20, 30],
            "published_time": [
                START - timedelta(days=2),
                START - timedelta(days=1),
                START - timedelta(hours=1),
            ],
        }
    )

    rows = candidate_rows(tuning)
    assert rows.select("impression_id", "article_id", "clicked").rows() == [
        (7, 10, 1), (7, 20, 0), (7, 30, 0)
    ]
    counts, last_click, event_count = historical_popularity(history, START)
    assert (counts, last_click, event_count) == ({20: 2}, START - timedelta(hours=2), 2)
    assert popularity_scores([10, 20, 30], counts) == [0.0, 2.0, 0.0]

    result = compare_baselines(tuning, articles, history, START)
    # ID control: 10,20,30. Popularity: 20,10,30. Recency: 30,20,10.
    # The clicked article 10 is therefore at ranks 1, 2, and 3.
    assert result.article_id_control.mean_mrr == pytest.approx(1.0)
    assert result.popularity.mean_mrr == pytest.approx(1 / 2)
    assert result.recency.mean_mrr == pytest.approx(1 / 3)
    assert result.article_id_control.mean_ndcg_at_10 == pytest.approx(1.0)
    assert result.popularity.mean_ndcg_at_10 == pytest.approx(1 / log2(3))
    assert result.recency.mean_ndcg_at_10 == pytest.approx(1 / log2(4))
    assert result.popularity_changes_id_order == 1
    assert result.id_better_than_popularity_impressions == 1


def test_zero_history_count_turns_popularity_into_id_tie_break() -> None:
    history = pl.DataFrame(
        {
            "article_id_fixed": [[99]],
            "impression_time_fixed": [[START - timedelta(hours=1)]],
        }
    )
    tuning = pl.DataFrame(
        {
            "impression_id": [8],
            "impression_time": [START + timedelta(days=1)],
            "article_ids_inview": [[30, 10, 20]],
            "article_ids_clicked": [[30]],
        }
    )
    articles = pl.DataFrame(
        {
            "article_id": [10, 20, 30],
            "published_time": [START - timedelta(days=1)] * 3,
        }
    )

    result = compare_baselines(tuning, articles, history, START)
    assert result.all_zero_popularity_impressions == 1
    assert result.candidates_without_history_count == 3
    assert result.popularity.mean_ndcg_at_10 == result.article_id_control.mean_ndcg_at_10
    assert result.popularity.mean_mrr == result.article_id_control.mean_mrr
    # Article 30 was first in the input list, but the ID tie-break ranks it third.
    assert result.popularity.mean_mrr == pytest.approx(1 / 3)
