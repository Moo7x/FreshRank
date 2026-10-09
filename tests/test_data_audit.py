from pathlib import Path

import polars as pl

from freshrank.data_audit import audit_training_data


def test_training_audit_detects_group_and_time_problems(tmp_path: Path) -> None:
    bundle = tmp_path / "ebnerd_demo"
    train = bundle / "train"
    train.mkdir(parents=True)

    pl.DataFrame(
        {
            "impression_id": [10, 11, 12],
            "impression_time": [
                "2023-01-02 10:00:00",
                "2023-01-02 11:00:00",
                "2023-01-02 12:00:00",
            ],
            "article_ids_inview": [[1, 2], [2, 2], [4]],
            "article_ids_clicked": [[2], [3], [4, 4]],
            "user_id": [100, 101, 102],
        }
    ).with_columns(pl.col("impression_time").str.to_datetime()).write_parquet(
        train / "behaviors.parquet"
    )
    pl.DataFrame(
        {
            "user_id": [100, 101],
            "impression_time_fixed": [
                ["2023-01-01 08:00:00"],
                ["2023-01-01 09:00:00"],
            ],
            "article_id_fixed": [[1], [2]],
            "read_time_fixed": [[5.0], [6.0]],
            "scroll_percentage_fixed": [[50.0], [60.0]],
        }
    ).with_columns(
        pl.col("impression_time_fixed").list.eval(pl.element().str.to_datetime())
    ).write_parquet(train / "history.parquet")
    pl.DataFrame(
        {
            "article_id": [1, 2, 4],
            "published_time": [
                "2023-01-01 07:00:00",
                "2023-01-01 08:00:00",
                "2023-01-03 08:00:00",
            ],
            "last_modified_time": [
                "2023-01-04 07:00:00",
                "2023-01-04 08:00:00",
                "2023-01-04 09:00:00",
            ],
        }
    ).with_columns(
        pl.col("published_time", "last_modified_time").str.to_datetime()
    ).write_parquet(bundle / "articles.parquet")

    result = audit_training_data(bundle)

    assert result["impressions"] == 3
    assert result["candidate_appearances"] == 5
    assert result["duplicate_candidate_impressions"] == 1
    assert result["multiple_click_impressions"] == 1
    assert result["multiple_distinct_click_impressions"] == 0
    assert result["multiple_click_impressions_without_duplicates"] == 0
    assert result["duplicate_click_impressions"] == 1
    assert result["clicked_ids_outside_candidates"] == 1
    assert result["behavior_users_missing_history"] == 1
    assert result["candidate_rows_published_after_impression"] == 1
    assert result["impressions_with_future_published_candidates"] == 1
    assert result["future_published_candidate_articles"] == 1
    assert result["shortest_future_publication_lead"] == "20:00:00"
    assert result["longest_future_publication_lead"] == "20:00:00"
    assert result["history_events_at_or_after_first_impression"] == 0
