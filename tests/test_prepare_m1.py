"""Tests for whole-impression quarantine and chronological splitting."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from freshrank.prepare_m1 import prepare_training


def test_quarantine_preserves_groups_and_valid_multiple_clicks() -> None:
    start = datetime(2023, 5, 1, 12)
    rows = [
        (day, start + timedelta(days=day - 1), [1, 2], [1, 2] if day == 4 else [1])
        for day in range(1, 6)
    ]
    rows += [
        (10, start, [1, 2], [1, 1]),
        (11, start, [1, 99], [1]),
        (12, start, [1, 99], [1, 1]),
    ]
    behaviors = pl.DataFrame(
        rows,
        schema=["impression_id", "impression_time", "article_ids_inview", "article_ids_clicked"],
        orient="row",
    )
    articles = pl.DataFrame(
        {"article_id": [1, 2, 99], "published_time": [start - timedelta(days=1)] * 2 + [start + timedelta(minutes=1)]}
    )

    result = prepare_training(behaviors, articles)

    assert set(result.fit["impression_id"].to_list()) == {1, 2, 3}
    assert set(result.tune["impression_id"].to_list()) == {4, 5}
    assert set(result.quarantine["impression_id"].to_list()) == {10, 11, 12}
    assert (result.duplicate_click_impressions, result.future_publication_impressions, result.overlapping_reasons) == (2, 2, 1)
    assert result.tuning_start.isoformat() == "2023-05-04"
    assert result.tune.filter(pl.col("impression_id") == 4)["article_ids_clicked"].to_list() == [[1, 2]]
    assert result.fit["impression_time"].max() < result.tune["impression_time"].min()


def test_missing_article_reference_fails_closed() -> None:
    start = datetime(2023, 5, 1)
    behaviors = pl.DataFrame(
        {
            "impression_id": [1],
            "impression_time": [start],
            "article_ids_inview": [[404]],
            "article_ids_clicked": [[404]],
        }
    )
    articles = pl.DataFrame({"article_id": [1], "published_time": [start]})
    with pytest.raises(ValueError, match="reference is missing"):
        prepare_training(behaviors, articles)
