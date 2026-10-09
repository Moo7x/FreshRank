"""Tests for impression-to-candidate conversion."""

from datetime import datetime

import polars as pl
import pytest

from freshrank.ranking_data import candidate_rows


def test_candidate_rows_keep_groups_and_multiple_positives() -> None:
    behaviors = pl.DataFrame(
        {
            "impression_id": [1, 2],
            "impression_time": [datetime(2023, 5, 1), datetime(2023, 5, 2)],
            "article_ids_inview": [[10, 20, 30], [20, 40]],
            "article_ids_clicked": [[20, 30], []],
        }
    )
    rows = candidate_rows(behaviors)
    assert rows.select("impression_id", "article_id", "clicked").rows() == [
        (1, 10, 0),
        (1, 20, 1),
        (1, 30, 1),
        (2, 20, 0),
        (2, 40, 0),
    ]
    assert "article_ids_clicked" not in rows.columns
    assert "candidate_position" not in rows.columns


def test_candidate_rows_reject_repeated_impression_id() -> None:
    behaviors = pl.DataFrame(
        {
            "impression_id": [1, 1],
            "impression_time": [datetime(2023, 5, 1)] * 2,
            "article_ids_inview": [[10], [20]],
            "article_ids_clicked": [[10], [20]],
        }
    )
    with pytest.raises(ValueError, match="one row per impression"):
        candidate_rows(behaviors)
