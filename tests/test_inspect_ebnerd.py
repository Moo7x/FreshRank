from pathlib import Path

import polars as pl
import pytest

from scripts.inspect_ebnerd import (
    EXPECTED_FILES,
    inspect_bundle,
    summarize_training_impressions,
)


def test_inspection_reports_all_expected_files_without_rows(tmp_path: Path) -> None:
    root = tmp_path / "ebnerd_demo"
    for name in EXPECTED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"example_id": [1, 2]}).write_parquet(path)

    result = inspect_bundle(tmp_path)

    assert result["bundle"] == "ebnerd_demo"
    assert set(result["files"]) == set(EXPECTED_FILES)
    assert all(info["rows"] == 2 for info in result["files"].values())
    assert all(info["schema"] == {"example_id": "Int64"} for info in result["files"].values())
    assert "example_id" not in result


def test_missing_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / "articles.parquet").touch()

    with pytest.raises(FileNotFoundError, match="train/behaviors.parquet"):
        inspect_bundle(tmp_path)


def test_training_summary_uses_candidate_lists_and_clicks(tmp_path: Path) -> None:
    (tmp_path / "articles.parquet").touch()
    train = tmp_path / "train"
    train.mkdir()
    pl.DataFrame(
        {
            "impression_time": [
                "2023-05-25 10:00:00",
                "2023-05-25 11:00:00",
            ],
            "article_ids_inview": [[1, 2], [3, 4, 5]],
            "article_ids_clicked": [[2], []],
        }
    ).with_columns(pl.col("impression_time").str.to_datetime()).write_parquet(
        train / "behaviors.parquet"
    )

    summary = summarize_training_impressions(tmp_path)

    assert summary["impressions"] == 2
    assert summary["fewest_candidates"] == 2
    assert summary["most_candidates"] == 3
    assert summary["no_click_impressions"] == 1
    assert summary["first_impression"] == "2023-05-25 10:00:00"
