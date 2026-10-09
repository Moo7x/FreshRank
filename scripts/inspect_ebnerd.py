"""Print a local EB-NeRD bundle inventory without copying dataset rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


EXPECTED_FILES = (
    "articles.parquet",
    "train/behaviors.parquet",
    "train/history.parquet",
    "validation/behaviors.parquet",
    "validation/history.parquet",
)


def bundle_root(directory: Path) -> Path:
    """Accept the extracted bundle root or its immediate parent."""
    for candidate in (directory, directory / "ebnerd_demo"):
        if (candidate / "articles.parquet").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find articles.parquet in {directory} or its ebnerd_demo child"
    )


def inspect_bundle(directory: Path) -> dict[str, object]:
    root = bundle_root(directory)
    missing = [name for name in EXPECTED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected files: {', '.join(missing)}")

    files: dict[str, dict[str, object]] = {}
    for name in EXPECTED_FILES:
        path = root / name
        frame = pl.scan_parquet(path)
        files[name] = {
            "bytes": path.stat().st_size,
            "rows": frame.select(pl.len()).collect().item(),
            "schema": {key: str(dtype) for key, dtype in frame.collect_schema().items()},
        }
    return {"bundle": root.name, "files": files}


def summarize_training_impressions(directory: Path) -> dict[str, object]:
    """Summarize training requests without exposing reader or article records."""
    root = bundle_root(directory)
    behaviors = pl.scan_parquet(root / "train" / "behaviors.parquet")
    summary = behaviors.select(
        pl.len().alias("impressions"),
        pl.col("impression_time").min().alias("first_impression"),
        pl.col("impression_time").max().alias("last_impression"),
        pl.col("article_ids_inview").list.len().min().alias("fewest_candidates"),
        pl.col("article_ids_inview").list.len().max().alias("most_candidates"),
        (pl.col("article_ids_clicked").list.len() == 0).sum().alias("no_click_impressions"),
    ).collect().row(0, named=True)
    return {
        **summary,
        "first_impression": str(summary["first_impression"]),
        "last_impression": str(summary["last_impression"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Extracted EB-NeRD demo directory")
    args = parser.parse_args()
    try:
        result = inspect_bundle(args.directory)
        result["training_impressions"] = summarize_training_impressions(args.directory)
    except (FileNotFoundError, OSError, pl.exceptions.PolarsError) as exc:
        parser.exit(2, f"Dataset inspection failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
