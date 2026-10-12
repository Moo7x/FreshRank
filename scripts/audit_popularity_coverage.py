"""Explain demo popularity coverage with aggregate publication/history timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freshrank.coverage import audit_popularity_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Extracted EB-NeRD demo directory")
    parser.add_argument("prepared", type=Path, help="Directory from prepare_ebnerd_demo.py")
    args = parser.parse_args()
    try:
        tuning = pl.read_parquet(args.prepared / "tune_behaviors.parquet")
        start = pl.read_parquet(
            args.bundle / "train" / "behaviors.parquet", columns=["impression_time"]
        )["impression_time"].min()
        articles = pl.read_parquet(args.bundle / "articles.parquet", columns=["article_id", "published_time"])
        history = pl.read_parquet(
            args.bundle / "train" / "history.parquet", columns=["article_id_fixed", "impression_time_fixed"]
        )
        result = audit_popularity_coverage(tuning, articles, history, start)
    except (FileNotFoundError, OSError, ValueError, AssertionError, pl.exceptions.PolarsError) as exc:
        parser.exit(2, f"Coverage audit failed: {exc}\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
