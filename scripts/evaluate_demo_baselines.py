"""Compare M1 baseline rules on cleaned EB-NeRD demo tuning impressions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freshrank.baselines import compare_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Extracted EB-NeRD demo directory")
    parser.add_argument("prepared", type=Path, help="Directory from prepare_ebnerd_demo.py")
    args = parser.parse_args()
    try:
        tuning = pl.read_parquet(args.prepared / "tune_behaviors.parquet")
        behaviors = pl.read_parquet(args.bundle / "train" / "behaviors.parquet", columns=["impression_time"])
        articles = pl.read_parquet(args.bundle / "articles.parquet", columns=["article_id", "published_time"])
        history = pl.read_parquet(args.bundle / "train" / "history.parquet")
        result = compare_baselines(tuning, articles, history, behaviors["impression_time"].min())
    except (FileNotFoundError, OSError, ValueError, pl.exceptions.PolarsError) as exc:
        parser.exit(2, f"Baseline evaluation failed: {exc}\n")
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
