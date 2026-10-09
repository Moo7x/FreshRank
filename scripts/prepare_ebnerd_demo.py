"""Create a strict M1 training split from the local EB-NeRD demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freshrank.prepare_m1 import prepare_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Extracted EB-NeRD demo directory")
    parser.add_argument("output", type=Path, help="New directory for generated, ignored files")
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(2, f"Output already exists; refusing to overwrite: {args.output}\n")
    try:
        behaviors = pl.read_parquet(args.bundle / "train" / "behaviors.parquet")
        articles = pl.read_parquet(args.bundle / "articles.parquet")
        prepared = prepare_training(behaviors, articles)
    except (FileNotFoundError, OSError, ValueError, AssertionError, pl.exceptions.PolarsError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")

    summary = {
        "source": str(args.bundle),
        "policy": "Quarantine whole impressions with duplicate clicked IDs or candidates published after impression time",
        "raw_impressions": behaviors.height,
        "fit_impressions": prepared.fit.height,
        "tune_impressions": prepared.tune.height,
        "quarantined_impressions": prepared.quarantine.height,
        "duplicate_click_impressions": prepared.duplicate_click_impressions,
        "future_publication_impressions": prepared.future_publication_impressions,
        "overlapping_reasons": prepared.overlapping_reasons,
        "tuning_starts_on": prepared.tuning_start.isoformat(),
        "fit_last_impression": str(prepared.fit["impression_time"].max()),
        "tune_first_impression": str(prepared.tune["impression_time"].min()),
        "official_validation_used": False,
    }
    args.output.mkdir(parents=True)
    prepared.fit.write_parquet(args.output / "fit_behaviors.parquet")
    prepared.tune.write_parquet(args.output / "tune_behaviors.parquet")
    prepared.quarantine.write_parquet(args.output / "quarantined_behaviors.parquet")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
