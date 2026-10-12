"""Run M1 from training files to baselines, coverage and an identified run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freshrank.experiment import run_m1_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Extracted bundle containing articles.parquet and train/")
    parser.add_argument("output", type=Path, help="New ignored directory for this run")
    args = parser.parse_args()
    try:
        result = run_m1_experiment(args.bundle, args.output, lambda stage: print(stage, file=sys.stderr, flush=True))
    except (OSError, ValueError, AssertionError, pl.exceptions.PolarsError) as exc:
        parser.exit(2, f"M1 experiment failed: {exc}\n")
    print(json.dumps(
        {
            "run_directory": str(args.output),
            "preparation": result["preparation"],
            "article_id_control": result["baselines"]["article_id_control"],
            "popularity": result["baselines"]["popularity"],
            "recency": result["baselines"]["recency"],
            "analysis_seconds_excluding_output_writes": result["analysis_seconds_excluding_output_writes"],
            "official_validation_used": result["official_validation_used"],
        }, indent=2
    ))


if __name__ == "__main__":
    main()
