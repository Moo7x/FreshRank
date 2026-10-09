"""Run aggregate integrity checks on the local EB-NeRD training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

# Direct script execution does not automatically include the src-layout package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freshrank.data_audit import audit_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Extracted EB-NeRD demo directory")
    args = parser.parse_args()
    try:
        result = audit_training_data(args.bundle)
    except (FileNotFoundError, OSError, pl.exceptions.PolarsError) as exc:
        parser.exit(2, f"Training audit failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
