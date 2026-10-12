"""Run the existing M1 development rules and persist their inputs and results."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Callable

import polars as pl

from freshrank.baselines import compare_baselines
from freshrank.coverage import audit_popularity_coverage
from freshrank.data_audit import audit_training_data
from freshrank.prepare_m1 import prepare_training


INPUT_FILES = ("articles.parquet", "train/behaviors.parquet", "train/history.parquet")


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run_m1_experiment(
    bundle: Path,
    output: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run a fresh development experiment; never overwrite an existing run.

    Uses the existing provisional rules without choosing a new protocol. Only
    the three declared input files are read; official validation is excluded.
    Source-file fingerprints are checked again before any outputs are created.
    """
    if output.exists():
        raise FileExistsError(f"Run output already exists; choose a new directory: {output}")
    started = perf_counter()

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    announce("1/6 Fingerprint training inputs")
    source_files = {name: _fingerprint(bundle / name) for name in INPUT_FILES}

    announce("2/6 Audit training integrity")
    audit = audit_training_data(bundle)
    behaviors = pl.read_parquet(bundle / "train/behaviors.parquet")
    articles = pl.read_parquet(bundle / "articles.parquet", columns=["article_id", "published_time"])
    history = pl.read_parquet(bundle / "train/history.parquet")

    announce("3/6 Quarantine and split complete impressions")
    prepared = prepare_training(behaviors, articles)
    period_start = behaviors["impression_time"].min()

    announce("4/6 Score the same tuning candidates with all three rules")
    comparison = compare_baselines(prepared.tune, articles, history, period_start)

    announce("5/6 Attribute missing popularity counts")
    coverage = audit_popularity_coverage(prepared.tune, articles, history, period_start)
    if source_files != {name: _fingerprint(bundle / name) for name in INPUT_FILES}:
        raise ValueError("Source files changed during the run; no outputs were saved")

    source_root = Path(__file__).resolve().parent
    implementation_files = {
        path.name: _fingerprint(path)
        for path in sorted(source_root.glob("*.py"))
    }
    result = {
        "experiment": "M1 development baseline",
        "protocol_status": "Existing working rules; final evaluation protocol remains under review",
        "bundle_name": bundle.name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_seconds_excluding_output_writes": round(perf_counter() - started, 6),
        "runtime": {"python": platform.python_version(), "polars": pl.__version__},
        "source_files": source_files,
        "implementation_files": implementation_files,
        "official_validation_used": False,
        "audit": audit,
        "preparation": {
            "raw_impressions": behaviors.height,
            "fit_impressions": prepared.fit.height,
            "tune_impressions": prepared.tune.height,
            "quarantined_impressions": prepared.quarantine.height,
            "duplicate_click_impressions": prepared.duplicate_click_impressions,
            "future_publication_impressions": prepared.future_publication_impressions,
            "overlapping_reasons": prepared.overlapping_reasons,
            "tuning_start": prepared.tuning_start.isoformat(),
        },
        "baselines": asdict(comparison),
        "coverage": coverage,
    }

    announce("6/6 Save a new run and its provenance")
    encoded_result = json.dumps(result, indent=2, default=str) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    prepared.fit.write_parquet(output / "fit_behaviors.parquet")
    prepared.tune.write_parquet(output / "tune_behaviors.parquet")
    prepared.quarantine.write_parquet(output / "quarantined_behaviors.parquet")
    (output / "run.json").write_text(encoded_result, encoding="utf-8")
    return result
