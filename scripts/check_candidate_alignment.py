"""Teaching diagnostic: safe candidate reordering versus detached scores.

The deliberately broken branch is isolated here, never used by the experiment
runner. It demonstrates that equal list lengths do not prove ID/score alignment.
Only aggregate results are printed; no raw user or article records are emitted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freshrank.baselines import compare_baselines, recency_scores
from freshrank.metrics import score_impression, summarize_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("prepared", type=Path, help="Existing M1 run containing tune_behaviors.parquet")
    args = parser.parse_args()
    tuning = pl.read_parquet(args.prepared / "tune_behaviors.parquet")
    articles = pl.read_parquet(args.bundle / "articles.parquet", columns=["article_id", "published_time"])
    history = pl.read_parquet(args.bundle / "train/history.parquet")
    start = pl.read_parquet(args.bundle / "train/behaviors.parquet", columns=["impression_time"])["impression_time"].min()

    reference = compare_baselines(tuning, articles, history, start)
    reordered = tuning.with_columns(pl.col("article_ids_inview").list.reverse())
    safe = compare_baselines(reordered, articles, history, start)
    if reference != safe:
        raise AssertionError("Candidate storage order changed the complete baseline comparison")

    publication_times = dict(articles.iter_rows())
    broken_results = []
    changed = 0
    for row in tuning.select("impression_time", "article_ids_inview", "article_ids_clicked").iter_rows(named=True):
        article_ids = row["article_ids_inview"]
        scores = recency_scores(row["impression_time"], article_ids, publication_times)
        reversed_ids = article_ids[::-1]
        reversed_labels = [int(article_id in row["article_ids_clicked"]) for article_id in reversed_ids]
        # Intentional fault: scores retain the OLD order while IDs use the NEW one.
        broken = score_impression(reversed_ids, scores, reversed_labels)
        original_labels = [int(article_id in row["article_ids_clicked"]) for article_id in article_ids]
        original = score_impression(article_ids, scores, original_labels)
        broken_results.append(broken)
        changed += int(broken != original)

    print(json.dumps({
        "diagnostic": "Intentional ID/score misalignment; not an alternative recommender",
        "impressions": tuning.height,
        "safe_reordering_preserves_complete_comparison": reference == safe,
        "correct_recency": asdict(reference.recency),
        "deliberately_misaligned_recency": asdict(summarize_metrics(broken_results)),
        "impressions_with_changed_metrics": changed,
        "official_validation_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
