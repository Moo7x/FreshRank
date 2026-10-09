"""Print a synthetic worked example of the existing M1 history-count pipeline."""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freshrank.baselines import historical_popularity, popularity_scores


def main() -> None:
    cutoff = datetime(2023, 5, 18, 9)
    history = pl.DataFrame(
        {
            "user_id": [101, 202],
            "article_id_fixed": [[20, 10], [20, 30]],
            "impression_time_fixed": [
                [cutoff - timedelta(minutes=60), cutoff - timedelta(minutes=50)],
                [cutoff - timedelta(minutes=40), cutoff - timedelta(minutes=30)],
            ],
        }
    )
    print("SYNTHETIC LESSON DATA; not an EB-NeRD measurement")
    print("History rows:", history.to_dicts())
    print("History schema:", history.schema)

    # Explode both columns together to preserve each article/time pairing.
    events = history.select("article_id_fixed", "impression_time_fixed").explode(
        "article_id_fixed", "impression_time_fixed", empty_as_null=False
    )
    print("Event rows:", events.to_dicts())
    print("Shapes (rows, columns):", history.shape, "->", events.shape)

    invalid_events = events.filter(pl.col("impression_time_fixed") >= cutoff)
    print("Events at or after cutoff:", invalid_events.height)
    count_table = events.group_by("article_id_fixed").len().sort("article_id_fixed")
    print("Count rows:", count_table.to_dicts())

    # Call the production function as well: the lesson must match real behavior.
    counts, latest, event_count = historical_popularity(history, cutoff)
    print("Production counts:", counts)
    print("Latest accepted event:", latest.isoformat())
    print("Accepted event count:", event_count)
    print("Candidate IDs:", [10, 20, 30])
    print("Popularity scores:", popularity_scores([10, 20, 30], counts))

    # Deliberately supply a later event to observe the existing failure behavior.
    late_history = pl.DataFrame(
        {
            "article_id_fixed": [[30]],
            "impression_time_fixed": [[cutoff + timedelta(minutes=1)]],
        }
    )
    try:
        historical_popularity(late_history, cutoff)
    except ValueError as exc:
        print("Deliberate invalid input rejected:", exc)
    else:
        raise AssertionError("The timestamp guard failed to reject a later event")


if __name__ == "__main__":
    main()
