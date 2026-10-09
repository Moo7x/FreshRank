"""Integrity checks for the EB-NeRD training bundle."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _as_text(value: object) -> str | None:
    return None if value is None else str(value)


def audit_training_data(bundle: Path) -> dict[str, object]:
    """Return aggregate integrity evidence without exposing dataset records."""
    behaviors = pl.read_parquet(bundle / "train" / "behaviors.parquet")
    history = pl.read_parquet(bundle / "train" / "history.parquet")
    articles = pl.read_parquet(bundle / "articles.parquet")

    candidates = behaviors.select(
        "impression_id",
        "impression_time",
        pl.col("article_ids_inview").alias("candidate_id"),
    ).explode("candidate_id", empty_as_null=False)

    clicks = behaviors.select(
        "impression_id",
        pl.col("article_ids_clicked").alias("clicked_id"),
    ).explode("clicked_id", empty_as_null=False)

    candidate_articles = candidates.join(
        articles.select(
            pl.col("article_id").alias("candidate_id"),
            "published_time",
            "last_modified_time",
        ),
        on="candidate_id",
        how="left",
    )
    future_candidates = candidate_articles.filter(
        pl.col("published_time") > pl.col("impression_time")
    ).with_columns(
        (pl.col("published_time") - pl.col("impression_time")).alias(
            "publication_lead"
        )
    )

    history_lengths = history.with_columns(
        pl.col("article_id_fixed").list.len().alias("article_count"),
        pl.col("impression_time_fixed").list.len().alias("time_count"),
        pl.col("read_time_fixed").list.len().alias("read_count"),
        pl.col("scroll_percentage_fixed").list.len().alias("scroll_count"),
    )
    mismatched_history = history_lengths.filter(
        ~(
            (pl.col("article_count") == pl.col("time_count"))
            & (pl.col("article_count") == pl.col("read_count"))
            & (pl.col("article_count") == pl.col("scroll_count"))
        )
    )
    history_events = history.select(
        pl.col("impression_time_fixed").alias("history_time")
    ).explode("history_time", empty_as_null=False)

    earliest_impression = behaviors["impression_time"].min()
    latest_impression = behaviors["impression_time"].max()
    latest_history_event = history_events["history_time"].max()

    duplicate_candidates = behaviors.filter(
        pl.col("article_ids_inview").list.len()
        != pl.col("article_ids_inview").list.n_unique()
    )
    duplicate_clicks = behaviors.filter(
        pl.col("article_ids_clicked").list.len()
        != pl.col("article_ids_clicked").list.n_unique()
    )
    clicks_outside_candidates = clicks.join(
        candidates,
        left_on=["impression_id", "clicked_id"],
        right_on=["impression_id", "candidate_id"],
        how="anti",
    )
    missing_candidate_articles = (
        candidates.select("candidate_id")
        .unique()
        .join(
            articles.select(pl.col("article_id").alias("candidate_id")),
            on="candidate_id",
            how="anti",
        )
    )
    users_without_history = behaviors.select("user_id").unique().join(
        history.select("user_id"), on="user_id", how="anti"
    )

    return {
        "impressions": behaviors.height,
        "unique_impression_ids": behaviors["impression_id"].n_unique(),
        "candidate_appearances": candidates.height,
        "unique_candidate_articles": candidates["candidate_id"].n_unique(),
        "empty_candidate_impressions": behaviors.filter(
            pl.col("article_ids_inview").list.len() == 0
        ).height,
        "duplicate_candidate_impressions": duplicate_candidates.height,
        "no_click_impressions": behaviors.filter(
            pl.col("article_ids_clicked").list.len() == 0
        ).height,
        "multiple_click_impressions": behaviors.filter(
            pl.col("article_ids_clicked").list.len() > 1
        ).height,
        "multiple_distinct_click_impressions": behaviors.filter(
            pl.col("article_ids_clicked").list.n_unique() > 1
        ).height,
        "multiple_click_impressions_without_duplicates": behaviors.filter(
            (pl.col("article_ids_clicked").list.len() > 1)
            & (
                pl.col("article_ids_clicked").list.len()
                == pl.col("article_ids_clicked").list.n_unique()
            )
        ).height,
        "duplicate_click_impressions": duplicate_clicks.height,
        "clicked_ids_outside_candidates": clicks_outside_candidates.height,
        "candidate_article_references_missing": missing_candidate_articles.height,
        "behavior_users_missing_history": users_without_history.height,
        "history_rows_with_misaligned_lists": mismatched_history.height,
        "history_events_at_or_after_first_impression": history_events.filter(
            pl.col("history_time") >= earliest_impression
        ).height,
        "candidate_rows_published_after_impression": future_candidates.height,
        "impressions_with_future_published_candidates": future_candidates[
            "impression_id"
        ].n_unique(),
        "future_published_candidate_articles": future_candidates[
            "candidate_id"
        ].n_unique(),
        "shortest_future_publication_lead": _as_text(
            future_candidates["publication_lead"].min()
        ),
        "longest_future_publication_lead": _as_text(
            future_candidates["publication_lead"].max()
        ),
        "candidate_rows_modified_after_impression": candidate_articles.filter(
            pl.col("last_modified_time") > pl.col("impression_time")
        ).height,
        "first_impression": _as_text(earliest_impression),
        "last_impression": _as_text(latest_impression),
        "last_fixed_history_event": _as_text(latest_history_event),
    }
