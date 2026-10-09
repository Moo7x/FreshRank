"""Prepare an impression-level, chronological M1 development split."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class PreparedTraining:
    fit: pl.DataFrame
    tune: pl.DataFrame
    quarantine: pl.DataFrame
    tuning_start: date
    duplicate_click_impressions: int
    future_publication_impressions: int
    overlapping_reasons: int


def prepare_training(behaviors: pl.DataFrame, articles: pl.DataFrame) -> PreparedTraining:
    """Keep whole impressions; quarantine bad labels or unavailable candidates."""
    if behaviors.is_empty():
        raise ValueError("Training behaviors are empty")
    if behaviors["impression_id"].null_count() or behaviors["impression_id"].n_unique() != behaviors.height:
        raise ValueError("Training impression IDs must be non-null and unique")
    if articles["article_id"].null_count() or articles["article_id"].n_unique() != articles.height:
        raise ValueError("Article IDs must be non-null and unique")
    if behaviors["impression_time"].null_count() or articles["published_time"].null_count():
        raise ValueError("Missing timestamps require investigation before splitting")
    malformed_lists = behaviors.filter(
        (pl.col("article_ids_inview").list.len() == 0)
        | pl.col("article_ids_inview").is_null()
        | pl.col("article_ids_clicked").is_null()
        | (pl.col("article_ids_inview").list.len() != pl.col("article_ids_inview").list.n_unique())
    )
    candidate_pairs = behaviors.select("impression_id", pl.col("article_ids_inview").alias("article_id")).explode("article_id", empty_as_null=False)
    clicked_pairs = behaviors.select("impression_id", pl.col("article_ids_clicked").alias("article_id")).explode("article_id", empty_as_null=False)
    unmatched_clicks = clicked_pairs.join(candidate_pairs, on=["impression_id", "article_id"], how="anti")
    if not malformed_lists.is_empty() or not unmatched_clicks.is_empty():
        raise ValueError("Unexpected candidate or click structure; audit before preparing")

    future_ids = (
        behaviors.select("impression_id", "impression_time", pl.col("article_ids_inview").alias("article_id"))
        .explode("article_id", empty_as_null=False)
        .join(articles.select("article_id", "published_time"), on="article_id", how="left")
    )
    if future_ids["published_time"].null_count():
        raise ValueError("Candidate article reference is missing")
    future_ids = (
        future_ids.filter(pl.col("published_time") > pl.col("impression_time"))
        .select("impression_id")
        .unique()
        .with_columns(pl.lit(True).alias("future_published_candidate"))
    )
    flagged = (
        behaviors.with_columns(
            (pl.col("article_ids_clicked").list.len() != pl.col("article_ids_clicked").list.n_unique())
            .alias("duplicate_clicked_id")
        )
        .join(future_ids, on="impression_id", how="left")
        .with_columns(pl.col("future_published_candidate").fill_null(False))
    )
    quarantine = flagged.filter(pl.col("duplicate_clicked_id") | pl.col("future_published_candidate"))
    clean = flagged.filter(~(pl.col("duplicate_clicked_id") | pl.col("future_published_candidate"))).drop(
        "duplicate_clicked_id", "future_published_candidate"
    )
    dates = sorted(clean["impression_time"].dt.date().unique().to_list())
    if len(dates) < 3:
        raise ValueError("Need at least three distinct training dates for the two-date tuning split")
    tuning_start = dates[-2]
    fit = clean.filter(pl.col("impression_time").dt.date() < tuning_start)
    tune = clean.filter(pl.col("impression_time").dt.date() >= tuning_start)
    if fit.is_empty() or tune.is_empty() or fit.height + tune.height + quarantine.height != behaviors.height:
        raise AssertionError("Preparation lost impressions or produced an empty partition")

    return PreparedTraining(
        fit=fit,
        tune=tune,
        quarantine=quarantine,
        tuning_start=tuning_start,
        duplicate_click_impressions=quarantine.filter(pl.col("duplicate_clicked_id")).height,
        future_publication_impressions=quarantine.filter(pl.col("future_published_candidate")).height,
        overlapping_reasons=quarantine.filter(pl.col("duplicate_clicked_id") & pl.col("future_published_candidate")).height,
    )
