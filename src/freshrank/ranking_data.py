"""Convert clean EB-NeRD impressions into grouped ranking examples."""

from __future__ import annotations

import polars as pl


def candidate_rows(behaviors: pl.DataFrame) -> pl.DataFrame:
    """Return one candidate row per article, retaining its impression group.

    The click label is for offline training/evaluation only, never a scoring input.
    The source candidate-list order is shuffled and is deliberately not emitted.
    """
    if behaviors["impression_id"].n_unique() != behaviors.height:
        raise ValueError("Input must have one row per impression")
    if behaviors.select(pl.col("article_ids_inview").list.len().min()).item() == 0:
        raise ValueError("Every impression must have at least one candidate")

    rows = (
        behaviors.select(
            "impression_id",
            "impression_time",
            "article_ids_clicked",
            pl.col("article_ids_inview").alias("article_id"),
        )
        .explode("article_id", empty_as_null=False)
        .with_columns(
            pl.col("article_ids_clicked").list.contains(pl.col("article_id")).cast(pl.Int8).alias("clicked")
        )
        .drop("article_ids_clicked")
    )
    expected = behaviors["article_ids_inview"].list.len().sum()
    if rows.height != expected:
        raise AssertionError("Candidate expansion changed the row count")
    return rows
