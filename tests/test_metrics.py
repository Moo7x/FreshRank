"""Hand-checkable ranking-metric fixtures."""

from math import log2

import pytest

from freshrank.metrics import score_impression, summarize_metrics


def test_perfect_and_reversed_rankings() -> None:
    perfect = score_impression([10, 20, 30], [3.0, 2.0, 1.0], [1, 1, 0])
    reversed_order = score_impression([10, 20, 30], [1.0, 2.0, 3.0], [1, 1, 0])
    assert perfect.ndcg_at_10 == pytest.approx(1.0)
    assert perfect.mrr == 1.0
    assert reversed_order.ndcg_at_10 == pytest.approx(
        ((1 / log2(3)) + (1 / log2(4))) / (1 + (1 / log2(3)))
    )
    assert reversed_order.mrr == 0.5


def test_multiple_clicks_and_short_group_match_hand_calculation() -> None:
    result = score_impression([10, 20, 30], [3.0, 2.0, 1.0], [0, 1, 1])
    assert result.ndcg_at_10 == pytest.approx(
        ((1 / log2(3)) + (1 / log2(4))) / (1 + (1 / log2(3)))
    )
    assert result.mrr == 0.5


def test_ties_use_article_id_not_input_order() -> None:
    first = score_impression([20, 10], [1.0, 1.0], [1, 0])
    reordered_input = score_impression([10, 20], [1.0, 1.0], [0, 1])
    assert first == reordered_input
    assert first.ndcg_at_10 == pytest.approx(1 / log2(3))
    assert first.mrr == 0.5


def test_distinct_scores_and_labels_must_follow_their_article_ids() -> None:
    ids = [101, 205, 309, 412]
    scores = [4.0, 3.0, 2.0, 1.0]
    labels = [1, 0, 1, 0]
    original = score_impression(ids, scores, labels)
    aligned_rows = list(zip(ids, scores, labels, strict=True))[::-1]
    new_ids, new_scores, new_labels = map(list, zip(*aligned_rows, strict=True))
    assert score_impression(new_ids, new_scores, new_labels) == original
    # Same lengths, valid types, finite scores: the wrong pairing still passes
    # structural validation. Only a semantic/identity check catches the mistake.
    assert score_impression(new_ids, scores, new_labels) != original


def test_no_positive_is_excluded_and_reported() -> None:
    no_click = score_impression([10, 20], [1.0, 0.0], [0, 0])
    clicked = score_impression([10, 20], [1.0, 0.0], [1, 0])
    summary = summarize_metrics([no_click, clicked])
    assert no_click.ndcg_at_10 is None and no_click.mrr is None
    assert (summary.total_impressions, summary.evaluated_impressions, summary.no_positive_impressions) == (2, 1, 1)
    assert summary.mean_ndcg_at_10 == pytest.approx(1.0)
    assert summary.mean_mrr == pytest.approx(1.0)
    assert summarize_metrics([no_click]).mean_ndcg_at_10 is None


def test_ndcg_cutoff_does_not_hide_full_list_mrr() -> None:
    ids = list(range(11))
    result = score_impression(ids, [float(11 - i) for i in ids], [0] * 10 + [1])
    assert result.ndcg_at_10 == 0.0
    assert result.mrr == pytest.approx(1 / 11)


@pytest.mark.parametrize(
    "article_ids,scores,labels",
    [([], [], []), ([1, 2], [1.0], [1, 0]), ([1, 1], [1.0, 0.0], [1, 0]),
     ([1], [float("nan")], [1]), ([1], [1.0], [2])],
)
def test_invalid_inputs_fail(article_ids: list[int], scores: list[float], labels: list[int]) -> None:
    with pytest.raises(ValueError):
        score_impression(article_ids, scores, labels)
