# M1 ranking examples: from an impression to candidate rows

An **impression** is one recommendation opportunity. Its candidate articles must stay together because the task is to order *those articles for that opportunity*. The click list is observed afterward. It supplies labels for an offline experiment, not information available to the ranker when it scores candidates.

Expanding an impression produces one row per candidate, all sharing one `impression_id`. EB-NeRD shuffles the stored candidate list, so its order cannot establish actual display positions. The code does not create a position feature. The following records are synthetic.

## A small synthetic trace

Suppose one impression has candidate IDs `[10, 20, 30]` and later clicked IDs `[20, 30]`:

| impression_id | article_id | clicked |
| ---: | ---: | ---: |
| 7 | 10 | 0 |
| 7 | 20 | 1 |
| 7 | 30 | 1 |

The group ID stays `7` for all three rows. The two clicked articles are two positive labels, not two separate impressions. If there are no clicks, the group still has candidate rows, all with label `0`; the metric code counts and excludes these groups from its averages.

In `src/freshrank/ranking_data.py`, `explode("article_id")` turns each candidate list into rows. This expression creates the label:

```python
pl.col("article_ids_clicked").list.contains(pl.col("article_id")).cast(pl.Int8).alias("clicked")
```

For each row, it asks whether that row's article ID appears in the **same impression's** later click list. The code then drops the click list from the result, leaving the group key, time, article ID and offline label. Before using these rows to score a new impression, scoring code must exclude `clicked` too; it is a target, not a feature.

This is a ranking dataset rather than a standalone classification dataset: the question is not merely whether article `20` gets a click in isolation, but where it should appear relative to articles `10` and `30` *inside impression 7*. A ranking metric lets us test an ordering before computing any baseline result.

## How the first ranking metric works

For labels `[0, 1, 1]` in ranked order, the two clicked articles are at ranks 2 and 3. Binary discounted cumulative gain (DCG) gives a click at one-based rank `r` credit `1/log₂(r + 1)`. Thus the observed DCG is `1/log₂(3) + 1/log₂(4) ≈ 1.131`. The best ordering puts both clicks first: ideal DCG is `1 + 1/log₂(3) ≈ 1.631`. NDCG is observed DCG divided by ideal DCG, approximately `0.693`. The perfect order scores `1`. The `@10` version looks only at the first ten ranks; if the list is shorter, it uses the whole list.

`src/freshrank/metrics.py` scores one impression at a time. Equal predicted scores are ordered by ascending article ID, not by the shuffled input order. Its secondary MRR metric is `1 / (rank of first clicked article)` over the full candidate list. A no-click impression has no ideal click ordering, so both metric values are `None`; summary code counts and excludes such groups instead of treating them as zero-quality recommendations. Final results will macro-average valid impression scores so a large candidate list does not automatically outweigh a smaller impression.

These metric rules were tested on synthetic perfect, reversed, multi-click, tied, short, cutoff and no-positive examples. They are used in the [demo baseline diagnostic](M1_DEMO_BASELINE_REPORT.md). The final evaluation protocol remains under review.
