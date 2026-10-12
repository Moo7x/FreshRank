# M1 demo baseline diagnostic

Date: 1 October 2026. This is a first end-to-end **development diagnostic on the demo**, not an official holdout or a final benchmark. The small EB-NeRD bundle required for the main M1 experiment has not been downloaded. Official validation labels were not read by this run.

## What was compared

The cleaned training data was split chronologically before scoring: 20,378 earlier fitting impressions and 4,217 tuning impressions on 24–25 May 2023. These two rules do not fit a learned model on the fitting rows. They score each candidate in the same 4,217 tuning impressions:

- **Fixed-history popularity:** number of click events for the candidate article in `train/history.parquet`. The 258,318 history clicks end at 18 May 2023 06:59:51, before the first training behavior impression at 07:00:03. Counts remain unchanged across the behavior period. Unseen articles score zero. In-period behavior clicks and later article totals are excluded.
- **Recency:** negative article age in seconds, `-(impression_time - published_time)`. Higher scores mean newer articles. The scorer fails if an article is missing or published after the impression; the preparation step already quarantined known future-publication cases.

The later `article_ids_clicked` are used only as binary evaluation labels. Both rules use deterministic article-ID tie-breaking. NDCG@10 and full-list MRR are computed within each impression and macro-averaged across valid impressions. All 4,217 tuning impressions have at least one positive label; none were excluded. Run locally with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_demo_baselines.py data\ebnerd_demo data\processed\demo_m1
```

## Measured results

| Rule | Mean NDCG@10 | Mean MRR | Impressions where its NDCG beat the other main rule |
| --- | ---: | ---: | ---: |
| Fixed-history popularity | 0.377896 | 0.247148 | 1,547 |
| Recency | 0.438529 | 0.320545 | 2,061 |

The two NDCG values were equal on 609 impressions. The paired mean differences are descriptive only; no confidence interval or generalization claim is made here.

### Tie-break-only control

Because many popularity counts are zero, we also scored every candidate `0` and let the predeclared ascending-article-ID tie-breaker set the entire order. This is a diagnostic control, **not** a random or useful recommender; article IDs themselves may correlate with publication time or other properties.

| Rule | Mean NDCG@10 | Mean MRR |
| --- | ---: | ---: |
| Article-ID tie-break only | 0.376745 | 0.245507 |
| Fixed-history popularity | 0.377896 | 0.247148 |

Popularity changes the article-ID-only order in 1,570 of 4,217 impressions, yet it raises NDCG in only 26 and lowers it in 22; the two score identically on 4,169. Thus the demo popularity metric is almost the same as its tie-break control. The tiny difference should not be sold as substantive value from historical popularity. This is a sharper interpretation than simply comparing popularity with recency.

## Why the popularity rule is limited

Of 46,497 tuning candidate appearances, 40,615 (about 87.4%) have no fixed-history click count. There are 1,546 impressions (about 36.7%) in which *every* candidate scores zero popularity, so the predeclared article-ID tie-breaker determines the entire popularity ordering. Of 4,240 clicked candidate appearances, 4,184 (about 98.7%) have no fixed-history click count. This is consistent with a fast-changing news candidate pool, but the data alone do not prove that this is the only reason recency scored higher. The pre-period counts come from a sampled user cohort, not a production-wide traffic count.

The history timestamps establish an event-time boundary, not when a production feature pipeline would have received every event; the latest click is only 12 seconds before the behavior period. This baseline is a conservative offline proxy under the fixed-history assumption, not a demonstration of online feature availability. Neither offline score estimates live click-through rate, and the shuffled stored candidate order prevents reconstructing original display-position effects.

A [12 October coverage attribution](M1_POPULARITY_COVERAGE.md) finds that 39,007 of the 40,615 zero-count candidate appearances have recorded publication times after the latest fixed-history event. This explains a dominant temporal coverage limitation under the publication-time assumption; it does not establish a causal explanation for the metric difference.

## Verification and next work

As of 12 October, 38 tests pass, including future-history rejection, future-published-article rejection, unseen-article handling, tie-break-only control behavior, separation of later click labels from score inputs, coverage attribution, complete experiment orchestration and candidate-score alignment.

The complete demo can now be reproduced with one command:

```powershell
.\.venv\Scripts\python.exe scripts\run_m1_experiment.py data\ebnerd_demo data\processed\demo_run_01
```

Use a fresh output directory. The run saves cleaned partitions and an aggregate `run.json` with runtime versions and SHA-256 fingerprints of training inputs and package source. Two real-data runs on 12 October reproduced the same audit, preparation, baseline and coverage results with matching input/source fingerprints. Elapsed time and completion timestamps are not deterministic results. This is verification in the current environment, not a cross-platform reproducibility claim.

The demo result must be repeated on the small bundle before claiming the main M1 baseline is complete; official validation remains reserved for frozen configurations. The implemented rules remain provisional rather than a user-ratified final evaluation protocol.

## Candidate-alignment controlled failure (12 October)

Run the isolated teaching diagnostic on an existing prepared run:

```powershell
.\.venv\Scripts\python.exe scripts\check_candidate_alignment.py data\ebnerd_demo data\processed\demo_run_01
```

Reversing each candidate list and recomputing its features by article identity preserves the complete three-rule comparison on all 4,217 tuning impressions. A deliberately broken branch reverses candidate IDs but retains old-order recency scores; labels are still attached to the correct IDs. This silently assigns scores to the wrong articles while retaining valid types and list lengths.

The deliberate fault changes metric pairs on 3,907 impressions. Correct recency mean NDCG@10/MRR is 0.438529 / 0.320545; the broken branch scores 0.439302 / 0.318433. The slightly higher NDCG must not be interpreted as improvement: correctness requires identity alignment, not a preferred metric direction. This faulty branch is isolated from the experiment runner and is not an alternative recommender.

Two regression tests cover complete-comparison order invariance and aligned distinct-score/label permutations. The full suite now has 38 passing tests. Neither this diagnostic nor these fixtures change the evaluation protocol or access official validation.
