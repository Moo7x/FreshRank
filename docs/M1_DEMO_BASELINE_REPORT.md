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

## Verification and next work

Twenty-four tests pass, including future-history rejection, future-published-article rejection, unseen-article handling, tie-break-only control behavior, and separation of later click labels from score inputs. The demo result must be repeated on the small bundle before claiming the main M1 baseline is complete; official validation remains reserved for frozen configurations.
