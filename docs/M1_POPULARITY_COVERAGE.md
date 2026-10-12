# M1 static-popularity coverage attribution

Measured on 12 October 2026 using the cleaned demo tuning partition. This diagnostic explains coverage of the existing baseline; it changes no candidate sets, feature policy, split, metric or final evaluation protocol. Official validation was not read.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\audit_popularity_coverage.py data\ebnerd_demo data\processed\demo_m1
```

The command joins grouped tuning candidates to recorded publication times and pre-period history click counts, and prints aggregate JSON. Source records and generated dataset files remain local.

## Boundary and population

- Latest observed fixed-history event: 18 May 2023 06:59:51.
- Behavior period begins: 18 May 2023 07:00:03.
- Recorded fixed-history click events: 258,318.
- Cleaned tuning population: 4,217 impressions, 46,497 candidate appearances, 1,141 distinct candidate articles and 4,240 clicked candidate appearances.
- Appearance counts reflect repetition of articles across impressions. They are not unique-article counts. Outcome labels are used only to summarize observed clicked appearances, not to define a coverage group.

## Measured attribution

| Coverage group | Candidate appearances | Distinct candidate articles | Clicked candidate appearances |
| --- | ---: | ---: | ---: |
| Has an earlier-history count and was published by history end | 5,882 | 388 | 56 |
| No count; published after history end | 39,007 | 595 | 4,175 |
| No count; published by history end | 1,608 | 158 | 9 |
| Count exists despite publication after history end | 0 | 0 | 0 |
| Total | 46,497 | 1,141 | 4,240 |

Of the 40,615 appearances without counts, 39,007 / 40,615 = **96.0%** have publication times after the observed history end. The same group contains 4,175 / 4,240 = **98.5%** of the tuning clicked appearances. These percentages have different denominators; neither is live click-through rate.

## Interpretation and limits

Under the recorded publication-time assumption, an article published after the last history event could not have accumulated an eligible click in that history. The dominant gap is therefore consistent with the temporal limitation of a static feature. A larger reader sample with the same history boundary cannot create clicks before an article exists.

For the 1,608 older zero-count appearances, publication timing alone cannot explain absence. The data does not distinguish no exposure, no clicks from this sampled cohort, limited observation, or other causes. Zero means no recorded count in this source; it is not evidence of universal unpopularity.

This diagnosis does not establish why recency outperformed popularity, nor prove that a rolling popularity feature would improve ranking. Per-impression feedback availability and article metadata accuracy still need evidence before such a feature is implemented. History event timestamps alone do not establish feature ingestion time.

Dropping the zero-count candidates would change the ranking task. In particular, dropping the post-history-publication group would remove 4,175 of the 4,240 observed positive candidate appearances. Report coverage while preserving the same candidate population for comparisons.

## Integrity checks

- Article keys must be unique so a join cannot silently multiply candidates.
- Missing publication records cause an error, rather than disappearing through an inner join or becoming an invented age.
- Missing count matches become zero, reflecting the existing static-popularity rule.
- Publication after an impression causes an error; this command expects prepared data.
- A later history event is rejected by the existing baseline guard.
- Publication exactly at the last history timestamp belongs to the "by history end" category.
- An observed count alongside later publication has its own contradiction category, rather than being explained away.
- Category counts must reconcile to the unchanged candidate population.
- Synthetic tests change later click labels and verify that coverage attribution remains unchanged.

The current full suite passes: **38 tests**, including the six coverage-audit tests. Tests use synthetic records; this report includes aggregates only.
