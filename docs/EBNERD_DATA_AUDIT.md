# EB-NeRD data audit

Status: **demo training audit and strict development split completed**. Feature availability remains preliminary. A development baseline diagnostic has been measured; see [the report](M1_DEMO_BASELINE_REPORT.md). The final evaluation protocol remains under review.

## Source and access

- Dataset: [EB-NeRD official download page](https://recsys.eb.dk/)
- Field descriptions: [official dataset documentation](https://recsys.eb.dk/dataset/)
- Provider's [General License Terms](https://recsys.eb.dk/assets/pdf/general_license_terms.pdf)
- Reference implementation: [EB-NeRD benchmark repository](https://github.com/jppol-ai/ebnerd-benchmark)
- Download origin recorded by Windows: `https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip`

The provider asks the downloader to read and accept its terms and supply their name, email address and organizational affiliation before accessing the data. The terms restrict use to research, prohibit commercial exploitation, redistributing the data and attempts to identify users, and do not permit training language models from it. Results can be published only without exact dataset content. Dataset access and eventual deletion must follow the provider's terms. The benchmark code's MIT license is separate from the dataset license.

The inspected bundle was obtained on 26 September 2026. Extracted files are kept under the ignored local `data/ebnerd_demo/` directory. The archive and extracted Parquet files are excluded from Git.

| Download record | Value |
| --- | --- |
| Archive | `ebnerd_demo.zip` |
| Archive bytes | 21,499,083 |
| SHA-256 | `FB28B4F95249C6571F2FE52ED4083A810BA7DA651AA38123D16B232BBECA72C2` |
| Download date | 26 September 2026, according to the file timestamp |
| Source | Provider-controlled S3 URL above, from the downloaded file's Windows zone metadata |

The archive has the five expected Parquet files plus macOS metadata entries. Its paths were checked before extraction. The macOS metadata is irrelevant to analysis.

## Expected bundle shape

The official documentation says each bundle has training and validation behavior and history files, plus article information. Each behavior file represents seven days of impressions. Its accompanying history file contains clicks from a preceding 21 day period and is fixed during that behavior period. The demo's extracted paths match this structure. The official test set is separate and unnecessary for M1.

| File | Rows | Size (bytes) |
| --- | ---: | ---: |
| `articles.parquet` | 11,777 | 15,276,561 |
| `train/behaviors.parquet` | 24,724 | 1,132,621 |
| `train/history.parquet` | 1,590 | 2,560,135 |
| `validation/behaviors.parquet` | 25,356 | 1,207,912 |
| `validation/history.parquet` | 1,562 | 2,384,884 |

The training impression times run from 18 May 2023 07:00:03 through 25 May 2023 06:59:52. Candidate list lengths range from 5 to 100. All 24,724 training impressions have at least one entry in `article_ids_clicked`; this is an observed property of the demo, not an assumption about live traffic. Candidate-list order is shuffled and does not reveal original display position. Examples in the public walkthrough use synthetic records.

Observed column names include `impression_time`, `article_ids_inview`, and `article_ids_clicked` in behaviors; `article_id_fixed` and `impression_time_fixed` in history; and `published_time`, `last_modified_time`, and `total_pageviews` in articles. The full observed schemas can be reproduced with `scripts/inspect_ebnerd.py`.

## Initial training integrity results

Run these checks with:

```powershell
.\.venv\Scripts\python.exe scripts\audit_ebnerd_training.py data\ebnerd_demo
```

| Check | Observed result | Interpretation |
| --- | ---: | --- |
| Unique impression IDs | 24,724 / 24,724 | Each training row has a distinct impression ID. |
| Candidate appearances | 278,139 | This is the row count after expanding candidate lists. It is not an impression count. |
| Unique candidate articles | 2,478 | Articles recur across impressions. |
| Empty candidate lists | 0 | Every audited impression has candidates. |
| Impressions with duplicate candidates | 0 | Candidate lists contain no repeated article ID. |
| Clicked IDs outside their candidate set | 0 | Every clicked label points to a candidate in that impression. |
| Missing candidate-to-article references | 0 | Every candidate ID joins to the article table. |
| Behavior users missing fixed history | 0 | All 1,590 training users have a history row. |
| History rows with different parallel-list lengths | 0 | History article, time, read-time and scroll lists align structurally. |
| History events at or after the first behavior impression | 0 | The fixed histories end before the behavior period; the final history timestamp is 12 seconds before the first behavior timestamp. |
| Impressions with multiple distinct clicks | 73 | Multi-positive ranking cases are present. Seventy-one contain multiple clicks without a duplicated clicked ID. |
| Impressions with duplicate clicked IDs | 59 | These require a declared policy; they will not be silently deduplicated. |
| Impressions with a candidate published after the impression | 70 | Two articles account for these cases. Recorded publication leads range from 2 seconds to 44 minutes 14 seconds. This conflicts with strict prediction-time availability and requires quarantine or another justified policy. |
| Candidate appearances whose article was modified after the impression | 278,139 | The supplied article table is a later snapshot. Mutable article content cannot be assumed historically available in strict replay. |

The implemented conservative policy quarantines whole impressions containing duplicate clicked IDs or future-published candidates from the primary strict-replay experiment. Removing only one candidate would change the logged ranking task. Multi-click impressions with distinct clicked IDs remain valid and retain multiple positive labels. Other structural failures cause preparation to stop rather than silently proceed. The quarantine files retain the original rows plus two reason flags; the raw bundle is unchanged.

## M1 demo preparation and chronology

Reproduce the preparation with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ebnerd_demo.py data\ebnerd_demo data\processed\demo_m1
```

The command refuses to overwrite an existing output directory. Its generated Parquet files and `summary.json` live under ignored `data/processed/demo_m1/`; they must not be committed or redistributed. To repeat the run, choose a new output directory. The code reads **only the official training behaviors and articles**; it does not use official validation labels.

| Partition or reason | Impressions | Candidate appearances |
| --- | ---: | ---: |
| Fit: before 24 May 2023 | 20,378 | 230,187 |
| Tune: 24–25 May 2023 | 4,217 | 46,497 |
| Quarantine: excluded from fit and tune | 129 | 1,455 |
| Original training total | 24,724 | 278,139 |

The two quarantine reasons are 59 impressions with repeated clicked IDs and 70 with future-published candidates; they do not overlap in this demo. All partition IDs are disjoint. The fit set's last timestamp is 23 May 2023 23:42:12; the tuning set's first is 24 May 2023 00:00:29. The tuning set contains 3,445 impressions on 24 May and 772 on 25 May. This is a date-based split, so the final date is a partial day in the demo. These counts describe preparation. The current metric code explicitly counts and excludes no-positive groups from its averages.

## Preliminary field availability register

An impression is one recommendation opportunity. Its candidate list and eventual clicked list have different roles. The statuses below are conservative until we inspect the actual data and timestamp semantics.

| Source and field | Role at impression time | Initial rule or open question |
| --- | --- | --- |
| Behaviors: impression ID, user ID, impression time | Grouping and time keys | Keep all candidates in an impression together. IDs are references, not labels. |
| Behaviors: inview article IDs | Candidate set | Available for the logged impression; the provided list order is shuffled, so it is not a display position. |
| Behaviors: clicked article IDs | Outcome label | Use only for training and evaluation after the impression; never as a prediction feature. |
| Behaviors: next read time, next scroll percentage | Later outcomes | Exclude from prediction features. |
| Behaviors: current article ID, read time, scroll percentage | Context with uncertain timing | Check whether each value would already exist when the candidate ranking is requested. Exclude until established. |
| Behaviors: subscription, SSO, demographics, device | Possible context | Verify observation time, stability and missing values before using. The official documentation says subscription status was fixed when the dataset was created. |
| History: clicked article IDs and timestamps | Prior user history | Use only events known before the impression. Fixed pre-period history does not represent clicks that occurred within the behavior period. |
| Articles: publication time | Possible age feature | Use only if publication precedes the impression; compute age from the two timestamps. |
| Articles: title, body, categories, topics, embeddings and similar content | Possible content features | A later article edit or later model-generated representation can leak information. Exclude from strict historical replay until snapshot availability is established. |
| Articles: seven-day total inviews, pageviews and read time | Future aggregates | Exclude; totals include activity after early impressions. Recompute a historical count from earlier events if needed. |

## Pre-period popularity availability check

The provider describes `train/history.parquet` as users' clicked articles from the 21 days **before** the training behavior period, fixed rather than updated during that period. Locally, 1,590 training users contribute 258,318 history click events dated 27 April 2023 07:00:05 through 18 May 2023 06:59:51. The first training impression is 18 May 2023 07:00:03, 12 seconds later. These timestamps support a conservative *static* popularity baseline: count each article's clicks in this pre-period history once, then use that count unchanged for every training development impression. Do not count any clicked IDs from the fitting or tuning behavior rows as prediction-time features; their feedback-availability times are not established here.

The pre-period counts cover 1,087 of the 2,478 distinct article IDs in raw training candidate lists. In the cleaned tuning partition, 5,882 of 46,497 candidate appearances (about 12.7%) have an article with any pre-period history click; 388 of its 1,141 distinct candidate articles have such a count. Unseen articles would receive zero and tie-break by article ID under the current metric rule. This is a substantial cold-start limitation, not a reason to substitute future seven-day totals. A recency baseline can instead rank by `impression_time - published_time`, after the existing future-publication quarantine.

This working feature policy has been exercised in the demo baseline diagnostic; see [the measured results and limitations](M1_DEMO_BASELINE_REPORT.md). The dataset is a sampled cohort, so these history counts are not a claim about all real-world readers. The official validation labels remain reserved.

## Reproducible audit checks

1. Record archive checksum, extracted file list, file sizes and Parquet schemas.
2. Count impressions and candidates; inspect the range of impression times and candidate-list lengths.
3. Check whether clicked IDs belong to each impression's inview IDs, and document empty or malformed cases without silently fixing them.
4. Inspect sample impressions locally and confirm that the candidate order is shuffled as documented.
5. Compare article publication and modification times with impression times before deciding which article fields can support strict replay.
6. Keep all candidate records within their impression and use chronological development splits.

The current results are development diagnostics for fixed baseline rules; no learned model or official holdout result is claimed. The official validation data has only been checked for file structure and schema; its labels remain reserved for later frozen-model evaluation.
