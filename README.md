# FreshRank

FreshRank is a learning-first news recommendation project built with the [EB-NeRD dataset](https://recsys.eb.dk/). It starts with a trustworthy offline ranking experiment and will grow in measured stages toward model serving, streaming features, recovery testing and operational monitoring.

The project studies one central question:

> How do feature freshness and data pipeline failures affect recommendation quality, and how can predictions remain correct and reproducible?

## Project status

**Current milestone: M1 — data audit and trustworthy baseline.**

The reproducible workspace and initial EB-NeRD demo audit are complete. The demo contains 24,724 training impressions, 25,356 validation impressions and 11,777 article records. Two transparent baseline rules have been compared on cleaned **demo development data**; this is not a frozen-validation benchmark or a trained model.

| Milestone | Focus | Status |
| --- | --- | --- |
| M0 | Reproducible Python workspace and project orientation | Complete |
| M1 | Data audit, chronological evaluation, popularity and recency baselines | In progress |
| M2 | LightGBM learning-to-rank and experiment lineage | Planned |
| M3 | Small PyTorch history-aware ranker | Planned |
| M4 | FastAPI serving, containers, CI and rollback | Planned |
| M5 | Kafka/Spark replay, online features and recovery experiments | Planned |
| M6 | Analytical tables, backfills and observability | Planned |
| M7 | Reproducible demonstration and portfolio evidence | Planned |

Planned technologies describe the roadmap. They are not presented as implemented results or learned skills.

## Why the project is structured this way

Recommendation experiments are easy to overstate when future information leaks into features, candidate articles are separated from their impressions, or an offline result is described as a live business outcome. FreshRank treats data timing, evaluation validity and failure recovery as part of the modeling problem.

The project develops practical skills through executable examples, code walkthroughs and debugging exercises. Personal learning records stay local; the public repository documents implementation, experiments and limitations.

## Current artifacts

- A pinned Python 3.12 environment with Polars, DuckDB and pytest.
- A local bundle inspection command that reports file sizes, schemas, row counts and training aggregates without printing dataset records.
- An EB-NeRD data audit, a preliminary feature-availability register, and a strict training quarantine/development split.
- A grouped candidate-label transformation and tested impression-level NDCG@10/MRR metrics.
- A first reproducible demo development comparison of fixed-history popularity and article recency, with measured coverage limitations.
- Tests for package setup, data integrity, splitting, candidate labeling and ranking metrics.

The dataset itself is ignored by Git and will not be distributed through this repository.

## Local setup

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pytest
```

The environment currently targets Python 3.12. The official EB-NeRD benchmark supports Python 3.10+ and recommends Python 3.11. Current data commands use Polars. DuckDB is included in the environment for planned analytical work; the complete reference package is not installed.

## Dataset access and inspection

EB-NeRD is provided under its own [General License Terms](https://recsys.eb.dk/assets/pdf/general_license_terms.pdf). Read and accept those terms on the [official download page](https://recsys.eb.dk/) and obtain the demo bundle directly from the provider.

Extract the demo under `data/ebnerd_demo/`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_ebnerd.py data\ebnerd_demo
```

The expected local structure is:

```text
data/ebnerd_demo/
├── articles.parquet
├── train/
│   ├── behaviors.parquet
│   └── history.parquet
└── validation/
    ├── behaviors.parquet
    └── history.parquet
```

## Reproduce the demo experiment

After obtaining and extracting the dataset, run:

```powershell
.\.venv\Scripts\python.exe scripts\audit_ebnerd_training.py data\ebnerd_demo
.\.venv\Scripts\python.exe scripts\prepare_ebnerd_demo.py data\ebnerd_demo data\processed\demo_m1
.\.venv\Scripts\python.exe scripts\evaluate_demo_baselines.py data\ebnerd_demo data\processed\demo_m1
```

Preparation refuses to overwrite an existing output directory; choose a new directory when repeating it. All generated dataset files remain local under `data/`.

For an executable history-count example that uses synthetic data and requires no dataset download:

```powershell
.\.venv\Scripts\python.exe scripts\trace_history_counts.py
```

Measured results and their limits are recorded in the [demo baseline report](docs/M1_DEMO_BASELINE_REPORT.md).

## Current development evaluation

These are the implemented working rules for the demo diagnostic. The final evaluation protocol remains under review.

- Keep every candidate from one impression in the same split.
- Develop chronologically so later behavior cannot influence earlier predictions.
- Exclude post-click outcomes and future article totals from prediction features.
- Use NDCG@10 as the primary ranking metric and MRR as a secondary metric.
- Reserve the official labeled validation set for configurations frozen after development.
- Report weak or negative results when the evaluation is valid; do not invent success thresholds.

Offline ranking results cannot establish live click-through rate or revenue effects. Historical replay also does not establish production-scale performance.

## Repository map

```text
docs/EBNERD_DATA_AUDIT.md       Dataset record and field-availability audit
docs/M1_DEMO_BASELINE_REPORT.md Demo results, control experiment and limitations
scripts/inspect_ebnerd.py       Local, privacy-conscious bundle inspection
scripts/evaluate_demo_baselines.py  Reproduce the baseline comparison
src/freshrank/                  Project package
tests/                          Reproducibility and data-tooling tests
```

## Documentation

- [EB-NeRD data audit](docs/EBNERD_DATA_AUDIT.md)
- [M1 ranking-example walkthrough](docs/M1_RANKING_EXAMPLES.md)
- [M1 demo baseline diagnostic](docs/M1_DEMO_BASELINE_REPORT.md)
- [Draft strict-replay data contract](docs/adr/0001-strict-replay-feature-availability.md)

FreshRank adapts public research and the official EB-NeRD resources for learning. It does not claim a novel recommendation architecture.
