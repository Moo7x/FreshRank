"""Check complete M1 runs, provenance and refusal to overwrite results."""

from datetime import datetime, timedelta
import json
from pathlib import Path

import polars as pl
import pytest

import freshrank.experiment as experiment


def _bundle(root: Path) -> Path:
    start = datetime(2023, 5, 1, 12)
    (root / "train").mkdir(parents=True)
    pl.DataFrame(
        {
            "impression_id": [1, 2, 3, 4, 5],
            "user_id": [1] * 5,
            "impression_time": [start, start + timedelta(days=1), start + timedelta(days=2), start, start],
            "article_ids_inview": [[10, 20], [10, 20], [10, 20], [10, 20], [10, 30]],
            "article_ids_clicked": [[10], [20], [10, 20], [20, 20], [10]],
        }
    ).write_parquet(root / "train/behaviors.parquet")
    pl.DataFrame(
        {
            "article_id": [10, 20, 30],
            "published_time": [start - timedelta(days=1), start - timedelta(days=1), start + timedelta(days=1)],
            "last_modified_time": [start + timedelta(days=10)] * 3,
        }
    ).write_parquet(root / "articles.parquet")
    pl.DataFrame(
        {
            "user_id": [1],
            "article_id_fixed": [[10, 20]],
            "impression_time_fixed": [[start - timedelta(hours=2), start - timedelta(hours=1)]],
            "read_time_fixed": [[1.0, 1.0]],
            "scroll_percentage_fixed": [[50.0, 60.0]],
        }
    ).write_parquet(root / "train/history.parquet")
    return root


def test_full_run_records_inputs_and_outputs_without_validation_files(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    output = tmp_path / "run"
    stages = []
    result = experiment.run_m1_experiment(bundle, output, stages.append)
    assert result["preparation"]["raw_impressions"] == 5
    assert result["preparation"]["fit_impressions"] == 1
    assert result["preparation"]["tune_impressions"] == 2
    assert result["preparation"]["quarantined_impressions"] == 2
    assert result["baselines"]["popularity"]["evaluated_impressions"] == 2
    assert result["official_validation_used"] is False
    assert set(result["source_files"]) == set(experiment.INPUT_FILES)
    assert all(len(value["sha256"]) == 64 for value in result["source_files"].values())
    assert set(path.name for path in output.iterdir()) == {
        "run.json", "fit_behaviors.parquet", "tune_behaviors.parquet", "quarantined_behaviors.parquet"
    }
    # JSON represents datetime objects as text; compare the serialized forms.
    expected_baselines = json.loads(json.dumps(result["baselines"], default=str))
    assert json.loads((output / "run.json").read_text())["baselines"] == expected_baselines
    assert len(stages) == 6


def test_same_inputs_and_code_reproduce_results_in_fresh_runs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    first = experiment.run_m1_experiment(bundle, tmp_path / "run-a")
    second = experiment.run_m1_experiment(bundle, tmp_path / "run-b")
    for key in ("source_files", "implementation_files", "audit", "preparation", "baselines", "coverage"):
        assert first[key] == second[key]


def test_existing_run_is_preserved_even_if_inputs_are_missing(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "run.json"
    sentinel.write_text("keep this existing result", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        experiment.run_m1_experiment(tmp_path / "missing-bundle", output)
    assert sentinel.read_text() == "keep this existing result"


def test_source_changes_abort_before_creating_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    original = experiment._fingerprint
    calls = 0

    def altered_on_recheck(path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        fingerprint = original(path)
        if calls == 4:
            fingerprint["sha256"] = "0" * 64
        return fingerprint

    monkeypatch.setattr(experiment, "_fingerprint", altered_on_recheck)
    output = tmp_path / "aborted"
    with pytest.raises(ValueError, match="changed during"):
        experiment.run_m1_experiment(bundle, output)
    assert not output.exists()
