"""Leakage guarantees, checked on a dataset with known planted duplicates.

These run in CI on synthetic data. ``test_real_data_audit`` re-checks the real processed
split when it exists locally.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from defect_detection.cli import app
from defect_detection.config import load_config
from defect_detection.data.dedupe import load_thumbnails
from defect_detection.data.leakage_audit import run_audit
from defect_detection.data.split import EXCLUDED, EXTERNAL_TEST
from tests.conftest import PipelineRun, make_config
from tests.fixtures.synthetic import build_planted_dataset


def audit(run: PipelineRun, splits: pd.DataFrame) -> dict:
    d = run.config.dedupe
    return run_audit(
        run.manifest, splits, run.thumbs, d.similarity_threshold, d.info_threshold, d.shortlist_k
    )


def test_corrupt_file_is_not_in_manifest(official_run: PipelineRun) -> None:
    assert official_run.cases["corrupt"] not in set(official_run.manifest["rel_path"])


def test_exact_copy_is_collapsed(official_run: PipelineRun) -> None:
    assert official_run.row("exact_copy")["exclude_reason"] == "exact_duplicate"
    assert official_run.row("exact_original")["split"] in {"train", "val"}


def test_rotated_copy_of_test_image_is_removed_from_train(official_run: PipelineRun) -> None:
    assert official_run.row("rotated_test_source")["split"] == "test"
    leak = official_run.row("rotated_test_leak")
    assert leak["exclude_reason"] == "near_duplicate_of_official_test"


def test_label_conflict_is_excluded_entirely(official_run: PipelineRun) -> None:
    for case in ("label_conflict_def", "label_conflict_ok"):
        assert official_run.row(case)["exclude_reason"] == "exact_duplicate_label_conflict"


def test_near_duplicates_land_in_the_same_split(official_run: PipelineRun) -> None:
    dup, src = official_run.row("jpeg_near_dup"), official_run.row("jpeg_source")
    assert dup["cluster_id"] == src["cluster_id"]
    assert dup["split"] == src["split"] != EXCLUDED


def test_official_test_set_is_kept_intact(official_run: PipelineRun) -> None:
    test_rows = official_run.splits[official_run.splits["source_split"] == "test"]
    assert (test_rows["split"] == "test").all()
    assert official_run.split_report["strategy"] == "official"


def test_external_copy_of_train_image_is_excluded(official_run: PipelineRun) -> None:
    assert official_run.row("external_copy_source")["split"] in {"train", "val"}
    assert official_run.row("external_copy")["exclude_reason"] == "near_duplicate_of_train_or_val"
    assert official_run.row("external_unique")["split"] == EXTERNAL_TEST


def test_evaluation_rows_record_nearest_train_similarity(official_run: PipelineRun) -> None:
    splits = official_run.splits
    evaluated = splits["split"].isin(["test", EXTERNAL_TEST])
    assert splits.loc[evaluated, "nearest_train_similarity"].between(-1, 0.99).all()
    assert splits.loc[~evaluated, "nearest_train_similarity"].isna().all()


@pytest.mark.parametrize("run_name", ["official_run", "random_run"])
def test_audit_passes_on_pipeline_output(run_name: str, request: pytest.FixtureRequest) -> None:
    run: PipelineRun = request.getfixturevalue(run_name)
    result = audit(run, run.splits)
    assert result["passed"], result["checks"]
    assert result["test_set_size"] > 0
    assert result["external_test_set_size"] > 0


def test_random_strategy_keeps_planted_pairs_together(random_run: PipelineRun) -> None:
    assert random_run.split_report["strategy"] == "random"
    for a, b in [("rotated_test_leak", "rotated_test_source"), ("jpeg_near_dup", "jpeg_source")]:
        assert random_run.row(a)["split"] == random_run.row(b)["split"]


def _move_to(splits: pd.DataFrame, rel_path: str, split: str) -> pd.DataFrame:
    tampered = splits.copy()
    tampered.loc[tampered["rel_path"] == rel_path, ["split", "exclude_reason"]] = [split, ""]
    return tampered


def test_audit_catches_rotated_leak_regardless_of_cluster_ids(official_run: PipelineRun) -> None:
    """Put the rotated test copy back in train (its own cluster id): the content check fires."""
    tampered = _move_to(official_run.splits, official_run.cases["rotated_test_leak"], "train")
    result = audit(official_run, tampered)
    assert not result["passed"]
    assert result["checks"]["no_near_duplicates_across_splits"] is False
    assert result["checks"]["no_cluster_in_two_splits"] is True  # ids alone would miss it


def test_audit_catches_exact_duplicate_across_splits(official_run: PipelineRun) -> None:
    original_split = official_run.row("exact_original")["split"]
    other = "val" if original_split == "train" else "train"
    tampered = _move_to(official_run.splits, official_run.cases["exact_copy"], other)
    assert audit(official_run, tampered)["checks"]["no_sha256_in_two_splits"] is False


def test_audit_catches_external_copy(official_run: PipelineRun) -> None:
    tampered = _move_to(official_run.splits, official_run.cases["external_copy"], EXTERNAL_TEST)
    assert audit(official_run, tampered)["checks"]["no_near_duplicates_across_splits"] is False


def test_audit_catches_dropped_rows(official_run: PipelineRun) -> None:
    result = audit(official_run, official_run.splits.iloc[1:])
    assert result["checks"]["all_manifest_rows_accounted_for"] is False


def _write_config(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    config = make_config(tmp_path)
    cases = build_planted_dataset(config.data.raw_dir)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return config_file, cases, config.data.processed_dir


def test_cli_pipeline_end_to_end(tmp_path: Path) -> None:
    config_file, _, processed = _write_config(tmp_path)
    runner = CliRunner()
    for command in ("manifest", "split", "audit"):
        result = runner.invoke(app, [command, "--config", str(config_file)])
        assert result.exit_code == 0, result.output
    report = json.loads((processed / "leakage_audit.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert {"git_commit", "config_hash", "created_at", "test_set_sha256"} <= report.keys()


def test_cli_audit_exits_nonzero_on_leak(tmp_path: Path) -> None:
    config_file, cases, processed = _write_config(tmp_path)
    runner = CliRunner()
    for command in ("manifest", "split"):
        assert runner.invoke(app, [command, "--config", str(config_file)]).exit_code == 0
    splits_path = processed / "splits.csv"
    splits = pd.read_csv(splits_path, keep_default_na=False)
    _move_to(splits, cases["rotated_test_leak"], "train").to_csv(splits_path, index=False)
    result = runner.invoke(app, ["audit", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "FAIL  no_near_duplicates_across_splits" in result.output


REAL = Path("data/processed")


@pytest.mark.requires_data
@pytest.mark.skipif(not (REAL / "splits.csv").exists(), reason="real data not processed locally")
def test_real_data_audit() -> None:
    config = load_config(Path("configs/train.yaml"))
    manifest = pd.read_csv(REAL / "manifest.csv")
    splits = pd.read_csv(REAL / "splits.csv", keep_default_na=False)
    d = config.dedupe
    thumbs = load_thumbnails(config.data.raw_dir, manifest["rel_path"], d.coarse_size, d.fine_size)
    result = run_audit(
        manifest, splits, thumbs, d.similarity_threshold, d.info_threshold, d.shortlist_k
    )
    assert result["passed"], result["checks"]
