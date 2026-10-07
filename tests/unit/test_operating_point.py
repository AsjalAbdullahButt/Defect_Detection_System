"""End-to-end V3 on a tiny trained run: only validation data may be used."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from defect_detection.cli import app
from defect_detection.config import ProjectConfig
from defect_detection.core.model_meta import ModelMeta
from defect_detection.data import dataset as dataset_module
from defect_detection.training import trainer
from defect_detection.training.operating_point import CANDIDATE_META, calibrate_and_select
from tests.conftest import run_pipeline


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[ProjectConfig, Path]:
    root = tmp_path_factory.mktemp("v3")
    run = run_pipeline(root, with_official_test=True)
    run.config.data.processed_dir.mkdir(parents=True, exist_ok=True)
    run.splits.to_csv(run.config.data.processed_dir / "splits.csv", index=False)
    run_dir = run.config.train.output_dir / "run1"
    trainer.train(run.config, run_dir)
    (run.config.train.output_dir / "LATEST").write_text("run1", encoding="utf-8")
    return run.config, run_dir


def test_calibration_uses_validation_only(
    trained: tuple[ProjectConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run_dir = trained
    requested: list[str] = []
    original = dataset_module.SplitDataset.__init__

    def recording(self, splits_csv, raw_dir, split, *args, **kwargs):  # type: ignore[no-untyped-def]
        requested.append(split)
        original(self, splits_csv, raw_dir, split, *args, **kwargs)

    monkeypatch.setattr(dataset_module.SplitDataset, "__init__", recording)
    report = calibrate_and_select(config, run_dir)
    assert requested == ["val"]
    assert report["split_used"] == "val"


def test_outputs_and_candidate_meta(trained: tuple[ProjectConfig, Path]) -> None:
    config, run_dir = trained
    report = calibrate_and_select(config, run_dir)
    for name in ("calibration.json", "reliability.png", CANDIDATE_META):
        assert (run_dir / name).exists(), name
    meta = ModelMeta.model_validate_json((run_dir / CANDIDATE_META).read_text(encoding="utf-8"))
    assert meta.threshold == pytest.approx(report["threshold"])
    assert meta.temperature == pytest.approx(report["temperature"])
    assert meta.review_band.low <= meta.threshold <= meta.review_band.high
    assert meta.input.image_size == config.preprocess.image_size
    names = [p["name"] for p in report["operating_points"]]
    assert names == ["default_0.5", "f1_optimal", "recall_constrained"]
    recall = report["operating_points"][2]["recall"]
    assert recall >= config.operating_point.min_defect_recall


def test_cli_calibrate_uses_latest_run(
    trained: tuple[ProjectConfig, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = trained
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["calibrate", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "review band=" in result.output
    assert json.loads((tmp_path / "reports" / "calibration.json").read_text())["run"] == "run1"


def test_cli_calibrate_without_runs_fails(tmp_path: Path) -> None:
    from tests.conftest import make_config

    config = make_config(tmp_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    result = CliRunner().invoke(app, ["calibrate", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "train first" in result.output
