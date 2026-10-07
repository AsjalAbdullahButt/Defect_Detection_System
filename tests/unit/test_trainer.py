"""Trainer behaviour on the planted synthetic dataset with timm's tiny ``test_efficientnet``."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from typer.testing import CliRunner

from defect_detection.cli import app
from defect_detection.config import ProjectConfig
from defect_detection.data import dataset as dataset_module
from defect_detection.training import trainer
from defect_detection.training.model_factory import (
    classifier,
    create_model,
    set_backbone_frozen,
    train_mode,
)
from tests.conftest import run_pipeline


def prepared_config(root: Path) -> ProjectConfig:
    run = run_pipeline(root, with_official_test=True)
    run.config.data.processed_dir.mkdir(parents=True, exist_ok=True)
    run.splits.to_csv(run.config.data.processed_dir / "splits.csv", index=False)
    run.manifest.to_csv(run.config.data.processed_dir / "manifest.csv", index=False)
    return run.config


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> ProjectConfig:
    return prepared_config(tmp_path_factory.mktemp("trainer"))


def test_balanced_class_weights() -> None:
    weights = trainer.class_weights(np.array([30, 10]), "balanced")
    torch.testing.assert_close(weights, torch.tensor([40 / 60, 40 / 20]))
    assert trainer.class_weights(np.array([30, 10]), "none").tolist() == [1.0, 1.0]


def test_freezing_leaves_only_the_head_trainable(config: ProjectConfig) -> None:
    model = create_model(config.model)
    set_backbone_frozen(model, frozen=True)
    head = {id(p) for p in classifier(model).parameters()}
    for param in model.parameters():
        assert param.requires_grad == (id(param) in head)
    train_mode(model, backbone_frozen=True)
    bn = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bn
    assert not any(m.training for m in bn)  # ImageNet BN statistics preserved
    assert classifier(model).training
    set_backbone_frozen(model, frozen=False)
    assert all(p.requires_grad for p in model.parameters())


@pytest.fixture(scope="module")
def trained(
    config: ProjectConfig, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict, Path, set[str]]:
    requested: set[str] = set()
    original_init = dataset_module.SplitDataset.__init__

    def recording_init(self, splits_csv, raw_dir, split, *args, **kwargs):  # type: ignore[no-untyped-def]
        requested.add(split)
        original_init(self, splits_csv, raw_dir, split, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dataset_module.SplitDataset, "__init__", recording_init)
        run_dir = tmp_path_factory.mktemp("run")
        summary = trainer.train(config, run_dir)
    return summary, run_dir, requested


def test_training_never_touches_test(trained: tuple[dict, Path, set[str]]) -> None:
    _, _, requested = trained
    assert requested == {"train", "val"}


def test_training_outputs(trained: tuple[dict, Path, set[str]]) -> None:
    summary, run_dir, _ = trained
    for name in ("best.pt", "history.csv", "curves.png", "train_summary.json"):
        assert (run_dir / name).exists(), name
    history = pd.read_csv(run_dir / "history.csv")
    assert history["stage"].tolist()[:1] == ["head"]
    assert 2 <= len(history) <= 3
    assert {
        "val",
        "train_no_aug",
        "generalization_gap",
        "git_commit",
        "config_hash",
    } <= summary.keys()
    assert 0 <= summary["val"]["pr_auc"] <= 1


def test_checkpoint_loads_weights_only_and_carries_provenance(
    trained: tuple[dict, Path, set[str]], config: ProjectConfig
) -> None:
    _, run_dir, _ = trained
    checkpoint = trainer.load_checkpoint(run_dir / "best.pt")
    assert {
        "state_dict",
        "git_commit",
        "config_hash",
        "backbone",
        "config",
        "epoch",
    } <= checkpoint.keys()
    model = create_model(config.model)
    model.load_state_dict(checkpoint["state_dict"])


def test_training_is_reproducible(config: ProjectConfig, tmp_path: Path) -> None:
    trainer.train(config, tmp_path / "a")
    trainer.train(config, tmp_path / "b")
    a = pd.read_csv(tmp_path / "a" / "history.csv")
    b = pd.read_csv(tmp_path / "b" / "history.csv")
    pd.testing.assert_frame_equal(a.drop(columns="seconds"), b.drop(columns="seconds"))


def test_cli_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = prepared_config(tmp_path / "data")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # reports/ is written relative to the working directory
    result = CliRunner().invoke(app, ["train", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "PR-AUC=" in result.output
    latest = (config.train.output_dir / "LATEST").read_text(encoding="utf-8")
    assert (config.train.output_dir / latest / "best.pt").exists()
    assert json.loads((tmp_path / "reports" / "train_summary.json").read_text())["val"]
