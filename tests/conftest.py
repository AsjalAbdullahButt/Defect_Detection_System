"""Shared fixtures. All images are generated on the fly; no real data is committed."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest
import yaml

from defect_detection.config import ProjectConfig
from defect_detection.data.dedupe import Thumbnails, load_thumbnails
from defect_detection.data.manifest import build_manifest
from defect_detection.data.split import make_splits
from tests.fixtures.synthetic import build_planted_dataset, write_image

matplotlib.use("Agg")  # headless plotting in tests


@pytest.fixture
def casting_like_dataset(tmp_path: Path) -> Path:
    """Tiny casting-shaped tree: train/test x def_front/ok_front, plus a corrupt file."""
    root = tmp_path / "raw"
    seed = 0
    for split, n_def, n_ok in [("train", 3, 2), ("test", 2, 1)]:
        for class_dir, n in [("def_front", n_def), ("ok_front", n_ok)]:
            for i in range(n):
                seed += 1
                write_image(root / split / class_dir / f"img_{i}.jpeg", size=(300, 300), seed=seed)
    good = write_image(root / "train" / "ok_front" / "truncated.jpeg", size=(300, 300), seed=99)
    good.write_bytes(good.read_bytes()[:400])
    (root / "notes.txt").write_text("not an image", encoding="utf-8")
    return root


REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"


def make_config(root: Path, strategy: str = "auto", seed: int = 42) -> ProjectConfig:
    """The repo config with paths under ``root`` and a tiny, untrained model for speed."""
    raw = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    raw["seed"] = seed
    raw["data"].update(
        raw_dir=str(root / "raw"),
        processed_dir=str(root / "processed"),
        external_test_dirs=["external"],
    )
    raw["split"]["strategy"] = strategy
    raw["preprocess"]["image_size"] = 64
    raw["model"].update(backbone="test_efficientnet", pretrained=False)
    raw["train"].update(
        output_dir=str(root / "runs"),
        device="cpu",
        batch_size=8,
        num_workers=0,
        head_epochs=1,
        finetune_epochs=2,
    )
    return ProjectConfig.model_validate(raw)


@dataclass
class PipelineRun:
    """Outputs of manifest -> cluster -> split on a planted dataset."""

    config: ProjectConfig
    cases: dict[str, str]
    manifest: pd.DataFrame
    thumbs: Thumbnails
    splits: pd.DataFrame
    split_report: dict[str, Any]

    def row(self, case: str) -> pd.Series:
        """The splits.csv row for a planted case."""
        return self.splits.set_index("rel_path").loc[self.cases[case]]


def run_pipeline(root: Path, with_official_test: bool) -> PipelineRun:
    """Build a planted dataset under ``root`` and run the data pipeline in memory."""
    config = make_config(root)
    cases = build_planted_dataset(config.data.raw_dir, with_official_test)
    manifest, _ = build_manifest(config.data.raw_dir, config.data.class_aliases)
    d = config.dedupe
    thumbs = load_thumbnails(config.data.raw_dir, manifest["rel_path"], d.coarse_size, d.fine_size)
    splits, split_report = make_splits(
        manifest, thumbs, config.split, d, config.seed, config.data.external_test_dirs
    )
    return PipelineRun(config, cases, manifest, thumbs, splits, split_report)


@pytest.fixture(scope="module")
def official_run(tmp_path_factory: pytest.TempPathFactory) -> PipelineRun:
    """Planted dataset with an official test folder (official strategy)."""
    return run_pipeline(tmp_path_factory.mktemp("official"), with_official_test=True)


@pytest.fixture(scope="module")
def random_run(tmp_path_factory: pytest.TempPathFactory) -> PipelineRun:
    """Planted dataset without split folders (random strategy)."""
    return run_pipeline(tmp_path_factory.mktemp("random"), with_official_test=False)
