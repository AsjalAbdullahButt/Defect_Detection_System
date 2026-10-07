"""Train/serve skew guard (§3.8): the training data path without augmentation must produce
exactly the arrays the serving path produces from raw upload bytes."""

import io

import numpy as np
import pytest
from PIL import Image

from defect_detection.core.preprocessing import PreprocessSpec, preprocess
from defect_detection.data.augment import TrainTransform
from defect_detection.data.dataset import SplitDataset
from tests.conftest import PipelineRun


@pytest.fixture(scope="module")
def spec(official_run: PipelineRun) -> PreprocessSpec:
    p = official_run.config.preprocess
    out = official_run.config.data.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    official_run.splits.to_csv(out / "splits.csv", index=False)
    return PreprocessSpec(p.image_size, p.mean, p.std)


def serving_path(raw_bytes: bytes, spec: PreprocessSpec) -> np.ndarray:
    """What the API does: decode the uploaded bytes in memory, then core.preprocess."""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.load()
        return preprocess(img, spec)


@pytest.mark.parametrize("split", ["val", "test"])
def test_dataset_matches_serving_bit_for_bit(
    official_run: PipelineRun, spec: PreprocessSpec, split: str
) -> None:
    cfg = official_run.config.data
    dataset = SplitDataset(cfg.processed_dir / "splits.csv", cfg.raw_dir, split, spec)
    for index, rel_path in enumerate(dataset.rel_paths):
        tensor, _ = dataset[index]
        expected = serving_path((cfg.raw_dir / rel_path).read_bytes(), spec)
        assert np.array_equal(tensor.numpy(), expected), rel_path


@pytest.mark.parametrize("split", ["val", "test"])
def test_augmentation_is_refused_outside_train(
    official_run: PipelineRun, spec: PreprocessSpec, split: str
) -> None:
    cfg = official_run.config
    with pytest.raises(ValueError, match="only allowed on train"):
        SplitDataset(
            cfg.data.processed_dir / "splits.csv",
            cfg.data.raw_dir,
            split,
            spec,
            TrainTransform(spec, cfg.augment),
        )
