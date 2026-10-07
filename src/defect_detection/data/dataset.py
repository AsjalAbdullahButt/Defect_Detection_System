"""Torch Dataset over one split of splits.csv.

Only the train split may be augmented; val/test always use the deterministic core
preprocessing, and asking for anything else raises.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from defect_detection.core.preprocessing import FloatArray, PreprocessSpec, preprocess

Transform = Callable[[Image.Image], FloatArray]


class SplitDataset(Dataset[tuple[torch.Tensor, int]]):
    """Images of one split as ``(CHW float tensor, label)`` pairs."""

    def __init__(
        self,
        splits_csv: Path,
        raw_dir: Path,
        split: str,
        spec: PreprocessSpec,
        train_transform: Transform | None = None,
    ) -> None:
        if train_transform is not None and split != "train":
            raise ValueError(f"augmentation is only allowed on train, not {split!r}")
        table = pd.read_csv(splits_csv, keep_default_na=False)
        rows = table[table["split"] == split].sort_values("rel_path")
        if rows.empty:
            raise ValueError(f"split {split!r} is empty in {splits_csv}")
        self.rel_paths: list[str] = rows["rel_path"].tolist()
        self.labels: list[int] = rows["label"].astype(int).tolist()
        self.raw_dir = raw_dir
        self.spec = spec
        self.train_transform = train_transform

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.raw_dir / self.rel_paths[index]) as img:
            img.load()
            array = (
                self.train_transform(img)
                if self.train_transform is not None
                else preprocess(img, self.spec)
            )
        return torch.from_numpy(array), self.labels[index]

    def class_counts(self) -> np.ndarray:
        """Number of images per label id."""
        return np.bincount(self.labels, minlength=2)
