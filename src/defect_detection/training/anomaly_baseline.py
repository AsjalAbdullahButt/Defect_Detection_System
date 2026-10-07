"""PatchCore-style anomaly-detection baseline (Roth et al., 2022), written in-repo.

Idea: describe every small patch of a NORMAL image with pretrained mid-level CNN features and
store them in a memory bank. A test patch far from every stored patch is unusual; an image's
anomaly score is its most unusual patch. No defect images or labels are used for fitting,
which is the appeal for production lines where defects are rare or new.

Pipeline:
1. features: ImageNet ``backbone`` (not fine-tuned) at two depths (``out_indices``), each
   smoothed with 3x3 average pooling, upsampled to the larger map and concatenated;
2. memory bank: ``patches_per_image`` random patches per normal TRAIN image, reduced to
   ``coreset_size`` with greedy k-center selection (keeps coverage of rare normal patterns);
3. score: max over an image's patches of the distance to the nearest memory-bank patch.

The decision threshold is chosen on validation with the same policy as the classifier
(best precision at defect recall >= the configured target), then applied once to test.
Simplifications vs the paper: random patch subsampling before the coreset, no reweighting of
the nearest-neighbour score.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from defect_detection.config import AnomalyConfig, ProjectConfig
from defect_detection.core.constants import NORMAL_LABEL
from defect_detection.core.preprocessing import PreprocessSpec
from defect_detection.data.dataset import SplitDataset
from defect_detection.data.split import EXTERNAL_TEST
from defect_detection.provenance import provenance
from defect_detection.training.evaluate import classification_metrics
from defect_detection.training.thresholding import threshold_for_recall

FloatArray = npt.NDArray[np.float32]


class PatchFeatures(nn.Module):
    """Locally aware patch embeddings from two pretrained feature maps."""

    def __init__(self, cfg: AnomalyConfig, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=pretrained, features_only=True, out_indices=cfg.out_indices
        )
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, P, D) patch embeddings."""
        maps = [self.smooth(m) for m in self.backbone(x)]
        size = maps[0].shape[-2:]
        maps = [
            m if m.shape[-2:] == size else nn.functional.interpolate(m, size=size, mode="bilinear")
            for m in maps
        ]
        stacked = torch.cat(maps, dim=1)  # (B, D, h, w)
        return stacked.flatten(2).transpose(1, 2)  # (B, h*w, D)


def greedy_coreset(points: FloatArray, size: int, projection_dim: int, seed: int) -> FloatArray:
    """k-center greedy: repeatedly add the point farthest from everything chosen so far.

    Distances are computed in a random low-dimensional projection (Johnson-Lindenstrauss) to
    keep selection fast; the selected points keep their full dimension.
    """
    if len(points) <= size:
        return points
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(points.shape[1], projection_dim)).astype(np.float32)
    projected = points @ projection
    chosen = [int(rng.integers(len(points)))]
    min_dist = np.linalg.norm(projected - projected[chosen[0]], axis=1)
    for _ in range(size - 1):
        nxt = int(np.argmax(min_dist))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(projected - projected[nxt], axis=1))
    return points[np.array(chosen)]


class PatchCore:
    """Fit on normal images, score any images."""

    def __init__(self, cfg: AnomalyConfig, seed: int, pretrained: bool = True) -> None:
        self.cfg = cfg
        self.seed = seed
        self.features = PatchFeatures(cfg, pretrained).eval()
        self.memory: torch.Tensor | None = None

    def fit(self, normal_loader: DataLoader[tuple[torch.Tensor, int]]) -> dict[str, Any]:
        """Build the memory bank from normal images; returns size/timing info."""
        rng = np.random.default_rng(self.seed)
        start = time.perf_counter()
        sampled = []
        for images, _ in normal_loader:
            patches = self.features(images).numpy()  # (B, P, D)
            for per_image in patches:
                keep = rng.choice(
                    len(per_image),
                    size=min(self.cfg.patches_per_image, len(per_image)),
                    replace=False,
                )
                sampled.append(per_image[keep])
        pool = np.concatenate(sampled).astype(np.float32)
        bank = greedy_coreset(pool, self.cfg.coreset_size, self.cfg.projection_dim, self.seed)
        self.memory = torch.from_numpy(bank)
        return {
            "patch_pool": len(pool),
            "memory_bank": len(bank),
            "feature_dim": int(bank.shape[1]),
            "fit_seconds": round(time.perf_counter() - start, 1),
        }

    @torch.no_grad()
    def score(
        self, loader: DataLoader[tuple[torch.Tensor, int]]
    ) -> tuple[npt.NDArray[np.int64], FloatArray, float]:
        """(labels, anomaly scores, mean milliseconds per image incl. feature extraction)."""
        if self.memory is None:
            raise RuntimeError("call fit() first")
        labels, scores, seconds, count = [], [], 0.0, 0
        for images, targets in loader:
            start = time.perf_counter()
            patches = self.features(images)  # (B, P, D)
            # One image at a time: (P x memory) distances stay ~16 MB instead of ~1 GB per batch.
            batch_scores = [torch.cdist(p, self.memory).min(dim=1).values.max() for p in patches]
            scores.append(torch.stack(batch_scores).numpy())
            seconds += time.perf_counter() - start
            count += len(images)
            labels.append(targets.numpy())
        return (
            np.concatenate(labels).astype(np.int64),
            np.concatenate(scores).astype(np.float32),
            1000 * seconds / count,
        )


RESULT_FILE = "anomaly_baseline.json"


def run_anomaly_baseline(config: ProjectConfig, out_dir: Path) -> dict[str, Any]:
    """Fit on normal TRAIN images, pick the threshold on VAL, evaluate ONCE on both test sets."""
    result_path = out_dir / RESULT_FILE
    if result_path.exists():
        raise RuntimeError(
            f"{result_path} exists: the anomaly baseline was already evaluated on test"
        )
    pre = config.preprocess
    spec = PreprocessSpec(pre.image_size, pre.mean, pre.std)
    splits_csv, raw = config.data.processed_dir / "splits.csv", config.data.raw_dir

    def loader(split: str, normal_only: bool = False) -> DataLoader[tuple[torch.Tensor, int]]:
        dataset = SplitDataset(splits_csv, raw, split, spec)
        if normal_only:
            keep = [i for i, y in enumerate(dataset.labels) if y == NORMAL_LABEL]
            return DataLoader(Subset(dataset, keep), batch_size=config.train.batch_size)
        return DataLoader(dataset, batch_size=config.train.batch_size)

    torch.manual_seed(config.seed)
    model = PatchCore(config.anomaly, config.seed, pretrained=config.model.pretrained)
    fit_info = model.fit(loader("train", normal_only=True))
    val_labels, val_scores, _ = model.score(loader("val"))
    threshold = threshold_for_recall(
        val_labels, val_scores.astype(np.float64), config.operating_point.min_defect_recall
    )
    results: dict[str, Any] = {
        "fit": fit_info,
        "threshold_policy": "chosen on val: best precision at defect recall >= "
        f"{config.operating_point.min_defect_recall}",
        "threshold": threshold,
        "val": classification_metrics(val_labels, val_scores.astype(np.float64), threshold),
    }
    for split in ("test", EXTERNAL_TEST):
        labels, scores, ms = model.score(loader(split))
        results[split] = {
            **classification_metrics(labels, scores.astype(np.float64), threshold),
            "ms_per_image_cpu": round(ms, 2),
            "n": len(labels),
        }
    result_path.write_text(
        json.dumps({**provenance(config), **results}, indent=2), encoding="utf-8"
    )
    return results
