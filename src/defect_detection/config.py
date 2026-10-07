"""Typed, validated view of ``configs/train.yaml``.

Unknown keys are rejected (``extra="forbid"``) so a typo in the YAML fails loudly instead of
silently falling back to a default.
"""

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from defect_detection.core.constants import CLASS_NAMES
from defect_detection.core.hashing import sha256_bytes


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    """Where data lives and how raw class folders map to class names."""

    raw_dir: Path
    processed_dir: Path
    class_aliases: dict[str, str]
    # Top-level folders (relative to raw_dir) held out entirely as a second, external test set.
    external_test_dirs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _aliases_target_known_classes(self) -> "DataConfig":
        unknown = set(self.class_aliases.values()) - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"class_aliases map to unknown classes {sorted(unknown)}")
        return self


class DedupeConfig(_Strict):
    """Near-duplicate detection: rotation/flip-aligned correlation of grayscale thumbnails."""

    similarity_threshold: float = Field(gt=0, le=1)
    info_threshold: float = Field(gt=0, le=1)
    coarse_size: int = Field(ge=8, le=64)
    fine_size: int = Field(ge=32, le=512)
    shortlist_k: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def _info_is_looser(self) -> "DedupeConfig":
        if self.info_threshold > self.similarity_threshold:
            raise ValueError("info_threshold must be <= similarity_threshold")
        return self


class SplitConfig(_Strict):
    """How images are assigned to train/val/test."""

    strategy: Literal["auto", "official", "random"]
    train: float = Field(gt=0, lt=1)
    val: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)
    val_fraction_of_official_train: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "SplitConfig":
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("train + val + test ratios must sum to 1")
        return self


class PreprocessConfig(_Strict):
    """Deterministic preprocessing shared by training and serving."""

    image_size: int = Field(ge=32, le=1024)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class AugmentConfig(_Strict):
    """Train-only augmentations (mild: defects are small, so nothing may crop them away)."""

    rotation_degrees: float = Field(ge=0, le=45)
    hflip_p: float = Field(ge=0, le=1)
    vflip_p: float = Field(ge=0, le=1)
    brightness: float = Field(ge=0, le=1)
    contrast: float = Field(ge=0, le=1)
    blur_p: float = Field(ge=0, le=1)
    blur_sigma: tuple[float, float]
    noise_p: float = Field(ge=0, le=1)
    noise_std: float = Field(ge=0, le=0.2)
    translate: float = Field(ge=0, le=0.2)
    scale: tuple[float, float]


class ModelConfig(_Strict):
    """timm backbone selection."""

    backbone: str
    pretrained: bool
    drop_rate: float = Field(ge=0, lt=1)


class TrainConfig(_Strict):
    """Two-stage fine-tuning schedule."""

    output_dir: Path
    device: Literal["auto", "cpu", "cuda"]
    batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    head_epochs: int = Field(ge=0)
    head_lr: float = Field(gt=0)
    finetune_epochs: int = Field(ge=1)
    finetune_lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    early_stopping_patience: int = Field(ge=1)
    class_weighting: Literal["balanced", "none"]


class CostConfig(_Strict):
    """Relative per-image costs used only to illustrate trade-offs in reports."""

    missed_defect: float = Field(ge=0)
    false_alarm: float = Field(ge=0)
    review: float = Field(ge=0)


class OperatingPointConfig(_Strict):
    """Threshold policy, applied to calibrated validation probabilities."""

    ece_bins: int = Field(ge=5, le=50)
    min_defect_recall: float = Field(gt=0, le=1)
    review_low_recall: float = Field(gt=0, le=1)
    review_high_precision: float = Field(gt=0, le=1)
    costs: CostConfig

    @model_validator(mode="after")
    def _band_recall_not_looser(self) -> "OperatingPointConfig":
        if self.review_low_recall < self.min_defect_recall:
            raise ValueError("review_low_recall must be >= min_defect_recall")
        return self


class ProjectConfig(_Strict):
    """Root of configs/train.yaml."""

    seed: int
    data: DataConfig
    dedupe: DedupeConfig
    split: SplitConfig
    preprocess: PreprocessConfig
    augment: AugmentConfig
    model: ModelConfig
    train: TrainConfig
    operating_point: OperatingPointConfig


def load_config(path: Path) -> ProjectConfig:
    """Parse and validate a YAML config file."""
    with path.open(encoding="utf-8") as handle:
        return ProjectConfig.model_validate(yaml.safe_load(handle))


def config_hash(config: ProjectConfig) -> str:
    """Short, stable hash of the validated config (key order and formatting don't matter)."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return sha256_bytes(canonical.encode("utf-8"))[:12]
