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

    @model_validator(mode="after")
    def _aliases_target_known_classes(self) -> "DataConfig":
        unknown = set(self.class_aliases.values()) - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"class_aliases map to unknown classes {sorted(unknown)}")
        return self


class DedupeConfig(_Strict):
    """Near-duplicate detection settings."""

    phash_hamming_threshold: int = Field(ge=0, le=16)


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


def load_config(path: Path) -> ProjectConfig:
    """Parse and validate a YAML config file."""
    with path.open(encoding="utf-8") as handle:
        return ProjectConfig.model_validate(yaml.safe_load(handle))


def config_hash(config: ProjectConfig) -> str:
    """Short, stable hash of the validated config (key order and formatting don't matter)."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return sha256_bytes(canonical.encode("utf-8"))[:12]
