"""Train-only augmentations wrapped around the shared core preprocessing.

Order: ``prepare_image`` (core) -> geometric + photometric PIL augmentations -> ``to_float_array``
(core) -> optional Gaussian noise -> ``normalize`` (core). Only the middle steps are
training-specific, so with augmentation off the output equals ``core.preprocess`` exactly.

Randomness comes from torch's RNG (seeded per DataLoader worker), so runs are reproducible.
"""

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from defect_detection.config import AugmentConfig
from defect_detection.core.preprocessing import (
    FloatArray,
    PreprocessSpec,
    normalize,
    prepare_image,
    to_float_array,
)


class TrainTransform:
    """Callable PIL image -> augmented, normalised CHW float32 array."""

    def __init__(self, spec: PreprocessSpec, cfg: AugmentConfig) -> None:
        self.spec = spec
        self.cfg = cfg
        self.pil_ops = v2.Compose(
            [
                v2.RandomHorizontalFlip(cfg.hflip_p),
                v2.RandomVerticalFlip(cfg.vflip_p),
                # Rotation and the small shift/scale share one resampling step.
                v2.RandomAffine(
                    degrees=cfg.rotation_degrees,
                    translate=(cfg.translate, cfg.translate),
                    scale=cfg.scale,
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
                v2.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=5, sigma=cfg.blur_sigma)], p=cfg.blur_p
                ),
            ]
        )

    def __call__(self, img: Image.Image) -> FloatArray:
        """Augment one image."""
        augmented: Image.Image = self.pil_ops(prepare_image(img, self.spec))
        array = to_float_array(augmented)
        if torch.rand(1).item() < self.cfg.noise_p:
            noise = torch.randn(array.shape).numpy().astype(np.float32) * self.cfg.noise_std
            array = np.clip(array + noise, 0.0, 1.0, dtype=np.float32)
        return normalize(array, self.spec)
