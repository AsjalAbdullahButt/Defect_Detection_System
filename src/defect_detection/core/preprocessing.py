"""Single source of truth for image preprocessing (pure PIL + NumPy, no torch).

Training (without augmentation) and serving both call :func:`preprocess`, so the model sees
byte-identical inputs in both places; tests/leakage/test_preprocessing_parity.py proves it.
The pipeline is split into steps so training can insert augmentations between them:

    prepare_image (EXIF orientation, RGB, resize) -> to_float_array ([0, 1] HWC) -> normalize (CHW)
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

FloatArray = npt.NDArray[np.float32]

# Fixed so training and serving can't drift; bilinear is what timm's ImageNet weights expect.
RESAMPLE = Image.Resampling.BILINEAR


@dataclass(frozen=True)
class PreprocessSpec:
    """Everything needed to turn an image into model input; persisted in model_meta.json."""

    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def prepare_image(img: Image.Image, spec: PreprocessSpec) -> Image.Image:
    """Apply EXIF orientation, convert to 3-channel RGB and resize to a square.

    Direct resize (no crop) keeps the whole part in view; a centre crop could cut off a
    defect on the rim.
    """
    upright = ImageOps.exif_transpose(img)
    rgb = upright.convert("RGB")
    return rgb.resize((spec.image_size, spec.image_size), RESAMPLE)


def to_float_array(img: Image.Image) -> FloatArray:
    """HWC float32 array scaled to [0, 1]."""
    return np.asarray(img, dtype=np.float32) / np.float32(255.0)


def normalize(array: FloatArray, spec: PreprocessSpec) -> FloatArray:
    """Channel-wise ``(x - mean) / std``, then HWC -> CHW (the layout the network expects)."""
    mean = np.asarray(spec.mean, dtype=np.float32)
    std = np.asarray(spec.std, dtype=np.float32)
    return np.ascontiguousarray(((array - mean) / std).transpose(2, 0, 1))


def preprocess(img: Image.Image, spec: PreprocessSpec) -> FloatArray:
    """Full deterministic pipeline: PIL image -> normalised CHW float32 array."""
    return normalize(to_float_array(prepare_image(img, spec)), spec)
