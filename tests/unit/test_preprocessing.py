import io
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from defect_detection.config import AugmentConfig
from defect_detection.core.preprocessing import (
    PreprocessSpec,
    prepare_image,
    preprocess,
    to_float_array,
)
from defect_detection.data.augment import TrainTransform
from tests.conftest import make_config
from tests.fixtures.synthetic import smooth_image

SPEC = PreprocessSpec(image_size=64, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def test_output_shape_dtype_layout() -> None:
    out = preprocess(smooth_image(1, size=300), SPEC)
    assert out.shape == (3, 64, 64)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


def test_normalisation_uses_imagenet_constants() -> None:
    white = Image.new("RGB", (10, 10), (255, 255, 255))
    out = preprocess(white, SPEC)
    expected = (1 - np.array(SPEC.mean)) / np.array(SPEC.std)
    np.testing.assert_allclose(out[:, 0, 0], expected, rtol=1e-6)


def test_grayscale_and_rgba_become_three_channels() -> None:
    gray = to_float_array(prepare_image(Image.new("L", (20, 20), 128), SPEC))
    assert gray.shape == (64, 64, 3)
    assert np.all(gray[..., 0] == gray[..., 1])
    assert np.all(gray[..., 1] == gray[..., 2])
    rgba = preprocess(Image.new("RGBA", (20, 20), (10, 20, 30, 0)), SPEC)
    assert rgba.shape == (3, 64, 64)


def test_exif_orientation_is_applied() -> None:
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    img.paste((255, 255, 255), (32, 0, 64, 64))  # right half white
    exif = Image.Exif()
    exif[0x0112] = 3  # "rotate 180" orientation tag
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif, quality=95)
    upright = to_float_array(prepare_image(Image.open(io.BytesIO(buf.getvalue())), SPEC))
    assert upright[:, :16].mean() > 0.9  # after the 180 rotation the white half is on the left
    assert upright[:, -16:].mean() < 0.1


def test_preprocess_is_deterministic() -> None:
    img = smooth_image(3, size=128)
    assert np.array_equal(preprocess(img, SPEC), preprocess(img, SPEC))


def test_core_preprocessing_does_not_import_torch() -> None:
    code = "import sys, defect_detection.core.preprocessing; sys.exit('torch' in sys.modules)"
    # Fixed interpreter and code string: no untrusted input.
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603


def test_augmentation_is_reproducible_and_changes_pixels(tmp_path) -> None:
    config = make_config(tmp_path)
    transform = TrainTransform(SPEC, config.augment)
    img = smooth_image(4, size=128)
    torch.manual_seed(0)
    first = transform(img)
    torch.manual_seed(0)
    second = transform(img)
    assert first.shape == (3, 64, 64)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert not np.array_equal(first, preprocess(img, SPEC))


def test_disabled_augmentation_equals_core_preprocessing() -> None:
    """With every augmentation switched off, the train path reduces to core.preprocess."""
    off = AugmentConfig(
        rotation_degrees=0, hflip_p=0, vflip_p=0, brightness=0, contrast=0, blur_p=0,
        blur_sigma=(0.1, 1.0), noise_p=0, noise_std=0, translate=0, scale=(1.0, 1.0),
    )  # fmt: skip
    img = smooth_image(5, size=128)
    np.testing.assert_allclose(TrainTransform(SPEC, off)(img), preprocess(img, SPEC), atol=1e-6)
