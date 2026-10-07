"""Generators for tiny synthetic test images (no real data is ever committed)."""

from pathlib import Path

import numpy as np
from PIL import Image


def write_image(
    path: Path, size: tuple[int, int] = (32, 32), fmt: str = "JPEG", seed: int = 0
) -> Path:
    """Write a small random RGB image and return its path."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format=fmt)
    return path
