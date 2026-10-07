"""Generators for tiny synthetic test images (no real data is ever committed)."""

from pathlib import Path

import numpy as np
from PIL import Image


def write_image(
    path: Path, size: tuple[int, int] = (32, 32), fmt: str = "JPEG", seed: int = 0
) -> Path:
    """Write a small random-noise RGB image and return its path."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format=fmt)
    return path


def smooth_image(seed: int, size: int = 64) -> Image.Image:
    """Random low-frequency RGB image: an 8x8 random grid upsampled bicubically.

    Unlike pure noise, its pHash is stable under re-encoding, which mimics real photos;
    two different seeds give pHashes ~32 bits apart.
    """
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return Image.fromarray(grid).resize((size, size), Image.Resampling.BICUBIC)


def save(img: Image.Image, path: Path, fmt: str = "PNG", **kwargs: int) -> Path:
    """Save ``img`` to ``path`` (creating parents) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt, **kwargs)
    return path


def build_planted_dataset(root: Path, with_official_test: bool = True) -> dict[str, str]:
    """Casting-shaped dataset with known duplicates planted in it.

    Returns a mapping from case name to the rel_path(s) involved, for assertions:

    * ``exact_copy``: byte-identical copy of a train image inside train
    * ``rotated_test_leak``: test image rotated 90 degrees and placed in train
    * ``jpeg_near_dup``: JPEG re-encode of a train PNG
    * ``label_conflict``: identical bytes in both class folders
    * ``corrupt``: truncated file
    """
    train = "train" if with_official_test else "all"
    test = "test" if with_official_test else "all"
    for i in range(20):
        save(smooth_image(100 + i), root / train / "def_front" / f"d{i:02d}.png")
        save(smooth_image(200 + i), root / train / "ok_front" / f"o{i:02d}.png")
    for i in range(6):
        save(smooth_image(300 + i), root / test / "def_front" / f"td{i:02d}.png")
        save(smooth_image(400 + i), root / test / "ok_front" / f"to{i:02d}.png")

    exact = root / train / "def_front" / "d00_copy.png"
    exact.write_bytes((root / train / "def_front" / "d00.png").read_bytes())
    save(
        smooth_image(400).transpose(Image.Transpose.ROTATE_90),
        root / train / "ok_front" / "leak_rot.png",
    )
    save(
        smooth_image(201), root / train / "ok_front" / "o01_reencoded.jpeg", fmt="JPEG", quality=85
    )
    conflict = smooth_image(999)
    save(conflict, root / train / "def_front" / "conflict.png")
    save(conflict, root / train / "ok_front" / "conflict.png")
    corrupt = save(smooth_image(998), root / train / "ok_front" / "broken.png")
    corrupt.write_bytes(corrupt.read_bytes()[:60])

    return {
        "exact_copy": f"{train}/def_front/d00_copy.png",
        "exact_original": f"{train}/def_front/d00.png",
        "rotated_test_leak": f"{train}/ok_front/leak_rot.png",
        "rotated_test_source": f"{test}/ok_front/to00.png",
        "jpeg_near_dup": f"{train}/ok_front/o01_reencoded.jpeg",
        "jpeg_source": f"{train}/ok_front/o01.png",
        "label_conflict_def": f"{train}/def_front/conflict.png",
        "label_conflict_ok": f"{train}/ok_front/conflict.png",
        "corrupt": f"{train}/ok_front/broken.png",
    }
