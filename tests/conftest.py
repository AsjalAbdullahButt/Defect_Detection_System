"""Shared fixtures. All images are generated on the fly; no real data is committed."""

from pathlib import Path

import pytest

from tests.fixtures.synthetic import write_image


@pytest.fixture
def casting_like_dataset(tmp_path: Path) -> Path:
    """Tiny casting-shaped tree: train/test x def_front/ok_front, plus a corrupt file."""
    root = tmp_path / "raw"
    seed = 0
    for split, n_def, n_ok in [("train", 3, 2), ("test", 2, 1)]:
        for class_dir, n in [("def_front", n_def), ("ok_front", n_ok)]:
            for i in range(n):
                seed += 1
                write_image(root / split / class_dir / f"img_{i}.jpeg", size=(300, 300), seed=seed)
    good = write_image(root / "train" / "ok_front" / "truncated.jpeg", size=(300, 300), seed=99)
    good.write_bytes(good.read_bytes()[:400])
    (root / "notes.txt").write_text("not an image", encoding="utf-8")
    return root
