from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from defect_detection.data.dedupe import (
    UnionFind,
    cluster_images,
    find_near_duplicates,
    parse_phashes,
)
from defect_detection.data.manifest import dihedral_phashes
from tests.fixtures.synthetic import save, smooth_image


def _d8(tmp_path: Path, img: Image.Image, name: str, fmt: str = "PNG") -> str:
    return " ".join(dihedral_phashes(save(img, tmp_path / name, fmt=fmt)))


def test_every_dihedral_transform_is_detected(tmp_path: Path) -> None:
    base = smooth_image(1)
    transforms = list(Image.Transpose)  # the 7 non-identity rotations/flips
    rows = [_d8(tmp_path, base, "base.png")]
    rows += [_d8(tmp_path, base.transpose(t), f"t{k}.png") for k, t in enumerate(transforms)]
    pairs = find_near_duplicates(parse_phashes(pd.Series(rows)), threshold=4)
    linked_to_base = {p.j for p in pairs if p.i == 0}
    assert linked_to_base == set(range(1, len(rows)))
    assert all(p.via_transform for p in pairs if p.i == 0)


def test_unrelated_images_are_not_linked(tmp_path: Path) -> None:
    rows = [_d8(tmp_path, smooth_image(s), f"{s}.png") for s in range(30)]
    assert find_near_duplicates(parse_phashes(pd.Series(rows)), threshold=4) == []


def test_jpeg_reencode_is_a_near_duplicate(tmp_path: Path) -> None:
    img = smooth_image(5)
    rows = [_d8(tmp_path, img, "a.png"), _d8(tmp_path, img, "a.jpeg", fmt="JPEG")]
    (pair,) = find_near_duplicates(parse_phashes(pd.Series(rows)), threshold=4)
    assert not pair.via_transform


def test_union_find_is_transitive_and_order_independent() -> None:
    uf = UnionFind(5)
    uf.union(3, 4)
    uf.union(4, 1)
    assert uf.find(3) == uf.find(1) == 1
    assert uf.find(0) == 0


def test_cluster_images_groups_exact_and_chained_duplicates() -> None:
    identity = ["0" * 16, "0" * 15 + "f", "0" * 14 + "ff", "f" * 16]  # 0~1 (4 bits), 1~2 (4 bits)
    manifest = pd.DataFrame(
        {
            "rel_path": ["a", "b", "c", "d", "e"],
            "sha256": ["s1", "s2", "s3", "s4", "s4"],
            "label": [0, 0, 0, 1, 1],
            "phash_d8": [" ".join([h] * 8) for h in identity] + [" ".join(["f" * 16] * 8)],
        }
    )
    cluster_id, report = cluster_images(manifest, threshold=4)
    assert cluster_id.tolist() == ["g00000", "g00000", "g00000", "g00001", "g00001"]
    assert report["exact_duplicate_groups"] == 1
    assert report["clusters_with_conflicting_labels"] == []


def test_parse_phashes_shape() -> None:
    arr = parse_phashes(pd.Series([" ".join(["ff"] * 8), " ".join(["01"] * 8)]))
    assert arr.shape == (2, 8)
    assert arr.dtype == np.uint64
