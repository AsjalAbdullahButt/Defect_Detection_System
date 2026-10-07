from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from defect_detection.data.dedupe import (
    Thumbnails,
    UnionFind,
    cluster_images,
    dihedral,
    load_thumbnails,
    nearest_similarity,
    standardize,
    top_matches,
)
from tests.fixtures.synthetic import save, smooth_image


def thumbs_of(tmp_path: Path, images: list[Image.Image], fmt: str = "PNG") -> Thumbnails:
    paths = [
        save(img, tmp_path / f"{i}.{fmt.lower()}", fmt=fmt).name for i, img in enumerate(images)
    ]
    return load_thumbnails(tmp_path, pd.Series(paths), 32, 128)


def test_dihedral_gives_eight_distinct_variants() -> None:
    img = np.arange(16).reshape(4, 4)
    variants = dihedral(img)
    assert variants.shape == (8, 4, 4)
    assert len({v.tobytes() for v in variants}) == 8
    assert np.array_equal(variants[0], img)


def test_standardized_dot_is_correlation() -> None:
    a = np.random.default_rng(0).integers(0, 255, (16, 16))
    va, vb = standardize(a), standardize(a * 0.5 + 40)  # affine brightness change
    assert float(va @ vb) > 0.9999


def test_every_rotation_and_flip_is_matched(tmp_path: Path) -> None:
    base = smooth_image(1, size=96)
    images = [base] + [base.transpose(t) for t in Image.Transpose]
    thumbs = thumbs_of(tmp_path, images)
    index, corr, via = top_matches(thumbs.subset(np.array([0])), thumbs, k=8, same_set=False)
    others = {int(i) for i, c in zip(index[0], corr[0], strict=True) if c > 0.999 and i != 0}
    assert others == set(range(1, 8))
    assert via[0][index[0] != 0].all()  # every copy needed a transform to match


def test_unrelated_images_are_not_similar(tmp_path: Path) -> None:
    thumbs = thumbs_of(tmp_path, [smooth_image(s) for s in range(30)])
    _, corr, _ = top_matches(thumbs, thumbs, k=5, same_set=True)
    assert corr.max() < 0.9


def test_jpeg_reencode_is_near_duplicate(tmp_path: Path) -> None:
    img = smooth_image(5)
    png = thumbs_of(tmp_path / "a", [img])
    jpg = thumbs_of(tmp_path / "b", [img], fmt="JPEG")
    assert nearest_similarity(jpg, png, k=1)[0] > 0.995


def test_nearest_similarity_with_empty_target() -> None:
    empty = Thumbnails(np.zeros((0, 32, 32), np.uint8), np.zeros((0, 128, 128), np.uint8))
    one = Thumbnails(np.ones((1, 32, 32), np.uint8), np.ones((1, 128, 128), np.uint8))
    assert nearest_similarity(one, empty, k=3).tolist() == [-1.0]


def test_union_find_is_transitive_and_order_independent() -> None:
    uf = UnionFind(5)
    uf.union(3, 4)
    uf.union(4, 1)
    assert uf.find(3) == uf.find(1) == 1
    assert uf.find(0) == 0


def test_cluster_images_groups_exact_and_rotated_copies(tmp_path: Path) -> None:
    a, b = smooth_image(10), smooth_image(11)
    images = [a, a.transpose(Image.Transpose.ROTATE_180), b, smooth_image(12)]
    thumbs = thumbs_of(tmp_path, images)
    manifest = pd.DataFrame(
        {
            "rel_path": ["0", "1", "2", "3", "4"],
            "sha256": ["s0", "s1", "s2", "s3", "s3"],  # rows 3 and 4 are byte-identical
            "label": [0, 0, 1, 1, 1],
        }
    )
    thumbs5 = thumbs.subset(np.array([0, 1, 2, 3, 3]))
    cluster_id, report = cluster_images(manifest, thumbs5, threshold=0.99, k=4)
    assert cluster_id.tolist() == ["g00000", "g00000", "g00001", "g00002", "g00002"]
    assert report["exact_duplicate_groups"] == 1
    assert report["near_duplicate_links_only_via_rotation_or_flip"] >= 1
    assert report["clusters_with_conflicting_labels"] == []
