"""Group exact and near-duplicate images (same physical part) into clusters.

Near-duplicate signal: Pearson correlation between standardised grayscale images, maximised
over the 8 rotations/flips (dihedral group D4). It is computed in two passes so it stays fast:

1. coarse (32x32): every query variant against every target -> shortlist the k best targets;
2. fine (128x128): re-score only the shortlist, where small nicks and burrs become visible.

Two images are linked when byte-identical (same SHA-256) or when their fine correlation is
>= the threshold. Links are merged transitively with union-find, so A~B~C forms one cluster
even if A and C are not directly similar.

Why not pHash: in this dataset every image shows the same part, centred, on the same
background. A 64-bit pHash puts ~95% of images within 4 bits of another one, including
different parts with different labels, so it cannot tell copies from look-alikes (D-031).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image

UInt8Array = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

_CHUNK = 128  # query rows per block: keeps the fine pass at a few hundred MB


@dataclass(frozen=True)
class Thumbnails:
    """Grayscale thumbnails aligned row-for-row with the manifest."""

    coarse: UInt8Array  # (n, coarse, coarse)
    fine: UInt8Array  # (n, fine, fine)

    def subset(self, rows: npt.NDArray[np.intp]) -> "Thumbnails":
        """Thumbnails for the given manifest row positions."""
        return Thumbnails(self.coarse[rows], self.fine[rows])


def load_thumbnails(
    raw_dir: Path, rel_paths: pd.Series, coarse_size: int, fine_size: int
) -> Thumbnails:
    """Decode every image once and keep two small grayscale copies."""
    coarse = np.empty((len(rel_paths), coarse_size, coarse_size), dtype=np.uint8)
    fine = np.empty((len(rel_paths), fine_size, fine_size), dtype=np.uint8)
    for i, rel_path in enumerate(rel_paths):
        with Image.open(raw_dir / rel_path) as img:
            gray = img.convert("L")
        coarse[i] = np.asarray(gray.resize((coarse_size, coarse_size), Image.Resampling.BILINEAR))
        fine[i] = np.asarray(gray.resize((fine_size, fine_size), Image.Resampling.BILINEAR))
    return Thumbnails(coarse, fine)


def dihedral(images: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """(..., S, S) -> (..., 8, S, S): identity, 3 rotations, 4 reflections."""
    rot = [np.rot90(images, k, axes=(-2, -1)) for k in range(4)]
    flips = [np.swapaxes(r, -2, -1) for r in rot]  # transpose of each rotation = a reflection
    return np.stack(rot + flips, axis=-3)


def standardize(images: npt.NDArray[Any]) -> FloatArray:
    """Flatten the last two axes to zero-mean, unit-norm vectors: dot product == correlation."""
    flat = images.reshape(*images.shape[:-2], -1).astype(np.float32)
    flat -= flat.mean(axis=-1, keepdims=True)
    flat /= np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-6
    return flat


def top_matches(
    query: Thumbnails, target: Thumbnails, k: int, same_set: bool = False
) -> tuple[IntArray, FloatArray, npt.NDArray[np.bool_]]:
    """For each query, its k best targets: (target index, fine correlation, only-via-transform).

    ``same_set=True`` means query and target are the same images, so self-matches are skipped.
    ``only-via-transform`` is True when the identity orientation alone scored below the best
    rotated/flipped variant, i.e. the copy was rotated or flipped.
    """
    n_target = len(target.coarse) - int(same_set)
    k = min(k, n_target)
    target_coarse = standardize(target.coarse)
    n_query = len(query.coarse)
    index = np.zeros((n_query, k), dtype=np.int64)
    corr = np.zeros((n_query, k), dtype=np.float32)
    via_transform = np.zeros((n_query, k), dtype=bool)
    if k <= 0:
        return index, corr, via_transform
    for start in range(0, n_query, _CHUNK):
        stop = min(start + _CHUNK, n_query)
        q_coarse = standardize(dihedral(query.coarse[start:stop]))  # (b, 8, Dc)
        coarse = np.einsum("bvd,td->bvt", q_coarse, target_coarse).max(axis=1)
        if same_set:
            coarse[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        cand = np.argpartition(-coarse, k - 1, axis=1)[:, :k]
        q_fine = standardize(dihedral(query.fine[start:stop]))  # (b, 8, Df)
        t_fine = standardize(target.fine[cand])  # (b, k, Df)
        fine = np.einsum("bvd,bkd->bvk", q_fine, t_fine)  # (b, 8, k)
        index[start:stop] = cand
        corr[start:stop] = fine.max(axis=1)
        via_transform[start:stop] = fine[:, 0, :] < fine.max(axis=1)
    return index, corr, via_transform


def nearest_similarity(query: Thumbnails, target: Thumbnails, k: int) -> FloatArray:
    """Highest fine correlation of each query image to any target image (-1 if no targets)."""
    if len(target.coarse) == 0:
        return np.full(len(query.coarse), -1.0, dtype=np.float32)
    _, corr, _ = top_matches(query, target, k)
    best: FloatArray = corr.max(axis=1)
    return best


class UnionFind:
    """Disjoint-set forest with path halving."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """Root of x's set."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Merge the sets of a and b; the smaller root wins so results are order-independent."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_images(
    manifest: pd.DataFrame, thumbs: Thumbnails, threshold: float, k: int
) -> tuple[pd.Series, dict[str, Any]]:
    """Assign a ``cluster_id`` to every manifest row and summarise what was found.

    The manifest must be sorted by ``rel_path`` (positional index 0..n-1); cluster ids
    (``g00000`` ...) follow the position of each cluster's first member, so they are
    deterministic.
    """
    n = len(manifest)
    uf = UnionFind(n)
    for indices in manifest.groupby("sha256").indices.values():
        for other in indices[1:]:
            uf.union(int(indices[0]), int(other))
    index, corr, via = top_matches(thumbs, thumbs, k, same_set=True)
    rows, cols = np.nonzero(corr >= threshold)
    pair_corr = corr[rows, cols]
    pair_via = via[rows, cols]
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        uf.union(r, int(index[r, c]))

    roots = [uf.find(i) for i in range(n)]
    ordered_roots = {root: k_ for k_, root in enumerate(dict.fromkeys(roots))}
    cluster_id = pd.Series([f"g{ordered_roots[r]:05d}" for r in roots], index=manifest.index)

    sizes = cluster_id.value_counts()
    labels_per_cluster = manifest.groupby(cluster_id)["label"].nunique()
    exact_groups = manifest.groupby("sha256").size()
    nearest = corr.max(axis=1) if corr.size else np.zeros(n, dtype=np.float32)
    edges = [0.0, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0001]
    report: dict[str, Any] = {
        "method": "max rotation/flip-aligned correlation of standardised grayscale thumbnails",
        "threshold": threshold,
        "shortlist_k": k,
        "n_images": n,
        "n_clusters": int(sizes.size),
        "n_multi_image_clusters": int((sizes > 1).sum()),
        "images_in_multi_image_clusters": int(sizes[sizes > 1].sum()),
        "largest_cluster_size": int(sizes.max()) if n else 0,
        "cluster_size_histogram": {
            str(s): int(c) for s, c in sizes.value_counts().sort_index().items()
        },
        "exact_duplicate_groups": int((exact_groups > 1).sum()),
        "exact_duplicate_extra_copies": int((exact_groups - 1).sum()),
        "near_duplicate_links": len(rows),
        "near_duplicate_links_only_via_rotation_or_flip": int(pair_via.sum()),
        "nearest_neighbour_similarity_histogram": {
            f"[{lo}, {min(hi, 1.0)})": int(c)
            for lo, hi, c in zip(
                edges[:-1], edges[1:], np.histogram(nearest, edges)[0], strict=True
            )
        },
        "link_similarity_min": float(pair_corr.min()) if len(pair_corr) else None,
        "clusters_with_conflicting_labels": sorted(
            labels_per_cluster[labels_per_cluster > 1].index.tolist()
        ),
    }
    return cluster_id, report
