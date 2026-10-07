"""Group exact and near-duplicate images into clusters that must never be split apart.

Two images are linked when they are byte-identical (same SHA-256) or when the minimum Hamming
distance between one image's 8 rotation/flip pHashes and the other's identity pHash is at most
the threshold. Links are merged transitively with union-find, so a chain A~B~C becomes one
cluster even if A and C are not directly similar.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

_ROW_CHUNK = 64  # rows per vectorised block: 64 x 8 x n uint64 stays a few tens of MB


@dataclass(frozen=True)
class NearDuplicatePair:
    """A link between two manifest rows (i < j)."""

    i: int
    j: int
    distance: int
    via_transform: bool  # True if only a rotated/flipped variant was within the threshold


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


def parse_phashes(phash_d8: pd.Series) -> npt.NDArray[np.uint64]:
    """Turn the space-separated hex column into an (n, 8) uint64 array."""
    return np.array(
        [[int(h, 16) for h in row.split()] for row in phash_d8], dtype=np.uint64
    ).reshape(len(phash_d8), -1)


def min_dihedral_distances(
    query_d8: npt.NDArray[np.uint64], target_identity: npt.NDArray[np.uint64]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Distances between each query's 8 variants and each target's identity hash.

    Returns ``(min over variants, identity-only)``, both shaped (n_query, n_target).
    """
    xor = query_d8[:, :, None] ^ target_identity[None, None, :]
    dist = np.bitwise_count(xor)
    return dist.min(axis=1), dist[:, 0, :]


def find_near_duplicates(
    phash_d8: npt.NDArray[np.uint64], threshold: int
) -> list[NearDuplicatePair]:
    """All pairs (i < j) whose rotation/flip-aware pHash distance is <= threshold."""
    identity = phash_d8[:, 0]
    pairs: list[NearDuplicatePair] = []
    for start in range(0, len(phash_d8), _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, len(phash_d8))
        best, plain = min_dihedral_distances(phash_d8[start:stop], identity)
        rows, cols = np.nonzero(best <= threshold)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            i = start + r
            if c > i:  # each unordered pair once, no self-pairs
                pairs.append(
                    NearDuplicatePair(i, c, int(best[r, c]), bool(plain[r, c] > threshold))
                )
    return pairs


def cluster_images(manifest: pd.DataFrame, threshold: int) -> tuple[pd.Series, dict[str, object]]:
    """Assign a ``cluster_id`` to every manifest row and summarise what was found.

    The manifest must be sorted by ``rel_path``; cluster ids (``g00000`` ...) follow the
    position of each cluster's first member, so they are deterministic.
    """
    n = len(manifest)
    uf = UnionFind(n)
    for indices in manifest.groupby("sha256").indices.values():
        for other in indices[1:]:
            uf.union(int(indices[0]), int(other))
    near = find_near_duplicates(parse_phashes(manifest["phash_d8"]), threshold)
    for pair in near:
        uf.union(pair.i, pair.j)

    roots = [uf.find(i) for i in range(n)]
    ordered_roots = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    cluster_id = pd.Series([f"g{ordered_roots[r]:05d}" for r in roots], index=manifest.index)

    sizes = cluster_id.value_counts()
    labels_per_cluster = manifest.groupby(cluster_id)["label"].nunique()
    exact_groups = manifest.groupby("sha256").size()
    report: dict[str, object] = {
        "threshold": threshold,
        "n_images": n,
        "n_clusters": int(sizes.size),
        "n_multi_image_clusters": int((sizes > 1).sum()),
        "images_in_multi_image_clusters": int(sizes[sizes > 1].sum()),
        "largest_cluster_size": int(sizes.max()) if n else 0,
        "cluster_size_histogram": {
            str(k): int(v) for k, v in sizes.value_counts().sort_index().items()
        },
        "exact_duplicate_groups": int((exact_groups > 1).sum()),
        "exact_duplicate_extra_copies": int((exact_groups - 1).sum()),
        "near_duplicate_pairs": len(near),
        "near_duplicate_pairs_only_via_rotation_or_flip": sum(p.via_transform for p in near),
        "near_duplicate_distance_histogram": {
            str(d): sum(p.distance == d for p in near) for d in range(threshold + 1)
        },
        "clusters_with_conflicting_labels": sorted(
            labels_per_cluster[labels_per_cluster > 1].index.tolist()
        ),
    }
    return cluster_id, report
