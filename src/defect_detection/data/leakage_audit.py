"""Leakage audit: prove that no image, or near-copy of one, appears in more than one split.

The audit re-derives cross-split similarity from the pHashes instead of trusting the cluster
ids written by the split step, so a bug in clustering or splitting cannot hide leakage.
Checks (all must pass):

* every manifest image is either assigned to a split or excluded with a reason;
* no SHA-256 occurs in two splits;
* no cluster id occurs in two splits;
* no cross-split pair (test-vs-train/val, val-vs-train) is within the near-duplicate threshold,
  including rotated/flipped variants;
* every split contains both classes.

A looser "borderline" count (2x threshold) is reported for information only, to show how
sensitive the result is to the threshold choice.
"""

from typing import Any

import pandas as pd

from defect_detection.core.hashing import sha256_of_hashes
from defect_detection.data.dedupe import min_dihedral_distances, parse_phashes
from defect_detection.data.split import EXCLUDED, SPLITS, class_counts

_CHUNK = 64


def count_cross_pairs(
    query: pd.Series, target: pd.Series, threshold: int
) -> tuple[int, int | None]:
    """(number of query/target pairs within threshold, smallest distance seen)."""
    if query.empty or target.empty:
        return 0, None
    q, t = parse_phashes(query), parse_phashes(target)[:, 0]
    n_close, smallest = 0, 64
    for start in range(0, len(q), _CHUNK):
        best, _ = min_dihedral_distances(q[start : start + _CHUNK], t)
        n_close += int((best <= threshold).sum())
        smallest = min(smallest, int(best.min()))
    return n_close, smallest


def run_audit(manifest: pd.DataFrame, splits: pd.DataFrame, threshold: int) -> dict[str, Any]:
    """Run all checks; ``result["passed"]`` is True only if every check passed."""
    df = splits.merge(manifest[["rel_path", "phash_d8"]], on="rel_path", how="left")
    used = df[df["split"] != EXCLUDED]
    checks: dict[str, bool] = {}

    checks["all_manifest_rows_accounted_for"] = set(manifest["rel_path"]) == set(
        splits["rel_path"]
    ) and bool((splits.loc[splits["split"] == EXCLUDED, "exclude_reason"] != "").all())

    sha_splits = used.groupby("sha256")["split"].nunique()
    checks["no_sha256_in_two_splits"] = bool((sha_splits <= 1).all())

    cluster_splits = used.groupby("cluster_id")["split"].nunique()
    checks["no_cluster_in_two_splits"] = bool((cluster_splits <= 1).all())

    by_split = {s: used.loc[used["split"] == s, "phash_d8"] for s in SPLITS}
    cross_pairs: dict[str, dict[str, int | None]] = {}
    for a, b in [("test", "train"), ("test", "val"), ("val", "train")]:
        close, smallest = count_cross_pairs(by_split[a], by_split[b], threshold)
        borderline, _ = count_cross_pairs(by_split[a], by_split[b], 2 * threshold)
        cross_pairs[f"{a}_vs_{b}"] = {
            "pairs_within_threshold": close,
            "pairs_within_2x_threshold_info_only": borderline,
            "min_distance": smallest,
        }
    checks["no_near_duplicates_across_splits"] = all(
        v["pairs_within_threshold"] == 0 for v in cross_pairs.values()
    )

    counts = class_counts(used)
    checks["every_split_has_both_classes"] = all(n > 0 for c in counts.values() for n in c.values())

    test_hashes = used.loc[used["split"] == "test", "sha256"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "threshold": threshold,
        "cross_split_near_duplicates": cross_pairs,
        "counts": counts,
        "excluded": int((df["split"] == EXCLUDED).sum()),
        "test_set_sha256": sha256_of_hashes(test_hashes),
        "test_set_size": len(test_hashes),
    }
