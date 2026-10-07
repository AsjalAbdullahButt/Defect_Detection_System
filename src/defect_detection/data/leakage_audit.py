"""Leakage audit: prove that no image, or same-part copy of one, appears in two splits.

The audit re-derives cross-split similarity from image content instead of trusting the
cluster ids written by the split step, so a bug in clustering or splitting cannot hide
leakage. Checks (all must pass):

* every manifest image is either assigned to a split or excluded with a reason;
* no SHA-256 occurs in two splits;
* no cluster id occurs in two splits;
* no evaluation image (val, test, external_test) has a training-side match (train, and val for
  the test sets) at or above the near-duplicate threshold, including rotated/flipped copies;
* train, val and test contain both classes (external_test too, when present).

Counts above a looser ``info_threshold`` are reported for information only: they show how
much residual similarity remains just below the cut-off.
"""

from typing import Any

import numpy as np
import pandas as pd

from defect_detection.core.hashing import sha256_of_hashes
from defect_detection.data.dedupe import Thumbnails, nearest_similarity
from defect_detection.data.split import EXCLUDED, EXTERNAL_TEST, class_counts

# (evaluation split, split it must not resemble)
CROSS_CHECKS = [
    ("val", "train"),
    ("test", "train"),
    ("test", "val"),
    (EXTERNAL_TEST, "train"),
    (EXTERNAL_TEST, "val"),
]


def run_audit(
    manifest: pd.DataFrame,
    splits: pd.DataFrame,
    thumbs: Thumbnails,
    threshold: float,
    info_threshold: float,
    k: int,
) -> dict[str, Any]:
    """Run all checks; ``result["passed"]`` is True only if every check passed.

    ``thumbs`` rows must align with ``manifest`` rows.
    """
    position = pd.Series(np.arange(len(manifest)), index=manifest["rel_path"])
    used = splits[splits["split"] != EXCLUDED]
    checks: dict[str, bool] = {}

    checks["all_manifest_rows_accounted_for"] = set(manifest["rel_path"]) == set(
        splits["rel_path"]
    ) and bool((splits.loc[splits["split"] == EXCLUDED, "exclude_reason"] != "").all())
    checks["no_sha256_in_two_splits"] = bool((used.groupby("sha256")["split"].nunique() <= 1).all())
    checks["no_cluster_in_two_splits"] = bool(
        (used.groupby("cluster_id")["split"].nunique() <= 1).all()
    )

    def rows_of(split: str) -> np.ndarray:
        return position.loc[used.loc[used["split"] == split, "rel_path"]].to_numpy()

    cross: dict[str, dict[str, Any]] = {}
    for query, target in CROSS_CHECKS:
        q_rows, t_rows = rows_of(query), rows_of(target)
        if len(q_rows) == 0 or len(t_rows) == 0:
            continue
        sims = nearest_similarity(thumbs.subset(q_rows), thumbs.subset(t_rows), k)
        cross[f"{query}_vs_{target}"] = {
            "n_query": len(q_rows),
            "pairs_at_or_above_threshold": int((sims >= threshold).sum()),
            "at_or_above_info_threshold": int((sims >= info_threshold).sum()),
            "max_similarity": round(float(sims.max()), 5),
            "median_nearest_similarity": round(float(np.median(sims)), 5),
        }
    checks["no_near_duplicates_across_splits"] = all(
        v["pairs_at_or_above_threshold"] == 0 for v in cross.values()
    )

    counts = class_counts(used)
    has_external = sum(counts[EXTERNAL_TEST].values()) > 0
    required = ["train", "val", "test"] + ([EXTERNAL_TEST] if has_external else [])
    checks["every_split_has_both_classes"] = all(min(counts[s].values()) > 0 for s in required)

    test_hashes = used.loc[used["split"] == "test", "sha256"]
    external_hashes = used.loc[used["split"] == EXTERNAL_TEST, "sha256"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "threshold": threshold,
        "info_threshold": info_threshold,
        "cross_split_similarity": cross,
        "counts": counts,
        "excluded": int((splits["split"] == EXCLUDED).sum()),
        "test_set_sha256": sha256_of_hashes(test_hashes),
        "test_set_size": len(test_hashes),
        "external_test_set_sha256": sha256_of_hashes(external_hashes),
        "external_test_set_size": len(external_hashes),
    }
