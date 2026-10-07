"""Stratified, group-aware train/val/test split.

Rules, applied in order:

1. Exact duplicates (same SHA-256) collapse to one kept copy. If the copies carry different
   labels the label is unknowable, so every copy is excluded.
2. ``official`` strategy (dataset ships a test folder): the official test set stays the test
   set. Any non-test image in the same duplicate cluster as a test image is excluded, so
   comparability with the published split is kept and leakage is removed. Validation is carved
   from the remaining official-train clusters.
   ``random`` strategy: all clusters are allocated to train/val/test.
3. Allocation is per label (stratified) and per cluster (group-aware): clusters are shuffled
   with the seed and assigned in order until each split's share of images is reached, so a
   cluster never straddles two splits.

Excluded images stay in splits.csv with ``split="excluded"`` and a reason, never silently dropped.
"""

from typing import Literal

import numpy as np
import pandas as pd

from defect_detection.config import SplitConfig
from defect_detection.core.constants import CLASS_NAMES

EXCLUDED = "excluded"
SPLITS = ("train", "val", "test")
SPLITS_COLUMNS = [
    "rel_path",
    "sha256",
    "label",
    "class_name",
    "source_split",
    "cluster_id",
    "split",
    "exclude_reason",
]


def resolve_strategy(strategy: str, manifest: pd.DataFrame) -> Literal["official", "random"]:
    """``auto`` becomes ``official`` when the raw data has a test folder, else ``random``."""
    has_official_test = bool((manifest["source_split"] == "test").any())
    if strategy == "official" and not has_official_test:
        raise ValueError("split.strategy=official but the raw data has no test folder")
    if strategy == "official" or (strategy == "auto" and has_official_test):
        return "official"
    return "random"


def exact_duplicate_exclusions(manifest: pd.DataFrame) -> pd.Series:
    """Exclusion reason per row ('' = keep) after collapsing byte-identical files.

    The kept copy is the one in the official test folder if any (protects the test set),
    otherwise the first by path.
    """
    reason = pd.Series("", index=manifest.index)
    for _, group in manifest.groupby("sha256"):
        if len(group) == 1:
            continue
        if group["label"].nunique() > 1:
            reason[group.index] = "exact_duplicate_label_conflict"
            continue
        order = group.assign(_not_test=group["source_split"] != "test").sort_values(
            ["_not_test", "rel_path"]
        )
        reason[order.index[1:]] = "exact_duplicate"
    return reason


def allocate_clusters(
    sizes: pd.Series, labels: pd.Series, fractions: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign whole clusters to splits, per label, hitting ``fractions`` of images.

    ``sizes`` and ``labels`` are indexed by cluster id. A cluster goes to the split whose
    cumulative share covers the number of images already allocated before it.
    """
    rng = np.random.default_rng(seed)
    names = list(fractions)
    bounds = np.cumsum([fractions[n] for n in names])[:-1]
    assignment: dict[str, str] = {}
    for label in sorted(labels.unique()):
        ids = sorted(labels.index[labels == label])
        shuffled = [ids[k] for k in rng.permutation(len(ids))]
        counts = sizes[shuffled].to_numpy()
        share_before = (np.cumsum(counts) - counts) / counts.sum()
        for cid, share in zip(shuffled, share_before, strict=True):
            assignment[cid] = names[int(np.searchsorted(bounds, share, side="right"))]
    return assignment


def class_counts(assigned: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Images per split and class name, with zeros for missing combinations."""
    table = pd.crosstab(assigned["split"], assigned["class_name"]).to_dict(orient="index")
    return {s: {c: int(table.get(s, {}).get(c, 0)) for c in CLASS_NAMES} for s in SPLITS}


def make_splits(
    manifest: pd.DataFrame, cluster_id: pd.Series, config: SplitConfig, seed: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return (splits table, report). See module docstring for the rules."""
    df = manifest[["rel_path", "sha256", "label", "class_name", "source_split"]].copy()
    df["cluster_id"] = cluster_id
    df["exclude_reason"] = exact_duplicate_exclusions(manifest)
    df["split"] = ""
    strategy = resolve_strategy(config.strategy, manifest)

    kept = df["exclude_reason"] == ""
    if strategy == "official":
        is_test = kept & (df["source_split"] == "test")
        in_test_cluster = df["cluster_id"].isin(df.loc[is_test, "cluster_id"])
        df.loc[kept & in_test_cluster & ~is_test, "exclude_reason"] = (
            "near_duplicate_of_official_test"
        )
        df.loc[is_test, "split"] = "test"
        pool = kept & ~in_test_cluster
        v = config.val_fraction_of_official_train
        fractions = {"train": 1 - v, "val": v}
    else:
        pool = kept
        fractions = {"train": config.train, "val": config.val, "test": config.test}

    pool_df = df[pool]
    sizes = pool_df.groupby("cluster_id").size()
    # Stratify each cluster by its majority label (ties count as defective).
    labels = (pool_df.groupby("cluster_id")["label"].mean() >= 0.5).astype(int)
    assignment = allocate_clusters(sizes, labels, fractions, seed)
    df.loc[pool, "split"] = df.loc[pool, "cluster_id"].map(assignment)
    df.loc[df["exclude_reason"] != "", "split"] = EXCLUDED

    used = df[df["split"] != EXCLUDED]
    counts = class_counts(used)
    totals = {s: sum(counts[s].values()) for s in SPLITS}
    report: dict[str, object] = {
        "strategy": strategy,
        "seed": seed,
        "counts": counts,
        "fraction_of_used_images": {s: round(totals[s] / max(len(used), 1), 4) for s in SPLITS},
        "defect_rate": {s: round(counts[s]["defective"] / max(totals[s], 1), 4) for s in SPLITS},
        "excluded": {
            str(k): int(v)
            for k, v in df.loc[~df["exclude_reason"].eq(""), "exclude_reason"]
            .value_counts()
            .items()
        },
    }
    return df[SPLITS_COLUMNS], report
