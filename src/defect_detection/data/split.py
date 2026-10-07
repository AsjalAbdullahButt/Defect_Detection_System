"""Stratified, group-aware split with leakage cleaning.

Rules, applied in order:

1. Exact duplicates (same SHA-256) collapse to one kept copy (the official-test copy if any).
   If the copies carry different labels the label is unknowable, so every copy is excluded.
2. Folders in ``data.external_test_dirs`` are held out of everything below.
3. ``official`` strategy (the dataset ships a test folder): the official test set stays the
   test set, untouched. Every other image whose OWN best similarity to some test image is
   >= the threshold is excluded as a same-part copy of a test image.
   ``random`` strategy: no official test; test is allocated like train/val in step 5.
4. The remaining pool is clustered (union-find over exact + near-duplicate links), so all
   copies of one part stay together.
5. Allocation is per label (stratified) and per cluster (group-aware): clusters are shuffled
   with the seed and assigned in order until each split's share of images is reached.
6. External images whose own best similarity to any train/val image is >= the threshold are
   excluded; the rest become ``external_test``.
7. Every test/external_test image records its similarity to the nearest final train image,
   so V4 can report metrics by how close an image is to something seen in training.

Why direct matches for cleaning but clusters inside the pool: chains of links (A~B~C...) on
look-alike normal parts can join *different* parts (random pairs inside the largest chains
are only ~0.89 similar). Excluding whole chains would discard distinct normal parts; grouping
them within the pool merely keeps the chain in one split, which is the safe direction.

Excluded images stay in splits.csv with ``split="excluded"`` and a reason, never silently dropped.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd

from defect_detection.config import DedupeConfig, SplitConfig
from defect_detection.core.constants import CLASS_NAMES
from defect_detection.data.dedupe import Thumbnails, cluster_images, nearest_similarity

EXCLUDED = "excluded"
EXTERNAL_TEST = "external_test"
SPLITS = ("train", "val", "test", EXTERNAL_TEST)
SPLITS_COLUMNS = [
    "rel_path",
    "sha256",
    "label",
    "class_name",
    "source_split",
    "cluster_id",
    "split",
    "exclude_reason",
    "nearest_train_similarity",
]


def resolve_strategy(strategy: str, manifest: pd.DataFrame) -> Literal["official", "random"]:
    """``auto`` becomes ``official`` when the raw data has a test folder, else ``random``."""
    has_official_test = bool((manifest["source_split"] == "test").any())
    if strategy == "official" and not has_official_test:
        raise ValueError("split.strategy=official but the raw data has no test folder")
    if strategy == "official" or (strategy == "auto" and has_official_test):
        return "official"
    return "random"


def is_external(rel_paths: pd.Series, external_dirs: tuple[str, ...]) -> pd.Series:
    """True for images whose top-level folder is listed as an external test source."""
    return rel_paths.str.split("/").str[0].isin(external_dirs)


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


def _rows(mask: pd.Series) -> np.ndarray:
    return np.flatnonzero(mask.to_numpy())


def _exclude_matches(
    df: pd.DataFrame,
    thumbs: Thumbnails,
    candidates: pd.Series,
    reference: pd.Series,
    dedupe: DedupeConfig,
    reason: str,
) -> None:
    """Exclude candidate rows whose own best similarity to any reference row >= threshold."""
    cand_rows, ref_rows = _rows(candidates), _rows(reference)
    if len(cand_rows) == 0 or len(ref_rows) == 0:
        return
    sims = nearest_similarity(thumbs.subset(cand_rows), thumbs.subset(ref_rows), dedupe.shortlist_k)
    hit = cand_rows[sims >= dedupe.similarity_threshold]
    df.loc[df.index[hit], "exclude_reason"] = reason


def make_splits(
    manifest: pd.DataFrame,
    thumbs: Thumbnails,
    config: SplitConfig,
    dedupe: DedupeConfig,
    seed: int,
    external_dirs: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (splits table, report). ``thumbs`` rows align with ``manifest`` rows."""
    df = manifest[["rel_path", "sha256", "label", "class_name", "source_split"]].reset_index(
        drop=True
    )
    df["exclude_reason"] = exact_duplicate_exclusions(df)
    df["split"] = ""
    df["cluster_id"] = [f"s{i:05d}" for i in range(len(df))]  # singleton ids outside the pool
    df["nearest_train_similarity"] = np.nan
    external = is_external(df["rel_path"], external_dirs)
    strategy = resolve_strategy(config.strategy, df[~external])

    kept = df["exclude_reason"] == ""
    if strategy == "official":
        is_test = kept & ~external & (df["source_split"] == "test")
        _exclude_matches(
            df, thumbs, kept & ~external & ~is_test, is_test, dedupe,
            "near_duplicate_of_official_test",
        )  # fmt: skip
        df.loc[is_test, "split"] = "test"
        pool = (df["exclude_reason"] == "") & ~external & ~is_test
        v = config.val_fraction_of_official_train
        fractions = {"train": 1 - v, "val": v}
    else:
        pool = kept & ~external
        fractions = {"train": config.train, "val": config.val, "test": config.test}

    pool_rows = _rows(pool)
    pool_ids, dedupe_report = cluster_images(
        df.iloc[pool_rows].reset_index(drop=True),
        thumbs.subset(pool_rows),
        dedupe.similarity_threshold,
        dedupe.shortlist_k,
    )
    df.loc[df.index[pool_rows], "cluster_id"] = pool_ids.to_numpy()
    pool_df = df[pool]
    sizes = pool_df.groupby("cluster_id").size()
    # Stratify each cluster by its majority label (ties count as defective).
    labels = (pool_df.groupby("cluster_id")["label"].mean() >= 0.5).astype(int)
    assignment = allocate_clusters(sizes, labels, fractions, seed)
    df.loc[pool, "split"] = df.loc[pool, "cluster_id"].map(assignment)

    ext = (df["exclude_reason"] == "") & external
    _exclude_matches(
        df, thumbs, ext, df["split"].isin(["train", "val"]), dedupe,
        "near_duplicate_of_train_or_val",
    )  # fmt: skip
    df.loc[ext & (df["exclude_reason"] == ""), "split"] = EXTERNAL_TEST
    df.loc[df["exclude_reason"] != "", "split"] = EXCLUDED

    train_rows = _rows(df["split"] == "train")
    for name in ("test", EXTERNAL_TEST):
        rows = _rows(df["split"] == name)
        if len(rows) and len(train_rows):
            sims = nearest_similarity(
                thumbs.subset(rows), thumbs.subset(train_rows), dedupe.shortlist_k
            )
            df.loc[df.index[rows], "nearest_train_similarity"] = sims.astype(float)

    used = df[df["split"] != EXCLUDED]
    counts = class_counts(used)
    totals = {s: sum(counts[s].values()) for s in SPLITS}
    report: dict[str, Any] = {
        "strategy": strategy,
        "seed": seed,
        "counts": counts,
        "fraction_of_used_images": {s: round(totals[s] / max(len(used), 1), 4) for s in SPLITS},
        "defect_rate": {s: round(counts[s]["defective"] / max(totals[s], 1), 4) for s in SPLITS},
        "excluded": {
            str(k): int(v)
            for k, v in df.loc[df["exclude_reason"] != "", "exclude_reason"].value_counts().items()
        },
        "pool_clustering": dedupe_report,
    }
    return df[SPLITS_COLUMNS], report
