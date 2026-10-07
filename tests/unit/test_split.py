import numpy as np
import pandas as pd
import pytest

from defect_detection.config import SplitConfig
from defect_detection.data.split import (
    EXCLUDED,
    allocate_clusters,
    exact_duplicate_exclusions,
    make_splits,
    resolve_strategy,
)

RANDOM_CFG = SplitConfig(
    strategy="random", train=0.7, val=0.15, test=0.15, val_fraction_of_official_train=0.15
)


def synthetic_manifest(n_clusters: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Manifest-like table: clusters of 1-3 images, ~35% defective, unique sha per image."""
    rng = np.random.default_rng(seed)
    rows, clusters = [], []
    for c in range(n_clusters):
        label = int(rng.random() < 0.35)
        for k in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "rel_path": f"img_{c:04d}_{k}.png",
                    "sha256": f"sha_{c}_{k}",
                    "label": label,
                    "class_name": ["normal", "defective"][label],
                    "source_split": "(none)",
                }
            )
            clusters.append(f"g{c:05d}")
    return pd.DataFrame(rows), pd.Series(clusters)


def test_split_is_deterministic_for_a_seed() -> None:
    manifest, clusters = synthetic_manifest()
    a, _ = make_splits(manifest, clusters, RANDOM_CFG, seed=42)
    b, _ = make_splits(manifest, clusters, RANDOM_CFG, seed=42)
    c, _ = make_splits(manifest, clusters, RANDOM_CFG, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])


def test_split_ratios_and_stratification() -> None:
    manifest, clusters = synthetic_manifest()
    _, report = make_splits(manifest, clusters, RANDOM_CFG, seed=42)
    fractions = report["fraction_of_used_images"]
    for split, target in [("train", 0.7), ("val", 0.15), ("test", 0.15)]:
        assert fractions[split] == pytest.approx(target, abs=0.02)
    rates = report["defect_rate"].values()
    assert max(rates) - min(rates) < 0.04


def test_no_cluster_straddles_splits() -> None:
    manifest, clusters = synthetic_manifest()
    splits, _ = make_splits(manifest, clusters, RANDOM_CFG, seed=42)
    assert (splits.groupby("cluster_id")["split"].nunique() == 1).all()


def test_allocate_clusters_respects_fraction_boundaries() -> None:
    sizes = pd.Series(1, index=[f"c{i}" for i in range(100)])
    labels = pd.Series(0, index=sizes.index)
    assignment = allocate_clusters(sizes, labels, {"train": 0.8, "val": 0.2}, seed=1)
    assert pd.Series(assignment).value_counts().to_dict() == {"train": 80, "val": 20}


def test_exact_duplicates_keep_test_copy_and_drop_conflicts() -> None:
    manifest = pd.DataFrame(
        {
            "rel_path": ["a_train.png", "b_test.png", "c.png", "d.png"],
            "sha256": ["s1", "s1", "s2", "s2"],
            "label": [1, 1, 0, 1],
            "source_split": ["train", "test", "train", "train"],
        }
    )
    reasons = exact_duplicate_exclusions(manifest).tolist()
    assert reasons == [
        "exact_duplicate",
        "",
        "exact_duplicate_label_conflict",
        "exact_duplicate_label_conflict",
    ]


def test_resolve_strategy() -> None:
    with_test = pd.DataFrame({"source_split": ["train", "test"]})
    without = pd.DataFrame({"source_split": ["(none)"]})
    assert resolve_strategy("auto", with_test) == "official"
    assert resolve_strategy("auto", without) == "random"
    assert resolve_strategy("random", with_test) == "random"
    with pytest.raises(ValueError, match="no test folder"):
        resolve_strategy("official", without)


def test_excluded_rows_always_have_a_reason() -> None:
    manifest, clusters = synthetic_manifest(50)
    manifest.loc[1, "sha256"] = manifest.loc[0, "sha256"]
    splits, _ = make_splits(manifest, clusters, RANDOM_CFG, seed=42)
    excluded = splits[splits["split"] == EXCLUDED]
    assert len(excluded) >= 1
    assert (excluded["exclude_reason"] != "").all()
