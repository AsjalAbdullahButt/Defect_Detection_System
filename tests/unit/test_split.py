import numpy as np
import pandas as pd
import pytest
from PIL import Image

from defect_detection.config import DedupeConfig, SplitConfig
from defect_detection.data.dedupe import Thumbnails
from defect_detection.data.split import (
    EXCLUDED,
    allocate_clusters,
    exact_duplicate_exclusions,
    is_external,
    make_splits,
    resolve_strategy,
)

RANDOM_CFG = SplitConfig(
    strategy="random", train=0.7, val=0.15, test=0.15, val_fraction_of_official_train=0.15
)
DEDUPE = DedupeConfig(
    similarity_threshold=0.99, info_threshold=0.97, coarse_size=16, fine_size=32, shortlist_k=5
)


def synthetic_dataset(
    n_groups: int = 400, seed: int = 0
) -> tuple[pd.DataFrame, Thumbnails, list[int]]:
    """Groups of 1-3 identical-content images (distinct bytes), ~35% defective."""
    rng = np.random.default_rng(seed)
    rows, coarse, fine, group_of = [], [], [], []
    for g in range(n_groups):
        label = int(rng.random() < 0.35)
        grid = Image.fromarray(rng.integers(0, 256, (6, 6), dtype=np.uint8))
        c = np.asarray(grid.resize((16, 16), Image.Resampling.BICUBIC))
        f = np.asarray(grid.resize((32, 32), Image.Resampling.BICUBIC))
        for k in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "rel_path": f"all/img_{g:04d}_{k}.png",
                    "sha256": f"sha_{g}_{k}",
                    "label": label,
                    "class_name": ["normal", "defective"][label],
                    "source_split": "(none)",
                }
            )
            coarse.append(c)
            fine.append(f)
            group_of.append(g)
    return pd.DataFrame(rows), Thumbnails(np.stack(coarse), np.stack(fine)), group_of


@pytest.fixture(scope="module")
def dataset() -> tuple[pd.DataFrame, Thumbnails, list[int]]:
    return synthetic_dataset()


def test_split_is_deterministic_for_a_seed(dataset: tuple) -> None:
    manifest, thumbs, _ = dataset
    a, _ = make_splits(manifest, thumbs, RANDOM_CFG, DEDUPE, seed=42)
    b, _ = make_splits(manifest, thumbs, RANDOM_CFG, DEDUPE, seed=42)
    c, _ = make_splits(manifest, thumbs, RANDOM_CFG, DEDUPE, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])


def test_split_ratios_and_stratification(dataset: tuple) -> None:
    manifest, thumbs, _ = dataset
    _, report = make_splits(manifest, thumbs, RANDOM_CFG, DEDUPE, seed=42)
    fractions = report["fraction_of_used_images"]
    for split, target in [("train", 0.7), ("val", 0.15), ("test", 0.15)]:
        assert fractions[split] == pytest.approx(target, abs=0.03)
    rates = [report["defect_rate"][s] for s in ("train", "val", "test")]
    assert max(rates) - min(rates) < 0.06


def test_same_content_groups_never_straddle_splits(dataset: tuple) -> None:
    manifest, thumbs, group_of = dataset
    splits, report = make_splits(manifest, thumbs, RANDOM_CFG, DEDUPE, seed=42)
    assert (splits.assign(g=group_of).groupby("g")["split"].nunique() == 1).all()
    assert report["pool_clustering"]["n_clusters"] == len(set(group_of))


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


def test_is_external_matches_top_level_folder_only() -> None:
    paths = pd.Series(["casting_512x512/x/a.jpeg", "casting_data/casting_512x512/b.jpeg"])
    assert is_external(paths, ("casting_512x512",)).tolist() == [True, False]


def test_excluded_rows_always_have_a_reason(dataset: tuple) -> None:
    manifest, thumbs, _ = dataset
    manifest = manifest.copy()
    manifest.loc[1, "sha256"] = manifest.loc[0, "sha256"]
    splits, _ = make_splits(manifest, thumbs, RANDOM_CFG, DEDUPE, seed=42)
    excluded = splits[splits["split"] == EXCLUDED]
    assert len(excluded) >= 1
    assert (excluded["exclude_reason"] != "").all()
