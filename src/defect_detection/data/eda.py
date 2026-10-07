"""EDA helpers called by notebooks/01_eda.ipynb.

Test isolation is enforced here, not by notebook discipline: ``load_eda_frame`` can only
return train/val rows, and the only thing exposed about test is ``split_counts``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import roc_auc_score

from defect_detection.core.constants import CLASS_NAMES, DEFECTIVE_LABEL
from defect_detection.data.split import EXCLUDED

EDA_SPLITS = ("train", "val")


def split_counts(processed_dir: Path) -> pd.DataFrame:
    """Images per split x class (including test: counts only), plus exclusions by reason."""
    splits = pd.read_csv(processed_dir / "splits.csv", keep_default_na=False)
    return pd.crosstab(
        splits["split"].where(splits["split"] != EXCLUDED, "excluded:" + splits["exclude_reason"]),
        splits["class_name"],
        margins=True,
    )


def load_eda_frame(processed_dir: Path) -> pd.DataFrame:
    """Train + val rows joined with manifest metadata. Test rows are never loaded."""
    splits = pd.read_csv(processed_dir / "splits.csv", keep_default_na=False)
    manifest = pd.read_csv(processed_dir / "manifest.csv")
    frame = splits[splits["split"].isin(EDA_SPLITS)].merge(
        manifest[["rel_path", "width", "height", "mode", "size_bytes"]], on="rel_path"
    )
    if frame["split"].eq("test").any():  # defensive: the filter above makes this impossible
        raise AssertionError("test rows leaked into the EDA frame")
    return frame.reset_index(drop=True)


def image_stats(frame: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Add grayscale brightness (mean) and contrast (std) per image, both on a 0-255 scale."""
    brightness, contrast = [], []
    for rel_path in frame["rel_path"]:
        with Image.open(raw_dir / rel_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
    return frame.assign(brightness=brightness, contrast=contrast)


def lighting_shortcut(stats: pd.DataFrame) -> pd.DataFrame:
    """Can a global photometric statistic alone separate the classes?

    For each feature: per-class means, Cohen's d, and the ROC-AUC of using the raw feature as
    a defect score (reported as max(AUC, 1-AUC), so 0.5 = no signal, 1.0 = perfect shortcut).
    """
    rows = []
    is_defect = stats["label"] == DEFECTIVE_LABEL
    for feature in ("brightness", "contrast"):
        normal, defect = stats.loc[~is_defect, feature], stats.loc[is_defect, feature]
        pooled_std = np.sqrt((normal.var(ddof=1) + defect.var(ddof=1)) / 2)
        auc = float(roc_auc_score(is_defect, stats[feature]))
        rows.append(
            {
                "feature": feature,
                "mean_normal": round(float(normal.mean()), 2),
                "mean_defective": round(float(defect.mean()), 2),
                "cohens_d": round(float((defect.mean() - normal.mean()) / pooled_std), 3),
                "separability_auc": round(max(auc, 1 - auc), 3),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def feature_histograms(stats: pd.DataFrame) -> Figure:
    """Per-class histograms of brightness and contrast."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, feature in zip(axes, ("brightness", "contrast"), strict=True):
        for label, name in enumerate(CLASS_NAMES):
            ax.hist(stats.loc[stats["label"] == label, feature], bins=40, alpha=0.6, label=name)
        ax.set_xlabel(f"{feature} (grayscale, 0-255)")
        ax.set_ylabel("images")
        ax.legend()
    fig.tight_layout()
    return fig


def sample_grid(frame: pd.DataFrame, raw_dir: Path, per_class: int = 8, seed: int = 42) -> Figure:
    """Random images per class (rows = classes) for visual inspection."""
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(1.8 * per_class, 4))
    for label, name in enumerate(CLASS_NAMES):
        pool = frame[frame["label"] == label]
        picks = pool.sample(n=min(per_class, len(pool)), random_state=seed)
        for ax in axes[label]:
            ax.axis("off")
        for ax, rel_path in zip(axes[label], picks["rel_path"], strict=False):
            with Image.open(raw_dir / rel_path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{name}\n{Path(rel_path).name}", fontsize=7)
    fig.tight_layout()
    return fig


def cluster_grid(frame: pd.DataFrame, raw_dir: Path, max_clusters: int = 6) -> Figure | None:
    """Show the largest duplicate clusters side by side (augmented copies, label conflicts)."""
    sizes = frame["cluster_id"].value_counts()
    biggest = sizes[sizes > 1].head(max_clusters)
    if biggest.empty:
        return None
    width = int(min(biggest.max(), 8))
    fig, axes = plt.subplots(
        len(biggest), width, figsize=(1.8 * width, 2 * len(biggest)), squeeze=False
    )
    for row, cid in enumerate(biggest.index):
        members = frame[frame["cluster_id"] == cid].head(width)
        for ax in axes[row]:
            ax.axis("off")
        for ax, (_, m) in zip(axes[row], members.iterrows(), strict=False):
            with Image.open(raw_dir / m["rel_path"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{cid} {m['class_name']}\n{Path(m['rel_path']).name}", fontsize=7)
    fig.tight_layout()
    return fig
