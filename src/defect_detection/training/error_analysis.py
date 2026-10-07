"""Error analysis helpers for notebooks/02_error_analysis.ipynb: Grad-CAM, galleries, histograms.

Grad-CAM (Selvaraju et al.) answers "where did the network look?": take the last convolutional
feature map A (C x h x w), the gradient of the defect logit with respect to it, average the
gradient per channel to get a weight per channel, and keep the positive part of the weighted
sum of channels. Upsampled to the image, it highlights regions that raised P(defective).

timm models split cleanly into ``forward_features`` (the feature map) and ``forward_head``
(pool + classifier), so no hooks are needed and the same code works for every backbone.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from defect_detection.core.constants import CLASS_NAMES, DEFECTIVE_LABEL
from defect_detection.core.preprocessing import PreprocessSpec, prepare_image, preprocess


def grad_cam(model: nn.Module, image: Image.Image, spec: PreprocessSpec) -> npt.NDArray[np.float32]:
    """Heatmap in [0, 1] (image_size x image_size) of evidence FOR the defective class."""
    model.eval()
    x = torch.from_numpy(preprocess(image, spec)).unsqueeze(0)
    features = model.forward_features(x)  # type: ignore[operator]
    logits = model.forward_head(features)  # type: ignore[operator]
    (grads,) = torch.autograd.grad(logits[0, DEFECTIVE_LABEL], features)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
    cam = nn.functional.interpolate(
        cam, size=(spec.image_size, spec.image_size), mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam - cam.min()
    peak = cam.max()
    out: npt.NDArray[np.float32] = (cam / peak if peak > 0 else cam).detach().numpy()
    return out


def gallery(
    rows: pd.DataFrame,
    raw_dir: Path,
    model: nn.Module | None,
    spec: PreprocessSpec,
    title: str,
    per_row: int = 4,
) -> Figure:
    """Grid of images (with a Grad-CAM overlay when ``model`` is given) and their scores.

    ``rows`` needs the columns of test_predictions.csv.
    """
    n = max(len(rows), 1)
    n_rows = (n + per_row - 1) // per_row
    fig, axes = plt.subplots(n_rows, per_row, figsize=(3.0 * per_row, 3.3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows(), strict=False):
        with Image.open(raw_dir / r["rel_path"]) as img:
            img.load()
            shown = prepare_image(img, spec)
            ax.imshow(shown.convert("L"), cmap="gray")
            if model is not None:
                ax.imshow(grad_cam(model, img, spec), cmap="jet", alpha=0.35)
        review = " REVIEW" if r["needs_review"] else ""
        ax.set_title(
            f"{CLASS_NAMES[int(r['label'])]} p={r['p_defective']:.3f}{review}\n"
            f"sim={r['nearest_train_similarity']:.3f} bright={r['brightness']:.0f}\n"
            f"{Path(r['rel_path']).name}",
            fontsize=7,
        )
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def confidence_histograms(
    predictions: pd.DataFrame, threshold: float, band: tuple[float, float]
) -> Figure:
    """P(defective) per true class and split, with the threshold and review band marked."""
    splits = list(dict.fromkeys(predictions["split"]))
    fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 3.8), squeeze=False)
    bins = np.linspace(0, 1, 41)
    for ax, split in zip(axes[0], splits, strict=True):
        part = predictions[predictions["split"] == split]
        for label, name in enumerate(CLASS_NAMES):
            ax.hist(
                part.loc[part["label"] == label, "p_defective"], bins=bins, alpha=0.6, label=name
            )
        ax.axvspan(*band, color="grey", alpha=0.2, label="review band")
        ax.axvline(threshold, color="black", linestyle="--", label="threshold")
        ax.set_yscale("log")
        ax.set(title=split, xlabel="P(defective)", ylabel="images (log)")
        ax.legend()
    fig.tight_layout()
    return fig


def sample_correct(predictions: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    """Up to ``per_group`` correct predictions per (split, true class); small groups give fewer."""
    correct = predictions[predictions["correct"]]
    parts = [
        group.sample(min(per_group, len(group)), random_state=seed)
        for _, group in correct.groupby(["split", "label"], sort=False)
    ]
    return pd.concat(parts) if parts else correct.head(0)


def error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Every misclassified image with the attributes used to categorise it."""
    errors = predictions[~predictions["correct"]].copy()
    errors["error_type"] = np.where(errors["label"] == 1, "FN (missed defect)", "FP (false alarm)")
    columns = [
        "split",
        "error_type",
        "rel_path",
        "p_defective",
        "needs_review",
        "nearest_train_similarity",
        "brightness",
    ]
    return errors[columns].sort_values(["split", "error_type", "p_defective"])


def select_explain_cases(predictions: pd.DataFrame, per_kind: int = 2) -> pd.DataFrame:
    """The cases shown on the UI's Explain page.

    False negatives: the official test set's misses (lowest P(defective) first).
    False positives: the most confident false alarms on the external test set.
    """
    errors = predictions[~predictions["correct"]]
    fn = errors[(errors["split"] == "test") & (errors["label"] == 1)].nsmallest(
        per_kind, "p_defective"
    )
    fp = errors[(errors["split"] == "external_test") & (errors["label"] == 0)].nlargest(
        per_kind, "p_defective"
    )
    return pd.concat([fn.assign(kind="FN"), fp.assign(kind="FP")], ignore_index=True)


def save_explain_panel(
    row: pd.Series, raw_dir: Path, model: nn.Module, spec: PreprocessSpec, out_path: Path
) -> None:
    """Side-by-side PNG: the model's input view and the Grad-CAM overlay for one case."""
    with Image.open(raw_dir / row["rel_path"]) as img:
        img.load()
        shown = prepare_image(img, spec).convert("L")
        cam = grad_cam(model, img, spec)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 3.6))
    left.imshow(shown, cmap="gray")
    left.set_title("input (224x224)", fontsize=9)
    right.imshow(shown, cmap="gray")
    right.imshow(cam, cmap="jet", alpha=0.4)
    right.set_title("Grad-CAM: evidence for 'defective'", fontsize=9)
    for ax in (left, right):
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
