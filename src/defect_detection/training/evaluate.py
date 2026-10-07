"""Classification metrics with ``defective`` as the positive class.

V2 uses :func:`classification_metrics` for validation during training. The rest of the module
serves the one-shot test evaluation (V4): confusion matrices, bootstrap confidence intervals,
metrics by similarity-to-train bucket, a brightness-shortcut probe, and PR/ROC figures.
Everything here is plain NumPy/sklearn on arrays of labels and probabilities.
"""

from itertools import pairwise
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from defect_detection.core.constants import CLASS_NAMES

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

BOOTSTRAPPED = ("pr_auc", "roc_auc", "precision", "recall", "f1", "macro_f1", "accuracy")


def classification_metrics(
    labels: IntArray, defect_prob: FloatArray, threshold: float = 0.5
) -> dict[str, float]:
    """PR-AUC, ROC-AUC (threshold-free) and precision/recall/F1/accuracy at ``threshold``."""
    if len(np.unique(labels)) < 2:
        raise ValueError("metrics need both classes present")
    pred = (defect_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(labels, defect_prob)),
        "roc_auc": float(roc_auc_score(labels, defect_prob)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, pred)),
        "threshold": threshold,
    }


def confusion(labels: IntArray, defect_prob: FloatArray, threshold: float) -> dict[str, Any]:
    """Confusion counts (rows = true class, cols = predicted) and the row-normalised version."""
    matrix = confusion_matrix(labels, (defect_prob >= threshold).astype(int), labels=[0, 1])
    rows = matrix.sum(axis=1, keepdims=True)
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {
        "labels": list(CLASS_NAMES),
        "counts": matrix.tolist(),
        "row_normalized": np.round(matrix / np.maximum(rows, 1), 4).tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def bootstrap_ci(
    labels: IntArray,
    defect_prob: FloatArray,
    threshold: float,
    n_resamples: int,
    level: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs: resample images with replacement, recompute every metric.

    Resamples that happen to contain only one class are skipped (and counted), because the
    AUCs are undefined on them.
    """
    rng = np.random.default_rng(seed)
    point = classification_metrics(labels, defect_prob, threshold)
    samples: dict[str, list[float]] = {m: [] for m in BOOTSTRAPPED}
    skipped = 0
    n = len(labels)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if labels[idx].min() == labels[idx].max():
            skipped += 1
            continue
        resampled = classification_metrics(labels[idx], defect_prob[idx], threshold)
        for metric in BOOTSTRAPPED:
            samples[metric].append(resampled[metric])
    alpha = (1 - level) / 2
    result = {
        m: {
            "point": point[m],
            "low": float(np.quantile(samples[m], alpha)),
            "high": float(np.quantile(samples[m], 1 - alpha)),
        }
        for m in BOOTSTRAPPED
    }
    result["_meta"] = {"resamples": float(n_resamples), "skipped": float(skipped), "level": level}
    return result


def bucket_metrics(
    labels: IntArray,
    defect_prob: FloatArray,
    threshold: float,
    similarity: FloatArray,
    edges: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Errors by similarity of each image to its nearest training image.

    If accuracy is much higher in the high-similarity buckets, the headline number is partly
    "recognising copies"; the low-similarity buckets show performance on unfamiliar parts.
    """
    bounds = [*edges, np.inf]
    rows = []
    pred = defect_prob >= threshold
    for lo, hi in pairwise(bounds):
        mask = (similarity >= lo) & (similarity < hi)
        if not mask.any():
            continue
        y, p = labels[mask], pred[mask]
        defects, normals = y == 1, y == 0
        rows.append(
            {
                "similarity": f"[{lo}, {hi})" if np.isfinite(hi) else f">= {lo}",
                "n": int(mask.sum()),
                "n_defective": int(defects.sum()),
                "errors": int((p != (y == 1)).sum()),
                "error_rate": float((p != (y == 1)).mean()),
                "missed_defects": int((~p & defects).sum()),
                "false_alarms": int((p & normals).sum()),
            }
        )
    return rows


def brightness_probe(
    labels: IntArray, defect_prob: FloatArray, threshold: float, brightness: FloatArray
) -> dict[str, Any]:
    """Does the model lean on global brightness (the shortcut found in EDA)?

    Within each true class, brightness should not drive P(defective). A strong negative
    Spearman correlation inside the normal class (darker normals look "more defective") is
    the warning sign. Error counts per brightness quartile show where mistakes concentrate.
    """
    out: dict[str, Any] = {}
    for label, name in enumerate(CLASS_NAMES):
        mask = labels == label
        rho = spearmanr(brightness[mask], defect_prob[mask]).statistic if mask.sum() > 2 else np.nan
        out[f"spearman_brightness_vs_p_defect_within_{name}"] = float(rho)
    quartiles = np.quantile(brightness, [0.25, 0.5, 0.75])
    which = np.searchsorted(quartiles, brightness)
    errors = (defect_prob >= threshold) != (labels == 1)
    out["errors_by_brightness_quartile"] = [
        {
            "quartile": q + 1,
            "n": int((which == q).sum()),
            "errors": int(errors[which == q].sum()),
            "defect_rate": float(labels[which == q].mean()) if (which == q).any() else 0.0,
        }
        for q in range(4)
    ]
    return out


def curves_figure(sets: dict[str, tuple[IntArray, FloatArray]]) -> Figure:
    """PR and ROC curves for each evaluation set on shared axes."""
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (labels, probs) in sets.items():
        precision, recall, _ = precision_recall_curve(labels, probs)
        fpr, tpr, _ = roc_curve(labels, probs)
        ax_pr.plot(
            recall, precision, label=f"{name} (AP={average_precision_score(labels, probs):.4f})"
        )
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(labels, probs):.4f})")
    ax_pr.set(xlabel="recall (defective)", ylabel="precision", title="Precision-recall")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    for ax in (ax_pr, ax_roc):
        ax.legend(loc="lower left" if ax is ax_pr else "lower right")
    fig.tight_layout()
    return fig


def confusion_figure(matrices: dict[str, dict[str, Any]]) -> Figure:
    """Counts and row-normalised confusion matrices, one column per evaluation set."""
    fig, axes = plt.subplots(2, len(matrices), figsize=(4.2 * len(matrices), 7.5), squeeze=False)
    for col, (name, cm) in enumerate(matrices.items()):
        for row, key in enumerate(("counts", "row_normalized")):
            ax = axes[row][col]
            data = np.array(cm[key], dtype=float)
            ax.imshow(data, cmap="Blues")
            for (i, j), v in np.ndenumerate(data):
                ax.text(
                    j, i, f"{int(v)}" if key == "counts" else f"{v:.3f}", ha="center", va="center"
                )
            ax.set_xticks([0, 1], CLASS_NAMES)
            ax.set_yticks([0, 1], CLASS_NAMES)
            ax.set(xlabel="predicted", ylabel="true", title=f"{name} - {key.replace('_', ' ')}")
    fig.tight_layout()
    return fig
