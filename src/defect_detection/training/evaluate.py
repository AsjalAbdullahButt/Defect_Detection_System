"""Classification metrics with ``defective`` as the positive class.

V2 uses :func:`classification_metrics` for validation during training; V4 adds curves,
confusion matrices and bootstrap confidence intervals for the one-shot test evaluation.
"""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    labels: npt.NDArray[np.int64], defect_prob: npt.NDArray[np.float64], threshold: float = 0.5
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
        "accuracy": float(accuracy_score(labels, pred)),
        "threshold": threshold,
    }
