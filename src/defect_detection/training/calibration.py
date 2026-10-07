"""Temperature scaling, fit on validation logits only.

A single scalar T divides the logits: ``softmax(z / T)``. T > 1 softens overconfident
probabilities. Because dividing by T never changes which logit is larger or the ranking of
P(defective), accuracy and PR/ROC-AUC are unchanged; only the probabilities become honest.
That matters here because class-weighted training deliberately skews the probabilities.

Calibration is measured on P(defective) (the quantity the threshold acts on): equal-width
bins, observed defect rate per bin vs mean predicted probability.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from defect_detection.core.constants import DEFECTIVE_LABEL

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Search range for log(T): T in [0.05, 20] covers any realistic miscalibration.
_LOG_T_BOUNDS = (float(np.log(0.05)), float(np.log(20.0)))


def nll(logits: FloatArray, labels: IntArray, temperature: float) -> float:
    """Mean negative log-likelihood of the labels under ``softmax(logits / T)``."""
    log_probs = log_softmax(logits / temperature, axis=1)
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: FloatArray, labels: IntArray) -> float:
    """T minimising validation NLL (bounded 1-D search over log T, so T stays positive)."""
    result = minimize_scalar(
        lambda log_t: nll(logits, labels, float(np.exp(log_t))),
        bounds=_LOG_T_BOUNDS,
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(np.exp(result.x))


def is_separable(logits: FloatArray, labels: IntArray) -> bool:
    """True if every defect outscores every normal: NLL then has no finite minimiser in T."""
    margin = logits[:, DEFECTIVE_LABEL] - logits[:, 1 - DEFECTIVE_LABEL]
    return bool(margin[labels == DEFECTIVE_LABEL].min() > margin[labels != DEFECTIVE_LABEL].max())


def choose_temperature(logits: FloatArray, labels: IntArray) -> tuple[float, str]:
    """Fitted T, or T = 1 when validation is perfectly separable (fit would push T -> 0)."""
    if is_separable(logits, labels):
        return 1.0, "validation perfectly separable: NLL keeps falling as T -> 0, so T = 1 is kept"
    temperature = fit_temperature(logits, labels)
    if not np.exp(_LOG_T_BOUNDS[0]) * 1.01 < temperature < np.exp(_LOG_T_BOUNDS[1]) / 1.01:
        return temperature, (
            "WARNING: fit hit the search bound [0.05, 20]; the model carries little signal or "
            "validation is too small - do not trust these probabilities"
        )
    return temperature, "fitted by minimising validation NLL"


def defect_probability(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    """P(defective) after temperature scaling."""
    probs: FloatArray = softmax(logits / temperature, axis=1)[:, DEFECTIVE_LABEL]
    return probs


def reliability_bins(
    probs: FloatArray, labels: IntArray, n_bins: int
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Per equal-width bin: mean predicted P(defective), observed defect rate, count."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    safe = np.maximum(counts, 1)
    mean_pred = np.bincount(index, weights=probs, minlength=n_bins) / safe
    observed = np.bincount(index, weights=labels.astype(np.float64), minlength=n_bins) / safe
    return mean_pred, observed, counts.astype(np.int64)


def expected_calibration_error(probs: FloatArray, labels: IntArray, n_bins: int = 15) -> float:
    """Count-weighted mean |observed rate - predicted probability| over bins."""
    mean_pred, observed, counts = reliability_bins(probs, labels, n_bins)
    return float(np.sum(counts * np.abs(observed - mean_pred)) / counts.sum())


def reliability_diagram(
    labels: IntArray, before: FloatArray, after: FloatArray, n_bins: int, temperature: float
) -> Figure:
    """Side-by-side reliability curves before and after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, probs, title in (
        (axes[0], before, "before (T = 1)"),
        (axes[1], after, f"after (T = {temperature:.3f})"),
    ):
        mean_pred, observed, counts = reliability_bins(probs, labels, n_bins)
        used = counts > 0
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
        ax.plot(mean_pred[used], observed[used], marker="o", label="model")
        ece = expected_calibration_error(probs, labels, n_bins)
        ax.set_title(f"{title}   ECE = {ece:.4f}")
        ax.set_xlabel("predicted P(defective)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("observed defect rate (validation)")
    fig.tight_layout()
    return fig
