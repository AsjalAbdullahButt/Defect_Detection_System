"""Operating-point selection on calibrated validation probabilities.

Business framing: a missed defect (false negative) ships a faulty part to a customer; a false
alarm (false positive) scraps or re-inspects a good part. The first is far more expensive, so
the decision threshold is the one with the best precision among thresholds that still catch at
least ``min_recall`` of defects. A review band around it sends the least certain images to a
human instead of forcing an automatic decision:

    P < low            -> auto-pass as normal (very few defects get through here)
    low <= P < high    -> needs_review (a person decides)
    P >= high          -> auto-reject as defective (almost always really defective)

``predicted_class`` always comes from ``threshold``; the band only adds the review flag.
"""

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_recall_curve

from defect_detection.config import CostConfig

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class OperatingPoint:
    """Confusion counts and rates at one threshold."""

    name: str
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float


def operating_point(
    name: str, labels: IntArray, probs: FloatArray, threshold: float
) -> OperatingPoint:
    """Evaluate ``P(defective) >= threshold`` as the defect decision."""
    pred = probs >= threshold
    pos = labels == 1
    tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return OperatingPoint(
        name, threshold, tp, fp, tn, fn, precision, recall, f1, fp / max(fp + tn, 1)
    )


def _candidates(labels: IntArray, probs: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Precision/recall at every distinct score used as threshold (sklearn's PR curve)."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    return precision[:-1], recall[:-1], thresholds  # last point has no threshold


def threshold_for_recall(labels: IntArray, probs: FloatArray, min_recall: float) -> float:
    """Highest-precision threshold with recall >= min_recall.

    Ties (several thresholds with the same best precision, common when validation is nearly
    separable) go to the LOWEST one: it catches the most defects at no precision cost on val,
    and a missed defect costs far more than a false alarm.
    """
    precision, recall, thresholds = _candidates(labels, probs)
    ok = recall >= min_recall
    if not ok.any():
        raise ValueError(f"no threshold reaches recall {min_recall}")
    best = np.flatnonzero(ok & (precision == precision[ok].max()))
    return float(thresholds[best].min())


def f1_optimal_threshold(labels: IntArray, probs: FloatArray) -> float:
    """Threshold maximising F1 (reference point only)."""
    precision, recall, thresholds = _candidates(labels, probs)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_precision(
    labels: IntArray, probs: FloatArray, min_precision: float, floor: float
) -> float:
    """Lowest threshold >= ``floor`` whose precision is >= min_precision (else the max score)."""
    precision, _, thresholds = _candidates(labels, probs)
    ok = (thresholds >= floor) & (precision >= min_precision)
    return float(thresholds[ok].min()) if ok.any() else float(max(probs.max(), floor))


def max_threshold_with_recall(labels: IntArray, probs: FloatArray, min_recall: float) -> float:
    """Highest threshold that still keeps recall >= min_recall."""
    _, recall, thresholds = _candidates(labels, probs)
    ok = recall >= min_recall
    if not ok.any():
        raise ValueError(f"no threshold reaches recall {min_recall}")
    return float(thresholds[ok].max())


def review_band(
    labels: IntArray, probs: FloatArray, threshold: float, low_recall: float, high_precision: float
) -> tuple[float, float]:
    """[low, high) around ``threshold``.

    low: the highest threshold that still flags ``low_recall`` of defects, so almost no defect
    is auto-passed. high: the lowest threshold at or above ``threshold`` where flagged images
    are defective with precision ``high_precision``.
    """
    low = min(max_threshold_with_recall(labels, probs, low_recall), threshold)
    high = threshold_for_precision(labels, probs, high_precision, floor=threshold)
    return low, high


@dataclass(frozen=True)
class BandSummary:
    """What the review band does on a labelled set."""

    review_fraction: float
    n_review: int
    errors_at_threshold: int
    errors_caught_by_review: int
    auto_decided_errors: int
    missed_defects_auto_passed: int
    false_alarms_auto_rejected: int


def band_summary(
    labels: IntArray, probs: FloatArray, threshold: float, low: float, high: float
) -> BandSummary:
    """How the band changes outcomes: images routed to review, errors caught, errors left."""
    in_band = (probs >= low) & (probs < high)
    errors = (probs >= threshold) != (labels == 1)
    return BandSummary(
        review_fraction=float(in_band.mean()),
        n_review=int(in_band.sum()),
        errors_at_threshold=int(errors.sum()),
        errors_caught_by_review=int((errors & in_band).sum()),
        auto_decided_errors=int((errors & ~in_band).sum()),
        missed_defects_auto_passed=int(((probs < low) & (labels == 1)).sum()),
        false_alarms_auto_rejected=int(((probs >= high) & (labels == 0)).sum()),
    )


def expected_cost_per_1000(
    n: int, missed_defects: int, false_alarms: int, n_review: int, costs: CostConfig
) -> float:
    """Illustrative cost per 1000 parts using relative unit costs (not real money)."""
    total = (
        costs.missed_defect * missed_defects
        + costs.false_alarm * false_alarms
        + costs.review * n_review
    )
    return 1000 * total / n


def as_dict(point: OperatingPoint) -> dict[str, float | int | str]:
    """JSON-friendly operating point."""
    return asdict(point)
