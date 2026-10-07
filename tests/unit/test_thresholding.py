import numpy as np
import pytest

from defect_detection.config import CostConfig
from defect_detection.training import thresholding as th

# 10 defects, 10 normals with overlap in the middle.
LABELS = np.array([1] * 10 + [0] * 10)
DEFECT_PROBS = [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.45, 0.30]
NORMAL_PROBS = [0.65, 0.5, 0.4, 0.35, 0.2, 0.15, 0.1, 0.05, 0.03, 0.01]
PROBS = np.array([*DEFECT_PROBS, *NORMAL_PROBS])


def test_operating_point_counts() -> None:
    p = th.operating_point("x", LABELS, PROBS, 0.5)
    assert (p.tp, p.fn, p.fp, p.tn) == (8, 2, 2, 8)
    assert p.precision == pytest.approx(0.8)
    assert p.recall == pytest.approx(0.8)


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_threshold_for_recall_meets_the_constraint() -> None:
    t = th.threshold_for_recall(LABELS, PROBS, min_recall=1.0)
    assert 0.2 < t < 0.30  # inside the empty interval below the lowest defect (0.30)
    assert th.operating_point("t", LABELS, PROBS, t).recall == 1.0
    t90 = th.threshold_for_recall(LABELS, PROBS, min_recall=0.9)
    assert th.operating_point("t", LABELS, PROBS, t90).recall >= 0.9


def test_ties_resolve_to_the_max_margin_point() -> None:
    """Separable: every t in (0.3, 0.8] is perfect; the logit midpoint of the gap is chosen."""
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.8, 0.3, 0.2])
    expected = 1 / (1 + np.exp(-(logit(0.3) + logit(0.8)) / 2))
    t = th.threshold_for_recall(labels, probs, 0.99)
    assert t == pytest.approx(expected)
    same = th.operating_point("a", labels, probs, t), th.operating_point("b", labels, probs, 0.8)
    assert (same[0].tp, same[0].fp, same[0].fn) == (same[1].tp, same[1].fp, same[1].fn)
    assert th.max_threshold_with_recall(labels, probs, 0.99) == pytest.approx(expected)


def test_margin_threshold_without_lower_scores_is_unchanged() -> None:
    assert th.margin_threshold(np.array([0.4, 0.7]), 0.4) == 0.4


def test_f1_optimal_threshold() -> None:
    t = th.f1_optimal_threshold(LABELS, PROBS)
    best = th.operating_point("f1", LABELS, PROBS, t).f1
    for other in np.unique(PROBS):
        assert th.operating_point("o", LABELS, PROBS, float(other)).f1 <= best + 1e-12


def test_review_band_brackets_threshold_and_catches_errors() -> None:
    t = th.threshold_for_recall(LABELS, PROBS, 0.9)
    low, high = th.review_band(LABELS, PROBS, t, low_recall=1.0, high_precision=1.0)
    assert low <= t <= high
    band = th.band_summary(LABELS, PROBS, t, low, high)
    assert band.missed_defects_auto_passed == 0  # every defect is >= low
    assert band.false_alarms_auto_rejected == 0  # everything >= high is a defect
    assert band.errors_caught_by_review + band.auto_decided_errors == band.errors_at_threshold
    assert band.auto_decided_errors == 0


def test_unreachable_recall_raises() -> None:
    with pytest.raises(ValueError, match="no threshold"):
        th.max_threshold_with_recall(LABELS, PROBS, min_recall=1.01)


def test_expected_cost() -> None:
    costs = CostConfig(missed_defect=20, false_alarm=1, review=0.25)
    assert th.expected_cost_per_1000(100, 1, 3, 4, costs) == pytest.approx(
        1000 * (20 + 3 + 1) / 100
    )
