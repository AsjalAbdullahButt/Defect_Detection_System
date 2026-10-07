import numpy as np
import pytest

from defect_detection.training import calibration as cal


def overconfident_logits(n: int, factor: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from well-calibrated margins, logits inflated by ``factor``.

    The true temperature that undoes the inflation is exactly ``factor``.
    """
    rng = np.random.default_rng(seed)
    margin = rng.normal(0.0, 2.0, size=n)
    labels = (rng.random(n) < 1 / (1 + np.exp(-margin))).astype(np.int64)
    logits = np.stack([np.zeros(n), margin * factor], axis=1)
    return logits, labels


def test_fit_recovers_planted_temperature() -> None:
    logits, labels = overconfident_logits(20_000, factor=3.0)
    assert cal.fit_temperature(logits, labels) == pytest.approx(3.0, rel=0.05)


def test_calibration_reduces_ece_and_nll() -> None:
    logits, labels = overconfident_logits(20_000, factor=3.0)
    t = cal.fit_temperature(logits, labels)
    before, after = cal.defect_probability(logits), cal.defect_probability(logits, t)
    assert (
        cal.expected_calibration_error(after, labels)
        < cal.expected_calibration_error(before, labels) / 3
    )
    assert cal.nll(logits, labels, t) < cal.nll(logits, labels, 1.0)


def test_temperature_preserves_ranking() -> None:
    logits, _ = overconfident_logits(500, factor=2.0)
    before, after = cal.defect_probability(logits), cal.defect_probability(logits, 2.0)
    assert np.array_equal(np.argsort(before), np.argsort(after))


def test_separable_validation_keeps_t_equal_one() -> None:
    logits = np.array([[0.0, 3.0], [0.0, 2.0], [0.0, -1.0], [0.0, -4.0]])
    labels = np.array([1, 1, 0, 0])
    assert cal.is_separable(logits, labels)
    t, note = cal.choose_temperature(logits, labels)
    assert t == 1.0
    assert "separable" in note


def test_fit_at_search_bound_is_flagged() -> None:
    rng = np.random.default_rng(1)
    logits = np.stack([np.zeros(200), rng.normal(0, 5, 200)], axis=1)  # pure noise
    labels = rng.integers(0, 2, 200)
    t, note = cal.choose_temperature(logits, labels)
    assert t > 19
    assert note.startswith("WARNING")


def test_reliability_bins_account_for_every_sample() -> None:
    probs = np.array([0.05, 0.1, 0.55, 0.95, 0.99])
    labels = np.array([0, 0, 1, 1, 1])
    mean_pred, observed, counts = cal.reliability_bins(probs, labels, n_bins=10)
    assert counts.sum() == 5
    assert observed[9] == 1.0
    assert mean_pred[9] == pytest.approx(0.97)


def test_ece_is_zero_for_perfectly_calibrated_bins() -> None:
    probs = np.array([0.25] * 4 + [0.75] * 4)
    labels = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert cal.expected_calibration_error(probs, labels, n_bins=4) == pytest.approx(0.0)
