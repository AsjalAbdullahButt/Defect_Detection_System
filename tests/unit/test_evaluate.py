import numpy as np
import pytest
from matplotlib.figure import Figure

from defect_detection.training import evaluate as ev

LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
PROBS = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.1, 0.05, 0.0])


def test_confusion_counts_and_normalisation() -> None:
    cm = ev.confusion(LABELS, PROBS, 0.5)
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (5, 1, 1, 3)
    assert cm["counts"] == [[5, 1], [1, 3]]
    assert cm["row_normalized"][1] == [0.25, 0.75]


def test_classification_metrics_include_macro_f1() -> None:
    m = ev.classification_metrics(LABELS, PROBS, 0.5)
    f1_defect, f1_normal = 2 * 3 / (2 * 3 + 1 + 1), 2 * 5 / (2 * 5 + 1 + 1)
    assert m["f1"] == pytest.approx(f1_defect)
    assert m["macro_f1"] == pytest.approx((f1_defect + f1_normal) / 2)


def test_bootstrap_ci_brackets_point_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 300)
    probs = np.clip(labels * 0.6 + rng.normal(0.2, 0.25, 300), 0, 1)
    a = ev.bootstrap_ci(labels, probs, 0.5, n_resamples=300, level=0.95, seed=1)
    b = ev.bootstrap_ci(labels, probs, 0.5, n_resamples=300, level=0.95, seed=1)
    assert a == b
    for metric in ev.BOOTSTRAPPED:
        assert a[metric]["low"] <= a[metric]["point"] <= a[metric]["high"]
    assert a["roc_auc"]["high"] - a["roc_auc"]["low"] < 0.15


def test_bootstrap_of_perfect_classifier_is_degenerate() -> None:
    labels = np.array([0] * 50 + [1] * 50)
    ci = ev.bootstrap_ci(labels, labels.astype(float), 0.5, n_resamples=200, level=0.95, seed=0)
    assert ci["roc_auc"] == {"point": 1.0, "low": 1.0, "high": 1.0}


def test_bucket_metrics_partition_all_images() -> None:
    sims = np.array([0.5, 0.96, 0.96, 0.975, 0.985, 0.985, 0.995, 0.6, 0.97, 0.99])
    rows = ev.bucket_metrics(LABELS, PROBS, 0.5, sims, (0.0, 0.95, 0.97, 0.98, 0.99))
    assert sum(r["n"] for r in rows) == len(LABELS)
    assert sum(r["errors"] for r in rows) == 2
    assert rows[-1]["similarity"] == ">= 0.99"


def test_brightness_probe_detects_reliance_on_brightness() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([0] * 200 + [1] * 200)
    brightness = rng.uniform(120, 160, 400)
    probs = np.clip(
        1 - (brightness - 120) / 40 + 0.05 * rng.normal(size=400), 0, 1
    )  # darker -> defect
    probe = ev.brightness_probe(labels, probs, 0.5, brightness)
    assert probe["spearman_brightness_vs_p_defect_within_normal"] < -0.8
    assert sum(q["n"] for q in probe["errors_by_brightness_quartile"]) == 400


def test_sample_correct_handles_small_and_empty_groups() -> None:
    import pandas as pd

    from defect_detection.training.error_analysis import sample_correct

    predictions = pd.DataFrame(
        {
            "split": ["test"] * 5 + ["external_test"] * 2,
            "label": [1, 1, 1, 0, 0, 1, 0],
            "correct": [True, True, True, True, False, True, False],  # no correct external normal
        }
    )
    sample = sample_correct(predictions, per_group=2, seed=0)
    counts = sample.groupby(["split", "label"]).size().to_dict()
    assert counts == {("test", 1): 2, ("test", 0): 1, ("external_test", 1): 1}
    assert sample["correct"].all()


def test_figures_render() -> None:
    sets = {"a": (LABELS, PROBS), "b": (LABELS[::-1], PROBS)}
    assert isinstance(ev.curves_figure(sets), Figure)
    cms = {k: ev.confusion(*v, 0.5) for k, v in sets.items()}
    assert isinstance(ev.confusion_figure(cms), Figure)
