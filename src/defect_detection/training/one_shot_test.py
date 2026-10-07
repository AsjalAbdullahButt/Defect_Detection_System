"""V4: evaluate the final model ONCE on the official test set and the external test set.

Guards that make "once" real rather than a promise:

* a ``TEST_EVALUATED.json`` marker in the run directory blocks a second evaluation;
* the leakage audit must have passed, and the test/external-test SHA-256 fingerprints
  recomputed from splits.csv must equal the ones the audit recorded;
* temperature, threshold and review band come from the validation-only candidate meta (V3);
  nothing here is fit or tuned on test.

Writes into the run directory: ``test_metrics.json``, ``test_predictions.csv``,
``test_curves.png``, ``test_confusion.png`` and the marker.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from defect_detection.config import ProjectConfig
from defect_detection.core.hashing import sha256_of_hashes
from defect_detection.core.model_meta import ModelMeta
from defect_detection.data.eda import image_stats
from defect_detection.data.split import EXTERNAL_TEST
from defect_detection.provenance import provenance
from defect_detection.training import evaluate as ev
from defect_detection.training.calibration import defect_probability
from defect_detection.training.operating_point import CANDIDATE_META
from defect_detection.training.thresholding import band_summary
from defect_detection.training.trainer import load_trained_model, resolve_device, split_logits

MARKER = "TEST_EVALUATED.json"
EVAL_SPLITS = ("test", EXTERNAL_TEST)


class AlreadyEvaluatedError(RuntimeError):
    """Raised when a run's test sets have already been evaluated."""


def verify_test_identity(processed_dir: Path) -> dict[str, str]:
    """Check the audit passed and the test sets are exactly the audited ones."""
    audit = json.loads((processed_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise RuntimeError("leakage audit did not pass; refusing to evaluate on test")
    splits = pd.read_csv(processed_dir / "splits.csv", keep_default_na=False)
    hashes = {
        "test_set_sha256": sha256_of_hashes(splits.loc[splits["split"] == "test", "sha256"]),
        "external_test_set_sha256": sha256_of_hashes(
            splits.loc[splits["split"] == EXTERNAL_TEST, "sha256"]
        ),
    }
    for key, value in hashes.items():
        if audit[key] != value:
            raise RuntimeError(f"{key} differs from the audited one; re-run `make audit`")
    return hashes


def evaluate_once(config: ProjectConfig, run_dir: Path) -> dict[str, Any]:
    """Evaluate both test sets with the run's validation-chosen operating point."""
    marker = run_dir / MARKER
    if marker.exists():
        when = json.loads(marker.read_text(encoding="utf-8"))["evaluated_at"]
        raise AlreadyEvaluatedError(
            f"{run_dir.name} was already evaluated on test at {when}; the test set is used once."
        )
    meta = ModelMeta.model_validate_json((run_dir / CANDIDATE_META).read_text(encoding="utf-8"))
    hashes = verify_test_identity(config.data.processed_dir)

    device = resolve_device(config.train.device)
    trained = load_trained_model(run_dir, device)
    splits = pd.read_csv(config.data.processed_dir / "splits.csv", keep_default_na=False)
    similarity = pd.to_numeric(
        splits.set_index("rel_path")["nearest_train_similarity"], errors="coerce"
    )
    ec = config.evaluation
    results: dict[str, Any] = {}
    curve_sets, matrices, frames = {}, {}, []
    for split in EVAL_SPLITS:
        paths, labels, logits = split_logits(config, trained, split, device)
        probs = defect_probability(logits, meta.temperature)
        sims = similarity.loc[paths].to_numpy(dtype=float)
        brightness = image_stats(pd.DataFrame({"rel_path": paths}), config.data.raw_dir)[
            "brightness"
        ].to_numpy()
        cm = ev.confusion(labels, probs, meta.threshold)
        band = band_summary(
            labels, probs, meta.threshold, meta.review_band.low, meta.review_band.high
        )
        results[split] = {
            "n": len(labels),
            "n_defective": int(labels.sum()),
            "at_operating_threshold": ev.bootstrap_ci(
                labels, probs, meta.threshold, ec.bootstrap_resamples, ec.ci_level, config.seed
            ),
            "at_0.5_reference": ev.classification_metrics(labels, probs, 0.5),
            "confusion": cm,
            "review_band": asdict(band),
            "by_similarity_to_train": ev.bucket_metrics(
                labels, probs, meta.threshold, sims, ec.similarity_buckets
            ),
            "brightness_probe": ev.brightness_probe(labels, probs, meta.threshold, brightness),
        }
        curve_sets[split] = (labels, probs)
        matrices[split] = cm
        pred = probs >= meta.threshold
        frames.append(
            pd.DataFrame(
                {
                    "rel_path": paths,
                    "split": split,
                    "label": labels,
                    "p_defective": probs,
                    "predicted": pred.astype(int),
                    "correct": pred == (labels == 1),
                    "needs_review": (probs >= meta.review_band.low)
                    & (probs < meta.review_band.high),
                    "nearest_train_similarity": sims,
                    "brightness": brightness,
                }
            )
        )

    for name, fig in (
        ("test_curves.png", ev.curves_figure(curve_sets)),
        ("test_confusion.png", ev.confusion_figure(matrices)),
    ):
        fig.savefig(run_dir / name, dpi=120)
        plt.close(fig)
    pd.concat(frames).to_csv(run_dir / "test_predictions.csv", index=False)
    metrics = {
        **provenance(config),
        "run": run_dir.name,
        "model_version": meta.model_version,
        "temperature": meta.temperature,
        "threshold": meta.threshold,
        "review_band": meta.review_band.model_dump(),
        **hashes,
        "results": results,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    marker.write_text(
        json.dumps({"evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"), **hashes}),
        encoding="utf-8",
    )
    return metrics
