"""V3 pipeline: calibrate a trained run and choose its operating point, on validation only.

Reads ``<run_dir>/best.pt`` and writes, next to it:

* ``calibration.json``: temperature, ECE/NLL before and after, thresholds, review band,
  operating-point table and illustrative cost table;
* ``reliability.png``: reliability diagram before/after temperature scaling;
* ``model_meta.candidate.json``: validated :class:`ModelMeta` for export (V5).

Architecture and preprocessing come from the checkpoint (they must match training); data
paths and the threshold policy come from the current config.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from defect_detection.config import ModelConfig, PreprocessConfig, ProjectConfig, config_hash
from defect_detection.core.model_meta import InputSpec, ModelMeta, ReviewBand
from defect_detection.core.preprocessing import PreprocessSpec
from defect_detection.data.dataset import SplitDataset
from defect_detection.training import calibration as cal
from defect_detection.training import thresholding as th
from defect_detection.training.evaluate import classification_metrics
from defect_detection.training.model_factory import create_model
from defect_detection.training.trainer import (
    CHECKPOINT_NAME,
    collect_logits,
    load_checkpoint,
    resolve_device,
)

CANDIDATE_META = "model_meta.candidate.json"


def calibrate_and_select(config: ProjectConfig, run_dir: Path) -> dict[str, Any]:
    """Fit temperature, pick threshold and review band on val; write reports + candidate meta."""
    checkpoint = load_checkpoint(run_dir / CHECKPOINT_NAME)
    trained = checkpoint["config"]
    model_cfg = ModelConfig.model_validate({**trained["model"], "pretrained": False})
    pre = PreprocessConfig.model_validate(trained["preprocess"])
    spec = PreprocessSpec(pre.image_size, pre.mean, pre.std)

    device = resolve_device(config.train.device)
    model = create_model(model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    val = SplitDataset(config.data.processed_dir / "splits.csv", config.data.raw_dir, "val", spec)
    loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        val, batch_size=config.train.batch_size * 2, num_workers=config.train.num_workers
    )
    labels, logits = collect_logits(model, loader, device)

    op = config.operating_point
    temperature, temperature_note = cal.choose_temperature(logits, labels)
    before = cal.defect_probability(logits)
    after = cal.defect_probability(logits, temperature)

    threshold = th.threshold_for_recall(labels, after, op.min_defect_recall)
    f1_threshold = th.f1_optimal_threshold(labels, after)
    low, high = th.review_band(
        labels, after, threshold, op.review_low_recall, op.review_high_precision
    )
    band = th.band_summary(labels, after, threshold, low, high)
    points = [
        th.operating_point("default_0.5", labels, after, 0.5),
        th.operating_point("f1_optimal", labels, after, f1_threshold),
        th.operating_point("recall_constrained", labels, after, threshold),
    ]
    n = len(labels)
    costs = {p.name: th.expected_cost_per_1000(n, p.fn, p.fp, 0, op.costs) for p in points}
    costs["recall_constrained+review_band"] = th.expected_cost_per_1000(
        n,
        band.missed_defects_auto_passed,
        band.false_alarms_auto_rejected,
        band.n_review,
        op.costs,
    )

    chosen = points[2]
    ranking = classification_metrics(labels, after, threshold)
    meta = ModelMeta(
        model_version=run_dir.name,
        backbone=model_cfg.backbone,
        input=InputSpec(image_size=pre.image_size, mean=pre.mean, std=pre.std),
        temperature=temperature,
        threshold=threshold,
        review_band=ReviewBand(low=low, high=high),
        threshold_policy=(
            f"highest precision with validation defect recall >= {op.min_defect_recall}; "
            f"review band: recall >= {op.review_low_recall} below, "
            f"precision >= {op.review_high_precision} above"
        ),
        val_metrics={
            "pr_auc": ranking["pr_auc"],
            "roc_auc": ranking["roc_auc"],
            "precision": chosen.precision,
            "recall": chosen.recall,
            "f1": chosen.f1,
            "ece": cal.expected_calibration_error(after, labels, op.ece_bins),
            "review_fraction": band.review_fraction,
        },
        git_commit=checkpoint["git_commit"],
        config_hash=checkpoint["config_hash"],
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    report: dict[str, Any] = {
        "run": run_dir.name,
        "calibrate_config_hash": config_hash(config),
        "split_used": "val",
        "n_val": n,
        "n_val_defective": int(labels.sum()),
        "temperature": temperature,
        "temperature_note": temperature_note,
        "val_separable": cal.is_separable(logits, labels),
        "ece": {
            "before": cal.expected_calibration_error(before, labels, op.ece_bins),
            "after": cal.expected_calibration_error(after, labels, op.ece_bins),
        },
        "nll": {
            "before": cal.nll(logits, labels, 1.0),
            "after": cal.nll(logits, labels, temperature),
        },
        "threshold": threshold,
        "f1_optimal_threshold": f1_threshold,
        "review_band": {"low": low, "high": high, **asdict(band)},
        "operating_points": [th.as_dict(p) for p in points],
        "cost_per_1000_parts_illustrative": costs,
        "cost_units": op.costs.model_dump(),
    }

    fig = cal.reliability_diagram(labels, before, after, op.ece_bins, temperature)
    fig.savefig(run_dir / "reliability.png", dpi=120)
    plt.close(fig)
    (run_dir / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / CANDIDATE_META).write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return report
