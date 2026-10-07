"""Command-line entry point: ``defect-detection <command>``.

Each command is a thin wrapper around a package function so the same logic is reachable
from tests, notebooks and the Makefile.
"""

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from defect_detection.config import ProjectConfig, config_hash, load_config
from defect_detection.data.dedupe import Thumbnails, load_thumbnails
from defect_detection.data.inventory import format_report, scan_dataset, summarize
from defect_detection.data.leakage_audit import run_audit
from defect_detection.data.manifest import build_manifest
from defect_detection.data.split import make_splits
from defect_detection.provenance import provenance

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


@app.command()
def inspect(
    data_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Raw dataset root.")
    ] = Path("data/raw"),
    json_out: Annotated[
        Path | None, typer.Option(help="Also write the summary as JSON here.")
    ] = None,
    tree_depth: Annotated[int, typer.Option(min=0, help="Depth of the printed tree.")] = 3,
) -> None:
    """Inventory a dataset: tree, counts per split/class, formats, sizes, corrupt files."""
    inventory = scan_dataset(data_dir)
    if not inventory.images:
        typer.echo(f"No images found under {data_dir}.", err=True)
        raise typer.Exit(code=1)
    summary = summarize(inventory, tree_depth=tree_depth)
    typer.echo(format_report(summary))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        typer.echo(f"\nSummary written to {json_out}")


def _not_yet(version: str) -> None:
    typer.echo(f"Not implemented yet: arrives in {version}.", err=True)
    raise typer.Exit(code=2)


ConfigOption = Annotated[
    Path, typer.Option("--config", exists=True, dir_okay=False, help="Project YAML config.")
]
DEFAULT_CONFIG = Path("configs/train.yaml")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"wrote {path}")


@app.command()
def manifest(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Build manifest.csv: SHA-256 + rotation/flip pHashes per image (corrupt files excluded)."""
    config = load_config(config_path)
    table, inventory = build_manifest(config.data.raw_dir, config.data.class_aliases)
    out = config.data.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "manifest.csv", index=False)
    typer.echo(f"wrote {out / 'manifest.csv'} ({len(table)} images)")
    _write_json(
        out / "manifest_report.json",
        {
            **provenance(config),
            "n_images": len(table),
            "n_corrupt_excluded": len(inventory.corrupt),
            "corrupt_files": [r.rel_path for r in inventory.corrupt],
            "counts": table.groupby(["source_split", "class_name"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index"),
        },
    )


def _thumbnails(config: ProjectConfig, manifest: pd.DataFrame) -> Thumbnails:
    d = config.dedupe
    return load_thumbnails(config.data.raw_dir, manifest["rel_path"], d.coarse_size, d.fine_size)


@app.command()
def split(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Cluster same-part copies and write the group-aware stratified splits.csv."""
    config = load_config(config_path)
    out = config.data.processed_dir
    table = pd.read_csv(out / "manifest.csv")
    splits, split_report = make_splits(
        table,
        _thumbnails(config, table),
        config.split,
        config.dedupe,
        config.seed,
        config.data.external_test_dirs,
    )
    splits.to_csv(out / "splits.csv", index=False)
    typer.echo(f"wrote {out / 'splits.csv'}")
    _write_json(
        out / "split_report.json",
        {**provenance(config), **split_report},
    )
    typer.echo(
        json.dumps(
            {"counts": split_report["counts"], "excluded": split_report["excluded"]}, indent=2
        )
    )


@app.command()
def audit(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Re-check splits for leakage from image content; exits 1 if any check fails."""
    config = load_config(config_path)
    out = config.data.processed_dir
    table = pd.read_csv(out / "manifest.csv")
    splits = pd.read_csv(out / "splits.csv", keep_default_na=False)
    d = config.dedupe
    result = run_audit(
        table,
        splits,
        _thumbnails(config, table),
        d.similarity_threshold,
        d.info_threshold,
        d.shortlist_k,
    )
    _write_json(out / "leakage_audit.json", {**provenance(config), **result})
    for name, ok in result["checks"].items():
        typer.echo(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for pair, stats in result["cross_split_similarity"].items():
        typer.echo(f"  {pair:<28} {stats}")
    if not result["passed"]:
        typer.echo("Leakage audit FAILED", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Leakage audit passed. test_set_sha256={result['test_set_sha256']}")


@app.command()
def train(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Two-stage fine-tuning; selects the checkpoint with the best validation PR-AUC."""
    # Imported lazily so data commands don't pay torch's import time.
    from defect_detection.training.trainer import train as run_training

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    config = load_config(config_path)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.train.output_dir / f"{stamp}_{config_hash(config)}"
    summary = run_training(config, run_dir)
    (config.train.output_dir / "LATEST").write_text(run_dir.name, encoding="utf-8")
    reports = Path("reports")
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "curves.png", reports / "figures" / "train_curves.png")
    _write_json(reports / "train_summary.json", summary)
    for name in ("val", "train_no_aug"):
        m = summary[name]
        typer.echo(
            f"{name:<13} PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
            f"F1@0.5={m['f1']:.4f} loss={m['loss']:.4f}"
        )
    typer.echo(f"best checkpoint: {run_dir / 'best.pt'}")


def _resolve_run(config_output_dir: Path, run: Path | None) -> Path:
    if run is not None:
        return run
    latest = config_output_dir / "LATEST"
    if not latest.exists():
        typer.echo(f"No --run given and {latest} does not exist; train first.", err=True)
        raise typer.Exit(code=1)
    return config_output_dir / latest.read_text(encoding="utf-8").strip()


@app.command()
def calibrate(
    config_path: ConfigOption = DEFAULT_CONFIG,
    run: Annotated[
        Path | None, typer.Option(file_okay=False, help="Run directory (default: runs/LATEST).")
    ] = None,
) -> None:
    """Temperature scaling, threshold and review band on VALIDATION; writes candidate meta."""
    from defect_detection.training.operating_point import CANDIDATE_META, calibrate_and_select

    config = load_config(config_path)
    run_dir = _resolve_run(config.train.output_dir, run)
    report = calibrate_and_select(config, run_dir)
    reports = Path("reports")
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "reliability.png", reports / "figures" / "reliability.png")
    _write_json(reports / "calibration.json", report)
    band = report["review_band"]
    typer.echo(f"T={report['temperature']:.4f} ({report['temperature_note']})")
    typer.echo(
        f"ECE {report['ece']['before']:.4f} -> {report['ece']['after']:.4f}   "
        f"NLL {report['nll']['before']:.4f} -> {report['nll']['after']:.4f}"
    )
    typer.echo(
        f"threshold={report['threshold']:.4f}  F1-optimal={report['f1_optimal_threshold']:.4f}"
    )
    typer.echo(
        f"review band=[{band['low']:.4f}, {band['high']:.4f})  "
        f"routes {band['review_fraction']:.1%} of val to review, "
        f"catches {band['errors_caught_by_review']}/{band['errors_at_threshold']} errors"
    )
    for point in report["operating_points"]:
        typer.echo(
            f"  {point['name']:<20} t={point['threshold']:.4f} P={point['precision']:.4f} "
            f"R={point['recall']:.4f} FN={point['fn']} FP={point['fp']}"
        )
    typer.echo(f"candidate meta: {run_dir / CANDIDATE_META}")


@app.command()
def evaluate(
    config_path: ConfigOption = DEFAULT_CONFIG,
    run: Annotated[
        Path | None, typer.Option(file_okay=False, help="Run directory (default: runs/LATEST).")
    ] = None,
) -> None:
    """ONE-SHOT evaluation on the official and external test sets (refuses a second run)."""
    from defect_detection.training.one_shot_test import AlreadyEvaluatedError, evaluate_once

    config = load_config(config_path)
    run_dir = _resolve_run(config.train.output_dir, run)
    try:
        metrics = evaluate_once(config, run_dir)
    except AlreadyEvaluatedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    reports = Path("reports")
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    for name in ("test_curves.png", "test_confusion.png"):
        shutil.copyfile(run_dir / name, reports / "figures" / name)
    shutil.copyfile(run_dir / "test_predictions.csv", reports / "test_predictions.csv")
    _write_json(reports / "metrics.json", metrics)
    for split, result in metrics["results"].items():
        m = result["at_operating_threshold"]
        cm = result["confusion"]
        typer.echo(f"{split} (n={result['n']}, threshold={metrics['threshold']:.4f})")
        for name in ("pr_auc", "roc_auc", "precision", "recall", "f1", "macro_f1"):
            typer.echo(
                f"  {name:<9} {m[name]['point']:.4f}  [{m[name]['low']:.4f}, {m[name]['high']:.4f}]"
            )
        typer.echo(f"  confusion tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")


@app.command("anomaly-baseline")
def anomaly_baseline(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """PatchCore baseline: fit on normal train, threshold on val, ONE evaluation on test sets."""
    from defect_detection.training.anomaly_baseline import RESULT_FILE, run_anomaly_baseline

    config = load_config(config_path)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    try:
        results = run_anomaly_baseline(config, reports)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"memory bank: {results['fit']}")
    for split in ("val", "test", "external_test"):
        m = results[split]
        typer.echo(
            f"{split:<14} ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
            + (f" {m['ms_per_image_cpu']} ms/img" if "ms_per_image_cpu" in m else "")
        )
    typer.echo(f"wrote {reports / RESULT_FILE}")


@app.command()
def export() -> None:
    """Export to ONNX with metadata and checksums (V5)."""
    _not_yet("V5")


@app.command()
def benchmark() -> None:
    """CPU latency/throughput benchmark (V5)."""
    _not_yet("V5")
