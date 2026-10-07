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

from defect_detection.config import config_hash, load_config
from defect_detection.data.dedupe import cluster_images
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


@app.command()
def split(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Cluster duplicates and write the group-aware stratified splits.csv."""
    config = load_config(config_path)
    out = config.data.processed_dir
    table = pd.read_csv(out / "manifest.csv")
    cluster_id, dedupe_report = cluster_images(table, config.dedupe.phash_hamming_threshold)
    splits, split_report = make_splits(table, cluster_id, config.split, config.seed)
    splits.to_csv(out / "splits.csv", index=False)
    typer.echo(f"wrote {out / 'splits.csv'}")
    _write_json(
        out / "split_report.json",
        {**provenance(config), "dedupe": dedupe_report, "split": split_report},
    )
    typer.echo(json.dumps(split_report["counts"], indent=2))


@app.command()
def audit(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Re-check splits for leakage; exits 1 if any check fails."""
    config = load_config(config_path)
    out = config.data.processed_dir
    table = pd.read_csv(out / "manifest.csv")
    splits = pd.read_csv(out / "splits.csv", keep_default_na=False)
    result = run_audit(table, splits, config.dedupe.phash_hamming_threshold)
    _write_json(out / "leakage_audit.json", {**provenance(config), **result})
    for name, ok in result["checks"].items():
        typer.echo(f"  {'PASS' if ok else 'FAIL'}  {name}")
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
def evaluate() -> None:
    """One-shot test-set evaluation (V4)."""
    _not_yet("V4")


@app.command()
def export() -> None:
    """Export to ONNX with metadata and checksums (V5)."""
    _not_yet("V5")


@app.command()
def benchmark() -> None:
    """CPU latency/throughput benchmark (V5)."""
    _not_yet("V5")
