"""Command-line entry point: ``defect-detection <command>``.

Each command is a thin wrapper around a package function so the same logic is reachable
from tests, notebooks and the Makefile.
"""

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from defect_detection.config import load_config
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
def train() -> None:
    """Two-stage fine-tuning of the classifier (V2)."""
    _not_yet("V2")


@app.command()
def calibrate() -> None:
    """Temperature scaling and threshold selection on validation (V3)."""
    _not_yet("V3")


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
