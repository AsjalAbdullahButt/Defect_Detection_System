"""Command-line entry point: ``defect-detection <command>``.

Each command is a thin wrapper around a package function so the same logic is reachable
from tests, notebooks and the Makefile.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from defect_detection.data.inventory import format_report, scan_dataset, summarize

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


@app.command()
def manifest() -> None:
    """Build manifest.csv with SHA-256 and perceptual hashes (V1)."""
    _not_yet("V1")


@app.command()
def split() -> None:
    """Create the group-aware stratified split (V1)."""
    _not_yet("V1")


@app.command()
def audit() -> None:
    """Run the leakage audit across splits (V1)."""
    _not_yet("V1")


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
