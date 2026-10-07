"""Scan a raw dataset folder: count images per split/class, validate them, flag corrupt files.

The scan is read-only. Every image is opened twice: ``Image.verify()`` checks the container
structure, then a fresh ``load()`` fully decodes the pixels, which catches truncated files that
``verify()`` alone lets through.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

# Folder names that denote an existing split, normalised to train/val/test.
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
NO_SPLIT = "(none)"


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for one image file; ``error`` is set when the file cannot be decoded."""

    rel_path: str
    split: str
    class_dir: str
    extension: str
    size_bytes: int
    format: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DatasetMatch:
    """A known public dataset the scanned folder appears to be, with the evidence for it."""

    name: str
    confidence: str
    evidence: list[str]
    licence: str
    source: str


@dataclass
class Inventory:
    """Result of scanning a dataset root."""

    root: Path
    images: list[ImageRecord] = field(default_factory=list)
    other_files: list[str] = field(default_factory=list)

    @property
    def corrupt(self) -> list[ImageRecord]:
        """Images that failed verification or decoding."""
        return [r for r in self.images if r.error is not None]

    @property
    def valid(self) -> list[ImageRecord]:
        """Images that decoded successfully."""
        return [r for r in self.images if r.error is None]

    def counts(self) -> dict[str, dict[str, int]]:
        """Image counts as ``{split: {class_dir: n}}``, sorted for stable output."""
        counter = Counter((r.split, r.class_dir) for r in self.images)
        result: dict[str, dict[str, int]] = {}
        for (split, class_dir), n in sorted(counter.items()):
            result.setdefault(split, {})[class_dir] = n
        return result


def detect_split(rel_parts: tuple[str, ...]) -> str:
    """Return the normalised split named by the first matching directory, else ``NO_SPLIT``."""
    for part in rel_parts:
        split = SPLIT_ALIASES.get(part.lower())
        if split is not None:
            return split
    return NO_SPLIT


def inspect_image(path: Path, root: Path) -> ImageRecord:
    """Read metadata for one image and fully decode it to detect corruption."""
    rel = path.relative_to(root)
    base = ImageRecord(
        rel_path=rel.as_posix(),
        split=detect_split(rel.parts[:-1]),
        class_dir=rel.parent.name if rel.parent != Path() else NO_SPLIT,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
    )
    try:
        with Image.open(path) as img:
            fmt, (width, height), mode = img.format, img.size, img.mode
            img.verify()
        with Image.open(path) as img:
            img.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        return replace(base, error=f"{type(exc).__name__}: {exc}")
    return replace(base, format=fmt, width=width, height=height, mode=mode)


def scan_dataset(root: Path) -> Inventory:
    """Walk ``root`` recursively and inspect every file with an image extension."""
    inventory = Inventory(root=root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            inventory.images.append(inspect_image(path, root))
        else:
            inventory.other_files.append(path.relative_to(root).as_posix())
    return inventory


def directory_tree(root: Path, max_depth: int = 3) -> list[str]:
    """Indented directory listing with the number of files directly inside each directory."""
    lines: list[str] = []

    def walk(directory: Path, depth: int) -> None:
        children = sorted(directory.iterdir())
        n_files = sum(1 for c in children if c.is_file())
        indent = "  " * depth
        lines.append(f"{indent}{directory.name}/  ({n_files} files)")
        if depth < max_depth:
            for child in children:
                if child.is_dir():
                    walk(child, depth + 1)

    walk(root, 0)
    return lines


# Reference figures for known datasets. They are only compared against what the scan finds;
# a match/mismatch is reported as evidence, never assumed.
_CASTING_TRAIN_TEST_COUNTS = {
    "train": {"def_front": 3758, "ok_front": 2875},
    "test": {"def_front": 453, "ok_front": 262},
}


def identify_dataset(inventory: Inventory) -> DatasetMatch | None:
    """Match the folder against known public datasets using folder names, sizes and counts."""
    class_dirs = {r.class_dir for r in inventory.images}
    sizes = Counter((r.width, r.height) for r in inventory.valid)
    dominant_size = sizes.most_common(1)[0][0] if sizes else None

    if {"def_front", "ok_front"} <= class_dirs:
        counts = inventory.counts()
        size_matches = dominant_size in {(300, 300), (512, 512)}
        counts_match = all(counts.get(s) == c for s, c in _CASTING_TRAIN_TEST_COUNTS.items())
        evidence = [
            "class folders 'def_front' and 'ok_front' present",
            f"dominant image size {dominant_size} "
            + ("matches" if size_matches else "differs from")
            + " the published 300x300 / 512x512 releases",
            "train/test per-class counts "
            + ("equal" if counts_match else "differ from")
            + f" the published 300x300 release {_CASTING_TRAIN_TEST_COUNTS}",
        ]
        return DatasetMatch(
            name="Real-life industrial dataset of casting product (submersible pump impellers)",
            confidence="high" if size_matches and counts_match else "medium",
            evidence=evidence,
            licence="CC BY-NC-ND 4.0 as listed on Kaggle (verify on the source page)",
            source="https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        )

    top_dirs = {Path(r.rel_path).parts[0] for r in inventory.images}
    if "good" in class_dirs and any("ground_truth" in r.rel_path for r in inventory.images):
        return DatasetMatch(
            name="MVTec AD",
            confidence="medium",
            evidence=[
                "'good' class folders",
                "'ground_truth' mask folders",
                f"top dirs: {sorted(top_dirs)}",
            ],
            licence="CC BY-NC-SA 4.0",
            source="https://www.mvtec.com/company/research/datasets/mvtec-ad",
        )
    return None


def summarize(inventory: Inventory, tree_depth: int = 3) -> dict[str, Any]:
    """JSON-serialisable summary of an inventory."""
    valid = inventory.valid
    widths = [r.width for r in valid if r.width is not None]
    heights = [r.height for r in valid if r.height is not None]
    sizes_bytes = [r.size_bytes for r in inventory.images]
    ext_format_mismatch = [
        r.rel_path
        for r in valid
        if r.format is not None and _canonical_ext(r.format) != _canonical_ext(r.extension)
    ]
    match = identify_dataset(inventory)
    return {
        "root": str(inventory.root),
        "n_images": len(inventory.images),
        "n_valid": len(valid),
        "n_corrupt": len(inventory.corrupt),
        "counts": inventory.counts(),
        "formats": dict(Counter(r.format for r in valid).most_common()),
        "modes": dict(Counter(r.mode for r in valid).most_common()),
        "channels": dict(Counter(_channels(r.mode) for r in valid).most_common()),
        "sizes_top10": {
            f"{w}x{h}": n
            for (w, h), n in Counter(zip(widths, heights, strict=True)).most_common(10)
        },
        "width_range": [min(widths), max(widths)] if widths else None,
        "height_range": [min(heights), max(heights)] if heights else None,
        "bytes": {
            "total": sum(sizes_bytes),
            "min": min(sizes_bytes, default=0),
            "max": max(sizes_bytes, default=0),
        },
        "extension_format_mismatch": ext_format_mismatch,
        "corrupt_files": [{"path": r.rel_path, "error": r.error} for r in inventory.corrupt],
        "other_files": inventory.other_files,
        "tree": directory_tree(inventory.root, tree_depth),
        "dataset_match": asdict(match) if match else None,
    }


def format_report(summary: dict[str, Any]) -> str:
    """Render a summary as a plain-text report for the terminal."""
    out = [f"Dataset root: {summary['root']}", "", "Directory tree:"]
    out += [f"  {line}" for line in summary["tree"]]
    out += ["", "Counts per split / class folder:"]
    total = 0
    for split, classes in summary["counts"].items():
        for class_dir, n in classes.items():
            out.append(f"  {split:<8} {class_dir:<24} {n:>7}")
            total += n
    out.append(f"  {'TOTAL':<33} {total:>7}")
    out += [
        "",
        f"Valid images: {summary['n_valid']}   Corrupt/unreadable: {summary['n_corrupt']}",
        f"Formats: {summary['formats']}",
        f"Modes: {summary['modes']}   Channels: {summary['channels']}",
        f"Sizes (top 10): {summary['sizes_top10']}",
        f"Width range: {summary['width_range']}   Height range: {summary['height_range']}",
        f"File bytes: {summary['bytes']}",
        f"Extension/format mismatches: {len(summary['extension_format_mismatch'])}",
        f"Non-image files: {len(summary['other_files'])}",
    ]
    out += [f"  CORRUPT {c['path']}: {c['error']}" for c in summary["corrupt_files"]]
    out += [f"  OTHER   {p}" for p in summary["other_files"][:20]]
    match = summary["dataset_match"]
    out += ["", "Dataset identification:"]
    if match is None:
        out.append("  No known public dataset signature matched.")
    else:
        out += [
            f"  {match['name']} (confidence: {match['confidence']})",
            *[f"    - {e}" for e in match["evidence"]],
            f"  Licence: {match['licence']}",
            f"  Source:  {match['source']}",
        ]
    return "\n".join(out)


def _canonical_ext(name: str) -> str:
    name = name.lower().lstrip(".")
    return {"jpg": "jpeg", "tif": "tiff"}.get(name, name)


def _channels(mode: str | None) -> int | None:
    return Image.getmodebands(mode) if mode else None
