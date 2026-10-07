"""Build manifest.csv: one row per decodable image with its SHA-256, label and metadata.

The manifest is created before any split so exact duplicates (same SHA-256) can be found
across the whole dataset. Near-duplicates are found from image content in dedupe.py.
"""

from pathlib import Path

import pandas as pd

from defect_detection.core.constants import CLASS_NAMES
from defect_detection.core.hashing import sha256_file
from defect_detection.data.inventory import Inventory, scan_dataset

MANIFEST_COLUMNS = [
    "rel_path",
    "class_dir",
    "class_name",
    "label",
    "source_split",
    "sha256",
    "width",
    "height",
    "mode",
    "size_bytes",
]


def resolve_classes(inventory: Inventory, class_aliases: dict[str, str]) -> dict[str, str]:
    """Map every class folder found to a class name; fail fast on unmapped folders."""
    found = {r.class_dir for r in inventory.valid}
    unmapped = sorted(found - class_aliases.keys())
    if unmapped:
        raise ValueError(
            f"Class folders {unmapped} are not in data.class_aliases; add them to the config."
        )
    return {d: class_aliases[d] for d in found}


def build_manifest(raw_dir: Path, class_aliases: dict[str, str]) -> tuple[pd.DataFrame, Inventory]:
    """Scan ``raw_dir`` and return (manifest of decodable images, full inventory).

    Corrupt files are left out of the manifest; the inventory keeps them for reporting.
    """
    inventory = scan_dataset(raw_dir)
    class_of = resolve_classes(inventory, class_aliases)
    rows = []
    for record in inventory.valid:
        path = raw_dir / record.rel_path
        class_name = class_of[record.class_dir]
        rows.append(
            {
                "rel_path": record.rel_path,
                "class_dir": record.class_dir,
                "class_name": class_name,
                "label": CLASS_NAMES.index(class_name),
                "source_split": record.split,
                "sha256": sha256_file(path),
                "width": record.width,
                "height": record.height,
                "mode": record.mode,
                "size_bytes": record.size_bytes,
            }
        )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return manifest.sort_values("rel_path", ignore_index=True), inventory
