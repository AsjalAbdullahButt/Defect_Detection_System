"""Build manifest.csv: one row per decodable image with SHA-256, perceptual hashes and metadata.

The manifest is created before any split so duplicates can be found across the whole dataset.
Each image stores the pHash of all 8 rotations/flips (the dihedral group D4; index 0 is the
untransformed image). Comparing one image's 8 variants with another image's identity hash
detects copies that were rotated or flipped, which is how augmented duplicates are spotted.
"""

from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

from defect_detection.core.constants import CLASS_NAMES
from defect_detection.core.hashing import sha256_file
from defect_detection.data.inventory import Inventory, scan_dataset

# Identity first, then rotations, then the four reflections.
DIHEDRAL_TRANSFORMS: tuple[Image.Transpose | None, ...] = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)

MANIFEST_COLUMNS = [
    "rel_path",
    "class_dir",
    "class_name",
    "label",
    "source_split",
    "sha256",
    "phash_d8",
    "width",
    "height",
    "mode",
    "size_bytes",
]


def dihedral_phashes(path: Path) -> list[str]:
    """64-bit pHash (16 hex chars) of the image under each of the 8 dihedral transforms."""
    with Image.open(path) as img:
        gray = img.convert("L")
    return [
        str(imagehash.phash(gray if t is None else gray.transpose(t))) for t in DIHEDRAL_TRANSFORMS
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
                "phash_d8": " ".join(dihedral_phashes(path)),
                "width": record.width,
                "height": record.height,
                "mode": record.mode,
                "size_bytes": record.size_bytes,
            }
        )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return manifest.sort_values("rel_path", ignore_index=True), inventory
