"""SHA-256 utilities for files, bytes and lists of hashes."""

import hashlib
from collections.abc import Iterable
from pathlib import Path

_CHUNK_BYTES = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks so large files never sit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_hashes(hashes: Iterable[str]) -> str:
    """Order-independent fingerprint of a set of hashes (e.g. "which images are in test")."""
    return sha256_bytes("\n".join(sorted(hashes)).encode("ascii"))


SHA256SUMS = "SHA256SUMS"


def write_sha256sums(directory: Path, filenames: Iterable[str]) -> Path:
    """Write ``<sha256>  <name>`` lines (the format ``sha256sum -c`` understands)."""
    lines = [f"{sha256_file(directory / name)}  {name}" for name in sorted(filenames)]
    path = directory / SHA256SUMS
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_sha256sums(directory: Path) -> list[str]:
    """Problems found when checking ``directory/SHA256SUMS`` (an empty list means all good).

    A missing manifest, a missing listed file, a malformed line or a hash mismatch are all
    reported; serving refuses to start on any of them.
    """
    manifest = directory / SHA256SUMS
    if not manifest.is_file():
        return [f"{SHA256SUMS} missing"]
    problems: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, sep, name = line.partition("  ")
        if not sep or len(expected) != 64 or "/" in name or "\\" in name or name.startswith("."):
            problems.append(f"malformed line: {line!r}")
            continue
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} missing")
        elif sha256_file(path) != expected:
            problems.append(f"{name} checksum mismatch")
    return problems
