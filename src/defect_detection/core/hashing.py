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
