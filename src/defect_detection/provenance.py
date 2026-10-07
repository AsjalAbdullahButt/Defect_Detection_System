"""Provenance stamped into every generated artifact: git commit, config hash, timestamp."""

import shutil
import subprocess
from datetime import UTC, datetime

from defect_detection.config import ProjectConfig, config_hash


def git_commit() -> str:
    """Current commit SHA, suffixed with ``-dirty`` if tracked files have uncommitted changes."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        # Fixed argument lists, no user input: safe to run without a shell.
        sha = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def provenance(config: ProjectConfig) -> dict[str, str]:
    """Fields every report JSON carries so it can be traced back to code + config."""
    return {
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
