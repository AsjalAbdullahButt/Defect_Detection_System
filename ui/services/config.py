"""UI configuration from environment variables (read once per process).

The UI is a thin client: it needs the API URL and key, plus two read-only folders for static
content (pre-generated reports and optional sample images). Nothing else.
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class UiConfig:
    """Settings for one UI process. The API key never leaves the server side."""

    api_url: str
    api_key: str
    timeout_s: float
    reports_dir: Path
    samples_dir: Path
    batch_chunk: int

    @classmethod
    def from_env(cls) -> "UiConfig":
        """Build from DD_UI_* variables with local-development defaults."""
        return cls(
            api_url=os.environ.get("DD_UI_API_URL", "http://127.0.0.1:8000").rstrip("/"),
            api_key=os.environ.get("DD_UI_API_KEY", ""),
            timeout_s=float(os.environ.get("DD_UI_TIMEOUT_S", "15")),
            reports_dir=Path(os.environ.get("DD_UI_REPORTS_DIR", REPO_ROOT / "reports")),
            samples_dir=Path(
                os.environ.get("DD_UI_SAMPLES_DIR", REPO_ROOT / "ui" / "assets" / "samples")
            ),
            batch_chunk=int(os.environ.get("DD_UI_BATCH_CHUNK", "16")),
        )
