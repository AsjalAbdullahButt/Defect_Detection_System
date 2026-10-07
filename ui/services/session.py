"""Streamlit glue shared by all pages: cached config/client, theme injection, static reports."""

import json
from pathlib import Path
from typing import Any

import streamlit as st

from services.api_client import ApiClient, ApiError, ModelInfo
from services.config import UiConfig

STYLES = Path(__file__).resolve().parents[1] / "styles"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]


@st.cache_resource
def get_config() -> UiConfig:
    """Process-wide configuration."""
    return UiConfig.from_env()


@st.cache_resource
def get_client() -> ApiClient:
    """Process-wide HTTP client (connection pooling)."""
    config = get_config()
    return ApiClient(config.api_url, config.api_key, config.timeout_s)


@st.cache_data
def _css() -> str:
    return (STYLES / "tokens.css").read_text(encoding="utf-8") + (STYLES / "theme.css").read_text(
        encoding="utf-8"
    )


def render(html: str) -> None:
    """Render component HTML. Components escape every dynamic value themselves."""
    st.markdown(html, unsafe_allow_html=True)


def inject_theme() -> None:
    """Design tokens + theme CSS; switch tokens to light when the configured theme is dark.

    The theme comes from server config (``theme.base``, set in .streamlit/config.toml or
    STREAMLIT_THEME_BASE), not from ``st.context.theme``: the latter is unknown on a session's
    first run and would briefly render the wrong palette. Users cannot switch themes in-app
    (the main menu is hidden), so config is the single source of truth.
    """
    render(f"<style>{_css()}</style>")
    if st.get_option("theme.base") == "dark":
        render('<span class="dd-theme-dark" hidden></span>')


@st.cache_data(ttl=60, show_spinner=False)
def model_info() -> ModelInfo | None:
    """/v1/model, cached for a minute; None when the API is unreachable or rejects us."""
    try:
        return get_client().model_info()
    except ApiError:
        return None


@st.cache_data(show_spinner=False)
def load_report(relative: str) -> dict[str, Any] | None:
    """A JSON file under the reports folder, or None if it does not exist."""
    path = get_config().reports_dir / relative
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def report_path(relative: str) -> Path | None:
    """Path of a file under the reports folder, or None if missing."""
    path = get_config().reports_dir / relative
    return path if path.is_file() else None


def sample_images(limit: int = 6) -> list[Path]:
    """Optional demo samples (not committed: the dataset licence forbids redistribution)."""
    folder = get_config().samples_dir
    if not folder.is_dir():
        return []
    files = sorted(p for p in folder.iterdir() if p.suffix.lower().lstrip(".") in IMAGE_TYPES)
    return files[:limit]
