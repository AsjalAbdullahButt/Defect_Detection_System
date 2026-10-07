"""Clean-start pages and stale-result handling without a live API or private data."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from components.empty_state import empty_state
from services.api_client import ApiClient, ApiError
from services.session import get_config, load_report, model_info

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def offline_ui(monkeypatch, tmp_path):
    monkeypatch.setenv("DD_UI_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("DD_UI_SAMPLES_DIR", str(tmp_path))

    def offline(self):
        raise ApiError(None, "unreachable", "The API is not reachable.", None)

    monkeypatch.setattr(ApiClient, "model_info", offline)
    for cached in (get_config, load_report, model_info):
        cached.clear()
    yield
    for cached in (get_config, load_report, model_info):
        cached.clear()


@pytest.mark.parametrize(
    ("page", "text"),
    [
        ("app.py", "Your result will appear here"),
        ("pages/2_Batch.py", "Ready for your batch"),
        ("pages/3_Model.py", "No evaluation report yet"),
        ("pages/4_Explain.py", "No explanation panels yet"),
    ],
)
def test_clean_start_pages(offline_ui, page, text):
    app = AppTest.from_file(str(ROOT / "ui" / page)).run(timeout=20)
    assert not app.exception
    assert any(text in item.value for item in app.markdown)


def test_clearing_inspect_upload_removes_previous_result(offline_ui):
    app = AppTest.from_file(str(ROOT / "ui" / "app.py"))
    app.session_state["inspect_image"] = b"old image"
    app.session_state["inspect_digest"] = "old digest"
    app.session_state["inspect_result"] = "old result"
    app.run(timeout=20)
    assert not app.exception
    assert "inspect_result" not in app.session_state
    assert "inspect_image" not in app.session_state
    assert app.button[0].disabled


def test_empty_state_escapes_dynamic_content():
    html = empty_state('<script>alert("x")</script>', "<img src=x onerror=alert(1)>")
    assert "<script" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html
