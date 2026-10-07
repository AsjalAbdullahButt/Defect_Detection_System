"""UI components must escape every dynamic value (no XSS through API messages or filenames)."""

import pytest

from components.badges import class_badge, pill, review_badge, version_badge
from components.gauge import gauge
from components.hero import hero
from components.metric_tile import metric_tile, text_value, ticker, tiles
from components.result_card import error_banner, result_card, review_banner, verdict_tone

XSS = '<script>alert("x")</script><img src=x onerror=alert(1)>'


def assert_inert(html: str) -> None:
    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert "&lt;script&gt;" in html


def test_error_banner_escapes_api_message_and_request_id() -> None:
    html = error_banner(XSS, XSS)
    assert_inert(html)
    assert 'role="alert"' in html


@pytest.mark.parametrize(
    "render",
    [
        lambda: pill(XSS),
        lambda: version_badge(XSS),
        lambda: hero(XSS, XSS),
        lambda: metric_tile(XSS, text_value(XSS), XSS),
        lambda: text_value(XSS),
        lambda: result_card("defective", 0.9, False, 12.0, XSS),
        lambda: ticker(5, XSS),
    ],
)
def test_every_component_escapes(render) -> None:  # type: ignore[no-untyped-def]
    assert_inert(render())


def test_class_and_review_badges_use_icon_and_text() -> None:
    normal = class_badge("normal")
    assert "Normal" in normal
    assert "dd-pill--normal" in normal
    assert "✓" in normal  # colour is never the only signal
    assert "Needs review" in review_badge(True)
    assert review_badge(False) == ""


def test_unknown_class_falls_back_to_neutral() -> None:
    assert "dd-pill--" not in class_badge("weird")


@pytest.mark.parametrize(
    ("cls", "review", "tone"),
    [("normal", False, "normal"), ("defective", False, "defective"), ("defective", True, "review")],
)
def test_verdict_tone(cls: str, review: bool, tone: str) -> None:
    assert verdict_tone(cls, review) == tone
    assert f"dd-verdict--{tone}" in result_card(cls, 0.5, review, 10.0, "rid")


def test_ticker_has_plain_text_for_screen_readers() -> None:
    html = ticker(97, "%")
    assert "--dd-target:97" in html
    assert '<span class="dd-sr">97%</span>' in html


def test_gauge_positions_are_clamped_percentages() -> None:
    html = gauge(1.7, 0.9036, (0.1054, 0.9979))
    assert "left:100.00%" in html  # probability clamped to the scale
    assert "left:90.36%" in html  # threshold line
    assert "aria-label" in html


def test_tiles_and_banner_markup() -> None:
    assert tiles("a", "b") == '<div class="dd-tiles">ab</div>'
    assert "Needs human review" in review_banner()
