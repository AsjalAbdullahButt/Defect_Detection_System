"""Verdict card with the border-beam effect, plus banners for review and errors."""

from html import escape

from components.badges import ICONS
from components.metric_tile import ticker


def verdict_tone(predicted_class: str, needs_review: bool) -> str:
    """Beam colour: amber for review, otherwise the class colour."""
    if needs_review:
        return "review"
    return predicted_class if predicted_class in ("normal", "defective") else "review"


def result_card(
    predicted_class: str,
    confidence: float,
    needs_review: bool,
    latency_ms: float,
    request_id: str,
) -> str:
    """The main verdict: class, icon, confidence ticker, latency and request id."""
    tone = verdict_tone(predicted_class, needs_review)
    icon = ICONS.get(tone, "?")
    confidence_pct = round(confidence * 100)
    return (
        f'<section class="dd-card dd-verdict dd-verdict--{tone} dd-reveal" aria-live="polite">'
        '<div class="dd-verdict__label">'
        f'<span class="dd-verdict__icon" aria-hidden="true">{icon}</span>'
        f"<span>{escape(predicted_class.capitalize())}</span></div>"
        f'<div class="dd-tile__value">{ticker(confidence_pct, "%")} '
        '<span class="dd-muted" style="font-size:.9rem">confidence</span></div>'
        '<div class="dd-verdict__meta dd-mono">'
        f"latency {latency_ms:.1f} ms · request {escape(request_id)}</div>"
        "</section>"
    )


def review_banner() -> str:
    """Shown when the probability falls inside the review band."""
    return (
        '<div class="dd-banner" role="status"><strong aria-hidden="true">!</strong>'
        "<div><strong>Needs human review.</strong> The score is in the uncertain band, so "
        "a person should confirm this part before it ships or is scrapped.</div></div>"
    )


def error_banner(message: str, request_id: str | None) -> str:
    """Friendly API error; the message comes from the API and is always escaped."""
    rid = f'<div class="dd-mono dd-muted">request {escape(request_id)}</div>' if request_id else ""
    return (
        '<div class="dd-banner dd-banner--error" role="alert"><strong aria-hidden="true">✕</strong>'
        f"<div><strong>Request failed.</strong> {escape(message)}{rid}</div></div>"
    )
