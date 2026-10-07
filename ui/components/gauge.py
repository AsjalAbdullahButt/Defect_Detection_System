"""Probability-vs-threshold bar with the review band shaded."""

from html import escape


def _pct(value: float) -> str:
    return f"{min(max(value, 0.0), 1.0) * 100:.2f}%"


def gauge(probability: float, threshold: float, band: tuple[float, float]) -> str:
    """Horizontal 0-1 scale: marker = P(defective), line = threshold, hatched = review band."""
    low, high = band
    label = (
        f"P(defective) {probability:.4f}; threshold {threshold:.4f}; "
        f"review band {low:.4f} to {high:.4f}"
    )
    return (
        f'<div class="dd-gauge" role="img" aria-label="{escape(label)}">'
        '<div class="dd-gauge__track">'
        f'<div class="dd-gauge__band" style="left:{_pct(low)};'
        f'width:calc({_pct(high)} - {_pct(low)})"></div>'
        f'<div class="dd-gauge__threshold" style="left:{_pct(threshold)}"></div>'
        f'<div class="dd-gauge__marker" style="left:{_pct(probability)}"></div>'
        "</div>"
        '<div class="dd-gauge__legend">'
        "<span>0 · normal</span>"
        f'<span class="dd-mono">threshold {threshold:.3f} · review {low:.3f} to {high:.3f}</span>'
        "<span>defective · 1</span></div></div>"
    )
