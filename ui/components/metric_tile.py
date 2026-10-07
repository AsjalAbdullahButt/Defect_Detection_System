"""Simple metric tiles with accessible, static values."""

from html import escape


def ticker(value: int, suffix: str = "") -> str:
    """Static integer with a plain-text equivalent for screen readers."""
    shown = (
        f'<span class="dd-ticker" style="--dd-target:{int(value)}" '
        f'aria-hidden="true">{int(value)}</span>'
    )
    return f'{shown}{escape(suffix)}<span class="dd-sr">{int(value)}{escape(suffix)}</span>'


def metric_tile(label: str, value_html: str, sub: str = "") -> str:
    """One tile. ``value_html`` must be escaped component output (e.g. from ``ticker``)."""
    sub_html = f'<div class="dd-tile__sub">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="dd-tile"><div class="dd-tile__label">{escape(label)}</div>'
        f'<div class="dd-tile__value">{value_html}</div>{sub_html}</div>'
    )


def text_value(text: str) -> str:
    """Escaped plain value for a tile (no animation)."""
    return f'<span class="dd-mono">{escape(text)}</span>'


def tiles(*tiles_html: str) -> str:
    """Responsive grid of tiles."""
    return f'<div class="dd-tiles">{"".join(tiles_html)}</div>'
