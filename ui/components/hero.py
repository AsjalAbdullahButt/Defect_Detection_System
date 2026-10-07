"""Page header with the aurora background effect."""

from html import escape


def hero(title: str, subtitle: str, pills_html: str = "") -> str:
    """Title + subtitle; ``pills_html`` must already be escaped component output."""
    pills = f'<div class="dd-pills">{pills_html}</div>' if pills_html else ""
    return (
        f'<header class="dd-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p>{pills}</header>'
    )
