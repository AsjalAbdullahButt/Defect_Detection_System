"""Consistent empty states without internal paths or implementation details."""

from html import escape


def empty_state(title: str, message: str) -> str:
    """Render an escaped, accessible placeholder for content that is not ready."""
    return (
        '<section class="dd-empty" role="status">'
        '<div class="dd-empty__icon" aria-hidden="true">&#9678;</div>'
        f"<h3>{escape(title)}</h3><p>{escape(message)}</p></section>"
    )
