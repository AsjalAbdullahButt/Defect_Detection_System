"""Pills for class, review status and model version (icon + text, never colour alone)."""

from html import escape

ICONS = {"normal": "✓", "defective": "✕", "review": "!"}


def pill(text: str, tone: str = "neutral", mono: bool = False) -> str:
    """A rounded label. ``tone`` is one of normal / defective / review / neutral."""
    classes = ["dd-pill"]
    if tone in ICONS:
        classes.append(f"dd-pill--{tone}")
    if mono:
        classes.append("dd-pill--mono")
    icon = f'<span aria-hidden="true">{ICONS[tone]}</span>' if tone in ICONS else ""
    return f'<span class="{" ".join(classes)}">{icon}{escape(text)}</span>'


def class_badge(predicted_class: str) -> str:
    """'Normal' / 'Defective' pill in the class colour."""
    tone = predicted_class if predicted_class in ("normal", "defective") else "neutral"
    return pill(predicted_class.capitalize(), tone)


def review_badge(needs_review: bool) -> str:
    """Amber 'Needs review' pill, or an empty string."""
    return pill("Needs review", "review") if needs_review else ""


def version_badge(model_version: str) -> str:
    """Monospace model-version pill."""
    return pill(f"model {model_version}", "neutral", mono=True)
