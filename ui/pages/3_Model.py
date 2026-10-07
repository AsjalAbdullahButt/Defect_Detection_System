"""Model: what is deployed, its operating point, and its one-shot test results with CIs."""

import streamlit as st

from components.badges import version_badge
from components.empty_state import empty_state
from components.hero import hero
from components.metric_tile import metric_tile, text_value, tiles
from services.session import load_report, model_info, render, report_path

info = model_info()
render(
    hero(
        "Model & results",
        "See which model is connected and how it performed on held-out images.",
        version_badge(info.model_version) if info else "",
    )
)

if info is None:
    st.info("Connect the API to see the active model and its settings.")
else:
    low, high = info.review_band
    render(
        tiles(
            metric_tile("Backbone", text_value(info.backbone)),
            metric_tile("Threshold", text_value(f"{info.threshold:.4f}")),
            metric_tile("Temperature", text_value(f"{info.temperature:.3f}")),
            metric_tile("Review band", text_value(f"{low:.3f}-{high:.3f}")),
        )
    )
    st.caption(info.threshold_policy)

metrics = load_report("metrics.json")
if metrics is None:
    render(
        empty_state(
            "No evaluation report yet",
            "Results will appear after a model has been trained and evaluated.",
        )
    )
else:
    titles = {"test": "Official test set", "external_test": "External test set (512 px capture)"}
    for split, result in metrics["results"].items():
        st.subheader(f"{titles.get(split, split)} · n = {result['n']}")
        ci = result["at_operating_threshold"]
        render(
            tiles(
                *(
                    metric_tile(
                        name,
                        text_value(f"{ci[key]['point']:.3f}"),
                        f"95% CI {ci[key]['low']:.3f}-{ci[key]['high']:.3f}",
                    )
                    for key, name in (
                        ("precision", "Precision"),
                        ("recall", "Recall"),
                        ("f1", "F1"),
                        ("pr_auc", "PR-AUC"),
                    )
                )
            )
        )
    st.caption(
        "Defective is the positive class. Each test set was evaluated once; intervals are "
        "1,000-resample bootstrap CIs."
    )
    figures = [
        p
        for p in (report_path("figures/test_confusion.png"), report_path("figures/test_curves.png"))
        if p
    ]
    with st.expander("Confusion matrices and performance curves"):
        for figure in figures:
            st.image(str(figure), width="stretch")
