"""Explain: pre-generated Grad-CAM panels for missed defects and false alarms."""

import streamlit as st

from components.badges import class_badge, pill, review_badge
from components.empty_state import empty_state
from components.hero import hero
from components.metric_tile import metric_tile, text_value, tiles
from services.session import load_report, render, report_path

render(
    hero(
        "Understand the mistakes",
        "Explore missed defects and false alarms. "
        "Heatmaps highlight regions that influenced the model.",
    )
)

index = load_report("error_analysis/index.json")
if index is None or not index.get("cases"):
    render(
        empty_state(
            "No explanation panels yet",
            "Saved examples will appear here once error analysis is available.",
        )
    )
else:
    st.caption("Heatmaps show model attention, not a precise defect outline or proof of the cause.")
    st.caption(index.get("category_note", ""))
    titles = {"FN": "Missed defect (false negative)", "FP": "False alarm (false positive)"}
    cases = index["cases"]
    selected = st.selectbox(
        "Choose an example",
        range(len(cases)),
        format_func=lambda i: f"{i + 1}. {titles.get(cases[i]['kind'], cases[i]['kind'])}",
    )
    for case in [cases[selected]]:
        st.subheader(titles.get(case["kind"], case["kind"]))
        image_col, text_col = st.columns([3, 2], gap="large")
        panel = report_path(f"error_analysis/{case['file']}")
        if panel is not None:
            image_col.image(str(panel), width="stretch")
        with text_col:
            render(
                pill(case["split"].replace("_", " "), "neutral")
                + " " + class_badge(case["true_class"])
                + " " + review_badge(case["needs_review"])
            )  # fmt: skip
            render(
                tiles(
                    metric_tile("P(defective)", text_value(f"{case['defect_probability']:.4f}")),
                    metric_tile(
                        "Similarity to train", text_value(f"{case['nearest_train_similarity']:.3f}")
                    ),
                )
            )
            st.markdown(f"**Category:** {case.get('category', 'not categorised')}")
            st.caption(f"Image: {case['image']} (true class: {case['true_class']})")
