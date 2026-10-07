"""Explain: pre-generated Grad-CAM panels for missed defects and false alarms."""

import streamlit as st

from components.badges import class_badge, pill, review_badge
from components.hero import hero
from components.metric_tile import metric_tile, text_value, tiles
from services.session import load_report, render, report_path

render(
    hero(
        "Explain errors",
        "Where the model looked (Grad-CAM) on two missed defects and two false alarms "
        "from the one-shot test evaluation. Static, pre-generated panels.",
    )
)

index = load_report("error_analysis/index.json")
if index is None:
    st.info("No panels found (reports/error_analysis/index.json). Run `defect-detection explain`.")
else:
    st.caption(index.get("category_note", ""))
    titles = {"FN": "Missed defect (false negative)", "FP": "False alarm (false positive)"}
    for case in index["cases"]:
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
