"""Inspect: one image in, verdict out (class, confidence, review flag, gauge)."""

import hashlib

import streamlit as st

from components.badges import pill
from components.empty_state import empty_state
from components.gauge import gauge
from components.hero import hero
from components.metric_tile import metric_tile, text_value, ticker, tiles
from components.result_card import error_banner, result_card, review_banner
from services.api_client import ApiError, Prediction
from services.session import IMAGE_TYPES, get_client, model_info, render, sample_images

info = model_info()
status = pill("Model connected", "normal") if info else pill("API unavailable", "review")
render(
    hero(
        "Inspect a part",
        "Add a photo to check for defects. Get a clear result "
        "and see when a human review is needed.",
        status,
    )
)


def select_image(data: bytes, label: str) -> None:
    """Store a new image and forget the previous result."""
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.get("inspect_digest") != digest:
        st.session_state.update(
            inspect_digest=digest, inspect_image=data, inspect_label=label,
            inspect_result=None, inspect_error=None,
        )  # fmt: skip


left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1. Add an image")
    upload = st.file_uploader(
        "Image (JPEG, PNG, BMP or WebP, up to 5 MB)", type=IMAGE_TYPES, key="inspect_upload"
    )
    samples = {path.stem.replace("_", " "): path for path in sample_images()}
    chosen = (
        st.pills("Or try a sample", list(samples), selection_mode="single", key="inspect_sample")
        if samples
        else None
    )
    # One source per rerun: an upload wins over a sample.
    if upload is not None:
        select_image(upload.getvalue(), "uploaded image")
    elif chosen is not None:
        select_image(samples[chosen].read_bytes(), f"sample: {chosen}")
    else:
        for key in (
            "inspect_digest",
            "inspect_image",
            "inspect_label",
            "inspect_result",
            "inspect_error",
        ):
            st.session_state.pop(key, None)
    image = st.session_state.get("inspect_image")
    if image is not None:
        st.image(image, caption=st.session_state.get("inspect_label", ""), width="stretch")
    run = st.button("Inspect image", type="primary", disabled=image is None, width="stretch")
    st.caption("Use a clear, top-view photo of one cast impeller.")

with right:
    st.subheader("2. Review the result")
    if run and image is not None:
        with st.spinner("Inspecting..."):
            try:
                st.session_state.inspect_result = get_client().predict(image)
                st.session_state.inspect_error = None
            except ApiError as exc:
                st.session_state.inspect_result = None
                st.session_state.inspect_error = exc
    result: Prediction | None = st.session_state.get("inspect_result")
    error: ApiError | None = st.session_state.get("inspect_error")
    if result is not None:
        render(
            result_card(
                result.predicted_class,
                result.confidence,
                result.needs_review,
                result.latency_ms,
                result.request_id,
            )
        )
        if result.needs_review:
            render(review_banner())
        render(
            tiles(
                metric_tile("P(defective)", text_value(f"{result.defect_probability:.4f}")),
                metric_tile("Latency", ticker(round(result.latency_ms), " ms")),
            )
        )
        with st.expander("How this decision was made"):
            st.write(
                "The defect probability is compared with a threshold chosen on validation data."
            )
            if info is not None:
                render(gauge(result.defect_probability, result.threshold, info.review_band))
            st.caption(f"Model: {result.model_version}")
    elif error is not None:
        render(error_banner(error.detail, error.request_id))
    else:
        render(
            empty_state("Your result will appear here", "Add an image, then select Inspect image.")
        )
