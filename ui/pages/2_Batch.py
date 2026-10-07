"""Batch: several images at once, results table, summary and CSV export."""

import base64
import io

import pandas as pd
import streamlit as st
from PIL import Image

from components.badges import version_badge
from components.hero import hero
from components.metric_tile import metric_tile, ticker, tiles
from components.result_card import error_banner
from services.api_client import ApiError
from services.session import IMAGE_TYPES, get_client, get_config, model_info, render

info = model_info()
render(
    hero(
        "Batch inspection",
        "Upload several images; they are sent to the API in batches and come back in order.",
        version_badge(info.model_version) if info else "",
    )
)


def thumbnail(data: bytes) -> str:
    """Small PNG data URI for the table (display only; inference uses the original bytes)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.thumbnail((72, 72))
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="PNG")
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


files = st.file_uploader(
    "Images (JPEG, PNG, BMP or WebP, up to 5 MB each)",
    type=IMAGE_TYPES,
    accept_multiple_files=True,
    key="batch_upload",
)
if st.button("Run batch", type="primary", disabled=not files):
    chunk = get_config().batch_chunk
    rows: list[dict[str, object]] = []
    errors: list[ApiError] = []
    progress = st.progress(0.0, text="Starting...")
    for start in range(0, len(files), chunk):
        part = files[start : start + chunk]
        try:
            result = get_client().predict_batch([f.getvalue() for f in part])
            for item, upload in zip(result.results, part, strict=True):
                rows.append(
                    {
                        "image": thumbnail(upload.getvalue()),
                        "file": upload.name,
                        "class": item.predicted_class,
                        "p_defective": item.defect_probability,
                        "confidence": item.confidence,
                        "needs_review": item.needs_review,
                        "request_id": result.request_id,
                    }
                )
        except ApiError as exc:
            errors.append(exc)
        done = min(start + chunk, len(files))
        progress.progress(done / len(files), text=f"{done} / {len(files)} images")
    st.session_state.batch_rows = rows
    st.session_state.batch_errors = errors

rows = st.session_state.get("batch_rows", [])
for failure in st.session_state.get("batch_errors", []):
    render(error_banner(failure.detail, failure.request_id))
if rows:
    table = pd.DataFrame(rows)
    defective = int((table["class"] == "defective").sum())
    review = int(table["needs_review"].sum())
    render(
        tiles(
            metric_tile("Images", ticker(len(table))),
            metric_tile("Defective", ticker(defective)),
            metric_tile("Needs review", ticker(review)),
        )
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "image": st.column_config.ImageColumn("Image", width="small"),
            "p_defective": st.column_config.NumberColumn("P(defective)", format="%.4f"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
            "needs_review": st.column_config.CheckboxColumn("Needs review"),
        },
    )
    st.download_button(
        "Download CSV",
        table.drop(columns=["image"]).to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
