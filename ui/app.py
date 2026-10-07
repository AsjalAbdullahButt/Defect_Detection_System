"""Streamlit entry point: page config, theme, navigation.

Run from the ui/ folder (so .streamlit/config.toml applies):  streamlit run app.py
The UI talks to the API over HTTP only and imports nothing from src/.
"""

from pathlib import Path

import streamlit as st

from services.session import inject_theme

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(
    page_title="Defect Inspection",
    page_icon=str(ASSETS / "logo.svg"),
    layout="wide",
    initial_sidebar_state="auto",  # collapses on narrow screens so it never covers content
)
inject_theme()

navigation = st.navigation(
    [
        st.Page(
            "pages/1_Inspect.py",
            title="Inspect",
            icon=":material/center_focus_strong:",
            default=True,
        ),
        st.Page("pages/2_Batch.py", title="Batch", icon=":material/grid_view:"),
        st.Page("pages/3_Model.py", title="Model", icon=":material/insights:"),
        st.Page("pages/4_Explain.py", title="Explain", icon=":material/visibility:"),
    ]
)
st.logo(str(ASSETS / "logo.svg"))
navigation.run()
