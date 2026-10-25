"""Research Paper Explorer — application Streamlit (Chat RAG + Recherche).

Lancement : streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGES = Path(__file__).parent / "pages"

st.set_page_config(
    page_title="Research Paper Explorer", page_icon=":material/menu_book:", layout="wide"
)

nav = st.navigation(
    [
        st.Page(PAGES / "chat.py", title="Chat (RAG)", icon=":material/forum:", default=True),
        st.Page(PAGES / "search.py", title="Recherche", icon=":material/search:"),
    ]
)
nav.run()
