"""Research Paper Explorer — Streamlit entry point (Recherche + Chat RAG).

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGES = Path(__file__).parent / "pages"

st.set_page_config(
    page_title="Research Paper Explorer", page_icon=":material/menu_book:", layout="wide"
)


def home() -> None:
    from rpe.ui.common import get_resources, setup_page

    res = setup_page(":material/menu_book: Research Paper Explorer")
    st.markdown(
        "Exploration de la littérature arXiv (dataset Kaggle Cornell) : **recherche hybride** "
        "(BM25 + embeddings, fusion RRF) et **RAG avec citations vérifiées**. Depuis la page "
        "Recherche, chaque résultat peut être interrogé directement avec le RAG."
    )
    cfg, dev = res.cfg, res.dev
    checks = {
        "Corpus (papers.parquet)": cfg.papers_path(dev),
        "Index (embeddings + BM25 + Chroma)": cfg.index_dir(dev) / "bm25.pkl",
    }
    st.subheader("État des artefacts")
    for label, path in checks.items():
        icon = ":material/check_circle:" if path.exists() else ":material/radio_button_unchecked:"
        st.markdown(f"{icon} **{label}** — `{path.relative_to(cfg.root)}`")
    try:
        n = len(get_resources(dev).papers)
        st.metric("Papiers dans le corpus", f"{n:,}".replace(",", " "))
    except FileNotFoundError:
        st.info("Commencer par : `python -m rpe.cli prepare-data" + (" --dev`" if dev else "`"))


nav = st.navigation(
    [
        st.Page(home, title="Accueil", icon=":material/home:", default=True),
        st.Page(PAGES / "1_search.py", title="Recherche", icon=":material/search:"),
        st.Page(PAGES / "2_chat.py", title="Chat (RAG)", icon=":material/forum:"),
    ]
)
nav.run()
