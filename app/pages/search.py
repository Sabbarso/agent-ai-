"""Page Recherche : recherche hybride + filtres, avec passage direct au Chat RAG."""

import streamlit as st

from rpe.index.filters import SearchFilters, venue_base
from rpe.rag.qa import tldr
from rpe.ui.common import open_rag_chat, paper_card, require, run_safely, setup_page

res = setup_page(":material/search: Recherche")
searcher = require(res, "searcher")
df = searcher.papers.df

years = df["year"].dropna().astype(int)
col_q, col_mode = st.columns([4, 1])
query = col_q.text_input("Requête", placeholder="retrieval-augmented generation for QA")
mode = col_mode.selectbox("Mode", ["hybrid", "bm25", "dense"])

with st.expander("Filtres", icon=":material/filter_list:", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    y_min, y_max = int(years.min()), int(years.max())
    year_range = (
        c1.slider("Années", y_min, y_max, (y_min, y_max)) if y_min < y_max else (y_min, y_max)
    )
    cats = df["categories"].explode().value_counts().head(30).index.tolist()
    categories = c2.multiselect("Catégories", cats)
    venues = sorted({venue_base(v) for v in df["venue"].dropna()})
    venue = c3.selectbox("Venue", ["(toutes)", *venues])
    has_code = c4.checkbox("Avec code")
    accepted = c4.checkbox("Accepté")
    top_k = c4.number_input("Résultats", 5, 50, 10)

if query:
    filters = SearchFilters(
        year_min=year_range[0] if year_range[0] > y_min else None,
        year_max=year_range[1] if year_range[1] < y_max else None,
        categories=categories or None,
        venue=None if venue == "(toutes)" else venue,
        has_code=True if has_code else None,
        is_accepted=True if accepted else None,
    )
    results = run_safely(
        lambda: searcher.search(query, int(top_k), filters, mode=mode), "la recherche"
    )
    head, ask_all = st.columns([3, 1])
    head.caption(f"{len(results or [])} résultats")
    if res.llm is not None and ask_all.button(
        "Poser cette requête au RAG", icon=":material/forum:", type="primary", width="stretch"
    ):
        open_rag_chat(question=query)
    for r in results or []:
        with st.container(border=True):
            paper = searcher.papers.get(r.arxiv_id) or r.model_dump()
            paper_card(paper, r.snippet)
            key = f"tldr_{r.arxiv_id}"
            if key in st.session_state:
                st.info(st.session_state[key], icon=":material/summarize:")
            b1, b2, _ = st.columns([1, 1, 3])
            if res.llm is not None and b1.button(
                "Interroger avec le RAG", key=f"rag_{r.arxiv_id}", icon=":material/forum:"
            ):
                open_rag_chat(paper_id=r.arxiv_id)
            if (
                res.llm is not None
                and key not in st.session_state
                and b2.button("TL;DR", key=f"btn_{r.arxiv_id}", icon=":material/summarize:")
            ):
                st.session_state[key] = run_safely(lambda p=paper: tldr(p, res.llm), "le TL;DR")
                st.rerun()
else:
    st.info(
        "Saisir une requête. Les scores combinent BM25 et embeddings denses (RRF). "
        "Chaque résultat peut ensuite être interrogé avec le RAG.",
        icon=":material/info:",
    )
