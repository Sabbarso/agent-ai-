"""Page Chat RAG : questions sur le corpus ou sur un papier, citations cliquables.

Peut être ouverte depuis la page Recherche (papier pré-sélectionné ou question pré-remplie).
"""

import streamlit as st

from rpe.rag.qa import answer_question
from rpe.ui.common import (
    ASSISTANT_AVATAR,
    USER_AVATAR,
    linkify_citations,
    paper_card,
    require,
    require_llm,
    run_safely,
    setup_page,
)
from rpe.ui.compare_view import build_comparison, render_comparison
from rpe.ui.report_view import render_report_button

res = setup_page(":material/forum: Chat (RAG)")
searcher = require(res, "searcher")
llm = require_llm(res)

st.session_state.setdefault("rag_scope_kind", "Corpus")
c1, c2, c3 = st.columns([2, 2, 1])
scope_kind = c1.radio("Portée", ["Corpus", "Un papier"], horizontal=True, key="rag_scope_kind")
paper_id = (
    c2.text_input("arxiv_id", placeholder="2305.12345", key="rag_paper_id")
    if scope_kind == "Un papier"
    else ""
)
lang = c3.selectbox("Langue", ["en", "fr"], key="rag_lang")
scope = paper_id.strip() if scope_kind == "Un papier" and paper_id.strip() else "corpus"
if scope != "corpus":
    paper = searcher.papers.get(scope)
    if paper is None:
        st.warning(f"{scope} n'est pas dans le corpus.", icon=":material/warning:")
    else:
        with st.container(border=True):
            paper_card(paper)

history = st.session_state.setdefault("chat_history", [])
b1, b2 = st.columns([1, 1])
if b1.button("Effacer la conversation", icon=":material/delete:"):
    history.clear()
show_compare = b2.toggle("Afficher la comparaison", value=True, key="rag_compare")
for i, msg in enumerate(history):
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if show_compare and msg.get("comparison"):
            render_comparison(msg["comparison"], key=f"m{i}")
        if msg["role"] == "assistant":
            render_report_button(res, msg, key=f"m{i}")

question = st.chat_input("Poser une question…") or st.session_state.pop("rag_pending", None)
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)
    with (
        st.chat_message("assistant", avatar=ASSISTANT_AVATAR),
        st.spinner("Recherche et rédaction…"),
    ):
        ans = run_safely(
            lambda: answer_question(
                question, searcher, llm, scope=scope, chunks=res.chunks, lang=lang
            ),
            "la génération de la réponse",
        )
        if ans is not None:
            text = linkify_citations(ans.answer)
            st.markdown(text)
            if ans.invalid_citations:
                st.caption(
                    ":material/warning: Citations retirées (hors contexte) : "
                    + ", ".join(ans.invalid_citations)
                )
            with st.expander(f"Contexte récupéré ({len(ans.contexts)})", icon=":material/source:"):
                for c in ans.contexts:
                    st.markdown(f"**{c.ref}** — {c.title}")
                    st.caption(c.text[:400])
            comparison = None
            if show_compare and ans.citations:
                with st.spinner("Comparaison des papiers cités…"):
                    comparison = run_safely(
                        lambda: build_comparison(res, question, ans.citations, lang),
                        "la comparaison des papiers",
                    )
                if comparison is not None:
                    render_comparison(comparison, key=f"m{len(history)}")
            msg = {
                "role": "assistant",
                "content": text,
                "question": question,
                "answer": ans.answer,
                "citations": ans.citations,
                "lang": lang,
                "comparison": comparison.model_dump() if comparison else None,
            }
            render_report_button(res, msg, key=f"m{len(history)}")
            history.append(msg)
