"""Page Chat RAG : questions sur le corpus ou sur un papier, citations cliquables.

Les conversations sont enregistrées (data/conversations/) et listées dans la barre latérale :
on peut en créer une nouvelle, reprendre n'importe laquelle, la télécharger ou la supprimer.
La page peut être ouverte depuis Recherche (papier pré-sélectionné ou question pré-remplie).
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
from rpe.ui.conversations import ConversationStore, conversation_markdown
from rpe.ui.report_view import render_report_button

res = setup_page(":material/forum: Chat (RAG)")
store = ConversationStore(res.cfg.path(res.cfg.ui.conversations_dir))
if st.session_state.get("conv") is None:
    st.session_state["conv"] = store.new()
conv = st.session_state["conv"]

# ------------------------------------------------------------ sidebar: history
with st.sidebar:
    if st.button("Nouvelle conversation", icon=":material/add:", type="primary", width="stretch"):
        st.session_state["conv"] = store.new()
        st.rerun()
    saved = store.list()
    if saved:
        st.caption("Historique des conversations")
    for c in saved:
        active = c["id"] == conv["id"]
        if (
            st.button(
                c["title"],
                key=f"conv_{c['id']}",
                icon=":material/chat_bubble:" if active else ":material/chat_bubble_outline:",
                type="secondary",
                width="stretch",
                help=f"Dernière activité : {c['updated_at'].replace('T', ' ')}",
            )
            and not active
        ):
            st.session_state["conv"] = store.load(c["id"]) or store.new()
            st.rerun()
    if conv["messages"]:
        st.divider()
        st.download_button(
            "Télécharger la conversation",
            conversation_markdown(conv),
            file_name=f"conversation_{conv['id']}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )
        if st.button("Supprimer la conversation", icon=":material/delete:", width="stretch"):
            store.delete(conv["id"])
            st.session_state["conv"] = store.new()
            st.rerun()

searcher = require(res, "searcher")
llm = require_llm(res)

# ------------------------------------------------------------------- options
st.session_state.setdefault("rag_scope_kind", "Corpus")
with st.expander("Options", icon=":material/tune:"):
    c1, c2, c3 = st.columns([2, 2, 1])
    scope_kind = c1.radio("Portée", ["Corpus", "Un papier"], horizontal=True, key="rag_scope_kind")
    paper_id = (
        c2.text_input("arxiv_id", placeholder="2305.12345", key="rag_paper_id")
        if scope_kind == "Un papier"
        else ""
    )
    lang = c3.selectbox("Langue", ["fr", "en"], key="rag_lang")
    show_compare = st.toggle("Afficher la comparaison des papiers", value=True, key="rag_compare")
scope = paper_id.strip() if scope_kind == "Un papier" and paper_id.strip() else "corpus"
if scope != "corpus":
    paper = searcher.papers.get(scope)
    if paper is None:
        st.warning(f"{scope} n'est pas dans le corpus.", icon=":material/warning:")
    else:
        with st.container(border=True):
            paper_card(paper)

# ------------------------------------------------------------------ messages
history = conv["messages"]
for i, msg in enumerate(history):
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if show_compare and msg.get("comparison"):
            render_comparison(msg["comparison"], key=f"{conv['id']}_m{i}")
        if msg["role"] == "assistant":
            render_report_button(res, msg, key=f"{conv['id']}_m{i}")

question = st.chat_input("Poser une question…") or st.session_state.pop("rag_pending", None)
if question:
    history.append({"role": "user", "content": question})
    store.save(conv)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)
    with (
        st.chat_message("assistant", avatar=ASSISTANT_AVATAR),
        st.spinner("Recherche et rédaction…"),
    ):
        ans = run_safely(
            lambda: answer_question(
                question,
                searcher,
                llm,
                scope=scope,
                top_k=res.cfg.retrieval.rag_top_k,
                chunks=res.chunks,
                lang=lang,
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
                    render_comparison(comparison, key=f"{conv['id']}_m{len(history)}")
            msg = {
                "role": "assistant",
                "content": text,
                "question": question,
                "answer": ans.answer,
                "citations": ans.citations,
                "lang": lang,
                "comparison": comparison.model_dump(mode="json") if comparison else None,
            }
            render_report_button(res, msg, key=f"{conv['id']}_m{len(history)}")
            history.append(msg)
            store.save(conv)
            if len(history) == 2:
                st.rerun()  # show the new conversation in the sidebar history
