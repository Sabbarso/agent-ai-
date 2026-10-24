"""Shared Streamlit helpers: cached resources, error handling, paper cards, citations."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

from rpe.agent.tools import Resources
from rpe.config import get_config
from rpe.llm.client import LLMClient, get_llm
from rpe.rag.citations import ARXIV_ID_PATTERN

T = TypeVar("T")
_CITE_RE = re.compile(rf"\[arXiv:\s*({ARXIV_ID_PATTERN})((?:\s*§[^\],]+)?)\]")
_MULTI_RE = re.compile(r"\[(arXiv:[^\]]+,[^\]]+)\]")

CSS = """
<style>
.rpe-badge {display:inline-block;padding:1px 8px;margin-right:6px;border-radius:10px;
            font-size:0.78rem;background:#eef2f7;color:#334155;border:1px solid #d5dde8;}
.rpe-badge.code {background:#e8f5e9;color:#1b5e20;border-color:#c8e6c9;}
.rpe-badge.venue {background:#e3f2fd;color:#0d47a1;border-color:#bbdefb;}
.rpe-meta {color:#64748b;font-size:0.85rem;}
</style>
"""

BUILD_HINTS = {
    "papers": "python -m rpe.cli prepare-data",
    "searcher": "python -m rpe.cli build-index",
    "dense": "python -m rpe.cli build-index",
}
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/smart_toy:"
CHAT_PAGE = "pages/2_chat.py"


def open_rag_chat(question: str | None = None, paper_id: str | None = None) -> None:
    """Go to the Chat RAG page, scoped to one paper and/or with a question to answer."""
    st.session_state["chat_history"] = []
    st.session_state["rag_scope_kind"] = "Un papier" if paper_id else "Corpus"
    if paper_id:
        st.session_state["rag_paper_id"] = paper_id
    if question:
        st.session_state["rag_pending"] = question
    st.switch_page(CHAT_PAGE)


@st.cache_resource(show_spinner="Chargement des index…")
def get_resources(dev: bool) -> Resources:
    return Resources(get_config(), dev=dev, llm=get_llm(get_config()))


def full_index_ready() -> bool:
    return (get_config().index_dir(False) / "bm25.pkl").exists()


def default_dev_mode() -> bool:
    """Start in --dev mode when only the dev artefacts are built."""
    return not full_index_ready() and (get_config().index_dir(True) / "bm25.pkl").exists()


def setup_page(title: str) -> Resources:
    """Common page header + sidebar (dev toggle, LLM status); returns the resources."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    # Widget state is reset when switching pages: keep the mode in a plain session key.
    if "dev_mode" not in st.session_state:
        st.session_state["dev_mode"] = default_dev_mode()

    def _sync() -> None:
        st.session_state["dev_mode"] = st.session_state["_dev_toggle"]

    with st.sidebar:
        dev = st.toggle(
            "Mode --dev (petit échantillon)",
            value=st.session_state["dev_mode"],
            key="_dev_toggle",
            on_change=_sync,
        )
        if dev and not full_index_ready():
            st.caption("Index complet pas encore construit : mode --dev activé automatiquement.")
        llm = get_resources(dev).llm
        if llm is None:
            st.warning("LLM non configuré (.env) : fonctions génératives désactivées.")
        else:
            st.caption(f"LLM : `{llm.model}`")
    return get_resources(dev)


def require(res: Resources, attr: str) -> Any:
    """Return `res.<attr>` or stop the page with a clear message if not built."""
    try:
        return getattr(res, attr)
    except FileNotFoundError as exc:
        hint = BUILD_HINTS.get(attr, "")
        suffix = " --dev" if res.dev else ""
        st.error(f"**Ressource non disponible** : {exc}")
        if not res.dev and (res.cfg.index_dir(True) / "bm25.pkl").exists():
            st.info("Les artefacts --dev existent : activer « Mode --dev » dans la barre latérale.")
        if hint:
            st.code(" && ".join(part + suffix for part in hint.split(" && ")), language="bash")
        st.stop()


def require_llm(res: Resources) -> LLMClient:
    if res.llm is None:
        st.error(
            "Cette fonctionnalité nécessite un LLM : renseigner LLM_BASE_URL, LLM_MODEL et "
            "LLM_API_KEY dans `.env`, puis relancer l'application."
        )
        st.stop()
    return res.llm


def run_safely(fn: Callable[[], T], what: str) -> T | None:
    """Run `fn`, showing a readable error instead of a traceback."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - UI boundary
        st.error(f"Erreur pendant {what} : {type(exc).__name__}: {exc}")
        return None


def linkify_citations(text: str) -> str:
    """`[arXiv:ID]` -> clickable Markdown links to arxiv.org (multi-id blocks too)."""

    def split_multi(m: re.Match[str]) -> str:
        return "".join(f"[{part.strip()}]" for part in m.group(1).split(","))

    text = _MULTI_RE.sub(split_multi, text)
    return _CITE_RE.sub(
        lambda m: f"[arXiv:{m.group(1)}{m.group(2)}](https://arxiv.org/abs/{m.group(1)})", text
    )


def badges(paper: dict[str, Any]) -> str:
    out = []
    if paper.get("year"):
        out.append(f'<span class="rpe-badge">{paper["year"]}</span>')
    if paper.get("venue"):
        out.append(f'<span class="rpe-badge venue">{paper["venue"]}</span>')
    if paper.get("has_code"):
        url = paper.get("github_url") or "#"
        out.append(f'<a class="rpe-badge code" href="{url}" target="_blank">code</a>')
    return " ".join(out)


def paper_card(paper: dict[str, Any], snippet: str | None = None) -> None:
    aid = paper["arxiv_id"]
    st.markdown(f"**[{paper['title']}](https://arxiv.org/abs/{aid})**")
    authors = ", ".join((paper.get("authors_list") or paper.get("authors") or [])[:5])
    st.markdown(
        f'{badges(paper)} <span class="rpe-meta">arXiv:{aid} · {authors}</span>',
        unsafe_allow_html=True,
    )
    if snippet:
        st.caption(snippet)
