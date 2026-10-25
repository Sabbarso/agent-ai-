"""Streamlit rendering of the paper comparison shown under a Chat RAG answer."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from rpe.pdf.reader import COMPARE_SECTIONS, pdf_excerpt
from rpe.rag.compare import (
    NOT_MENTIONED,
    RELEVANCE_LABELS,
    ComparisonResult,
    PaperComparison,
    add_synthesis,
    compare_papers,
    comparison_frame,
    comparison_markdown,
)
from rpe.resources import Resources

NOTE = (
    "Comparaison basée sur les résumés (et le texte complet lorsqu'il est disponible). "
    "Les cellules 'Non mentionné' indiquent une information absente du texte."
)
GREY = "color: #9aa0a6"


def build_comparison(
    res: Resources, question: str, citations: list[str], lang: str
) -> ComparisonResult:
    """Comparison of the cited papers + synthesis (never raises on LLM errors)."""
    cfg = res.cfg

    def pdf_text(aid: str) -> str | None:
        return pdf_excerpt(cfg, aid, COMPARE_SECTIONS, cfg.compare.pdf_chars)

    result = compare_papers(
        question,
        citations,
        res.papers,
        res.llm,
        pdf_text=pdf_text,
        cache_dir=cfg.path(cfg.compare.cache_dir),
        max_papers=cfg.compare.max_papers,
    )
    add_synthesis(result, res.llm, lang)
    return result


def _grey_low(row: pd.Series) -> list[str]:
    low = row["Pertinence"] == RELEVANCE_LABELS["low"]
    return [GREY if low else "" for _ in row]


def _paper_header(c: PaperComparison) -> str:
    year = f" ({c.year})" if c.year else ""
    head = f"**[arXiv:{c.arxiv_id}](https://arxiv.org/abs/{c.arxiv_id})** {c.title}{year}"
    if c.relevance == "low":
        return f":gray[{head}] :gray-badge[peu pertinent]"
    return head


def _value(text: str, c: PaperComparison) -> str:
    if text == NOT_MENTIONED or c.relevance == "low":
        return f":gray[{text}]"
    return text


def _items(values: list[str], c: PaperComparison) -> str:
    return _value(", ".join(values), c)


def render_comparison(data: dict[str, Any] | ComparisonResult, key: str) -> None:
    """Table + tabs + synthesis + exports. `data` may come from session_state (dict)."""
    result = data if isinstance(data, ComparisonResult) else ComparisonResult(**data)
    if not result.comparisons:
        return
    st.markdown("#### :material/compare: Comparaison des papiers")
    for w in result.warnings:
        st.warning(w, icon=":material/warning:")

    frame = comparison_frame(result).drop(columns=["Métriques", "Source"])
    st.dataframe(
        frame.style.apply(_grey_low, axis=1),
        hide_index=True,
        width="stretch",
        row_height=96,
        column_config={
            "arXiv": st.column_config.LinkColumn(
                "arXiv", display_text=r"https://arxiv\.org/abs/(.+)", width="small"
            ),
            "Papier": st.column_config.TextColumn("Papier", width="large"),
            "Méthode": st.column_config.TextColumn("Méthode", width="large"),
            "Datasets": st.column_config.TextColumn("Datasets", width="medium"),
            "Résultats": st.column_config.TextColumn("Résultats", width="large"),
            "Limitations": st.column_config.TextColumn("Limitations", width="medium"),
            "Pertinence": st.column_config.TextColumn("Pertinence", width="small"),
        },
    )
    st.caption(NOTE)

    t_method, t_data, t_res, t_lim = st.tabs(
        ["Méthodes", "Datasets & métriques", "Résultats", "Limitations"]
    )
    for c in result.comparisons:
        header = _paper_header(c)
        t_method.markdown(
            f"{header}  \n*Problème :* {_value(c.problem, c)}  \n*Méthode :* {_value(c.method, c)}"
        )
        t_data.markdown(
            f"{header}  \n*Datasets :* {_items(c.datasets, c)}  \n"
            f"*Métriques :* {_items(c.metrics, c)}"
        )
        t_res.markdown(f"{header}  \n{_value(c.results, c)}")
        t_lim.markdown(f"{header}  \n" + "  \n".join(f"- {_value(x, c)}" for x in c.limitations))

    if result.synthesis:
        from rpe.ui.common import linkify_citations

        st.markdown("**Synthèse comparative**")
        st.markdown(linkify_citations(result.synthesis))

    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button(
        "Exporter (.md)",
        comparison_markdown(result),
        file_name="comparaison.md",
        mime="text/markdown",
        icon=":material/download:",
        key=f"cmp_md_{key}",
    )
    d2.download_button(
        "Exporter (.csv)",
        comparison_frame(result).to_csv(index=False).encode("utf-8"),
        file_name="comparaison.csv",
        mime="text/csv",
        icon=":material/table:",
        key=f"cmp_csv_{key}",
    )
