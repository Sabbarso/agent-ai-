"""'Générer le rapport PDF' option shown under each Chat answer (after the comparison)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from rpe.agent.tools import Resources
from rpe.kg.extract import cache_path, load_cache
from rpe.pdf.reader import COMPARE_SECTIONS, pdf_excerpt
from rpe.rag.compare import ComparisonResult
from rpe.reports.chat_report import ChatReport, build_chat_report, report_from_comparison


def make_report(res: Resources, msg: dict[str, Any]) -> ChatReport:
    """Reuse the comparison already shown in the Chat if any, else compute it."""
    if msg.get("comparison"):
        result = ComparisonResult(**msg["comparison"])
        return report_from_comparison(msg["question"], msg["answer"], result)
    cfg = res.cfg

    def pdf_text(aid: str) -> str | None:
        return pdf_excerpt(cfg, aid, COMPARE_SECTIONS, cfg.compare.pdf_chars)

    return build_chat_report(
        msg["question"],
        msg["answer"],
        msg.get("citations", []),
        res.papers,
        res.llm,
        extractions=load_cache(cache_path(cfg, res.dev)),
        pdf_text=pdf_text,
        cache_dir=cfg.path(cfg.compare.cache_dir),
        max_papers=cfg.compare.max_papers,
        lang=msg.get("lang", "en"),
    )


def render_report_button(res: Resources, msg: dict[str, Any], key: str) -> None:
    """Button -> (spinner) report -> download button; the PDF is cached per message."""
    if not msg.get("question") or not msg.get("citations"):
        return
    cache_key = f"report_{key}"
    report: ChatReport | None = st.session_state.get(cache_key)
    clicked = report is None and st.button(
        "Générer le rapport PDF", key=f"btn_{cache_key}", icon=":material/picture_as_pdf:"
    )
    if clicked:
        with st.spinner("Préparation du rapport…"):
            try:
                report = make_report(res, msg)
            except Exception as exc:  # noqa: BLE001 - UI boundary
                st.error(f"Erreur pendant la génération du rapport : {exc}")
                return
        st.session_state[cache_key] = report
    if report is not None:
        st.download_button(
            "Télécharger le rapport PDF",
            report.pdf,
            file_name=report.filename,
            mime="application/pdf",
            icon=":material/download:",
            key=f"dl_{cache_key}",
        )
        if report.warnings:
            st.warning(" ".join(report.warnings), icon=":material/warning:")
