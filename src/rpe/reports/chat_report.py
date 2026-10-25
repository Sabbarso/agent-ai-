"""Chat answer -> PDF report (comparison computed only when the report is requested)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from rpe.data.store import PaperStore
from rpe.llm.client import LLMClient
from rpe.rag.compare import ComparisonResult, add_synthesis, compare_papers
from rpe.reports.pdf_report import build_pdf, report_filename


class ChatReport(BaseModel):
    pdf: bytes
    filename: str
    n_papers: int
    warnings: list[str] = Field(default_factory=list)


def build_chat_report(
    question: str,
    answer: str,
    citations: Sequence[str],
    papers: PaperStore,
    llm: LLMClient | None,
    pdf_text: Callable[[str], str | None] | None = None,
    cache_dir: Path | None = None,
    max_papers: int = 5,
    lang: str = "en",
) -> ChatReport:
    """Compare the CITED papers, write the synthesis, render the PDF (never fails on LLM errors)."""
    result = compare_papers(question, citations, papers, llm, pdf_text, cache_dir, max_papers)
    add_synthesis(result, llm, lang)
    return report_from_comparison(question, answer, result)


def report_from_comparison(question: str, answer: str, result: ComparisonResult) -> ChatReport:
    """PDF from an already computed comparison (e.g. the one displayed in the Chat): no LLM."""
    now = datetime.now()
    pdf = build_pdf(question, answer, result.comparisons, result.synthesis, now)
    return ChatReport(
        pdf=pdf,
        filename=report_filename(question, now),
        n_papers=len(result.comparisons),
        warnings=result.warnings,
    )
