"""Generate docs/exemple_rapport_chat.pdf from fake data (no LLM, no index needed).

Run: python scripts/make_sample_report.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rpe.rag.compare import NOT_MENTIONED, PaperComparison
from rpe.reports.pdf_report import build_pdf

OUT = Path(__file__).resolve().parents[1] / "docs" / "exemple_rapport_chat.pdf"

PAPERS = [
    PaperComparison(
        arxiv_id="2401.00001",
        title="Retrieval-Augmented Generation for Open-Domain QA "
        "with $\\alpha$-Calibrated Retrievers",
        authors=["Zoë Müller", "José Núñez", "Łukasz Kowalski", "Ana Borges"],
        year=2024,
        venue="ACL",
        problem="Reducing hallucinations of LLMs on open-domain question answering.",
        method="A dense retriever feeds top-k passages to a generator; retrieval scores are "
        "calibrated with a temperature α learned on a validation set.",
        datasets=["Natural Questions", "TriviaQA"],
        metrics=["EM", "F1"],
        results="EM 45.2 on Natural Questions (+3.1 over the baseline).",
        limitations=["English only", "Retriever trained on Wikipedia"],
        source="abstract",
        relevance="high",
    ),
    PaperComparison(
        arxiv_id="2403.00002",
        title="Self-Checking Generation: Detecting Hallucinations without External Knowledge",
        authors=["Hélène Dubois", "Kenji Sato"],
        year=2024,
        venue="EMNLP Findings",
        problem="Detecting hallucinated statements in generated text.",
        method="Sample several answers and measure their consistency (self-consistency).",
        datasets=["WikiBio"],
        metrics=["AUC-PR"],
        results=NOT_MENTIONED,
        limitations=[NOT_MENTIONED],
        source="extraction",
        relevance="medium",
    ),
    PaperComparison(
        arxiv_id="2505.00003",
        title="Speech Translation for Low-Resource Languages",
        authors=["Amina Diallo"],
        year=2025,
        problem="Speech translation for languages with little data.",
        method="Cascade ASR + MT with data augmentation.",
        datasets=["CoVoST 2"],
        metrics=["BLEU"],
        results=NOT_MENTIONED,
        limitations=["Only 3 language pairs"],
        source="pdf",
        relevance="low",
    ),
]

ANSWER = (
    "Retrieval reduces hallucinations by grounding the generator in retrieved passages "
    "[arXiv:2401.00001]. Consistency checks between several sampled answers can also flag "
    "unsupported statements without external knowledge [arXiv:2403.00002].\n\n"
    "Calibrating the retriever's scores (temperature α) further improves exact match."
)
SYNTHESIS = (
    "Both papers address hallucinations, one by grounding generation in retrieved passages "
    "[arXiv:2401.00001] and the other by checking self-consistency [arXiv:2403.00002]. They "
    "do not share datasets. The retrieval-based approach seems best suited to the question "
    "because it reports a gain on an open-domain QA benchmark [arXiv:2401.00001]. The third "
    "paper is about speech translation and is only marginally related [arXiv:2505.00003]. "
    "Limitations are rarely stated in the abstracts."
)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = build_pdf(
        "How can retrieval reduce hallucinations in LLMs?",
        ANSWER,
        PAPERS,
        SYNTHESIS,
        datetime(2026, 10, 24, 12, 0),
    )
    OUT.write_bytes(pdf)
    print(f"Written: {OUT} ({len(pdf) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
