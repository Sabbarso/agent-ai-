import re
from datetime import datetime

import pytest

pytest.importorskip("reportlab")

from rpe.llm.client import LLMClient  # noqa: E402
from rpe.rag.compare import PaperComparison  # noqa: E402
from rpe.reports.chat_report import build_chat_report  # noqa: E402
from rpe.reports.pdf_report import (  # noqa: E402
    build_pdf,
    clean_latex,
    report_filename,
    rl_text,
)
from tests.fakes import ScriptedLLM  # noqa: E402

_PAGE_RE = re.compile(rb"/Type\s*/Page(?!s)")


def n_pages(pdf: bytes) -> int:
    return len(_PAGE_RE.findall(pdf))


def comparison(i: int, long: bool = False, **kw) -> PaperComparison:
    text = ("Très longue description avec accents é è à ç & <balises> " * 40) if long else ""
    return PaperComparison(
        arxiv_id=f"2401.{i:05d}",
        title=f"Paper {i}: $\\alpha$-Tuning for \\emph{{Ré}}trieval & <QA>",
        authors=["Zoë Müller", "José Núñez", "Łukasz Kowalski", "Ana B."],
        year=2024,
        venue="ACL" if i % 2 else None,
        problem=text or "Question answering.",
        method=text or "Dense retrieval + generator.",
        datasets=["SQuAD", "Natural Questions"],
        metrics=["EM", "F1"],
        results=text or "EM 45.2 on NQ",
        limitations=[text] if text else [],
        source="abstract",
        **kw,
    )


ANSWER = "RAG helps [arXiv:2401.00001]. See also [arXiv:2401.00002 §Method] & <others>."


def test_build_pdf_valid_multi_page():
    comps = [comparison(1), comparison(2, relevance="low"), comparison(3)]
    pdf = build_pdf(
        "How does RAG work?",
        ANSWER,
        comps,
        "Both [arXiv:2401.00001].",
        datetime(2026, 10, 24, 12, 0),
    )
    assert pdf.startswith(b"%PDF") and n_pages(pdf) > 1
    assert b"https://arxiv.org/abs/2401.00001" in pdf  # clickable citation links


def test_special_characters_do_not_crash():
    q = "Qu'est-ce que l'élève apprend ? $\\alpha$ & <tag> \\'ecole"
    pdf = build_pdf(q, "Réponse : α < β & γ > δ $x^2$ \\textbf{gras}", [comparison(1)], "")
    assert pdf.startswith(b"%PDF")
    assert clean_latex("$\\alpha$-Tuning \\emph{fast} \\'ecole") == "α-Tuning fast école"
    assert rl_text("a & b <c>") == "a &amp; b &lt;c&gt;"


def test_eight_long_papers_no_layout_error():
    comps = [comparison(i, long=True) for i in range(1, 9)]
    pdf = build_pdf("q", ANSWER * 20, comps, "synthèse " * 200)
    assert pdf.startswith(b"%PDF") and n_pages(pdf) >= 4


def test_report_filename():
    name = report_filename("How can retrieval reduce hallucinations?", datetime(2026, 1, 2))
    assert name == "rapport_retrieval-reduce-hallucination_2026-01-02.pdf"


def test_chat_report_only_cited_ids_one_batch_call(paper_store):
    def router(messages):
        if "EACH paper" in messages[-1]["content"]:
            return {"papers": [{"arxiv_id": "2302.00002", "method": "Back-translation."}]}
        return "Synthèse [arXiv:2302.00002]."

    fake = ScriptedLLM(router)
    llm = LLMClient("m", completion_fn=fake)
    rep = build_chat_report(
        "translation",
        "Answer [arXiv:2302.00002].",
        ["2302.00002"],
        paper_store,
        llm,
    )
    assert rep.n_papers == 1 and rep.pdf.startswith(b"%PDF")
    assert len(fake.calls) == 2  # one batched extraction + one synthesis
    assert b"2405.00005" not in rep.pdf  # uncited papers never appear


def test_chat_report_survives_llm_failure(paper_store):
    llm = LLMClient("m", completion_fn=lambda m, **k: "not json", max_parse_retries=0)
    rep = build_chat_report("q", "A [arXiv:2302.00002].", ["2302.00002"], paper_store, llm)
    assert rep.pdf.startswith(b"%PDF") and rep.warnings


def _chat_message_script(root: str) -> None:
    """Mimics the Chat page's assistant message + the report button."""
    import streamlit as st

    from rpe.config import Config
    from rpe.data.store import PaperStore
    from rpe.llm.client import LLMClient
    from rpe.resources import Resources
    from rpe.ui.report_view import render_report_button
    from tests.fakes import ScriptedLLM, make_papers_df

    res = Resources(
        Config(root=root),
        llm=LLMClient(
            "m",
            completion_fn=ScriptedLLM(
                lambda m: {"papers": []} if "EACH paper" in m[-1]["content"] else "Synth."
            ),
        ),
    )
    res.__dict__["papers"] = PaperStore(make_papers_df())
    msg = {
        "role": "assistant",
        "content": "Answer [arXiv:2302.00002]",
        "question": "q",
        "answer": "Answer [arXiv:2302.00002]",
        "citations": ["2302.00002"],
    }
    st.markdown(msg["content"])
    with st.expander("Contexte récupéré (1)"):
        st.caption("ctx")
    render_report_button(res, msg, key="m1")


def test_pdf_option_adds_only_a_button(tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_chat_message_script, args=(str(tmp_path),), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert [b.label for b in at.button] == ["Générer le rapport PDF"]
    assert len(at.markdown) == 1 and len(at.expander) == 1
    assert not at.dataframe and not at.tabs and not at.table
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.dataframe and not at.tabs  # the PDF option itself only adds buttons
    assert at.session_state["report_m1"].pdf.startswith(b"%PDF")


def test_pdf_reuses_displayed_comparison_without_llm(paper_store):
    from rpe.config import Config
    from rpe.rag.compare import ComparisonResult
    from rpe.resources import Resources
    from rpe.ui.report_view import make_report

    fake = ScriptedLLM(lambda m: "should not be called")
    res = Resources(Config(), llm=LLMClient("m", completion_fn=fake))
    res.__dict__["papers"] = paper_store
    shown = ComparisonResult(
        question="q", comparisons=[comparison(1)], synthesis="S [arXiv:2401.00001]."
    )
    msg = {
        "question": "q",
        "answer": "A [arXiv:2401.00001]",
        "citations": ["2401.00001"],
        "comparison": shown.model_dump(),
    }
    rep = make_report(res, msg)
    assert rep.pdf.startswith(b"%PDF") and rep.n_papers == 1 and fake.calls == []
