from rpe.llm.client import LLMClient
from rpe.rag.compare import (
    NOT_MENTIONED,
    add_synthesis,
    cited_ids,
    compare_papers,
    comparison_frame,
    comparison_markdown,
    relevance,
)
from rpe.rag.qa import answer_question
from tests.fakes import ScriptedLLM

ANSWER = "Back-translation helps [arXiv:2302.00002]; metrics compared [arXiv:2405.00005]."


def llm_of(fn) -> LLMClient:
    return LLMClient("m", completion_fn=ScriptedLLM(fn), sleep=lambda s: None)


def batch_router(messages):
    user = messages[-1]["content"]
    if "comparison facts of EACH paper" in user:
        papers = []
        for aid in ("2302.00002", "2405.00005", "2301.00001"):
            if f"arXiv:{aid}" in user:
                papers.append(
                    {
                        "arxiv_id": f"arXiv:{aid}",
                        "problem": "Low-resource MT.",
                        "method": "Back-translation.",
                        "datasets": ["WMT"],
                        "metrics": ["BLEU"],
                        "results": "",
                        "limitations": [],
                    }
                )
        return {"papers": papers}
    return "Both study translation [arXiv:2302.00002] [arXiv:2405.00005] [arXiv:9999.00001]."


def test_existing_answer_unchanged_and_only_cited_ids_compared(searcher):
    fake = ScriptedLLM(lambda m: ANSWER)
    ans = answer_question("low-resource translation", searcher, LLMClient("m", completion_fn=fake))
    before = ans.model_dump()
    res = compare_papers(
        "low-resource translation", ans.citations, searcher.papers, llm_of(batch_router)
    )
    assert ans.model_dump() == before  # the RAG answer object is never modified
    assert [c.arxiv_id for c in res.comparisons] == ["2302.00002", "2405.00005"]
    assert len(ans.contexts) > 2  # retrieved but not cited papers are not compared


def test_missing_fields_are_not_mentioned(paper_store):
    res = compare_papers("translation", ["2302.00002"], paper_store, llm_of(batch_router))
    c = res.comparisons[0]
    assert c.results == NOT_MENTIONED and c.limitations == [NOT_MENTIONED]
    assert c.method == "Back-translation." and c.source == "abstract"


def test_single_batch_call_and_disk_cache(paper_store, tmp_path):
    fake = ScriptedLLM(batch_router)
    llm = LLMClient("m", completion_fn=fake)
    ids = ["2302.00002", "2405.00005"]
    compare_papers("q", ids, paper_store, llm, cache_dir=tmp_path)
    assert len(fake.calls) == 1  # one batched extraction for all missing papers
    compare_papers("q2", ids, paper_store, llm, cache_dir=tmp_path)
    assert len(fake.calls) == 1  # second time: per-arxiv_id disk cache


def test_pdf_source_used_when_available(paper_store):
    seen = {}

    def router(messages):
        seen["prompt"] = messages[-1]["content"]
        return batch_router(messages)

    res = compare_papers(
        "q",
        ["2302.00002"],
        paper_store,
        llm_of(router),
        pdf_text=lambda aid: "[Method] We back-translate monolingual data.",
    )
    assert res.comparisons[0].source == "pdf" and "full-text excerpt" in seen["prompt"]


def test_invented_numbers_in_results_are_dropped(paper_store):
    def router(messages):
        return {"papers": [{"arxiv_id": "2302.00002", "results": "BLEU improves by 7.3 points"}]}

    res = compare_papers("q", ["2302.00002"], paper_store, llm_of(router))
    assert res.comparisons[0].results == NOT_MENTIONED


def test_synthesis_citations_validated(paper_store):
    llm = llm_of(batch_router)
    res = compare_papers("translation", ["2302.00002", "2405.00005"], paper_store, llm)
    add_synthesis(res, llm)
    assert set(res.synthesis_citations) == {"2302.00002", "2405.00005"}
    assert res.invalid_citations == ["9999.00001"] and "9999.00001" not in res.synthesis


def test_llm_failure_keeps_table(paper_store):
    llm = LLMClient("m", completion_fn=lambda m, **k: "not json", max_parse_retries=0)
    res = compare_papers("q", ["2302.00002"], paper_store, llm)
    assert len(res.comparisons) == 1 and res.warnings
    assert res.comparisons[0].method == NOT_MENTIONED


def test_low_relevance_last_and_helpers(paper_store):
    assert relevance("machine translation", paper_store.get("2302.00002")) == "high"
    assert relevance("protein folding", paper_store.get("2302.00002")) == "low"
    res = compare_papers(
        "machine translation", ["2403.00003", "2302.00002"], paper_store, llm_of(batch_router)
    )
    assert [c.arxiv_id for c in res.comparisons] == ["2302.00002", "2403.00003"]
    assert cited_ids(["a", "b", "a", "c"], 2) == ["a", "b"]
    frame = comparison_frame(res)
    assert frame.loc[1, "Pertinence"] == "peu pertinent"
    assert "arxiv.org/abs/2302.00002" in comparison_markdown(res)
