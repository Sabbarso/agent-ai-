import pytest

from rpe.llm.client import (
    LLMClient,
    LLMError,
    StructuredOutputError,
    TransientLLMError,
    parse_json,
)
from rpe.llm.schemas import PaperExtraction, ReadingSheet
from rpe.rag.citations import extract_citations, normalize_id, validate_citations
from rpe.rag.qa import IDK_MESSAGE, answer_question
from rpe.rag.reading_sheet import generate_reading_sheet, reading_sheet_markdown
from tests.fakes import ScriptedLLM


def make_llm(fn, tmp_path=None, **kw) -> LLMClient:
    return LLMClient(
        "fake-model",
        completion_fn=fn,
        cache_dir=tmp_path,
        sleep=lambda s: None,
        **kw,
    )


# ------------------------------------------------------------ citations


def test_extract_citations_formats():
    text = (
        "A [arXiv:2301.00001]. B [arXiv:2302.00002v2, arXiv:2301.00001] "
        "C [arXiv:2403.00003 §Method]"
    )
    assert extract_citations(text) == ["2301.00001", "2302.00002", "2403.00003"]
    assert normalize_id("cs/0101001v3") == "cs/0101001"


def test_validate_citations_remove():
    text = (
        "X holds [arXiv:2301.00001]. Y holds [arXiv:9999.99999]. "
        "Z [arXiv:2301.00001, arXiv:1111.11111]."
    )
    rep = validate_citations(text, ["2301.00001"])
    assert rep.valid == ["2301.00001"]
    assert rep.invalid == ["9999.99999", "1111.11111"]
    assert "9999" not in rep.text and "1111" not in rep.text
    assert "Y holds." in rep.text and "Z [arXiv:2301.00001]." in rep.text
    assert rep.n_citations == 4 and rep.valid_rate == pytest.approx(0.5)


def test_validate_citations_flag_and_sections():
    rep = validate_citations(
        "A [arXiv:2301.00001 §Results] B [arXiv:2222.22222]", ["2301.00001"], mode="flag"
    )
    assert "[arXiv:2301.00001 §Results]" in rep.text
    assert "[unverified: arXiv:2222.22222]" in rep.text


def test_validate_citations_drops_invented_sections():
    text = "A [arXiv:2301.00001 §3.3] B [arXiv:2301.00001 §Method]"
    rep = validate_citations(text, ["2301.00001"], allowed_sections={"2301.00001": ["Method"]})
    assert rep.text == "A [arXiv:2301.00001] B [arXiv:2301.00001 §Method]"


def test_validate_citations_no_citation():
    rep = validate_citations("nothing cited", ["2301.00001"])
    assert rep.valid == [] and rep.valid_rate == 1.0


# --------------------------------------------------------------- client


def test_parse_json_tolerant():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json")


def test_structured_retries_on_invalid_json(tmp_path):
    fake = ScriptedLLM(["not json at all", {"tasks": ["QA"], "contribution_type": "New Method"}])
    llm = make_llm(fake, tmp_path)
    out = llm.structured("extract", PaperExtraction)
    assert out.tasks == ["QA"] and out.contribution_type == "new_method"
    assert llm.stats.n_parse_failures == 1 and len(fake.calls) == 2
    # The corrective message contains the error and the previous reply.
    assert fake.calls[1][-2]["content"] == "not json at all"


def test_structured_gives_up(tmp_path):
    llm = make_llm(ScriptedLLM(["bad"]), tmp_path, max_parse_retries=1)
    with pytest.raises(StructuredOutputError):
        llm.structured("x", PaperExtraction)


def test_cache_avoids_redundant_calls(tmp_path):
    fake = ScriptedLLM(["hello"])
    llm = make_llm(fake, tmp_path)
    assert llm.chat("hi") == "hello" and llm.chat("hi") == "hello"
    assert len(fake.calls) == 1 and llm.stats.n_cache_hits == 1
    llm2 = make_llm(fake, tmp_path)  # cache persists on disk
    assert llm2.chat("hi") == "hello" and len(fake.calls) == 1


def test_retry_with_backoff(tmp_path):
    attempts = {"n": 0}
    delays: list[float] = []

    def flaky(messages, **kw):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientLLMError("rate limited")
        return "ok"

    llm = LLMClient("m", completion_fn=flaky, sleep=delays.append, backoff_seconds=1.0)
    assert llm.chat("q") == "ok"
    assert delays == [1.0, 2.0] and llm.stats.n_retries == 2


def test_retry_exhausted():
    def down(messages, **kw):
        raise TransientLLMError("down")

    llm = LLMClient("m", completion_fn=down, sleep=lambda s: None, max_retries=2)
    with pytest.raises(LLMError):
        llm.chat("q")
    assert llm.stats.n_calls == 3


def test_extraction_schema_cleaning():
    e = PaperExtraction.model_validate(
        {
            "tasks": ["QA", " qa ", None, ""],
            "datasets": "SQuAD",
            "contribution_type": "weird",
            "claimed_results": [{"metric": "F1", "value": 91.2}, "junk"],
        }
    )
    assert e.tasks == ["QA"] and e.datasets == ["SQuAD"] and e.contribution_type == "other"
    assert e.claimed_results[0].value == "91.2" and len(e.claimed_results) == 1


# ------------------------------------------------------------------ RAG


def test_rag_answer_valid_citations(searcher, tmp_path):
    def router(messages):
        ctx = messages[-1]["content"]
        assert "arXiv:2302.00002" in ctx
        return "Back-translation helps [arXiv:2302.00002]. Also [arXiv:0000.00000]."

    res = answer_question(
        "low-resource machine translation", searcher, make_llm(ScriptedLLM(router))
    )
    assert res.citations == ["2302.00002"] and res.invalid_citations == ["0000.00000"]
    assert "0000.00000" not in res.answer and not res.insufficient


def test_rag_insufficient_context(searcher):
    res = answer_question(
        "protein folding", searcher, make_llm(ScriptedLLM(["INSUFFICIENT_CONTEXT"]))
    )
    assert res.insufficient and res.answer == IDK_MESSAGE["en"]


def test_rag_without_valid_citation_becomes_idk(searcher):
    res = answer_question(
        "translation", searcher, make_llm(ScriptedLLM(["Made up [arXiv:1234.56789]"])), lang="fr"
    )
    assert res.insufficient and res.answer == IDK_MESSAGE["fr"]


def test_rag_paper_scope(searcher):
    res = answer_question(
        "what is evaluated?",
        searcher,
        make_llm(ScriptedLLM(["SQuAD [arXiv:2301.00001]"])),
        scope="2301.00001",
    )
    assert [c.arxiv_id for c in res.contexts] == ["2301.00001"] and res.citations == ["2301.00001"]
    with pytest.raises(KeyError):
        answer_question("q", searcher, make_llm(ScriptedLLM(["x"])), scope="0000.0000")


# -------------------------------------------------------- reading sheet


SHEET = {
    "summary": "A RAG system.",
    "contributions": ["retriever + generator"],
    "method": "Dense retrieval.",
    "results": "Better QA.",
    "limitations": ["English only (inferred)"],
    "comprehension_questions": ["Q1?", "Q2?", "Q3?", "Q4?"],
}


def test_reading_sheet_fr(paper_store):
    fake = ScriptedLLM([SHEET])
    paper = paper_store.get("2301.00001")
    sheet = generate_reading_sheet(paper, make_llm(fake), lang="fr", full_text="1 Introduction ...")
    assert isinstance(sheet, ReadingSheet) and len(sheet.comprehension_questions) == 3
    prompt = fake.calls[0][-1]["content"]
    assert "French" in prompt and "FULL-TEXT" in prompt
    md = reading_sheet_markdown(sheet, paper, lang="fr")
    assert "## Questions de compréhension" in md and "[arXiv:2301.00001]" in md


def test_reading_sheet_requires_three_questions(paper_store):
    bad = dict(SHEET, comprehension_questions=["only one?"])
    llm = make_llm(ScriptedLLM([bad]), max_parse_retries=0)
    with pytest.raises(StructuredOutputError):
        generate_reading_sheet(paper_store.get("2301.00001"), llm)
