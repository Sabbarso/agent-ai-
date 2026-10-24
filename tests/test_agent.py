import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from rpe.agent.digest import build_arxiv_query, fetch_latest_arxiv, make_digest, save_digest
from rpe.agent.gaps import cooccurrence, find_gaps, heatmap_frame, justify_gaps
from rpe.agent.graph import run_agent
from rpe.agent.recommend import centroid, recommend_similar
from rpe.agent.sota import build_sota_graph, generate_sota, mmr_select, score_candidates
from rpe.kg.build_graph import build_graph
from rpe.kg.graph_store import NetworkXGraphStore, node_id
from rpe.kg.normalize import EntityNormalizer
from rpe.llm.client import LLMClient
from rpe.llm.schemas import PaperExtraction
from tests.fakes import ScriptedLLM


def llm_of(fn) -> LLMClient:
    return LLMClient("m", completion_fn=ScriptedLLM(fn), sleep=lambda s: None)


# --------------------------------------------------------------- recommend


def test_recommend_excludes_inputs_and_explains(numpy_store, paper_store):
    recs = recommend_similar(["2301.00001", "2506.00006"], numpy_store, paper_store, k=3)
    ids = [r.arxiv_id for r in recs]
    assert len(ids) == 3 and not {"2301.00001", "2506.00006"} & set(ids)
    assert all(f"[arXiv:{r.arxiv_id}]" in r.explanation for r in recs)


def test_recommend_validates_input(numpy_store, paper_store):
    with pytest.raises(ValueError):
        recommend_similar([], numpy_store, paper_store)
    with pytest.raises(ValueError):
        recommend_similar(["a", "b", "c", "d", "e", "f"], numpy_store, paper_store)
    with pytest.raises(ValueError, match="Unknown"):
        recommend_similar(["9999.99999"], numpy_store, paper_store)


def test_recommend_llm_explanations_are_validated(numpy_store, paper_store):
    def router(messages):
        return {
            "items": [
                {
                    "arxiv_id": "2404.00004",
                    "explanation": "Same QA checks [arXiv:2404.00004] [arXiv:1111.11111]",
                }
            ]
        }

    recs = recommend_similar(["2301.00001"], numpy_store, paper_store, k=5, llm=llm_of(router))
    expl = {r.arxiv_id: r.explanation for r in recs}
    if "2404.00004" in expl:
        assert "1111.11111" not in expl["2404.00004"]


def test_centroid_normalised():
    import numpy as np

    c = centroid([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert abs(np.linalg.norm(c) - 1) < 1e-6


# -------------------------------------------------------------------- gaps


def gap_graph() -> NetworkXGraphStore:
    """Tasks T1/T2 and methods M1/M2 are frequent; (T1, M2) never co-occur."""
    g = NetworkXGraphStore()
    combos = [("T1", "M1")] * 6 + [("T2", "M2")] * 6 + [("T2", "M1")] * 2
    for i, (t, m) in enumerate(combos):
        pid = node_id("Paper", f"2401.{i:05d}")
        g.add_node(pid, "Paper", f"paper {i}", year=2024)
        for nid, typ, rel in (
            (f"task:{t.lower()}", "Task", "ADDRESSES"),
            (f"method:{m.lower()}", "Method", "PROPOSES_OR_USES"),
        ):
            g.add_node(nid, typ, t if typ == "Task" else m)
            g.add_edge(pid, nid, rel)
    return g


def test_find_gaps_rare_but_expected():
    gaps = find_gaps(gap_graph(), min_task_papers=5, min_method_papers=5, max_cooccurrence=0)
    pairs = [(g.task, g.method) for g in gaps]
    assert ("T1", "M2") in pairs
    assert ("T1", "M1") not in pairs and ("T2", "M2") not in pairs
    g = next(x for x in gaps if (x.task, x.method) == ("T1", "M2"))
    assert g.n_both == 0 and g.expected > 0 and g.task_papers and g.method_papers


def test_gaps_task_filter_and_heatmap():
    store = gap_graph()
    assert all(g.task == "T2" for g in find_gaps(store, 5, 5, 0, task="T2"))
    frame = heatmap_frame(cooccurrence(store))
    assert frame.loc["T1", "M1"] == 6 and frame.loc["T1", "M2"] == 0


def test_gap_justification_is_hypothesis(paper_store):
    gaps = find_gaps(gap_graph(), 5, 5, 0)
    out = justify_gaps(gaps[:1], paper_store, None)
    assert out[0].hypothesis.lower().startswith("hypothesis")


# ------------------------------------------------------------------ digest


def test_build_arxiv_query():
    assert build_arxiv_query("LLM agents") == "all:LLM AND all:agents"
    assert build_arxiv_query('"tool use"', ["cs.CL", "cs.AI"]) == (
        '(cat:cs.CL OR cat:cs.AI) AND (all:"tool use")'
    )


class FakeArxivClient:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def results(self, search):
        for i, age in enumerate([1, 5, 30, 50]):  # hours
            yield SimpleNamespace(
                get_short_id=lambda i=i: f"2610.0000{i}v1",
                title=f"Agent paper {i}",
                summary="An agent  paper.\n",
                authors=[SimpleNamespace(name="A. B")],
                published=self.now - timedelta(hours=age),
                categories=["cs.CL"],
                entry_id=f"http://arxiv.org/abs/2610.0000{i}v1",
            )


def test_fetch_latest_and_digest(tmp_path):
    now = datetime(2026, 10, 24, 12, tzinfo=UTC)
    papers = fetch_latest_arxiv("LLM agents", days=1, client=FakeArxivClient(now), now=now)
    assert [p.arxiv_id for p in papers] == ["2610.00000", "2610.00001"]

    def router(messages):
        return {
            "themes": [
                {
                    "theme": "Agents",
                    "summary": "Agents [arXiv:2610.00000] [arXiv:9999.0001]",
                    "arxiv_ids": ["2610.00000"],
                }
            ]
        }

    d = make_digest("LLM agents", 1, papers, llm_of(router))
    assert d.invalid_citations == ["9999.0001"]
    assert [t.theme for t in d.themes] == ["Agents", "Other"]  # unassigned paper kept
    path = save_digest(d, tmp_path)
    assert path.read_text(encoding="utf-8").count("arxiv.org/abs/") == 2


def test_digest_empty():
    d = make_digest("nothing", 1, [], None)
    assert "No new papers" in d.markdown


# ------------------------------------------------------------------- agent


def make_tools(seen):
    from langchain_core.tools import StructuredTool

    def search_papers(query: str) -> str:
        """Search."""
        seen.append(query)
        return json.dumps([{"arxiv_id": "2301.00001", "title": "RAG for QA"}])

    return [StructuredTool.from_function(search_papers)]


def test_agent_tool_then_answer_with_validated_citations():
    seen: list[str] = []
    replies = [
        {"thought": "search", "action": "search_papers", "args": {"query": "rag"}},
        {
            "thought": "done",
            "action": "final_answer",
            "answer": "RAG helps QA [arXiv:2301.00001]; also [arXiv:2402.99999].",
        },
    ]
    res = run_agent("What is RAG?", make_tools(seen), llm_of(replies), max_iterations=3)
    assert seen == ["rag"] and [s.tool for s in res.steps] == ["search_papers"]
    assert res.citations == ["2301.00001"] and res.invalid_citations == ["2402.99999"]
    assert "2402.99999" not in res.answer


def test_agent_respects_max_iterations():
    seen: list[str] = []

    def router(messages):
        if "Maximum number of tool calls" in messages[-1]["content"]:
            return {"action": "final_answer", "answer": "Partial [arXiv:2301.00001]"}
        return {"action": "search_papers", "args": {"query": f"q{len(messages)}"}}

    res = run_agent("loop forever", make_tools(seen), llm_of(router), max_iterations=2)
    assert len(res.steps) == 2 and len(seen) == 2 and res.hit_max_iterations
    assert res.citations == ["2301.00001"]


def test_agent_unknown_tool_is_reported():
    replies = [
        {"action": "nope", "args": {}},
        {"action": "final_answer", "answer": "no citation"},
    ]
    res = run_agent("q", make_tools([]), llm_of(replies), max_iterations=3)
    assert "unknown tool" in res.steps[0].result_summary


# -------------------------------------------------------------------- SOTA


def test_score_and_mmr():
    cands = {
        "a": {"best_rank": 1, "hits": 2, "year": 2025, "is_accepted": True, "has_code": True},
        "b": {"best_rank": 5, "hits": 1, "year": 2020, "is_accepted": False, "has_code": False},
    }
    s = score_candidates(cands, 2025)
    assert s["a"] > s["b"]
    assert mmr_select(s, 1) == ["a"]


def test_sota_workflow_nodes_and_citations(searcher, papers_df):
    def router(messages):
        user = messages[-1]["content"]
        if "plan a literature review" in user:
            return {"subquestions": ["question answering retrieval", "machine translation"]}
        if "Group the following papers" in user:
            return {"groups": [{"name": "Retrieval", "arxiv_ids": ["2301.00001", "2506.00006"]}]}
        return (
            "## Introduction\nRAG [arXiv:2301.00001]. Fake [arXiv:2999.00001].\n"
            "## Approaches\n### Retrieval\nDense [arXiv:2506.00006]."
        )

    ext = {"2301.00001": PaperExtraction(datasets=["SQuAD"], metrics=["EM"])}
    graph = build_graph(papers_df, ext, EntityNormalizer())
    wf = build_sota_graph(searcher, llm_of(router), max_papers=4, min_papers=2, graph=graph)
    res = generate_sota(wf, "retrieval-augmented generation")
    assert [t["node"] for t in res.trace] == [
        "plan",
        "search",
        "select",
        "group",
        "write",
        "validate",
    ]
    assert len(res.selected) <= 4
    assert set(res.citations) == {"2301.00001", "2506.00006"} and res.invalid_citations == [
        "2999.00001"
    ]
    assert "## References" in res.markdown and "2999.00001" not in res.markdown
    assert any(g["name"] == "Other approaches" for g in res.groups) or len(res.selected) == 2


def test_sota_replans_when_too_few_papers(searcher):
    calls = {"plan": 0}

    def router(messages):
        user = messages[-1]["content"]
        if "plan a literature review" in user:
            calls["plan"] += 1
            return {"subquestions": ["zzz"]}
        if "Group" in user:
            return {"groups": []}
        return "## Introduction\nNothing."

    wf = build_sota_graph(searcher, llm_of(router), max_papers=10, min_papers=50, max_iterations=2)
    res = generate_sota(wf, "topic")
    assert calls["plan"] == 2  # bounded by max_iterations
    assert [t["node"] for t in res.trace].count("plan") == 2


def test_sota_without_llm(searcher):
    res = generate_sota(build_sota_graph(searcher, None, max_papers=3, min_papers=1), "translation")
    assert res.citations and "## References" in res.markdown
