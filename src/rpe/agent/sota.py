"""State-of-the-art (literature review) generator as a LangGraph workflow.

plan (sub-questions) -> search (hybrid, per sub-question) -> select (relevance, recency,
venue, code, MMR diversity; loops back to plan with feedback if too few papers, bounded by
`sota_max_iterations`) -> group (approaches) -> write (Markdown) -> validate (citations,
references section generated from the validated ids only).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rpe.index.hybrid import HybridSearcher
from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import SotaGrouping, SotaPlan
from rpe.rag.citations import validate_citations

logger = logging.getLogger(__name__)
WEIGHTS = {"relevance": 0.55, "recency": 0.2, "venue": 0.15, "code": 0.1}


class SotaState(TypedDict, total=False):
    topic: str
    lang: str
    iteration: int
    feedback: str
    subquestions: list[str]
    candidates: dict[str, dict[str, Any]]
    selected: list[str]
    groups: list[dict[str, Any]]
    draft: str
    markdown: str
    citations: list[str]
    invalid_citations: list[str]
    trace: list[dict[str, Any]]


class SotaResult(BaseModel):
    topic: str
    markdown: str
    subquestions: list[str] = Field(default_factory=list)
    selected: list[str] = Field(default_factory=list)
    groups: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


def score_candidates(cands: dict[str, dict[str, Any]], year_now: int) -> dict[str, float]:
    """Weighted score in [0, 1]: relevance (best rank), recency, venue, code."""
    years = [c["year"] for c in cands.values() if c.get("year")]
    y_min = min(years) if years else year_now
    span = max(1, year_now - y_min)
    out = {}
    for aid, c in cands.items():
        relevance = 1.0 / c["best_rank"] ** 0.5 + 0.1 * (c["hits"] - 1)
        recency = ((c.get("year") or y_min) - y_min) / span
        out[aid] = (
            WEIGHTS["relevance"] * min(1.0, relevance)
            + WEIGHTS["recency"] * recency
            + WEIGHTS["venue"] * float(bool(c.get("is_accepted")))
            + WEIGHTS["code"] * float(bool(c.get("has_code")))
        )
    return out


def mmr_select(
    scores: dict[str, float],
    k: int,
    vectors: dict[str, np.ndarray] | None = None,
    lam: float = 0.7,
) -> list[str]:
    """Maximal Marginal Relevance on embeddings (plain top-k when vectors are missing)."""
    ranked = sorted(scores, key=lambda a: (-scores[a], a))
    if not vectors or any(a not in vectors for a in ranked):
        return ranked[:k]
    chosen: list[str] = []
    pool = ranked[: max(k * 4, k)]
    while pool and len(chosen) < k:

        def mmr(a: str) -> float:
            sim = max((float(vectors[a] @ vectors[c]) for c in chosen), default=0.0)
            return lam * scores[a] - (1 - lam) * sim

        best = max(pool, key=lambda a: (mmr(a), -ranked.index(a)))
        chosen.append(best)
        pool.remove(best)
    return chosen


def references_section(ids: list[str], papers: dict[str, dict[str, Any]]) -> str:
    lines = ["## References"]
    for aid in ids:
        p = papers.get(aid, {})
        meta = ", ".join(str(x) for x in (p.get("year"), p.get("venue")) if x)
        lines.append(
            f"- [arXiv:{aid}] {p.get('title', aid)}"
            + (f" ({meta})" if meta else "")
            + f". https://arxiv.org/abs/{aid}"
        )
    return "\n".join(lines)


def build_sota_graph(
    searcher: HybridSearcher,
    llm: LLMClient | None,
    max_papers: int = 25,
    min_papers: int = 8,
    max_iterations: int = 2,
    per_query: int = 15,
    vectors: dict[str, np.ndarray] | None = None,
    graph: Any = None,
) -> Any:
    papers = searcher.papers

    def trace(state: SotaState, node: str, detail: Any) -> list[dict[str, Any]]:
        return state.get("trace", []) + [{"node": node, "detail": detail}]

    def plan(state: SotaState) -> SotaState:
        topic = state["topic"]
        subqs: list[str] = []
        if llm is not None:
            try:
                out = llm.structured(
                    prompts.SOTA_PLAN_USER.format(
                        topic=topic, n=4, feedback=state.get("feedback", "")
                    ),
                    SotaPlan,
                )
                subqs = out.subquestions[:6]
            except LLMError as exc:
                logger.warning("SOTA planning failed: %s", exc)
        if not subqs:
            subqs = [
                topic,
                f"{topic} benchmark evaluation",
                f"{topic} survey",
                f"{topic} limitations challenges",
            ]
        if topic not in subqs:
            subqs = [topic, *subqs]
        return {
            "subquestions": subqs,
            "iteration": state.get("iteration", 0) + 1,
            "trace": trace(state, "plan", subqs),
        }

    def search(state: SotaState) -> SotaState:
        cands = dict(state.get("candidates", {}))
        for q in state["subquestions"]:
            for rank, r in enumerate(searcher.search(q, top_k=per_query), 1):
                c = cands.setdefault(
                    r.arxiv_id,
                    {
                        "best_rank": rank,
                        "hits": 0,
                        "year": r.year,
                        "is_accepted": r.is_accepted,
                        "has_code": r.has_code,
                        "queries": [],
                    },
                )
                c["best_rank"] = min(c["best_rank"], rank)
                c["hits"] += 1
                c["queries"].append(q)
        return {"candidates": cands, "trace": trace(state, "search", {"n_candidates": len(cands)})}

    def select(state: SotaState) -> SotaState:
        cands = state["candidates"]
        years = [c["year"] for c in cands.values() if c.get("year")]
        scores = score_candidates(cands, max(years) if years else 2025)
        chosen = mmr_select(scores, max_papers, vectors)
        feedback = ""
        if len(chosen) < min_papers:
            feedback = (
                f"Previous queries {state['subquestions']} found only {len(chosen)} "
                "papers: propose broader or different queries."
            )
        return {
            "selected": chosen,
            "feedback": feedback,
            "trace": trace(state, "select", {"n_selected": len(chosen)}),
        }

    def after_select(state: SotaState) -> str:
        if len(state["selected"]) < min_papers and state.get("iteration", 1) < max_iterations:
            return "plan"
        return "group"

    def group(state: SotaState) -> SotaState:
        sel = state["selected"]
        groups: list[dict[str, Any]] = []
        if llm is not None and sel:
            try:
                out = llm.structured(
                    prompts.SOTA_GROUP_USER.format(
                        topic=state["topic"], papers=prompts.format_papers(papers.many(sel), 300)
                    ),
                    SotaGrouping,
                )
                seen: set[str] = set()
                for g in out.groups:
                    ids = [
                        i
                        for i in (x.replace("arXiv:", "").strip() for x in g.arxiv_ids)
                        if i in sel and i not in seen
                    ]
                    seen.update(ids)
                    if ids:
                        groups.append(
                            {"name": g.name, "description": g.description, "arxiv_ids": ids}
                        )
                rest = [i for i in sel if i not in seen]
                if rest:
                    groups.append(
                        {"name": "Other approaches", "description": "", "arxiv_ids": rest}
                    )
            except LLMError as exc:
                logger.warning("SOTA grouping failed: %s", exc)
        if not groups:
            groups = [{"name": "Selected papers", "description": "", "arxiv_ids": sel}]
        return {
            "groups": groups,
            "trace": trace(
                state, "group", [f"{g['name']} ({len(g['arxiv_ids'])})" for g in groups]
            ),
        }

    def datasets_metrics(sel: list[str]) -> str:
        if graph is None:
            return "(knowledge graph not available)"
        counts: dict[str, Counter[str]] = {"Datasets": Counter(), "Metrics": Counter()}
        for aid in sel:
            for rel, label in (("EVALUATED_ON", "Datasets"), ("MEASURED_BY", "Metrics")):
                counts[label].update(
                    graph.node(n).get("name", n) for n in graph.neighbors(f"paper:{aid}", rel)
                )
        return "\n".join(
            f"{k}: " + (", ".join(f"{n} ({c})" for n, c in v.most_common(10)) or "none")
            for k, v in counts.items()
        )

    def write(state: SotaState) -> SotaState:
        sel, groups = state["selected"], state["groups"]
        group_txt = "\n".join(
            f"- {g['name']}: {', '.join('arXiv:' + i for i in g['arxiv_ids'])}" for g in groups
        )
        if llm is None:
            body = [f"# State of the art: {state['topic']}", "", "## Approaches"]
            for g in groups:
                body.append(f"### {g['name']}")
                body += [
                    f"- {p['title']} [arXiv:{p['arxiv_id']}]" for p in papers.many(g["arxiv_ids"])
                ]
            draft = "\n".join(body)
        else:
            draft = llm.chat(
                prompts.SOTA_WRITE_USER.format(
                    topic=state["topic"],
                    groups=group_txt,
                    datasets_metrics=datasets_metrics(sel),
                    context=prompts.format_papers(papers.many(sel), 600),
                    language=prompts.language_instruction(state.get("lang", "en")),
                ),
                system=prompts.SOTA_WRITE_SYSTEM,
                max_tokens=4096,
            )
            if not draft.lstrip().startswith("#"):
                draft = f"# State of the art: {state['topic']}\n\n{draft}"
        return {"draft": draft, "trace": trace(state, "write", {"chars": len(draft)})}

    def validate(state: SotaState) -> SotaState:
        rep = validate_citations(state["draft"], state["selected"])
        meta = {p["arxiv_id"]: p for p in papers.many(rep.valid)}
        md = rep.text.rstrip() + "\n\n" + references_section(rep.valid, meta) + "\n"
        return {
            "markdown": md,
            "citations": rep.valid,
            "invalid_citations": rep.invalid,
            "trace": trace(state, "validate", {"valid": len(rep.valid), "invalid": rep.invalid}),
        }

    g = StateGraph(SotaState)
    for name, fn in [
        ("plan", plan),
        ("search", search),
        ("select", select),
        ("group", group),
        ("write", write),
        ("validate", validate),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "select")
    g.add_conditional_edges("select", after_select, {"plan": "plan", "group": "group"})
    g.add_edge("group", "write")
    g.add_edge("write", "validate")
    g.add_edge("validate", END)
    return g.compile()


def _initial(topic: str, lang: str) -> SotaState:
    return {"topic": topic, "lang": lang, "iteration": 0, "candidates": {}, "trace": []}


def stream_sota(graph: Any, topic: str, lang: str = "en") -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (node, update) as the workflow runs (live trace for the UI)."""
    for update in graph.stream(_initial(topic, lang), stream_mode="updates"):
        yield from update.items()


def generate_sota(graph: Any, topic: str, lang: str = "en") -> SotaResult:
    state = graph.invoke(_initial(topic, lang), {"recursion_limit": 30})
    return SotaResult(
        topic=topic,
        markdown=state.get("markdown", ""),
        subquestions=state.get("subquestions", []),
        selected=state.get("selected", []),
        groups=state.get("groups", []),
        citations=state.get("citations", []),
        invalid_citations=state.get("invalid_citations", []),
        trace=state.get("trace", []),
    )
