"""Research-gap detector on the Task x Method co-occurrence matrix of the graph.

A gap candidate is a (task, method) pair whose task AND method are both frequent
(>= min papers) but that co-occur rarely (<= max_cooccurrence). Score = expected / (observed+1)
with expected = n_task * n_method / N (independence). The LLM explains each candidate as a
HYPOTHESIS, citing papers of the task and of the method as evidence.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field

from rpe.data.store import PaperStore
from rpe.kg.graph_store import GraphStore
from rpe.kg.graphrag import find_entity
from rpe.kg.normalize import EntityNormalizer
from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import GapJustification
from rpe.rag.citations import validate_citations

logger = logging.getLogger(__name__)


class Gap(BaseModel):
    task: str
    method: str
    n_task: int
    n_method: int
    n_both: int
    expected: float
    score: float
    task_papers: list[str] = Field(default_factory=list)
    method_papers: list[str] = Field(default_factory=list)
    hypothesis: str = ""
    rationale: str = ""
    citations: list[str] = Field(default_factory=list)


@dataclass
class CooccurrenceData:
    pairs: Counter[tuple[str, str]]  # (task node, method node) -> number of papers (sparse)
    task_papers: dict[str, list[str]]
    method_papers: dict[str, list[str]]
    n_papers: int
    names: dict[str, str]


def cooccurrence(store: GraphStore) -> CooccurrenceData:
    """Count papers per task, per method, and per (task, method) pair."""
    pairs: Counter[tuple[str, str]] = Counter()
    task_papers: dict[str, list[str]] = {}
    method_papers: dict[str, list[str]] = {}
    n = 0
    for p in store.nodes("Paper"):
        tasks = store.neighbors(p, "ADDRESSES")
        methods = store.neighbors(p, "PROPOSES_OR_USES")
        if not tasks or not methods:
            continue
        n += 1
        aid = p.split(":", 1)[1]
        for t in tasks:
            task_papers.setdefault(t, []).append(aid)
        for m in methods:
            method_papers.setdefault(m, []).append(aid)
        pairs.update((t, m) for t in tasks for m in methods)
    names = {nid: store.node(nid).get("name", nid) for nid in [*task_papers, *method_papers]}
    return CooccurrenceData(pairs, task_papers, method_papers, n, names)


def _recent_first(ids: list[str]) -> list[str]:
    return sorted(ids, reverse=True)  # new-style arXiv ids sort chronologically


def find_gaps(
    store: GraphStore,
    min_task_papers: int = 8,
    min_method_papers: int = 8,
    max_cooccurrence: int = 1,
    top_n: int = 10,
    task: str | None = None,
    normalizer: EntityNormalizer | None = None,
    data: CooccurrenceData | None = None,
) -> list[Gap]:
    """Rank rare-but-expected (task, method) combinations."""
    data = data or cooccurrence(store)
    if data.n_papers == 0:
        return []
    tasks = [t for t, ps in data.task_papers.items() if len(ps) >= min_task_papers]
    if task:
        nid = find_entity(store, task, "Task", normalizer)
        tasks = [nid] if nid in data.task_papers else []
    methods = [m for m, ps in data.method_papers.items() if len(ps) >= min_method_papers]
    gaps = []
    for t in tasks:
        nt = len(data.task_papers[t])
        for m in methods:
            nb = data.pairs.get((t, m), 0)
            nm = len(data.method_papers[m])
            expected = nt * nm / data.n_papers
            if nb > max_cooccurrence or expected <= nb:
                continue
            gaps.append(
                Gap(
                    task=data.names[t],
                    method=data.names[m],
                    n_task=nt,
                    n_method=nm,
                    n_both=nb,
                    expected=round(expected, 2),
                    score=round(expected / (nb + 1), 3),
                    task_papers=_recent_first(data.task_papers[t])[:3],
                    method_papers=_recent_first(data.method_papers[m])[:3],
                )
            )
    gaps.sort(key=lambda g: (-g.score, g.task, g.method))
    return gaps[:top_n]


def template_justification(g: Gap) -> tuple[str, str]:
    hyp = f"Hypothesis: applying {g.method} to {g.task} may be under-explored in this corpus."
    cites = ", ".join(f"arXiv:{i}" for i in g.task_papers[:2] + g.method_papers[:2])
    why = (
        f"{g.task} appears in {g.n_task} papers and {g.method} in {g.n_method}, but only "
        f"{g.n_both} combine them (about {g.expected:.1f} expected if independent). This may "
        f"also reflect sampling or extraction noise [{cites}]."
    )
    return hyp, why


def justify_gaps(gaps: list[Gap], papers: PaperStore, llm: LLMClient | None) -> list[Gap]:
    """Fill `hypothesis` / `rationale` (LLM, citations validated; template fallback)."""
    for g in gaps:
        evidence = g.task_papers + g.method_papers
        hyp, why = template_justification(g)
        if llm is not None:
            prompt = prompts.GAP_USER.format(
                task=g.task,
                n_task=g.n_task,
                method=g.method,
                n_method=g.n_method,
                n_both=g.n_both,
                expected=g.expected,
                task_papers=prompts.format_papers(papers.many(g.task_papers), 300),
                method_papers=prompts.format_papers(papers.many(g.method_papers), 300),
            )
            try:
                out = llm.structured(prompt, GapJustification, system=prompts.GAP_SYSTEM)
                hyp, why = out.hypothesis, out.rationale
            except LLMError as exc:
                logger.warning("Gap justification failed (%s); using template", exc)
        rep = validate_citations(why, evidence)
        g.hypothesis, g.rationale, g.citations = hyp, rep.text, rep.valid
    return gaps


def heatmap_frame(
    data: CooccurrenceData, top_tasks: int = 15, top_methods: int = 20
) -> pd.DataFrame:
    """Task x Method counts restricted to the most frequent rows/columns (display names)."""
    t = sorted(data.task_papers, key=lambda k: -len(data.task_papers[k]))[:top_tasks]
    m = sorted(data.method_papers, key=lambda k: -len(data.method_papers[k]))[:top_methods]
    frame = pd.DataFrame(
        [[data.pairs.get((ti, mi), 0) for mi in m] for ti in t], index=t, columns=m, dtype=int
    )
    return frame.rename(index=data.names, columns=data.names)
