"""Typed LangChain tools exposed to the agent, backed by lazily-loaded resources.

Every tool returns a compact JSON string; errors (missing index, unknown id, ...) are returned
as `{"error": ...}` so that the agent can recover instead of crashing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from functools import cached_property
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from rpe.config import Config
from rpe.data.store import PaperStore
from rpe.index.filters import SearchFilters
from rpe.llm.client import LLMClient, get_llm

logger = logging.getLogger(__name__)
MAX_TOOL_CHARS = 4000


class Resources:
    """Lazy container for indexes, graph, topics and LLM (shared by CLI, agent and UI)."""

    def __init__(self, cfg: Config, dev: bool = False, llm: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.dev = dev
        self._llm = llm

    @cached_property
    def llm(self) -> LLMClient | None:
        return self._llm or get_llm(self.cfg)

    @cached_property
    def papers(self) -> PaperStore:
        return PaperStore.load(self.cfg, self.dev)

    @cached_property
    def searcher(self) -> Any:
        from rpe.index.hybrid import load_searcher

        return load_searcher(self.cfg, self.dev)

    @cached_property
    def dense(self) -> Any:
        from rpe.index.embed import load_embeddings
        from rpe.index.vector_store import NumpyVectorStore

        matrix, ids = load_embeddings(self.cfg.index_dir(self.dev))
        return NumpyVectorStore(matrix, ids, self.papers.df.loc[ids])

    @cached_property
    def graph(self) -> Any:
        from rpe.kg.build_graph import load_graph

        return load_graph(self.cfg, self.dev)

    @cached_property
    def normalizer(self) -> Any:
        from rpe.kg.build_graph import load_normalizer

        return load_normalizer(self.cfg)

    @cached_property
    def topics(self) -> Any:
        from rpe.topics.bertopic_model import load_topics

        return load_topics(self.cfg, self.dev)

    @cached_property
    def chunks(self) -> Any:
        """Full-text chunk store (phase 7) or None if not built."""
        try:
            from rpe.pdf.chunks import ChunkStore

            store = ChunkStore(self.cfg, self.dev)
            return store if store.count() else None
        except Exception as exc:  # noqa: BLE001 - optional component
            logger.debug("No chunk store: %s", exc)
            return None


def _dump(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if len(text) <= MAX_TOOL_CHARS else text[: MAX_TOOL_CHARS - 20] + '..."[truncated]'


def _safe(fn: Callable[..., Any]) -> Callable[..., str]:
    def wrapper(**kwargs: Any) -> str:
        try:
            return _dump(fn(**kwargs))
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            logger.warning("Tool %s failed: %s", fn.__name__, exc)
            return _dump({"error": f"{type(exc).__name__}: {exc}"})

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ----------------------------------------------------------- arg schemas


class SearchArgs(BaseModel):
    query: str = Field(description="search query")
    year_min: int | None = Field(None, description="minimum publication year")
    year_max: int | None = Field(None, description="maximum publication year")
    category: str | None = Field(None, description="arXiv category, e.g. cs.CL")
    venue: str | None = Field(None, description="venue, e.g. ACL, NeurIPS")
    has_code: bool | None = Field(None, description="only papers with a GitHub link")
    top_k: int = Field(8, ge=1, le=20)


class PaperArgs(BaseModel):
    arxiv_id: str = Field(description="arXiv id, e.g. 2305.12345")


class GraphArgs(BaseModel):
    entity: str = Field(description="entity name, e.g. SQuAD or machine translation")
    relation: str = Field(
        "ADDRESSES",
        description="WRITTEN_BY | ADDRESSES | PROPOSES_OR_USES | EVALUATED_ON | MEASURED_BY | "
        "IN_CATEGORY | CO_AUTHOR | CITES",
    )
    target_type: str = Field(
        "Paper", description="Paper | Method | Dataset | Metric | Task | Author"
    )


class RecommendArgs(BaseModel):
    arxiv_ids: list[str] = Field(description="1 to 5 arXiv ids the user likes")
    k: int = Field(5, ge=1, le=20)


class TrendsArgs(BaseModel):
    topic: str | None = Field(None, description="topic keywords, or null for the overview")


class GapsArgs(BaseModel):
    task: str | None = Field(None, description="task name, or null for all tasks")


class LatestArgs(BaseModel):
    query: str
    days: int = Field(7, ge=1, le=60)


# ------------------------------------------------------------------ tools


def build_tools(res: Resources) -> list[BaseTool]:
    def search_papers(
        query: str,
        year_min: int | None = None,
        year_max: int | None = None,
        category: str | None = None,
        venue: str | None = None,
        has_code: bool | None = None,
        top_k: int = 8,
    ) -> Any:
        """Hybrid (BM25 + dense) search over the local arXiv corpus, with optional filters."""
        f = SearchFilters(
            year_min=year_min,
            year_max=year_max,
            categories=[category] if category else None,
            venue=venue,
            has_code=has_code,
        )
        return [
            r.model_dump(include={"arxiv_id", "title", "year", "venue", "has_code", "snippet"})
            for r in res.searcher.search(query, top_k=top_k, filters=f)
        ]

    def get_paper(arxiv_id: str) -> Any:
        """Metadata and abstract of one paper of the corpus."""
        p = res.papers.get(arxiv_id.replace("arXiv:", "").strip())
        if p is None:
            return {"error": f"unknown arxiv_id {arxiv_id}"}
        keep = (
            "arxiv_id",
            "title",
            "abstract",
            "authors_list",
            "year",
            "venue",
            "is_accepted",
            "github_url",
            "categories",
        )
        return {k: p.get(k) for k in keep}

    def graph_query(entity: str, relation: str = "ADDRESSES", target_type: str = "Paper") -> Any:
        """Query the knowledge graph: papers or entities linked to an entity by a relation."""
        from rpe.kg.graphrag import graph_query as gq

        r = gq(
            res.graph,
            entity,
            relation,
            target_type=target_type,
            limit=20,
            normalizer=res.normalizer,
        )
        papers = [
            {"arxiv_id": p["arxiv_id"], "title": p["title"], "year": p.get("year")}
            for p in res.papers.many(r.papers[:10])
        ]
        return {
            "matched": r.entity_name,
            "type": r.entity_type,
            "relation": r.relation,
            "n_papers": len(r.papers),
            "papers": papers,
            "entities": r.entities[:20],
        }

    def recommend_similar(arxiv_ids: list[str], k: int = 5) -> Any:
        """Papers similar to 1-5 given papers (embedding centroid), with explanations."""
        from rpe.agent.recommend import recommend_similar as rec

        return [r.model_dump() for r in rec(arxiv_ids, res.dense, res.papers, k, res.llm)]

    def get_trends(topic: str | None = None) -> Any:
        """Topic trends: emerging/declining topics, or the monthly series of one topic."""
        from rpe.topics.trends import build_trend_report, match_topics, trends_config

        topics, info = res.topics
        ids = match_topics(info, topic) if topic else None
        rep = build_trend_report(topics, res.papers.df, trends_config(res.cfg.trends, res.dev), ids)
        out = rep.model_dump(include={"recent_months", "emerging", "declining"})
        out["topics"] = [t.model_dump() for t in rep.topics[:10]]
        if topic:
            out["series"] = rep.series[-36:]
        return out

    def find_research_gaps(task: str | None = None) -> Any:
        """Candidate research gaps (rare Task x Method combinations), as hypotheses."""
        from rpe.agent.gaps import find_gaps, justify_gaps

        g = res.cfg.gaps
        gaps = find_gaps(
            res.graph,
            g.min_task_papers,
            g.min_method_papers,
            g.max_cooccurrence,
            min(g.top_n, 5),
            task,
            res.normalizer,
        )
        return [x.model_dump() for x in justify_gaps(gaps, res.papers, res.llm)]

    def read_pdf(arxiv_id: str) -> Any:
        """Full text of a paper by section (downloads the arXiv PDF if needed)."""
        from rpe.pdf.reader import read_paper_sections

        sections = read_paper_sections(res.cfg, arxiv_id.replace("arXiv:", "").strip())
        return {"arxiv_id": arxiv_id, "sections": {s.name: s.text[:700] for s in sections}}

    def fetch_latest_arxiv(query: str, days: int = 7) -> Any:
        """Newest papers on arXiv for a query over the last N days (live arXiv API)."""
        from rpe.agent.digest import fetch_latest_arxiv as fetch

        papers = fetch(query, days, max_results=15, categories=res.cfg.digest.categories)
        return [p.model_dump(include={"arxiv_id", "title", "published"}) for p in papers]

    specs = [
        (search_papers, SearchArgs),
        (get_paper, PaperArgs),
        (graph_query, GraphArgs),
        (recommend_similar, RecommendArgs),
        (get_trends, TrendsArgs),
        (find_research_gaps, GapsArgs),
        (read_pdf, PaperArgs),
        (fetch_latest_arxiv, LatestArgs),
    ]
    return [
        StructuredTool.from_function(
            func=_safe(fn), name=fn.__name__, description=fn.__doc__ or "", args_schema=schema
        )
        for fn, schema in specs
    ]


def describe_tools(tools: list[BaseTool]) -> str:
    """Tool list for the agent prompt: name(args): description."""
    lines = []
    for t in tools:
        props = (t.args_schema.model_json_schema() if t.args_schema else {}).get("properties", {})
        args = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {t.name}({args}): {t.description}")
    return "\n".join(lines)
