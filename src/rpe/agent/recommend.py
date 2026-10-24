"""Recommendation: centroid of 1-5 papers' embeddings -> nearest neighbours (inputs excluded)."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from rpe.data.store import PaperStore
from rpe.index.bm25 import tokenize
from rpe.index.vector_store import NumpyVectorStore
from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import Explanations
from rpe.rag.citations import validate_citations

logger = logging.getLogger(__name__)
MAX_SEEDS = 5


class Recommendation(BaseModel):
    arxiv_id: str
    title: str
    score: float
    year: int | None = None
    venue: str | None = None
    explanation: str = ""


def centroid(vectors: Sequence[np.ndarray]) -> np.ndarray:
    c = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm else c


def keyword_explanation(rec: dict, seeds: Sequence[dict], top: int = 3) -> str:
    """Fallback explanation from shared salient terms (no LLM)."""
    rec_terms = Counter(tokenize(rec["title"] + " " + rec.get("abstract", "")))
    best_seed, best_shared = seeds[0], []
    for s in seeds:
        seed_terms = set(tokenize(s["title"] + " " + s.get("abstract", "")))
        shared = [t for t, _ in rec_terms.most_common() if t in seed_terms][:top]
        if len(shared) > len(best_shared):
            best_seed, best_shared = s, shared
    terms = ", ".join(f"'{t}'" for t in best_shared) or "related themes"
    return (
        f"Shares {terms} with [arXiv:{best_seed['arxiv_id']}], "
        f"closest in embedding space [arXiv:{rec['arxiv_id']}]."
    )


def explain_with_llm(recs: list[dict], seeds: list[dict], llm: LLMClient) -> dict[str, str]:
    prompt = prompts.RECOMMEND_USER.format(
        seeds=prompts.format_papers(seeds, 400), recs=prompts.format_papers(recs, 400)
    )
    out = llm.structured(prompt, Explanations, system=prompts.RECOMMEND_SYSTEM)
    allowed = [p["arxiv_id"] for p in recs + seeds]
    return {
        e.arxiv_id.replace("arXiv:", "").strip(): validate_citations(e.explanation, allowed).text
        for e in out.items
    }


def recommend_similar(
    arxiv_ids: Sequence[str],
    dense: NumpyVectorStore,
    papers: PaperStore,
    k: int = 5,
    llm: LLMClient | None = None,
) -> list[Recommendation]:
    """k nearest papers to the centroid of `arxiv_ids`, each with a one-sentence reason."""
    ids = list(dict.fromkeys(i.strip() for i in arxiv_ids if i.strip()))
    if not 1 <= len(ids) <= MAX_SEEDS:
        raise ValueError(f"Give between 1 and {MAX_SEEDS} arxiv_ids (got {len(ids)})")
    vectors = [v for i in ids if (v := dense.vector(i)) is not None]
    missing = [i for i in ids if dense.vector(i) is None]
    if missing:
        raise ValueError(f"Unknown arxiv_ids (not in the index): {', '.join(missing)}")
    hits = [(i, s) for i, s in dense.query(centroid(vectors), k + len(ids)) if i not in ids][:k]
    recs = papers.many(i for i, _ in hits)
    seeds = papers.many(ids)
    explanations: dict[str, str] = {}
    if llm is not None:
        try:
            explanations = explain_with_llm(recs, seeds, llm)
        except LLMError as exc:
            logger.warning("LLM explanations failed (%s); using keyword overlap", exc)
    scores = dict(hits)
    return [
        Recommendation(
            arxiv_id=p["arxiv_id"],
            title=p["title"],
            score=round(scores[p["arxiv_id"]], 4),
            year=p.get("year"),
            venue=p.get("venue"),
            explanation=explanations.get(p["arxiv_id"]) or keyword_explanation(p, seeds),
        )
        for p in recs
    ]
