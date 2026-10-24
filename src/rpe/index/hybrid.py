"""Hybrid search: BM25 + dense, fused with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from rpe.config import Config, load_env
from rpe.data.store import PaperStore
from rpe.index.bm25 import BM25Index, tokenize
from rpe.index.embed import Embedder, load_embeddings
from rpe.index.filters import SearchFilters
from rpe.index.vector_store import ChromaVectorStore, DenseIndex, NumpyVectorStore

SearchMode = Literal["hybrid", "bm25", "dense"]
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


class SearchResult(BaseModel):
    arxiv_id: str
    title: str
    score: float
    snippet: str
    year: int | None = None
    venue: str | None = None
    has_code: bool = False
    is_accepted: bool = False
    authors: list[str] = Field(default_factory=list)
    published_date: str | None = None
    primary_category: str | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)), ranks from 1.

    Ties are broken by best individual rank, then by id (deterministic).
    """
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best[doc_id] = min(best.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


def make_snippet(abstract: str, query: str, max_chars: int = 300) -> str:
    """Abstract sentence sharing the most terms with the query (truncated)."""
    q = set(tokenize(query))
    sentences = [s for s in _SENT_RE.split(abstract or "") if s]
    if not sentences:
        return ""
    best = max(sentences, key=lambda s: len(q.intersection(tokenize(s))))
    if not q.intersection(tokenize(best)):
        best = abstract
    return best if len(best) <= max_chars else best[: max_chars - 1].rsplit(" ", 1)[0] + "…"


class HybridSearcher:
    """Search over a paper store with BM25, dense, or both (RRF)."""

    def __init__(
        self,
        papers: PaperStore,
        bm25: BM25Index,
        dense: DenseIndex,
        embedder: Embedder,
        rrf_k: int = 60,
        candidates: int = 100,
    ) -> None:
        self.papers = papers
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidates = candidates
        self._bm25_meta = papers.df.loc[bm25.ids].reset_index(drop=True)

    def _bm25_ranking(self, query: str, filters: SearchFilters | None, n: int) -> list[str]:
        mask = None if filters is None or filters.is_empty() else filters.mask(self._bm25_meta)
        return [i for i, _ in self.bm25.search(query, n, mask)]

    def _dense_ranking(self, query: str, filters: SearchFilters | None, n: int) -> list[str]:
        return [i for i, _ in self.dense.query(self.embedder.embed_query(query), n, filters)]

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[SearchResult]:
        """Ranked `SearchResult`s; scores are RRF scores (hybrid) or rank-based."""
        n = max(self.candidates, top_k)
        bm25 = self._bm25_ranking(query, filters, n) if mode in ("hybrid", "bm25") else []
        dense = self._dense_ranking(query, filters, n) if mode in ("hybrid", "dense") else []
        rankings = [r for r in (bm25, dense) if r] if mode == "hybrid" else [bm25 or dense]
        fused = rrf_fuse(rankings, self.rrf_k)[:top_k]
        bm25_pos = {i: r for r, i in enumerate(bm25, 1)}
        dense_pos = {i: r for r, i in enumerate(dense, 1)}
        return [self._to_result(i, s, query, bm25_pos.get(i), dense_pos.get(i)) for i, s in fused]

    def _to_result(
        self, arxiv_id: str, score: float, query: str, bm25_rank: int | None, dense_rank: int | None
    ) -> SearchResult:
        p = self.papers.get(arxiv_id) or {"title": arxiv_id, "abstract": ""}
        pub = p.get("published_date")
        return SearchResult(
            arxiv_id=arxiv_id,
            title=p["title"],
            score=round(score, 6),
            snippet=make_snippet(p.get("abstract") or "", query),
            year=p.get("year"),
            venue=p.get("venue"),
            has_code=bool(p.get("has_code")),
            is_accepted=bool(p.get("is_accepted")),
            authors=list(p.get("authors_list") or []),
            published_date=str(pub) if pub else None,
            primary_category=p.get("primary_category"),
            bm25_rank=bm25_rank,
            dense_rank=dense_rank,
        )


def make_embedder(cfg: Config) -> Embedder:
    env = load_env()
    return Embedder(
        env.embedding_model, cfg.embeddings.batch_size, cfg.embeddings.query_instruction
    )


def load_searcher(cfg: Config, dev: bool = False, backend: str = "chroma") -> HybridSearcher:
    """Load papers + indexes built by `build-index` (clear error if missing)."""
    papers = PaperStore.load(cfg, dev)
    index_dir = cfg.index_dir(dev)
    bm25 = BM25Index.load(index_dir / "bm25.pkl")
    dense: DenseIndex
    if backend == "chroma":
        dense = ChromaVectorStore(index_dir / "chroma")
        if dense.count() == 0:
            raise FileNotFoundError(f"Empty Chroma index in {index_dir}. Run: rpe build-index")
    else:
        matrix, ids = load_embeddings(index_dir)
        dense = NumpyVectorStore(matrix, ids, papers.df.loc[ids])
    return HybridSearcher(
        papers, bm25, dense, make_embedder(cfg), cfg.retrieval.rrf_k, cfg.retrieval.candidates
    )
