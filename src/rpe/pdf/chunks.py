"""Chroma collection "chunks" (full-text passages) + CITES edges from references."""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from rpe.config import Config
from rpe.index.embed import Embedder
from rpe.index.hybrid import make_embedder
from rpe.index.vector_store import ChromaVectorStore
from rpe.kg.graph_store import GraphStore, node_id
from rpe.pdf.download import pdf_dir, pdf_path
from rpe.pdf.parse import chunk_sections, extract_arxiv_references, parse_pdf
from rpe.rag.qa import ContextItem

logger = logging.getLogger(__name__)


class ChunkStore:
    """Full-text chunks of downloaded papers; implements `rag.qa.ChunkRetriever`."""

    def __init__(self, cfg: Config, dev: bool = False, embedder: Embedder | None = None) -> None:
        self.cfg = cfg
        self.store = ChromaVectorStore(cfg.index_dir(dev) / "chroma", collection="chunks")
        self._embedder = embedder
        self._papers_with_chunks: set[str] | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = make_embedder(self.cfg)
        return self._embedder

    def count(self) -> int:
        return self.store.count()

    def indexed_papers(self) -> set[str]:
        if self._papers_with_chunks is None:
            metas = self.store.collection.get(include=["metadatas"])["metadatas"] or []
            self._papers_with_chunks = {m["arxiv_id"] for m in metas}
        return self._papers_with_chunks

    def has_chunks(self, arxiv_id: str) -> bool:
        return arxiv_id in self.indexed_papers()

    def retrieve(self, arxiv_id: str, query: str, k: int = 5) -> list[ContextItem]:
        vec = self.embedder.embed_query(query)
        res = self.store.collection.query(
            query_embeddings=[vec.tolist()],
            n_results=k,
            where={"arxiv_id": arxiv_id},
            include=["documents", "metadatas"],
        )
        return [
            ContextItem(
                ref=f"arXiv:{arxiv_id} §{m['section']}",
                arxiv_id=arxiv_id,
                title=m.get("title", ""),
                text=doc,
            )
            for doc, m in zip(res["documents"][0], res["metadatas"][0], strict=True)
        ]

    def add_paper(self, arxiv_id: str, title: str, chunks: list) -> int:
        if not chunks:
            return 0
        vecs = self.embedder.embed_documents([c.text for c in chunks])
        metas = [{"arxiv_id": arxiv_id, "section": c.section, "title": title} for c in chunks]
        n = self.store.add([c.chunk_id for c in chunks], vecs, metas, [c.text for c in chunks])
        self.indexed_papers().add(arxiv_id)
        return n


def index_pdfs(
    cfg: Config, dev: bool = False, graph: GraphStore | None = None, progress: bool = True
) -> dict[str, int]:
    """Parse every downloaded PDF of the corpus, index its chunks, add CITES edges."""
    from rpe.data.store import PaperStore

    papers = PaperStore.load(cfg, dev)
    store = ChunkStore(cfg, dev)
    stats = {"papers": 0, "chunks": 0, "skipped": 0, "failed": 0, "cites": 0}
    for aid in tqdm(papers.ids, desc="Index PDFs", disable=not progress):
        pdf: Path = pdf_path(cfg, aid)
        if not pdf.exists():
            continue
        try:
            sections = parse_pdf(pdf, aid, pdf_dir(cfg))
        except Exception as exc:  # noqa: BLE001 - corrupted PDFs are skipped
            logger.warning("Cannot parse %s: %s", pdf, exc)
            stats["failed"] += 1
            continue
        if graph is not None:
            for ref in extract_arxiv_references(sections, aid):
                if ref in papers:
                    graph.add_edge(node_id("Paper", aid), node_id("Paper", ref), "CITES")
                    stats["cites"] += 1
        if store.has_chunks(aid):
            stats["skipped"] += 1
            continue
        chunks = chunk_sections(aid, sections, cfg.pdfs.chunk_chars, cfg.pdfs.chunk_overlap)
        stats["chunks"] += store.add_paper(aid, papers.get(aid)["title"], chunks)
        stats["papers"] += 1
    return stats
