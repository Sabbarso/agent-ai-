"""`build-index`: embeddings -> Chroma -> BM25 (each step resumable)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from rpe.config import Config
from rpe.data.store import PaperStore
from rpe.index.bm25 import BM25Index
from rpe.index.embed import build_embeddings, embedding_text
from rpe.index.filters import paper_metadata
from rpe.index.hybrid import make_embedder
from rpe.index.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    n_papers: int
    embedding_dim: int
    n_added_chroma: int
    chroma_count: int
    bm25_rebuilt: bool


def build_index(cfg: Config, dev: bool = False, force: bool = False) -> IndexReport:
    papers = PaperStore.load(cfg, dev)
    df = papers.df
    ids = df["arxiv_id"].tolist()
    texts = [embedding_text(t, a) for t, a in zip(df["title"], df["abstract"], strict=True)]
    out = cfg.index_dir(dev)

    embedder = make_embedder(cfg)
    matrix = build_embeddings(
        ids, texts, embedder, out, cfg.embeddings.checkpoint_every, force=force
    )

    store = ChromaVectorStore(out / "chroma")
    if force and store.count():
        store.client.delete_collection("papers")
        store = ChromaVectorStore(out / "chroma")
    metas = [paper_metadata(r) for r in df.to_dict("records")]
    added = store.add(ids, matrix, metas, documents=df["title"].tolist())

    bm25_path, bm25_ids = out / "bm25.pkl", out / "bm25_ids.json"
    rebuilt = (
        force
        or not bm25_path.exists()
        or (not bm25_ids.exists() or json.loads(bm25_ids.read_text()) != ids)
    )
    if rebuilt:
        BM25Index.build(ids, texts).save(bm25_path)
        bm25_ids.write_text(json.dumps(ids))
    return IndexReport(len(ids), int(matrix.shape[1]), added, store.count(), rebuilt)
