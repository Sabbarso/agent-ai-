from __future__ import annotations

import pytest

from rpe.data.store import PaperStore
from rpe.index.bm25 import BM25Index
from rpe.index.embed import Embedder, embedding_text
from rpe.index.hybrid import HybridSearcher
from rpe.index.vector_store import NumpyVectorStore
from tests.fakes import HashEncoder, make_papers_df


@pytest.fixture
def papers_df():
    return make_papers_df()


@pytest.fixture
def paper_store(papers_df):
    return PaperStore(papers_df)


@pytest.fixture
def embedder():
    return Embedder("fake-hash", batch_size=8, encoder=HashEncoder())


@pytest.fixture
def numpy_store(papers_df, embedder):
    texts = [embedding_text(t, a) for t, a in zip(papers_df.title, papers_df.abstract, strict=True)]
    matrix = embedder.embed_documents(texts)
    return NumpyVectorStore(matrix, papers_df.arxiv_id.tolist(), papers_df)


@pytest.fixture
def searcher(paper_store, papers_df, numpy_store, embedder):
    texts = [embedding_text(t, a) for t, a in zip(papers_df.title, papers_df.abstract, strict=True)]
    bm25 = BM25Index.build(papers_df.arxiv_id.tolist(), texts)
    return HybridSearcher(paper_store, bm25, numpy_store, embedder, rrf_k=60, candidates=20)
