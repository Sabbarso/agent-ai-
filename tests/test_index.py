import json

import numpy as np
import pytest

from rpe.index.bm25 import BM25Index, tokenize
from rpe.index.embed import Embedder, build_embeddings, load_embeddings
from rpe.index.filters import SearchFilters, paper_metadata
from rpe.index.hybrid import make_snippet, rrf_fuse
from tests.fakes import HashEncoder


def test_tokenize_removes_stopwords_and_plurals():
    assert tokenize("The Models of Translation, and BLEU!") == ["model", "translation", "bleu"]


def test_rrf_fuse_basic_and_ties():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [i for i, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    # a and b tie on score and best rank -> id order
    assert [i for i, _ in fused] == ["a", "b", "c", "d"]


def test_rrf_rewards_agreement():
    fused = dict(rrf_fuse([["x", "y"], ["y", "z"]], k=1))
    assert fused["y"] > fused["x"] > fused["z"]


def test_bm25_search(papers_df):
    texts = (papers_df.title + ". " + papers_df.abstract).tolist()
    idx = BM25Index.build(papers_df.arxiv_id.tolist(), texts)
    top = [i for i, _ in idx.search("machine translation", 3)]
    assert set(top[:2]) == {"2302.00002", "2405.00005"}
    assert idx.search("zzzz", 3) == []


def test_bm25_save_load(tmp_path, papers_df):
    idx = BM25Index.build(papers_df.arxiv_id.tolist(), papers_df.title.tolist())
    idx.save(tmp_path / "bm25.pkl")
    again = BM25Index.load(tmp_path / "bm25.pkl")
    assert again.search("translation", 2) == idx.search("translation", 2)


def test_filters_mask(papers_df):
    def ids(f):
        return papers_df.arxiv_id[f.mask(papers_df)].tolist()

    assert ids(SearchFilters(year_min=2024)) == [
        "2403.00003",
        "2404.00004",
        "2405.00005",
        "2506.00006",
    ]
    assert ids(SearchFilters(has_code=True, year_max=2024)) == ["2301.00001", "2403.00003"]
    assert ids(SearchFilters(venue="EMNLP")) == ["2302.00002"]
    assert ids(SearchFilters(venue="EMNLP Findings")) == ["2302.00002"]
    assert ids(SearchFilters(categories=["cs.AI", "cs.IR"])) == ["2403.00003", "2506.00006"]
    assert SearchFilters().is_empty()


def test_filters_chroma_where():
    assert SearchFilters().to_chroma_where() is None
    assert SearchFilters(has_code=True).to_chroma_where() == {"has_code": True}
    where = SearchFilters(year_min=2024, categories=["cs.CL", "cs.AI"]).to_chroma_where()
    assert where == {
        "$and": [{"year": {"$gte": 2024}}, {"$or": [{"cat_cs.CL": True}, {"cat_cs.AI": True}]}]
    }


def test_paper_metadata_no_none(papers_df):
    meta = paper_metadata(papers_df.to_dict("records")[3])
    assert None not in meta.values() and meta["venue"] == "" and meta["cat_cs.CL"] is True


def test_hybrid_search_relevance_and_filters(searcher):
    res = searcher.search("machine translation BLEU", top_k=3)
    assert res[0].arxiv_id in {"2302.00002", "2405.00005"}
    assert res[0].bm25_rank is not None and res[0].dense_rank is not None
    filtered = searcher.search("question answering", top_k=5, filters=SearchFilters(year_min=2025))
    assert [r.arxiv_id for r in filtered] == ["2506.00006"]
    coded = searcher.search("language model", top_k=10, filters=SearchFilters(has_code=True))
    assert all(r.has_code for r in coded)


def test_hybrid_search_deterministic(searcher):
    a = [r.model_dump() for r in searcher.search("retrieval question answering", top_k=5)]
    b = [r.model_dump() for r in searcher.search("retrieval question answering", top_k=5)]
    assert a == b


def test_search_modes(searcher):
    for mode in ("bm25", "dense", "hybrid"):
        assert searcher.search("translation", top_k=2, mode=mode)


def test_snippet_picks_relevant_sentence():
    abstract = "First sentence about cats. Second sentence on machine translation quality."
    assert make_snippet(abstract, "translation").startswith("Second")


def test_build_embeddings_resumable(tmp_path):
    emb = Embedder("fake", batch_size=2, encoder=HashEncoder(16))
    ids, texts = ["a", "b", "c"], ["neural translation", "translation quality", "speech models"]
    m1 = build_embeddings(ids, texts, emb, tmp_path, checkpoint_every=2)
    assert m1.shape == (3, 16)
    assert len(list((tmp_path / "emb_parts").glob("part_*.npy"))) == 2
    (tmp_path / "embeddings.npy").unlink()  # simulate crash after parts were written
    m2 = build_embeddings(ids, texts, emb, tmp_path, checkpoint_every=2)
    np.testing.assert_allclose(m1, m2)
    matrix, loaded_ids = load_embeddings(tmp_path)
    assert loaded_ids == ids and json.loads((tmp_path / "ids.json").read_text()) == ids
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5)


def test_chroma_store_roundtrip(tmp_path, papers_df, numpy_store):
    pytest.importorskip("chromadb")
    from rpe.index.vector_store import ChromaVectorStore

    store = ChromaVectorStore(tmp_path / "chroma")
    metas = [paper_metadata(r) for r in papers_df.to_dict("records")]
    ids = papers_df.arxiv_id.tolist()
    assert store.add(ids, numpy_store.matrix, metas) == 6
    assert store.add(ids, numpy_store.matrix, metas) == 0  # resumable
    q = numpy_store.matrix[1]
    assert store.query(q, 1)[0][0] == ids[1]
    res = store.query(q, 5, SearchFilters(has_code=True))
    assert {i for i, _ in res} <= {"2301.00001", "2403.00003", "2506.00006"}


def test_parquet_roundtrip_list_columns(tmp_path, papers_df):
    """List columns come back from Parquet as numpy arrays: metadata/graph must cope."""
    import pandas as pd

    from rpe.kg.build_graph import build_graph
    from rpe.kg.normalize import EntityNormalizer

    papers_df.to_parquet(tmp_path / "p.parquet", index=False)
    df = pd.read_parquet(tmp_path / "p.parquet")
    meta = paper_metadata(df.to_dict("records")[1])
    assert meta["cat_cs.LG"] is True
    g = build_graph(df, {}, EntityNormalizer())
    assert g.count_by_type()["Author"] >= 3
