import math

import pytest

from rpe.eval.extraction_eval import evaluate_extraction, prf
from rpe.eval.rag_eval import evaluate_rag
from rpe.eval.retrieval_eval import evaluate_retrieval, ndcg_at_k, recall_at_k, reciprocal_rank
from rpe.kg.normalize import DEFAULT_ALIASES, EntityNormalizer
from rpe.llm.client import LLMClient
from rpe.llm.schemas import PaperExtraction
from tests.fakes import ScriptedLLM


def test_retrieval_metrics():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["b", "z"], 2) == 0.5
    assert reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["z"]) == 0.0
    assert ndcg_at_k(ranked, ["a"], 10) == 1.0
    assert ndcg_at_k(ranked, ["b"], 10) == pytest.approx(1 / math.log2(3))


def test_evaluate_retrieval_table(searcher):
    queries = [
        {"query": "machine translation BLEU", "relevant": ["2302.00002"]},
        {"query": "dense passage retrieval", "relevant": ["2506.00006"]},
    ]
    table = evaluate_retrieval(searcher, queries)
    assert list(table.index) == ["bm25", "dense", "hybrid"]
    assert table.loc["bm25", "Recall@10"] == 1.0 and 0 < table.loc["hybrid", "MRR"] <= 1


def test_prf_edge_cases():
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)
    assert prf(1, 2, 4) == pytest.approx((0.5, 0.25, 1 / 3))


def test_extraction_eval_uses_normalisation():
    gold = [
        {
            "arxiv_id": "1",
            "gold": {"datasets": ["SQuAD"], "tasks": ["QA"], "methods": [], "metrics": ["F1"]},
        }
    ]
    preds = {
        "1": PaperExtraction(
            datasets=["SQuAD v1.1", "GLUE"], tasks=["question answering"], metrics=["F1 score"]
        )
    }
    table = evaluate_extraction(gold, preds, EntityNormalizer(DEFAULT_ALIASES))
    assert table.loc["datasets", "recall"] == 1.0 and table.loc["datasets", "precision"] == 0.5
    assert table.loc["tasks", "f1"] == 1.0 and table.loc["metrics", "f1"] == 1.0


def test_rag_eval_counts_hallucinated_ids(searcher):
    def router(messages):
        system = messages[0]["content"]
        if "strict evaluator" in system:
            return {"score": 0.5, "unsupported_claims": ["x"]}
        if "Rate how well" in messages[-1]["content"]:
            return {"score": 4}
        return "Claim [arXiv:2302.00002] and fake [arXiv:9999.99999]."

    llm = LLMClient("m", completion_fn=ScriptedLLM(router))
    rows, agg = evaluate_rag(["translation?"], searcher, llm)
    assert (
        agg["valid_citation_rate"] == 0.5 and agg["faithfulness"] == 0.5 and agg["relevance"] == 4
    )
