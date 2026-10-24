"""Retrieval evaluation: Recall@k, MRR, nDCG@k for BM25 vs dense vs hybrid."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

MODES = ("bm25", "dense", "hybrid")


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    return len(rel.intersection(ranked[:k])) / len(rel) if rel else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    return next((1.0 / i for i, d in enumerate(ranked, 1) if d in rel), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@k."""
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


def load_queries(path: Path, validated_only: bool = False) -> list[dict[str, Any]]:
    """JSONL rows: {"query", "relevant": [ids], "validated": bool}."""
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r.get("validated") or not validated_only]


def evaluate_retrieval(
    searcher: Any, queries: list[dict[str, Any]], modes: Sequence[str] = MODES, k_max: int = 10
) -> pd.DataFrame:
    """Mean metrics per mode (rows) over all queries."""
    rows = []
    for mode in modes:
        metrics: dict[str, list[float]] = {
            "Recall@5": [],
            "Recall@10": [],
            "MRR": [],
            "nDCG@10": [],
        }
        for q in queries:
            ranked = [r.arxiv_id for r in searcher.search(q["query"], top_k=k_max, mode=mode)]
            rel = q["relevant"]
            metrics["Recall@5"].append(recall_at_k(ranked, rel, 5))
            metrics["Recall@10"].append(recall_at_k(ranked, rel, 10))
            metrics["MRR"].append(reciprocal_rank(ranked, rel))
            metrics["nDCG@10"].append(ndcg_at_k(ranked, rel, 10))
        rows.append(
            {
                "mode": mode,
                **{m: round(sum(v) / len(v), 4) if v else 0.0 for m, v in metrics.items()},
            }
        )
    return pd.DataFrame(rows).set_index("mode")


def retrieval_chart(table: pd.DataFrame, path: Path) -> Path:
    import plotly.express as px

    long = table.reset_index().melt(id_vars="mode", var_name="metric", value_name="value")
    fig = px.bar(
        long,
        x="metric",
        y="value",
        color="mode",
        barmode="group",
        title="Retrieval: BM25 vs dense vs hybrid",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    return path
