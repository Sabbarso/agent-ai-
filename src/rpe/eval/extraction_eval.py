"""Extraction evaluation: precision / recall / F1 per field against a hand-annotated gold set.

Gold JSONL rows: {"arxiv_id", "gold": {tasks, methods, datasets, metrics}, "validated": true}.
Entities are compared after normalisation (`EntityNormalizer.key`), so "SQuAD v1.1" == "SQuAD".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from rpe.kg.normalize import EntityNormalizer
from rpe.llm.schemas import PaperExtraction

FIELDS = {"tasks": "task", "methods": "method", "datasets": "dataset", "metrics": "metric"}


def load_gold(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r.get("validated")]


def _keys(items: list[str], kind: str, norm: EntityNormalizer) -> set[str]:
    return {k for i in items if (k := norm.key(i, kind))}


def prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    r = tp / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate_extraction(
    gold: list[dict[str, Any]],
    predictions: Mapping[str, PaperExtraction],
    normalizer: EntityNormalizer,
) -> pd.DataFrame:
    """Micro-averaged P/R/F1 per field over gold papers that have a prediction."""
    rows = []
    for field, kind in FIELDS.items():
        tp = n_pred = n_gold = 0
        for g in gold:
            pred = predictions.get(g["arxiv_id"])
            if pred is None:
                continue
            gk = _keys(g["gold"].get(field, []), kind, normalizer)
            pk = _keys(getattr(pred, field), kind, normalizer)
            tp, n_pred, n_gold = tp + len(gk & pk), n_pred + len(pk), n_gold + len(gk)
        p, r, f = prf(tp, n_pred, n_gold)
        rows.append(
            {
                "field": field,
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f, 3),
                "gold": n_gold,
                "predicted": n_pred,
            }
        )
    return pd.DataFrame(rows).set_index("field")
