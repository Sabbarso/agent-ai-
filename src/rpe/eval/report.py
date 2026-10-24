"""`evaluate --all`: run the available evaluations and write eval/report.md."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rpe.config import Config
from rpe.eval.extraction_eval import evaluate_extraction, load_gold
from rpe.eval.rag_eval import evaluate_rag, load_questions
from rpe.eval.retrieval_eval import evaluate_retrieval, load_queries, retrieval_chart
from rpe.kg.extract import cache_path, load_cache

logger = logging.getLogger(__name__)


def _table(df: pd.DataFrame) -> str:
    return df.to_markdown() if hasattr(df, "to_markdown") and _has_tabulate() else df.to_string()


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


def _retrieval(cfg: Config, res: Any, report: Path) -> list[str]:
    path = cfg.path(cfg.eval.datasets_dir) / "queries.jsonl"
    queries = load_queries(path)
    if not queries:
        return ["*No query set: run `python -m rpe.cli make-eval-sets`.*"]
    n_valid = sum(1 for q in queries if q.get("validated"))
    table = evaluate_retrieval(res.searcher, queries)
    chart = retrieval_chart(table, report.parent / "retrieval.html")
    return [
        f"{len(queries)} queries ({n_valid} validated by hand; known-item relevance otherwise).",
        "",
        "```",
        table.to_string(),
        "```",
        "",
        f"Chart: [{chart.name}]({chart.name})",
    ]


def _extraction(cfg: Config, res: Any) -> list[str]:
    gold = load_gold(cfg.path(cfg.eval.datasets_dir) / "extraction_gold.jsonl")
    if not gold:
        return [
            "*No validated gold set yet: annotate `extraction_gold_todo.jsonl` by hand, set "
            '`"validated": true` and save it as `extraction_gold.jsonl`.*'
        ]
    preds = load_cache(cache_path(cfg, res.dev))
    table = evaluate_extraction(gold, preds, res.normalizer)
    covered = sum(1 for g in gold if g["arxiv_id"] in preds)
    return [
        f"{len(gold)} gold abstracts, {covered} with a prediction.",
        "",
        "```",
        table.to_string(),
        "```",
    ]


def _rag(cfg: Config, res: Any, n_max: int = 20) -> list[str]:
    if res.llm is None:
        return ["*LLM not configured: RAG evaluation skipped.*"]
    questions = load_questions(cfg.path(cfg.eval.datasets_dir) / "rag_questions.jsonl")[:n_max]
    if not questions:
        return ["*No RAG question set: run `python -m rpe.cli make-eval-sets`.*"]
    rows, agg = evaluate_rag(questions, res.searcher, res.llm)
    lines = [f"{len(questions)} questions; judge = same LLM (strict faithfulness prompt).", ""]
    lines += [f"- **{k}**: {v}" for k, v in agg.items()]
    lines += ["", "```", rows.drop(columns=["question"]).to_string(), "```"]
    return lines


def run_evaluation(
    cfg: Config, res: Any, retrieval: bool = True, extraction: bool = True, rag: bool = True
) -> Path:
    report = cfg.path(cfg.eval.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation report",
        f"*Generated {datetime.now():%Y-%m-%d %H:%M}{' (dev corpus)' if res.dev else ''}*",
        "",
    ]
    sections = [
        ("Retrieval (BM25 vs dense vs hybrid)", retrieval, lambda: _retrieval(cfg, res, report)),
        ("Knowledge extraction", extraction, lambda: _extraction(cfg, res)),
        ("RAG", rag, lambda: _rag(cfg, res)),
    ]
    for title, enabled, fn in sections:
        if not enabled:
            continue
        lines.append(f"## {title}")
        try:
            lines += fn()
        except FileNotFoundError as exc:
            lines.append(f"*Skipped: {exc}*")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
