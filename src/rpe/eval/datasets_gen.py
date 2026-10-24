"""Semi-automatic drafts of the evaluation sets (to be validated by hand).

- queries.jsonl: 30 known-item queries (LLM-written, or keyword fallback); `relevant` holds
  the source paper; annotators may add other relevant ids and set `"validated": true`.
- extraction_gold_todo.jsonl: 50 abstracts pre-filled with the current LLM extraction.
  Annotators correct `gold`, set `"validated": true`, and save as extraction_gold.jsonl.
  Only validated rows are used: an LLM pre-annotation is never treated as gold.
- rag_questions.jsonl: 20 open questions on the corpus domain.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rpe.index.bm25 import tokenize
from rpe.kg.extract import cache_path, load_cache
from rpe.llm import prompts
from rpe.llm.client import LLMError
from rpe.llm.schemas import GeneratedQuery

logger = logging.getLogger(__name__)

RAG_QUESTIONS = [
    "How is retrieval-augmented generation used to reduce hallucinations?",
    "Which methods are proposed to detect hallucinations in large language models?",
    "How do LLM agents use external tools?",
    "What techniques improve machine translation for low-resource languages?",
    "How is chain-of-thought prompting evaluated on reasoning benchmarks?",
    "What are common approaches to evaluate large language model outputs automatically?",
    "How are large language models aligned with human preferences?",
    "Which approaches are used for multilingual named entity recognition?",
    "How do papers compress or quantize large language models?",
    "What datasets are used to evaluate question answering systems?",
    "How is in-context learning analysed in recent work?",
    "What methods are used for long-context modeling in transformers?",
    "How do recent papers address bias and fairness in language models?",
    "Which approaches combine knowledge graphs with language models?",
    "How are speech recognition models adapted to new languages or domains?",
    "What are the main approaches to text summarization evaluation?",
    "How do papers study the safety or jailbreaking of language models?",
    "Which methods are used for code generation with language models?",
    "How are multimodal (vision-language) models evaluated?",
    "What parameter-efficient fine-tuning methods are proposed?",
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    return path


def keyword_query(title: str, n: int = 5) -> str:
    return " ".join(tokenize(title)[:n])


def make_queries(res: Any, n: int = 30) -> list[dict[str, Any]]:
    df = res.papers.df
    sample = df[df["venue"].notna()] if df["venue"].notna().sum() >= n else df
    sample = sample.sample(n=min(n, len(sample)), random_state=res.cfg.data.seed)
    rows = []
    for p in sample.to_dict("records"):
        query = keyword_query(p["title"])
        if res.llm is not None:
            try:
                query = res.llm.structured(
                    prompts.QUERY_GEN_USER.format(title=p["title"], abstract=p["abstract"][:800]),
                    GeneratedQuery,
                ).query
            except LLMError as exc:
                logger.warning("Query generation failed: %s", exc)
        rows.append(
            {
                "query": query,
                "relevant": [p["arxiv_id"]],
                "source_title": p["title"],
                "validated": False,
            }
        )
    return rows


def make_extraction_todo(res: Any, n: int = 50) -> list[dict[str, Any]]:
    cache = load_cache(cache_path(res.cfg, res.dev))
    ids = (
        list(cache)[:n]
        if len(cache) >= n
        else res.papers.df.sample(n=min(n, len(res.papers)), random_state=res.cfg.data.seed)[
            "arxiv_id"
        ].tolist()
    )
    rows = []
    for aid in ids:
        p = res.papers.get(aid)
        pre = cache.get(aid)
        gold = (
            pre.model_dump(include={"tasks", "methods", "datasets", "metrics"})
            if pre
            else {"tasks": [], "methods": [], "datasets": [], "metrics": []}
        )
        rows.append(
            {
                "arxiv_id": aid,
                "title": p["title"],
                "abstract": p["abstract"],
                "gold": gold,
                "validated": False,
            }
        )
    return rows


def make_eval_sets(cfg: Any, res: Any, force: bool = False) -> list[Path]:
    d = cfg.path(cfg.eval.datasets_dir)
    out = []
    targets = {
        "queries.jsonl": lambda: make_queries(res),
        "extraction_gold_todo.jsonl": lambda: make_extraction_todo(res),
        "rag_questions.jsonl": lambda: [{"question": q} for q in RAG_QUESTIONS],
    }
    for name, build in targets.items():
        path = d / name
        if path.exists() and not force:
            logger.info("%s exists (use --force to regenerate)", path)
            continue
        out.append(_write_jsonl(path, build()))
    return out
