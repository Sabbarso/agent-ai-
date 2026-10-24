"""RAG evaluation: valid-citation rate, faithfulness and relevance (LLM-as-judge).

RAGAS could be plugged in as an option; it is not a dependency of the project.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import JudgeVerdict, RelevanceVerdict
from rpe.rag.qa import generate_answer, retrieve_context

logger = logging.getLogger(__name__)


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [json.loads(line)["question"] for line in path.read_text("utf-8").splitlines() if line]


def judge(
    question: str, context: str, answer: str, llm: LLMClient
) -> tuple[float | None, int | None]:
    """(faithfulness in [0,1], relevance 1-5) or Nones if the judge fails."""
    try:
        f = llm.structured(
            prompts.JUDGE_FAITHFULNESS_USER.format(
                context=context, question=question, answer=answer
            ),
            JudgeVerdict,
            system=prompts.JUDGE_FAITHFULNESS_SYSTEM,
        ).score
        r = llm.structured(
            prompts.JUDGE_RELEVANCE_USER.format(question=question, answer=answer), RelevanceVerdict
        ).score
        return f, r
    except LLMError as exc:
        logger.warning("Judge failed: %s", exc)
        return None, None


def evaluate_rag(
    questions: list[str],
    searcher: Any,
    llm: LLMClient,
    judge_llm: LLMClient | None = None,
    top_k: int = 8,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-question rows + aggregate metrics.

    valid_citation_rate = valid cited ids / all cited ids in the raw LLM answer (before
    `validate_citations` removes the invalid ones), i.e. 1 - hallucinated-id rate. An honest
    "I don't know" without citations counts as 1.
    """
    judge_llm = judge_llm or llm
    rows = []
    for q in questions:
        contexts = retrieve_context(q, searcher, "corpus", top_k)
        ans = generate_answer(q, contexts, llm)
        n_cited = len(ans.citations) + len(ans.invalid_citations)
        rate = len(ans.citations) / n_cited if n_cited else (1.0 if ans.insufficient else 0.0)
        ctx = prompts.format_papers([c.model_dump() for c in contexts], 800)
        faith, rel = judge(q, ctx, ans.answer, judge_llm)
        rows.append(
            {
                "question": q,
                "insufficient": ans.insufficient,
                "n_citations": n_cited,
                "valid_citation_rate": round(rate, 3),
                "faithfulness": faith,
                "relevance": rel,
            }
        )
    df = pd.DataFrame(rows)
    agg = {
        "valid_citation_rate": float(df["valid_citation_rate"].mean()) if len(df) else 0.0,
        "faithfulness": float(df["faithfulness"].dropna().mean()) if len(df) else 0.0,
        "relevance": float(df["relevance"].dropna().mean()) if len(df) else 0.0,
        "idk_rate": float(df["insufficient"].mean()) if len(df) else 0.0,
    }
    return df, {k: round(v, 3) for k, v in agg.items()}
