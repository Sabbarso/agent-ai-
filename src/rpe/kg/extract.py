"""LLM knowledge extraction from abstracts (batched, cached in JSONL, resumable).

Selection: up to `extraction.max_papers` papers, accepted-or-with-code first, then random
(seed). Each batch of `extraction.batch_size` abstracts is sent in one call; papers missing
from a batch reply are retried one by one. Results are appended to the JSONL cache after
every batch; errors go to `<cache>_errors.jsonl` and are retried on the next run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rpe.config import Config
from rpe.data.store import PaperStore
from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import BatchExtraction, PaperExtraction

logger = logging.getLogger(__name__)
DEV_MAX_PAPERS = 50


@dataclass
class ExtractionReport:
    n_selected: int
    n_cached: int
    n_new: int
    n_errors: int


def select_papers(df: pd.DataFrame, max_papers: int, seed: int = 42) -> list[str]:
    """Accepted or with-code papers first (random order), then the others."""
    prio_mask = df["is_accepted"].astype(bool) | df["has_code"].astype(bool)
    prio = df[prio_mask].sample(frac=1.0, random_state=seed)["arxiv_id"].tolist()
    rest = df[~prio_mask].sample(frac=1.0, random_state=seed)["arxiv_id"].tolist()
    return (prio + rest)[:max_papers]


def load_cache(path: Path) -> dict[str, PaperExtraction]:
    """Successful extractions from the JSONL cache (last line wins)."""
    out: dict[str, PaperExtraction] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            out[rec["arxiv_id"]] = PaperExtraction.model_validate(rec["extraction"])
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Skipping corrupt cache line in %s", path)
    return out


def _append_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _paper_block(p: dict) -> str:
    return f"### arXiv:{p['arxiv_id']}\nTitle: {p['title']}\nAbstract: {p['abstract']}"


def extract_one(paper: dict, llm: LLMClient) -> PaperExtraction:
    prompt = prompts.EXTRACTION_SINGLE_USER.format(paper=_paper_block(paper))
    return llm.structured(prompt, PaperExtraction, system=prompts.EXTRACTION_SYSTEM)


def extract_batch(papers: Sequence[dict], llm: LLMClient) -> dict[str, PaperExtraction]:
    """One call for the batch; ids missing from the reply are extracted individually."""
    wanted = {p["arxiv_id"] for p in papers}
    results: dict[str, PaperExtraction] = {}
    if len(papers) > 1:
        prompt = prompts.EXTRACTION_BATCH_USER.format(
            papers="\n\n".join(_paper_block(p) for p in papers)
        )
        try:
            batch = llm.structured(prompt, BatchExtraction, system=prompts.EXTRACTION_SYSTEM)
            for item in batch.papers:
                aid = item.arxiv_id.replace("arXiv:", "").strip()
                if aid in wanted:
                    results[aid] = PaperExtraction.model_validate(
                        item.model_dump(exclude={"arxiv_id"})
                    )
        except LLMError as exc:
            logger.warning("Batch extraction failed (%s); falling back to single calls", exc)
    for p in papers:
        if p["arxiv_id"] not in results:
            results[p["arxiv_id"]] = extract_one(p, llm)
    return results


def cache_path(cfg: Config, dev: bool = False) -> Path:
    return cfg.artifact(cfg.extraction.cache_path, dev)


def run_extraction(
    cfg: Config,
    llm: LLMClient,
    dev: bool = False,
    limit: int | None = None,
    progress: bool = True,
) -> ExtractionReport:
    papers = PaperStore.load(cfg, dev)
    max_papers = limit or (DEV_MAX_PAPERS if dev else cfg.extraction.max_papers)
    selected = select_papers(papers.df, max_papers, cfg.data.seed)
    path = cache_path(cfg, dev)
    err_path = path.with_name(path.stem + "_errors.jsonl")
    cached = load_cache(path)
    todo = [i for i in selected if i not in cached]
    bs = max(1, cfg.extraction.batch_size)
    n_new = n_err = 0
    for start in tqdm(range(0, len(todo), bs), desc="Extracting", disable=not progress):
        batch = papers.many(todo[start : start + bs])
        try:
            res = extract_batch(batch, llm)
        except LLMError as exc:
            # Batch-level failure: retry each paper alone so one bad abstract costs one paper.
            res = {}
            for p in batch:
                try:
                    res[p["arxiv_id"]] = extract_one(p, llm)
                except LLMError as e:
                    n_err += 1
                    _append_jsonl(err_path, [{"arxiv_id": p["arxiv_id"], "error": str(e)[:500]}])
            logger.debug("Batch error: %s", exc)
        _append_jsonl(path, [{"arxiv_id": a, "extraction": e.model_dump()} for a, e in res.items()])
        n_new += len(res)
    return ExtractionReport(len(selected), len(selected) - len(todo), n_new, n_err)
