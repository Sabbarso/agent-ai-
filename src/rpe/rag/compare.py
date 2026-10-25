"""Structured comparison of the papers cited in a RAG answer (shown under the Chat answer).

Only the ids CITED in the validated answer are compared (max `max_papers`, order of appearance).
Source of each paper's facts: the PDF sections Method/Experiments/Results/Conclusion/
Limitations if the PDF was downloaded, otherwise the abstract. All papers are extracted in
ONE batched LLM call, cached on disk per arxiv_id (no new call for an already seen paper).
Anti-hallucination: missing facts become "Non mentionné dans le résumé"; results whose
numbers do not appear in the source text are discarded; the synthesis citations are checked
with `validate_citations()`. Relevance is a deterministic lexical score w.r.t. the question.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from rpe.data.store import PaperStore
from rpe.index.bm25 import tokenize
from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import ComparisonBatch, ComparisonItem
from rpe.rag.citations import normalize_id, validate_citations

logger = logging.getLogger(__name__)

NOT_MENTIONED = "Non mentionné dans le résumé"
MAX_PAPERS = 5
_EMPTY = {
    "",
    "-",
    "none",
    "n/a",
    "na",
    "null",
    "unknown",
    "not mentioned",
    "not stated",
    "not specified",
    "not reported",
    NOT_MENTIONED.lower(),
}
RELEVANCE_ORDER = {"high": 0, "medium": 1, "low": 2}
RELEVANCE_LABELS = {"high": "élevée", "medium": "moyenne", "low": "peu pertinent"}

Source = Literal["abstract", "pdf"]
Relevance = Literal["high", "medium", "low"]


_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?")
_CITATION_RE = re.compile(r"\[[^\]]*arXiv[^\]]*\]", re.IGNORECASE)


def unknown_numbers(text: str, source: str) -> list[str]:
    """Numbers in `text` (citations excluded) that do not appear in `source`."""
    allowed = {n.lstrip("+").replace(",", ".") for n in _NUMBER_RE.findall(source)}
    found = _NUMBER_RE.findall(_CITATION_RE.sub(" ", text))
    return sorted({n for n in found if n.lstrip("+").replace(",", ".") not in allowed})


def _is_empty(value: Any) -> bool:
    return value is None or str(value).strip().strip(".").lower() in _EMPTY


class PaperComparison(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    problem: str = NOT_MENTIONED
    method: str = NOT_MENTIONED
    datasets: list[str] = Field(default_factory=lambda: [NOT_MENTIONED])
    metrics: list[str] = Field(default_factory=lambda: [NOT_MENTIONED])
    results: str = NOT_MENTIONED
    limitations: list[str] = Field(default_factory=lambda: [NOT_MENTIONED])
    source: Source = "abstract"
    relevance: Relevance = "medium"

    @field_validator("problem", "method", "results", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return NOT_MENTIONED if _is_empty(v) else str(v).strip()

    @field_validator("datasets", "metrics", "limitations", mode="before")
    @classmethod
    def _items(cls, v: Any) -> list[str]:
        items = [v] if isinstance(v, str) else list(v or [])
        kept = list(dict.fromkeys(str(i).strip() for i in items if not _is_empty(i)))
        return kept or [NOT_MENTIONED]


class ComparisonResult(BaseModel):
    question: str
    comparisons: list[PaperComparison] = Field(default_factory=list)
    synthesis: str = ""
    synthesis_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- helpers


def cited_ids(citations: Iterable[str], max_papers: int = MAX_PAPERS) -> list[str]:
    """Unique cited ids, in order of appearance, capped at `max_papers`."""
    return list(dict.fromkeys(normalize_id(c) for c in citations))[:max_papers]


def relevance(question: str, paper: Mapping[str, Any]) -> Relevance:
    """Share of the question's terms found in the paper's title + abstract."""
    q = set(tokenize(question))
    if not q:
        return "medium"
    doc = set(tokenize(f"{paper.get('title', '')} {paper.get('abstract', '')}"))
    share = len(q & doc) / len(q)
    return "high" if share >= 0.5 else "medium" if share >= 0.25 else "low"


def _meta(paper: Mapping[str, Any], question: str) -> dict[str, Any]:
    venue = paper.get("venue")
    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper.get("title", ""),
        "authors": list(paper.get("authors_list") or []),
        "year": paper.get("year"),
        "venue": venue if isinstance(venue, str) and venue else None,
        "relevance": relevance(question, paper),
    }


def from_item(
    paper: Mapping[str, Any],
    item: ComparisonItem | None,
    source: Source,
    text: str,
    question: str,
) -> PaperComparison:
    """Comparison row from an LLM item; results with numbers absent from `text` are dropped."""
    data = item.model_dump(exclude={"arxiv_id"}) if item else {}
    results = data.get("results", "")
    if results and unknown_numbers(results, text):
        logger.info("Dropping results of %s: numbers not in the source text", paper["arxiv_id"])
        results = ""
    return PaperComparison(**_meta(paper, question), **{**data, "results": results}, source=source)


class ComparisonCache:
    """One JSON file per (arxiv_id, source) holding the LLM-extracted comparison fields."""

    def __init__(self, directory: Path | None) -> None:
        self.dir = directory

    def _path(self, aid: str, source: str) -> Path | None:
        return self.dir / f"{aid.replace('/', '_')}__{source}.json" if self.dir else None

    def get(self, aid: str, source: str) -> ComparisonItem | None:
        p = self._path(aid, source)
        if p is None or not p.exists():
            return None
        return ComparisonItem.model_validate(json.loads(p.read_text(encoding="utf-8")))

    def put(self, item: ComparisonItem, source: str) -> None:
        p = self._path(item.arxiv_id, source)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(item.model_dump_json(), encoding="utf-8")


def extract_batch(
    todo: Sequence[tuple[Mapping[str, Any], Source, str]], llm: LLMClient
) -> dict[str, ComparisonItem]:
    """ONE LLM call for every paper of `todo` (paper, source, text)."""
    blocks = "\n\n".join(
        f"### arXiv:{p['arxiv_id']} ({'full-text excerpt' if src == 'pdf' else 'abstract'})\n"
        f"Title: {p.get('title', '')}\nTEXT: {text}"
        for p, src, text in todo
    )
    out = llm.structured(
        prompts.COMPARE_EXTRACT_USER.format(papers=blocks),
        ComparisonBatch,
        system=prompts.COMPARE_EXTRACT_SYSTEM,
    )
    wanted = {p["arxiv_id"] for p, _, _ in todo}
    items = {}
    for it in out.papers:
        aid = normalize_id(it.arxiv_id.replace("arXiv:", ""))
        if aid in wanted:
            items[aid] = it.model_copy(update={"arxiv_id": aid})
    return items


# ------------------------------------------------------------------ main


def compare_papers(
    question: str,
    citations: Iterable[str],
    papers: PaperStore,
    llm: LLMClient | None,
    pdf_text: Callable[[str], str | None] | None = None,
    cache_dir: Path | None = None,
    max_papers: int = MAX_PAPERS,
) -> ComparisonResult:
    """Comparison rows of the cited papers (low-relevance papers last)."""
    cache = ComparisonCache(cache_dir)
    warnings: list[str] = []
    rows: dict[str, PaperComparison] = {}
    todo: list[tuple[Mapping[str, Any], Source, str]] = []

    ids = [i for i in cited_ids(citations, max_papers) if i in papers]
    for aid in ids:
        paper = papers.get(aid) or {}
        text = None
        if pdf_text is not None:
            try:
                text = pdf_text(aid)
            except Exception as exc:  # noqa: BLE001 - a broken PDF falls back to the abstract
                logger.warning("PDF excerpt failed for %s: %s", aid, exc)
        source: Source = "pdf" if text else "abstract"
        text = text or paper.get("abstract", "")
        if (cached := cache.get(aid, source)) is not None:
            rows[aid] = from_item(paper, cached, source, text, question)
        else:
            todo.append((paper, source, text))

    if todo:
        items: dict[str, ComparisonItem] = {}
        if llm is None:
            warnings.append("LLM non configuré : comparaison limitée aux données disponibles.")
        else:
            try:
                items = extract_batch(todo, llm)
            except LLMError as exc:
                warnings.append(f"Extraction LLM indisponible ({exc}) : données partielles.")
        for paper, source, text in todo:
            aid = paper["arxiv_id"]
            if aid in items:
                cache.put(items[aid], source)
            elif llm is not None and not warnings:
                warnings.append("Certains papiers n'ont pas été renvoyés par le LLM.")
            rows[aid] = from_item(paper, items.get(aid), source, text, question)

    ordered = sorted((rows[i] for i in ids), key=lambda r: r.relevance == "low")
    return ComparisonResult(question=question, comparisons=ordered, warnings=warnings)


def comparison_table_text(comparisons: Sequence[PaperComparison]) -> str:
    return "\n".join(
        f"- arXiv:{c.arxiv_id} | {c.title} ({c.year}) | problem: {c.problem} | method: "
        f"{c.method} | datasets: {', '.join(c.datasets)} | metrics: {', '.join(c.metrics)} | "
        f"results: {c.results} | limitations: {'; '.join(c.limitations)}"
        for c in comparisons
    )


def compare_synthesis(
    question: str, comparisons: Sequence[PaperComparison], llm: LLMClient, lang: str = "en"
) -> tuple[str, list[str], list[str]]:
    """4-6 sentence comparative synthesis; returns (text, valid ids, invalid ids)."""
    raw = llm.chat(
        prompts.COMPARE_SYNTHESIS_USER.format(
            question=question,
            table=comparison_table_text(comparisons),
            language=prompts.language_instruction(lang),
        ),
        system=prompts.COMPARE_SYNTHESIS_SYSTEM,
    )
    rep = validate_citations(raw, [c.arxiv_id for c in comparisons])
    return rep.text.strip(), rep.valid, rep.invalid


def add_synthesis(result: ComparisonResult, llm: LLMClient | None, lang: str = "en") -> None:
    """Fill `result.synthesis` in place; failures become a warning."""
    if llm is None or not result.comparisons:
        return
    try:
        text, valid, invalid = compare_synthesis(result.question, result.comparisons, llm, lang)
        result.synthesis, result.synthesis_citations, result.invalid_citations = (
            text,
            valid,
            invalid,
        )
    except LLMError as exc:
        result.warnings.append(f"Synthèse comparative indisponible ({exc}).")


# ---------------------------------------------------------------- export


def comparison_frame(result: ComparisonResult) -> pd.DataFrame:
    """One row per paper (used for the table and the CSV export)."""
    return pd.DataFrame(
        [
            {
                "arXiv": f"https://arxiv.org/abs/{c.arxiv_id}",
                "Papier": f"{c.title} ({c.year})" if c.year else c.title,
                "Méthode": c.method,
                "Datasets": ", ".join(c.datasets),
                "Métriques": ", ".join(c.metrics),
                "Résultats": c.results,
                "Limitations": "; ".join(c.limitations),
                "Pertinence": RELEVANCE_LABELS[c.relevance],
                "Source": c.source,
            }
            for c in result.comparisons
        ]
    )


def comparison_markdown(result: ComparisonResult) -> str:
    lines = [f"# Comparaison des papiers\n\n**Question :** {result.question}\n"]
    for c in result.comparisons:
        lines += [
            f"## [arXiv:{c.arxiv_id}](https://arxiv.org/abs/{c.arxiv_id}) {c.title}"
            + (f" ({c.year})" if c.year else ""),
            f"- **Pertinence :** {RELEVANCE_LABELS[c.relevance]} · **Source :** {c.source}",
            f"- **Problème :** {c.problem}",
            f"- **Méthode :** {c.method}",
            f"- **Datasets :** {', '.join(c.datasets)}",
            f"- **Métriques :** {', '.join(c.metrics)}",
            f"- **Résultats :** {c.results}",
            f"- **Limitations :** {'; '.join(c.limitations)}",
            "",
        ]
    if result.synthesis:
        lines += ["## Synthèse comparative", result.synthesis, ""]
    return "\n".join(lines)
