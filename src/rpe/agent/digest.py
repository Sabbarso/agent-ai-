"""Research watch: newest arXiv papers on a query (arXiv API) -> grouped Markdown digest."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import DigestGrouping, DigestTheme
from rpe.rag.citations import normalize_id, validate_citations

logger = logging.getLogger(__name__)
_TERM_RE = re.compile(r'"[^"]+"|\S+')


class LatestPaper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published: datetime
    categories: list[str] = Field(default_factory=list)
    url: str


class Digest(BaseModel):
    query: str
    days: int
    generated_at: datetime
    papers: list[LatestPaper] = Field(default_factory=list)
    themes: list[DigestTheme] = Field(default_factory=list)
    markdown: str = ""
    invalid_citations: list[str] = Field(default_factory=list)


def build_arxiv_query(query: str, categories: Iterable[str] | None = None) -> str:
    """`LLM agents` -> `(cat:cs.CL OR cat:cs.AI) AND (all:LLM AND all:agents)`."""
    terms = " AND ".join(f"all:{t}" for t in _TERM_RE.findall(query.strip()))
    cats = [f"cat:{c}" for c in categories or []]
    if cats:
        return f"({' OR '.join(cats)}) AND ({terms})"
    return terms


def fetch_latest_arxiv(
    query: str,
    days: int = 1,
    max_results: int = 50,
    categories: Iterable[str] | None = None,
    client: Any = None,
    now: datetime | None = None,
) -> list[LatestPaper]:
    """Papers submitted in the last `days` days, newest first (arXiv API, polite client)."""
    import arxiv

    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    search = arxiv.Search(
        query=build_arxiv_query(query, categories),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = client or arxiv.Client(page_size=min(100, max_results), delay_seconds=3, num_retries=3)
    out = []
    for r in client.results(search):
        if r.published < cutoff:
            break
        out.append(
            LatestPaper(
                arxiv_id=normalize_id(r.get_short_id()),
                title=" ".join(r.title.split()),
                abstract=" ".join(r.summary.split()),
                authors=[a.name for a in r.authors],
                published=r.published,
                categories=list(r.categories),
                url=r.entry_id,
            )
        )
    return out


def group_papers(
    query: str, days: int, papers: list[LatestPaper], llm: LLMClient | None
) -> tuple[list[DigestTheme], list[str]]:
    """Themes (LLM or single fallback theme) with validated citations; returns invalid ids."""
    ids = [p.arxiv_id for p in papers]
    themes: list[DigestTheme] = []
    invalid: list[str] = []
    if llm is not None:
        try:
            out = llm.structured(
                prompts.DIGEST_USER.format(
                    query=query,
                    days=days,
                    papers=prompts.format_papers([p.model_dump() for p in papers], 500),
                ),
                DigestGrouping,
                system=prompts.DIGEST_SYSTEM,
            )
            seen: set[str] = set()
            for t in out.themes:
                rep = validate_citations(t.summary, ids)
                invalid += rep.invalid
                members = [
                    i
                    for i in (normalize_id(x.replace("arXiv:", "")) for x in t.arxiv_ids)
                    if i in ids and i not in seen
                ]
                seen.update(members)
                themes.append(DigestTheme(theme=t.theme, summary=rep.text, arxiv_ids=members))
            rest = [i for i in ids if i not in seen]
            if rest:
                themes.append(DigestTheme(theme="Other", summary="", arxiv_ids=rest))
        except LLMError as exc:
            logger.warning("Digest grouping failed (%s); listing papers", exc)
            themes = []
    if not themes:
        themes = [DigestTheme(theme="New papers", summary="", arxiv_ids=ids)]
    return [t for t in themes if t.arxiv_ids], invalid


def digest_markdown(d: Digest) -> str:
    by_id = {p.arxiv_id: p for p in d.papers}
    lines = [
        f"# arXiv digest: {d.query}",
        f"*{d.generated_at:%Y-%m-%d %H:%M} UTC · last {d.days} day(s) · {len(d.papers)} papers*",
    ]
    if not d.papers:
        lines.append("\nNo new papers for this query in this period.")
    for t in d.themes:
        lines.append(f"\n## {t.theme}")
        if t.summary:
            lines.append(t.summary)
        for i in t.arxiv_ids:
            p = by_id[i]
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            lines.append(f"- **{p.title}** — {authors} [arXiv:{i}](https://arxiv.org/abs/{i})")
    return "\n".join(lines) + "\n"


def make_digest(query: str, days: int, papers: list[LatestPaper], llm: LLMClient | None) -> Digest:
    themes, invalid = group_papers(query, days, papers, llm) if papers else ([], [])
    d = Digest(
        query=query,
        days=days,
        generated_at=datetime.now(UTC),
        papers=papers,
        themes=themes,
        invalid_citations=invalid,
    )
    d.markdown = digest_markdown(d)
    return d


def save_digest(d: Digest, out_dir: Path) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", d.query.lower()).strip("-")[:50] or "digest"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{d.generated_at:%Y-%m-%d}_{slug}.md"
    path.write_text(d.markdown, encoding="utf-8")
    return path
