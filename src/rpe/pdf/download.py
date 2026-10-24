"""Polite PDF download from https://arxiv.org/pdf/{id} (delay, resume, skip existing)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from tqdm import tqdm

from rpe.config import Config

logger = logging.getLogger(__name__)
USER_AGENT = "ResearchPaperExplorer/0.1 (student project; polite crawler)"
HttpGet = Callable[[str], bytes]


def pdf_dir(cfg: Config) -> Path:
    return cfg.path(cfg.pdfs.dir)


def pdf_path(cfg: Config, arxiv_id: str) -> Path:
    return pdf_dir(cfg) / f"{arxiv_id.replace('/', '_')}.pdf"


def http_get(url: str, timeout: float = 60.0) -> bytes:
    import httpx

    r = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.content


def download_pdf(cfg: Config, arxiv_id: str, get: HttpGet = http_get) -> Path | None:
    """Download one PDF (atomic write); returns None if it is not a valid PDF."""
    out = pdf_path(cfg, arxiv_id)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    data = get(f"https://arxiv.org/pdf/{arxiv_id}")
    if not data.startswith(b"%PDF"):
        logger.warning("Not a PDF for %s (%d bytes)", arxiv_id, len(data))
        return None
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out


def select_pdf_candidates(
    cfg: Config, searcher, max_pdfs: int, demo_queries: Sequence[str] | None = None
) -> list[str]:
    """Priority: top hits of the demo queries, then accepted / with-code papers, then others."""
    df = searcher.papers.df
    ordered: dict[str, None] = {}
    for q in demo_queries if demo_queries is not None else cfg.pdfs.demo_queries:
        for r in searcher.search(q, top_k=20):
            ordered.setdefault(r.arxiv_id, None)
    prio = df[df["is_accepted"].astype(bool) | df["has_code"].astype(bool)]
    for aid in prio.sort_values("arxiv_id", ascending=False)["arxiv_id"]:
        ordered.setdefault(aid, None)
    for aid in df.sort_values("arxiv_id", ascending=False)["arxiv_id"]:
        ordered.setdefault(aid, None)
    return list(ordered)[:max_pdfs]


def fetch_pdfs(
    cfg: Config,
    ids: Sequence[str],
    get: HttpGet = http_get,
    sleep: Callable[[float], None] = time.sleep,
    progress: bool = True,
) -> dict[str, int]:
    """Download `ids`, skipping existing files; waits `delay_seconds` between requests."""
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    first = True
    for aid in tqdm(ids, desc="PDFs", disable=not progress):
        if pdf_path(cfg, aid).exists():
            stats["skipped"] += 1
            continue
        if not first:
            sleep(cfg.pdfs.delay_seconds)
        first = False
        try:
            ok = download_pdf(cfg, aid, get)
            stats["downloaded" if ok else "failed"] += 1
        except Exception as exc:  # noqa: BLE001 - network errors: continue with the next PDF
            logger.warning("Download failed for %s: %s", aid, exc)
            stats["failed"] += 1
    return stats
