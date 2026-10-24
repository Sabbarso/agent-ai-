"""Phase-1 pipeline: stream -> filter -> clean -> sample -> enrich -> Parquet."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rpe.config import Config
from rpe.data.clean import (
    clean_record,
    has_wanted_category,
    in_date_range,
    make_raw_filter,
    parse_categories,
    reservoir_sample,
    to_dataframe,
)
from rpe.data.enrich_comments import enrich_dataframe
from rpe.data.load import StreamStats, stream_records

logger = logging.getLogger(__name__)


@dataclass
class PrepareStats:
    """Counters reported by `prepare-data`."""

    stream: StreamStats = field(default_factory=StreamStats)
    n_category_match: int = 0
    n_malformed: int = 0
    n_in_date_range: int = 0
    n_kept: int = 0

    def to_dict(self) -> dict[str, int]:
        d = {k: v for k, v in asdict(self).items() if k != "stream"}
        d.update(
            n_lines=self.stream.n_lines,
            n_invalid=self.stream.n_invalid,
            n_prefiltered=self.stream.n_prefiltered,
        )
        return d


def iter_candidates(
    records: Iterable[dict[str, Any]], cfg: Config, stats: PrepareStats
) -> Iterator[dict[str, Any]]:
    """Cleaned rows matching the configured categories and date range."""
    data = cfg.data
    for rec in records:
        if not has_wanted_category(parse_categories(rec.get("categories")), data.categories):
            continue
        stats.n_category_match += 1
        row = clean_record(rec)
        if row is None:
            stats.n_malformed += 1
            continue
        if not in_date_range(row["published_date"], data.start_date, data.end_date):
            continue
        stats.n_in_date_range += 1
        yield row


def build_papers(
    cfg: Config, dev: bool = False, progress: bool = True
) -> tuple[pd.DataFrame, PrepareStats]:
    """Full scan of the raw file -> enriched DataFrame (sampled to the configured size)."""
    stats = PrepareStats()
    records = stream_records(
        cfg.raw_path,
        stats=stats.stream,
        raw_filter=make_raw_filter(cfg.data.categories),
        progress=progress,
    )
    k = cfg.data.sample_dev if dev else cfg.data.max_papers
    rows = reservoir_sample(iter_candidates(records, cfg, stats), k, seed=cfg.data.seed)
    rows.sort(key=lambda r: r["arxiv_id"])
    stats.n_kept = len(rows)
    return enrich_dataframe(to_dataframe(rows)), stats


def summarize_papers(df: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """Dataset statistics: counts per year, % with code / accepted, top venues."""
    n = len(df)

    def pct(s: pd.Series) -> float:
        return round(100 * float(s.mean()), 1) if n else 0.0

    return {
        "n_papers": n,
        "by_year": {int(y): int(c) for y, c in df["year"].value_counts().sort_index().items()},
        "pct_has_code": pct(df["has_code"]),
        "pct_accepted": pct(df["is_accepted"]),
        "pct_with_venue": pct(df["venue"].notna()),
        "top_venues": {str(v): int(c) for v, c in df["venue"].value_counts().head(top_n).items()},
    }


def stats_path(papers_path: Path) -> Path:
    return papers_path.with_suffix(".stats.json")


def run_prepare(
    cfg: Config, dev: bool = False, force: bool = False, progress: bool = True
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    """Build (or reuse) the papers Parquet; returns (path, df, report).

    Resumable: if the output already exists and `force` is False, it is reused.
    Output is written to a temporary file then renamed (no partial Parquet).
    """
    out = cfg.papers_path(dev)
    if out.exists() and not force:
        logger.info("Reusing existing %s (use --force to rebuild)", out)
        df = pd.read_parquet(out)
        sp = stats_path(out)
        prev = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        return out, df, {"scan": prev.get("scan"), "summary": summarize_papers(df), "reused": True}

    if not cfg.raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {cfg.raw_path}. Download it manually from Kaggle "
            "(Cornell-University/arxiv) and place it there."
        )
    df, stats = build_papers(cfg, dev=dev, progress=progress)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    report = {"scan": stats.to_dict(), "summary": summarize_papers(df), "reused": False}
    stats_path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out, df, report
