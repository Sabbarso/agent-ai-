"""Monthly topic trends, growth, emerging / declining topics and LLM synthesis.

Growth compares a topic's *share* of the corpus over the last `window` months with the
previous `window` months, so that the overall growth of arXiv does not count as a trend:

    growth = ((n_recent + 1) / (N_recent + 1)) / ((n_prev + 1) / (N_prev + 1)) - 1

(+1 smoothing avoids divisions by zero for new topics). Emerging: growth >= threshold and
n_recent >= min_volume. Declining: growth <= threshold and n_prev >= min_volume.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from rpe.config import TrendsConfig
from rpe.index.bm25 import tokenize
from rpe.llm import prompts
from rpe.llm.client import LLMClient
from rpe.rag.citations import validate_citations

_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?")
_CITATION_RE = re.compile(r"\[[^\]]*arXiv[^\]]*\]", re.IGNORECASE)


class TopicTrend(BaseModel):
    topic_id: int
    topic_label: str
    recent: int
    previous: int
    share_recent: float  # percent
    share_previous: float  # percent
    growth: float  # relative change of share, percent


class TrendReport(BaseModel):
    window: int
    recent_months: list[str]
    previous_months: list[str]
    topics: list[TopicTrend] = Field(default_factory=list)
    emerging: list[TopicTrend] = Field(default_factory=list)
    declining: list[TopicTrend] = Field(default_factory=list)
    series: list[dict[str, Any]] = Field(default_factory=list)  # topic_id, month, n, share


class TrendSummary(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    unknown_numbers: list[str] = Field(default_factory=list)


def trends_config(cfg: TrendsConfig, dev: bool = False) -> TrendsConfig:
    """Trend thresholds, with the smaller volume threshold of the --dev corpus."""
    return cfg.model_copy(update={"min_volume": cfg.dev_min_volume}) if dev else cfg


def with_months(topics: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    """Join topics with publication dates; adds a `month` column ("YYYY-MM")."""
    df = topics.merge(papers[["arxiv_id", "published_date"]], on="arxiv_id", how="inner")
    df["month"] = pd.to_datetime(df["published_date"]).dt.strftime("%Y-%m")
    return df


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Papers per (topic, month) and share of that month's papers."""
    counts = df.groupby(["topic_id", "topic_label", "month"]).size().rename("n").reset_index()
    totals = df.groupby("month").size().rename("total")
    counts = counts.join(totals, on="month")
    counts["share"] = counts["n"] / counts["total"]
    return counts.sort_values(["topic_id", "month"]).reset_index(drop=True)


def split_windows(months: list[str], window: int, drop_last: bool) -> tuple[list[str], list[str]]:
    months = sorted(set(months))
    if drop_last and months:
        months = months[:-1]
    return months[-window:], months[-2 * window : -window] if len(months) > window else []


def compute_growth(df: pd.DataFrame, window: int = 6, drop_last_month: bool = True) -> pd.DataFrame:
    """One row per topic with recent/previous counts, shares (%) and growth (%)."""
    recent_m, prev_m = split_windows(df["month"].tolist(), window, drop_last_month)
    n_recent_all = int(df["month"].isin(recent_m).sum())
    n_prev_all = int(df["month"].isin(prev_m).sum())
    rows = []
    for (tid, label), g in df.groupby(["topic_id", "topic_label"]):
        rec = int(g["month"].isin(recent_m).sum())
        prev = int(g["month"].isin(prev_m).sum())
        ratio = ((rec + 1) / (n_recent_all + 1)) / ((prev + 1) / (n_prev_all + 1))
        rows.append(
            {
                "topic_id": int(tid),
                "topic_label": str(label),
                "recent": rec,
                "previous": prev,
                "share_recent": round(100 * rec / n_recent_all, 2) if n_recent_all else 0.0,
                "share_previous": round(100 * prev / n_prev_all, 2) if n_prev_all else 0.0,
                "growth": round(100 * (ratio - 1), 1),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["recent_months"], out.attrs["previous_months"] = recent_m, prev_m
    return out


def emerging_topics(growth: pd.DataFrame, min_growth: float, min_volume: int) -> pd.DataFrame:
    """min_growth is a fraction (0.5 = +50 %)."""
    m = (growth["growth"] >= 100 * min_growth) & (growth["recent"] >= min_volume)
    return growth[m & (growth["topic_id"] != -1)].sort_values("growth", ascending=False)


def declining_topics(growth: pd.DataFrame, max_growth: float, min_volume: int) -> pd.DataFrame:
    m = (growth["growth"] <= 100 * max_growth) & (growth["previous"] >= min_volume)
    return growth[m & (growth["topic_id"] != -1)].sort_values("growth")


def match_topics(info: list[dict[str, Any]], query: str, limit: int = 3) -> list[int]:
    """Topic ids whose label/keywords share the most tokens with `query`."""
    q = set(tokenize(query))
    scored = []
    for t in info:
        if t["topic_id"] == -1:
            continue
        words = set(tokenize(t["label"] + " " + " ".join(t.get("keywords", []))))
        overlap = len(q & words)
        if overlap:
            scored.append((-overlap, -t["size"], t["topic_id"]))
    return [tid for *_, tid in sorted(scored)[:limit]]


def build_trend_report(
    topics: pd.DataFrame,
    papers: pd.DataFrame,
    cfg: TrendsConfig,
    topic_ids: list[int] | None = None,
) -> TrendReport:
    """Full trend report (optionally restricted to `topic_ids` for series/topics)."""
    df = with_months(topics, papers)
    growth = compute_growth(df, cfg.window_months, cfg.drop_last_month)
    emerging = emerging_topics(growth, cfg.emerging_growth, cfg.min_volume)
    declining = declining_topics(growth, cfg.declining_growth, cfg.min_volume)
    series = monthly_counts(df)
    shown = growth[growth["topic_id"] != -1]
    if topic_ids is not None:
        shown = shown[shown["topic_id"].isin(topic_ids)]
        series = series[series["topic_id"].isin(topic_ids)]
        emerging = emerging[emerging["topic_id"].isin(topic_ids)]
        declining = declining[declining["topic_id"].isin(topic_ids)]

    def as_trends(frame: pd.DataFrame) -> list[TopicTrend]:
        return [TopicTrend(**r) for r in frame.to_dict("records")]

    return TrendReport(
        window=cfg.window_months,
        recent_months=growth.attrs["recent_months"],
        previous_months=growth.attrs["previous_months"],
        topics=as_trends(shown.sort_values("recent", ascending=False)),
        emerging=as_trends(emerging),
        declining=as_trends(declining),
        series=series.assign(share=series["share"].round(4)).to_dict("records"),
    )


def trends_table(trends: list[TopicTrend]) -> str:
    lines = [
        "topic | recent papers | previous papers | share recent % | share previous % | growth %"
    ]
    lines += [
        f"{t.topic_label} | {t.recent} | {t.previous} | {t.share_recent} | {t.share_previous} | "
        f"{t.growth:+}"
        for t in trends
    ]
    return "\n".join(lines)


def unknown_numbers(text: str, allowed_source: str) -> list[str]:
    """Numbers in `text` (citations excluded) that do not appear in `allowed_source`."""
    allowed = {n.lstrip("+").replace(",", ".") for n in _NUMBER_RE.findall(allowed_source)}
    found = _NUMBER_RE.findall(_CITATION_RE.sub(" ", text))
    return sorted({n for n in found if n.lstrip("+").replace(",", ".") not in allowed})


def summarize_trends(
    report: TrendReport,
    llm: LLMClient,
    examples: dict[int, list[dict[str, Any]]],
    max_topics: int = 8,
) -> TrendSummary:
    """LLM synthesis restricted to the computed numbers; numbers and citations are checked."""
    selected = report.emerging[:max_topics] + report.declining[:max_topics]
    if not selected:
        selected = report.topics[:max_topics]
    data = (
        f"Recent window: {report.recent_months[0] if report.recent_months else '?'} to "
        f"{report.recent_months[-1] if report.recent_months else '?'}\n"
        "EMERGING:\n" + trends_table(report.emerging[:max_topics]) + "\n"
        "DECLINING:\n" + trends_table(report.declining[:max_topics]) + "\n"
        "LARGEST TOPICS:\n" + trends_table(report.topics[:max_topics])
    )
    ex_lines, allowed_ids = [], []
    for t in selected:
        for p in examples.get(t.topic_id, [])[:2]:
            ex_lines.append(f"- {t.topic_label}: [arXiv:{p['arxiv_id']}] {p['title']}")
            allowed_ids.append(p["arxiv_id"])
    prompt = prompts.TRENDS_USER.format(
        window=report.window, data=data, examples="\n".join(ex_lines) or "(none)"
    )
    text = llm.chat(prompt, system=prompts.TRENDS_SYSTEM)
    rep = validate_citations(text, allowed_ids)
    return TrendSummary(
        text=rep.text.strip(),
        citations=rep.valid,
        invalid_citations=rep.invalid,
        unknown_numbers=unknown_numbers(rep.text, prompt),
    )
