from datetime import date

import numpy as np
import pandas as pd

from rpe.config import TrendsConfig
from rpe.topics.bertopic_model import fallback_label, label_topic, representative_indices
from rpe.topics.trends import (
    build_trend_report,
    compute_growth,
    declining_topics,
    emerging_topics,
    match_topics,
    monthly_counts,
    split_windows,
    summarize_trends,
    unknown_numbers,
    with_months,
)
from tests.fakes import ScriptedLLM


def synthetic(months: int = 13) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Topic 0 grows, topic 1 declines, topic 2 is stable; 13th month is partial."""
    rows, papers = [], []
    k = 0
    for m in range(months):
        year, month = 2024 + m // 12, m % 12 + 1
        counts = {0: 2 + 3 * m, 1: max(1, 40 - 3 * m), 2: 20, -1: 5}
        for tid, n in counts.items():
            for _ in range(n):
                aid = f"p{k:05d}"
                k += 1
                rows.append({"arxiv_id": aid, "topic_id": tid, "topic_label": f"T{tid}"})
                papers.append({"arxiv_id": aid, "published_date": date(year, month, 1)})
    return pd.DataFrame(rows), pd.DataFrame(papers)


def test_split_windows_drops_partial_month():
    months = [f"2024-{m:02d}" for m in range(1, 13)] + ["2025-01"]
    recent, prev = split_windows(months, 6, drop_last=True)
    assert recent == [f"2024-{m:02d}" for m in range(7, 13)]
    assert prev == [f"2024-{m:02d}" for m in range(1, 7)]


def test_monthly_counts_shares_sum_to_one():
    topics, papers = synthetic()
    counts = monthly_counts(with_months(topics, papers))
    sums = counts.groupby("month")["share"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0)


def test_growth_and_emergence():
    topics, papers = synthetic()
    g = compute_growth(with_months(topics, papers), window=6, drop_last_month=True)
    by = g.set_index("topic_id")
    assert by.loc[0, "growth"] > 50 and by.loc[1, "growth"] < -30
    assert abs(by.loc[2, "growth"]) < 40
    em = emerging_topics(g, min_growth=0.5, min_volume=20)
    de = declining_topics(g, max_growth=-0.3, min_volume=20)
    assert em["topic_id"].tolist() == [0]
    assert de["topic_id"].tolist() == [1]
    assert -1 not in em["topic_id"].tolist() + de["topic_id"].tolist()


def test_min_volume_blocks_small_topics():
    topics, papers = synthetic()
    g = compute_growth(with_months(topics, papers))
    assert emerging_topics(g, 0.5, min_volume=10_000).empty


def test_growth_with_new_topic_no_division_by_zero():
    df = pd.DataFrame(
        {
            "topic_id": [5] * 3,
            "topic_label": ["new"] * 3,
            "month": ["2024-08", "2024-09", "2024-10"],
        }
    )
    g = compute_growth(df, window=2, drop_last_month=False)
    assert np.isfinite(g["growth"]).all()


def test_build_report_and_topic_filter():
    topics, papers = synthetic()
    cfg = TrendsConfig(window_months=6, min_volume=20)
    rep = build_trend_report(topics, papers, cfg)
    assert [t.topic_id for t in rep.emerging] == [0]
    only = build_trend_report(topics, papers, cfg, topic_ids=[2])
    assert {s["topic_id"] for s in only.series} == {2} and not only.emerging


def test_match_topics():
    info = [
        {
            "topic_id": 0,
            "label": "LLM Agents and Tool Use",
            "keywords": ["agent", "tool"],
            "size": 50,
        },
        {"topic_id": 1, "label": "Machine Translation", "keywords": ["translation"], "size": 80},
        {"topic_id": -1, "label": "Outliers", "keywords": ["agents"], "size": 999},
    ]
    assert match_topics(info, "agents") == [0]
    assert match_topics(info, "speech") == []


def test_unknown_numbers_detects_invented_figures():
    source = "T0 | 120 | 40 | 12.5 | 4.1 | +205.0"
    assert unknown_numbers("T0 grew by +205.0% with 120 papers [arXiv:2401.00001]", source) == []
    assert unknown_numbers("T0 grew by 300%", source) == ["300"]


def test_summarize_trends_checks_citations_and_numbers():
    from rpe.llm.client import LLMClient

    topics, papers = synthetic()
    rep = build_trend_report(topics, papers, TrendsConfig(min_volume=20))
    growth = rep.emerging[0].growth
    text = f"T0 is emerging ({growth:+}%) [arXiv:2401.00001]. Bad [arXiv:2999.99999]. Now 999."
    llm = LLMClient("m", completion_fn=ScriptedLLM([text]))
    s = summarize_trends(rep, llm, {0: [{"arxiv_id": "2401.00001", "title": "x"}]})
    assert s.citations == ["2401.00001"] and s.invalid_citations == ["2999.99999"]
    assert s.unknown_numbers == ["999"]


def test_representative_indices_closest_to_centroid():
    m = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.95, 0.05]], dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    reps = representative_indices(np.array([0, 1, 3]), m, k=2)
    assert set(reps) <= {0, 1, 3} and len(reps) == 2


def test_label_topic_llm_and_fallback():
    from rpe.llm.client import LLMClient

    llm = LLMClient(
        "m", completion_fn=ScriptedLLM([{"label": "Retrieval-Augmented Generation Methods"}])
    )
    assert label_topic(["rag"], ["t"], llm) == "Retrieval-Augmented Generation Methods"
    assert label_topic(["a", "b", "c", "d"], [], None) == fallback_label(["a", "b", "c"])
