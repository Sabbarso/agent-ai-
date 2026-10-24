"""Smoke tests: every Streamlit page renders without exception (missing artefacts -> clear
error message, not a traceback)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from rpe.ui.common import linkify_citations  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = sorted((APP / "pages").glob("*.py"))


def test_linkify_citations():
    out = linkify_citations("A [arXiv:2301.00001] B [arXiv:2302.00002, arXiv:2303.00003 §Method]")
    assert "[arXiv:2301.00001](https://arxiv.org/abs/2301.00001)" in out
    assert "(https://arxiv.org/abs/2303.00003)" in out and "§Method" in out


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders(page, tmp_path, monkeypatch):
    # Point the app to an empty project root: pages must fail gracefully.
    (tmp_path / "config.yaml").write_text("data:\n  categories: ['cs.CL']\n", encoding="utf-8")
    monkeypatch.setenv("RPE_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "")
    from rpe.config import get_config

    get_config.cache_clear()
    at = AppTest.from_file(str(page), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    get_config.cache_clear()


def test_home_renders(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("RPE_CONFIG", str(tmp_path / "config.yaml"))
    from rpe.config import get_config

    get_config.cache_clear()
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    get_config.cache_clear()


def test_render_comparison():
    def script():
        from rpe.rag.compare import ComparisonResult, PaperComparison
        from rpe.ui.compare_view import render_comparison

        res = ComparisonResult(
            question="q",
            comparisons=[
                PaperComparison(
                    arxiv_id="2302.00002",
                    title="MT",
                    year=2023,
                    method="BT",
                    relevance="high",
                    source="abstract",
                ),
                PaperComparison(arxiv_id="2403.00003", title="Agents", relevance="low"),
            ],
            synthesis="Both differ [arXiv:2302.00002].",
        )
        render_comparison(res.model_dump(), key="t")

    at = AppTest.from_function(script, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert len(at.dataframe) == 1 and len(at.tabs) == 4
    assert any("peu pertinent" in m.value for m in at.markdown)
