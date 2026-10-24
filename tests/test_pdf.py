from rpe.config import Config, PdfConfig
from rpe.pdf.download import fetch_pdfs, pdf_path
from rpe.pdf.parse import (
    Section,
    canonical_section,
    chunk_sections,
    chunk_text,
    extract_arxiv_references,
    split_sections,
)

PAPER_TEXT = """Some Title
Alice Bob
Abstract
We study X.
1 Introduction
Intro text about X.
2 Related Work
Prior work.
3 Method
Our method is simple.
4 Experiments
We test on SQuAD.
5 Results
It works well.
6 Conclusion
Done.
References
[1] A. Author. Paper. arXiv preprint arXiv:2301.00001, 2023.
[2] B. Author. Other. https://arxiv.org/abs/2405.00005v2
[3] C. Author. Self. arXiv:2506.00006
"""


def test_canonical_section_headings():
    assert canonical_section("1 Introduction") == "Introduction"
    assert canonical_section("3.  Methodology") == "Method"
    assert canonical_section("IV. EXPERIMENTAL SETUP") == "Experiments"
    assert canonical_section("References") == "References"
    assert canonical_section("Introduction of a new loss for training") is None


def test_split_sections():
    names = [s.name for s in split_sections(PAPER_TEXT)]
    assert names == [
        "Front matter",
        "Abstract",
        "Introduction",
        "Related Work",
        "Method",
        "Experiments",
        "Results",
        "Conclusion",
        "References",
    ]


def test_chunk_text_bounded_with_overlap():
    text = " ".join(f"Sentence number {i}." for i in range(300))
    chunks = chunk_text(text, max_chars=500, overlap=100)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) > 5
    assert chunks[0][-50:].split()[-1] in chunks[1]  # overlap keeps context


def test_chunk_sections_skips_references():
    secs = [Section("Method", "a " * 10), Section("References", "b " * 10)]
    chunks = chunk_sections("2301.00001", secs)
    assert [c.section for c in chunks] == ["Method"] and chunks[0].chunk_id == "2301.00001#0"


def test_extract_arxiv_references():
    refs = extract_arxiv_references(split_sections(PAPER_TEXT), self_id="2506.00006")
    assert refs == ["2301.00001", "2405.00005"]


def test_fetch_pdfs_polite_and_resumable(tmp_path):
    cfg = Config(pdfs=PdfConfig(dir="pdfs", delay_seconds=3), root=tmp_path)
    pdf_path(cfg, "2301.00001").parent.mkdir(parents=True)
    pdf_path(cfg, "2301.00001").write_bytes(b"%PDF existing")
    calls, sleeps = [], []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return b"<html>not a pdf</html>" if url.endswith("bad") else b"%PDF-1.5 data"

    stats = fetch_pdfs(
        cfg,
        ["2301.00001", "2302.00002", "2303.00003", "bad"],
        fake_get,
        sleeps.append,
        progress=False,
    )
    assert stats == {"downloaded": 2, "skipped": 1, "failed": 1}
    assert calls == [
        "https://arxiv.org/pdf/2302.00002",
        "https://arxiv.org/pdf/2303.00003",
        "https://arxiv.org/pdf/bad",
    ]
    assert sleeps == [3, 3]  # delay between consecutive requests only
    assert not pdf_path(cfg, "bad").exists()


def test_select_pdf_candidates_priority(tmp_path, searcher):
    from rpe.pdf.download import select_pdf_candidates

    cfg = Config(pdfs=PdfConfig(demo_queries=["machine translation"]), root=tmp_path)
    ids = select_pdf_candidates(cfg, searcher, max_pdfs=4)
    assert len(ids) == 4 and len(set(ids)) == 4
    assert ids[0] in {"2302.00002", "2405.00005"}  # best hit of the demo query first
