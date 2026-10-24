"""On-demand full-text access used by `read_pdf` and the reading sheet."""

from __future__ import annotations

from rpe.config import Config
from rpe.pdf.download import download_pdf, pdf_dir, pdf_path
from rpe.pdf.parse import Section, parse_pdf

FULL_TEXT_SECTIONS = (
    "Introduction",
    "Method",
    "Experiments",
    "Results",
    "Conclusion",
    "Limitations",
)


def read_paper_sections(cfg: Config, arxiv_id: str, download: bool = True) -> list[Section]:
    """Sections of a paper's PDF, downloading it once if needed."""
    pdf = pdf_path(cfg, arxiv_id)
    if not pdf.exists() and (not download or download_pdf(cfg, arxiv_id) is None):
        raise FileNotFoundError(f"No PDF available for {arxiv_id}")
    return parse_pdf(pdf, arxiv_id, pdf_dir(cfg))


COMPARE_SECTIONS = ("Method", "Experiments", "Results", "Conclusion", "Limitations")


def pdf_excerpt(cfg: Config, arxiv_id: str, names: tuple[str, ...], max_chars: int) -> str | None:
    """Selected sections of a local PDF as one text, or None if no local PDF / section."""
    if not pdf_path(cfg, arxiv_id).exists():
        return None
    sections = read_paper_sections(cfg, arxiv_id, download=False)
    budget = max_chars // max(1, len(names))
    parts = [f"[{s.name}] {s.text[:budget]}" for s in sections if s.name in names]
    return "\n\n".join(parts)[:max_chars] or None


def full_text_for_sheet(cfg: Config, arxiv_id: str, max_chars: int = 12000) -> str | None:
    """Main sections as one text (for the reading sheet), or None if no local PDF."""
    return pdf_excerpt(cfg, arxiv_id, FULL_TEXT_SECTIONS, max_chars)
