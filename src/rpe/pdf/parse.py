"""PDF text extraction (PyMuPDF), section splitting, chunking, arXiv reference extraction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rpe.rag.citations import normalize_id

CANONICAL_SECTIONS: list[tuple[str, str]] = [
    ("Abstract", r"abstract"),
    ("Introduction", r"introduction"),
    ("Related Work", r"related\s+works?|background|prior\s+work|literature\s+review"),
    ("Method", r"methods?|methodology|approach|proposed\s+method|model|framework|our\s+approach"),
    ("Experiments", r"experiments?|experimental\s+(?:setup|settings?|results)|evaluation|setup"),
    ("Results", r"results?(?:\s+and\s+discussion)?|analysis|discussion|ablations?"),
    ("Conclusion", r"conclusions?(?:\s+and\s+future\s+work)?|summary|future\s+work"),
    ("Limitations", r"limitations?|ethic(?:s|al)\s+(?:statement|considerations)"),
    ("References", r"references|bibliography"),
    ("Appendix", r"appendi(?:x|ces)|supplementary(?:\s+material)?"),
]
_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+|[IVX]+|[A-H])(?:\.\d+)*\.?\s+)?("
    + "|".join(f"(?:{p})" for _, p in CANONICAL_SECTIONS)
    + r")\s*:?\s*$",
    re.IGNORECASE,
)
_ARXIV_REF_RE = re.compile(
    r"(?:arxiv(?:\.org/(?:abs|pdf)/|\s*:\s*|\s+preprint\s+arxiv\s*:\s*)|abs/)"
    r"(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


@dataclass
class Section:
    name: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    section: str
    text: str


def canonical_section(heading: str) -> str | None:
    m = _HEADING_RE.match(heading)
    if not m:
        return None
    title = m.group(1)
    for name, pattern in CANONICAL_SECTIONS:
        if re.fullmatch(pattern, title.strip(), re.IGNORECASE):
            return name
    return None


def extract_text(path: Path) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def split_sections(text: str) -> list[Section]:
    """Split on heading-like short lines; text before the first heading is 'Front matter'."""
    sections: list[Section] = []
    current, buf = "Front matter", []
    for line in text.splitlines():
        stripped = line.strip()
        name = canonical_section(stripped) if 0 < len(stripped) <= 60 else None
        # Accept each canonical section once (later matches are probably subsections),
        # except "Appendix" which marks the end of the main text.
        if name and name not in {s.name for s in sections} and name != current:
            sections.append(Section(current, " ".join(buf).strip()))
            current, buf = name, []
        else:
            buf.append(stripped)
    sections.append(Section(current, " ".join(buf).strip()))
    merged: dict[str, str] = {}
    for s in sections:
        if s.text:
            merged[s.name] = (merged.get(s.name, "") + " " + s.text).strip()
    return [Section(k, re.sub(r"\s+", " ", v)) for k, v in merged.items()]


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Bounded chunks with overlap, cut on sentence boundaries when possible."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind(". ", start + max_chars // 2, end)
            end = cut + 1 if cut != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_sections(
    arxiv_id: str,
    sections: list[Section],
    max_chars: int = 1500,
    overlap: int = 200,
    skip: tuple[str, ...] = ("References", "Front matter"),
) -> list[Chunk]:
    out: list[Chunk] = []
    for s in sections:
        if s.name in skip:
            continue
        for piece in chunk_text(s.text, max_chars, overlap):
            out.append(Chunk(f"{arxiv_id}#{len(out)}", arxiv_id, s.name, piece))
    return out


def extract_arxiv_references(sections: list[Section], self_id: str | None = None) -> list[str]:
    """arXiv ids cited in the References section (or the whole text if absent)."""
    refs = next((s.text for s in sections if s.name == "References"), None)
    text = refs if refs is not None else " ".join(s.text for s in sections)
    ids = dict.fromkeys(normalize_id(m) for m in _ARXIV_REF_RE.findall(text))
    ids.pop(self_id or "", None)
    return list(ids)


def parsed_path(cfg_pdf_dir: Path, arxiv_id: str) -> Path:
    return cfg_pdf_dir / "parsed" / f"{arxiv_id.replace('/', '_')}.json"


def parse_pdf(pdf: Path, arxiv_id: str, cache_dir: Path) -> list[Section]:
    """Sections of a PDF, cached as JSON next to the PDFs."""
    cache = parsed_path(cache_dir, arxiv_id)
    if cache.exists():
        return [Section(**s) for s in json.loads(cache.read_text(encoding="utf-8"))]
    sections = split_sections(extract_text(pdf))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(s) for s in sections], ensure_ascii=False), "utf-8")
    return sections
