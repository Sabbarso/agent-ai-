"""Citation parsing and validation (`[arXiv:ID]`, `[arXiv:ID §Section]`).

`validate_citations()` is the anti-hallucination guard used by every generator: each cited
id must belong to the retrieved context; others are removed (or flagged).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

ARXIV_ID_PATTERN = r"(?:\d{4}\.\d{4,5}|[a-z][a-z.\-]*/\d{7})(?:v\d+)?"
_BLOCK_RE = re.compile(r"\[\s*arXiv\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)
_ITEM_RE = re.compile(rf"(?:arXiv\s*:\s*)?({ARXIV_ID_PATTERN})(\s*§\s*[^,;]+)?", re.IGNORECASE)
_VERSION_RE = re.compile(r"v\d+$")


def normalize_id(arxiv_id: str) -> str:
    """Strip whitespace and a version suffix: `2301.00001v2` -> `2301.00001`."""
    return _VERSION_RE.sub("", arxiv_id.strip())


def extract_citations(text: str) -> list[str]:
    """Unique cited ids in order of appearance."""
    seen: dict[str, None] = {}
    for block in _BLOCK_RE.finditer(text or ""):
        for item in _ITEM_RE.finditer(block.group(1)):
            seen.setdefault(normalize_id(item.group(1)), None)
    return list(seen)


@dataclass
class CitationReport:
    text: str
    valid: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    n_citations: int = 0  # citation occurrences (an id cited twice counts twice)
    n_invalid_citations: int = 0

    @property
    def valid_rate(self) -> float:
        """Share of valid citation occurrences (1.0 when nothing is cited)."""
        if not self.n_citations:
            return 1.0
        return (self.n_citations - self.n_invalid_citations) / self.n_citations


def validate_citations(
    text: str,
    allowed_ids: Iterable[str],
    mode: Literal["remove", "flag"] = "remove",
    allowed_sections: Mapping[str, Iterable[str]] | None = None,
) -> CitationReport:
    """Keep citations whose id is in `allowed_ids`; remove or flag the others.

    remove: `[arXiv:BAD]` disappears, `[arXiv:OK, arXiv:BAD]` -> `[arXiv:OK]`.
    flag:   invalid ids are rewritten as `[unverified: arXiv:BAD]`.
    allowed_sections: when given, a `§Section` not present in the context for that id is
    dropped (the id itself is kept): `[arXiv:OK §3.3]` -> `[arXiv:OK]`.
    """
    allowed = {normalize_id(i) for i in allowed_ids}
    sections = (
        {normalize_id(k): {s.lower() for s in v} for k, v in allowed_sections.items()}
        if allowed_sections is not None
        else None
    )
    valid: dict[str, None] = {}
    invalid: dict[str, None] = {}
    counts = {"n": 0, "bad": 0}

    def rewrite(block: re.Match[str]) -> str:
        kept, bad = [], []
        for item in _ITEM_RE.finditer(block.group(1)):
            aid = normalize_id(item.group(1))
            section = (item.group(2) or "").strip()
            name = section.lstrip("§").strip().lower()
            if section and sections is not None and name not in sections.get(aid, set()):
                section = ""
            ref = f"arXiv:{aid}" + (f" §{section.lstrip('§').strip()}" if section else "")
            counts["n"] += 1
            if aid in allowed:
                valid.setdefault(aid, None)
                kept.append(ref)
            else:
                counts["bad"] += 1
                invalid.setdefault(aid, None)
                bad.append(ref)
        out = f"[{', '.join(kept)}]" if kept else ""
        if mode == "flag" and bad:
            out += "".join(f"[unverified: {b}]" for b in bad)
        return out

    new_text = _BLOCK_RE.sub(rewrite, text or "")
    if mode == "remove":
        new_text = re.sub(r"[ \t]+([.,;:])", r"\1", new_text)
        new_text = re.sub(r"[ \t]{2,}", " ", new_text)
    return CitationReport(new_text, list(valid), list(invalid), counts["n"], counts["bad"])
