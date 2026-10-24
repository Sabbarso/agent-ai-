"""Cleaning and filtering of raw arXiv metadata records."""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import pandas as pd

T = TypeVar("T")

PAPER_COLUMNS: list[str] = [
    "arxiv_id",
    "title",
    "abstract",
    "authors_list",
    "n_authors",
    "categories",
    "primary_category",
    "published_date",
    "year",
    "month",
    "update_date",
    "n_versions",
    "comments",
    "journal_ref",
    "doi",
]

_WS_RE = re.compile(r"\s+")
_AUTHOR_SPLIT_RE = re.compile(r",\s*|\s+and\s+")

# LaTeX accent command -> Unicode combining mark.
_LATEX_ACCENTS = {
    "'": "\u0301",
    "`": "\u0300",
    "^": "\u0302",
    '"': "\u0308",
    "~": "\u0303",
    "=": "\u0304",
    ".": "\u0307",
    "c": "\u0327",
    "v": "\u030c",
    "u": "\u0306",
    "H": "\u030b",
}
# Matches \'a, \'{a}, \'{\i}, \"o, and letter commands with braces: \c{c}, \v{s}.
_LATEX_ACCENT_RE = re.compile(
    r"""\\(['`^"~=.])\s*\{?\\?([A-Za-z])\}?|\\([cvuH])\s*\{\\?([A-Za-z])\}"""
)


# --------------------------------------------------------------------------- text


def normalize_whitespace(text: str | None) -> str:
    """Collapse newlines and repeated spaces; `None` -> ""."""
    return _WS_RE.sub(" ", text).strip() if text else ""


def optional_text(text: str | None) -> str | None:
    """Normalized text, or `None` when empty."""
    return normalize_whitespace(text) or None


def strip_latex_accents(text: str) -> str:
    """Convert LaTeX accents (`\\'a`, `{\\"o}`) to Unicode and drop leftover braces."""

    def repl(m: re.Match[str]) -> str:
        accent = m.group(1) or m.group(3)
        letter = m.group(2) or m.group(4)
        return unicodedata.normalize("NFC", letter + _LATEX_ACCENTS[accent])

    return _LATEX_ACCENT_RE.sub(repl, text).replace("{", "").replace("}", "")


# --------------------------------------------------------------------- categories


def parse_categories(value: str | None) -> list[str]:
    """`"cs.CL cs.LG"` -> `["cs.CL", "cs.LG"]` (first one is the primary category)."""
    return value.split() if value else []


def matches_category(category: str, wanted: Iterable[str]) -> bool:
    """Exact match, or prefix match for wildcard entries such as `"cs.*"`."""
    for w in wanted:
        if w.endswith("*") and category.startswith(w[:-1]):
            return True
        if category == w:
            return True
    return False


def has_wanted_category(categories: Sequence[str], wanted: Iterable[str]) -> bool:
    """True if at least one category of the paper is wanted."""
    wanted = list(wanted)
    return any(matches_category(c, wanted) for c in categories)


def make_raw_filter(categories: Iterable[str]) -> Callable[[bytes], bool]:
    """Cheap byte-level pre-filter: the line must contain one of the category tokens."""
    tokens = [c.rstrip("*").encode() for c in categories]
    return lambda line: any(t in line for t in tokens)


# -------------------------------------------------------------------------- dates


def parse_published_date(versions: Any) -> date | None:
    """Publication date = `versions[0].created` (RFC 2822), or `None` if unusable."""
    try:
        return parsedate_to_datetime(versions[0]["created"]).date()
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def parse_iso_date(value: str | None) -> date | None:
    """`"2008-11-26"` -> date, `None` if missing or malformed."""
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def in_date_range(d: date | None, start: date | None, end: date | None) -> bool:
    """Inclusive range check; a missing date never matches."""
    if d is None:
        return False
    return (start is None or d >= start) and (end is None or d <= end)


# ------------------------------------------------------------------------ authors


def format_author(parts: Sequence[str]) -> str:
    """`["Balázs", "C.", ""]` (last, first, suffix) -> `"C. Balázs"`."""
    last, first, suffix = (list(parts) + ["", "", ""])[:3]
    name = " ".join(p for p in (first, last, suffix) if p and p.strip())
    return normalize_whitespace(strip_latex_accents(name))


def authors_from_record(record: dict[str, Any]) -> list[str]:
    """Authors as "First Last", preferring `authors_parsed` over the raw string."""
    parsed = record.get("authors_parsed")
    if parsed:
        names = [format_author(p) for p in parsed]
    else:
        raw = strip_latex_accents(normalize_whitespace(record.get("authors")))
        names = [normalize_whitespace(n) for n in _AUTHOR_SPLIT_RE.split(raw)]
    return [n for n in names if n]


# ------------------------------------------------------------------------- record


def clean_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Raw snapshot record -> cleaned row (see `PAPER_COLUMNS`); `None` if unusable."""
    arxiv_id = normalize_whitespace(record.get("id"))
    title = normalize_whitespace(record.get("title"))
    if not arxiv_id or not title:
        return None
    published = parse_published_date(record.get("versions"))
    categories = parse_categories(record.get("categories"))
    authors = authors_from_record(record)
    versions = record.get("versions") or []
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": normalize_whitespace(record.get("abstract")),
        "authors_list": authors,
        "n_authors": len(authors),
        "categories": categories,
        "primary_category": categories[0] if categories else None,
        "published_date": published,
        "year": published.year if published else None,
        "month": published.month if published else None,
        "update_date": parse_iso_date(record.get("update_date")),
        "n_versions": len(versions),
        "comments": optional_text(record.get("comments")),
        "journal_ref": optional_text(record.get("journal-ref")),
        "doi": optional_text(record.get("doi")),
    }


# ----------------------------------------------------------------------- sampling


def reservoir_sample(items: Iterable[T], k: int | None, seed: int = 42) -> list[T]:
    """Uniform sample of size `k` in one pass and O(k) memory (Algorithm R).

    Deterministic for a given input order and seed; `k=None` keeps everything.
    """
    if k is None:
        return list(items)
    rng = random.Random(seed)
    reservoir: list[T] = []
    for i, item in enumerate(items):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows -> DataFrame with stable column order and nullable integer dtypes."""
    df = pd.DataFrame(rows, columns=PAPER_COLUMNS)
    for col in ("year", "month"):
        df[col] = df[col].astype("Int64")
    for col in ("n_authors", "n_versions"):
        df[col] = df[col].astype("int64")
    return df
