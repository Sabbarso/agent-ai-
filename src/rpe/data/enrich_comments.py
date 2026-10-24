"""Enrichment from free-text metadata: `comments`, `journal_ref`, `abstract`.

Added columns
-------------
venue, venue_year, is_accepted
    Venue names are matched as acronyms (``ACL``, ``NeurIPS'23``, ``AAAI-24``) or
    long forms (``Empirical Methods in Natural Language Processing``), see
    `VENUE_PATTERNS`. Acronyms must not touch other letters, so ``NAACL`` and ``EACL``
    never match ``ACL``; digits may follow (``EMNLP2023``). URLs are removed first.
    If the same clause (split on ``;`` / newlines) mentions "Findings" or "workshop",
    the suffix `` Findings`` / `` Workshop`` is appended (e.g. ``EMNLP Findings``).
    When several venues appear, the first one in a clause containing an acceptance
    cue wins, otherwise the first one in the text.

    `is_accepted` is True when:
      * the venue clause has an acceptance cue (`ACCEPT_RE`: "accepted at/to/by",
        "to appear in", "published in", "camera-ready", "proceedings of", ...), or
      * a venue *with a year* is given and no negative cue (`NEGATIVE_RE`: "submitted",
        "under review", "submission", ...) is present, or
      * `journal_ref` is non-null (the paper has a publication reference).
    The abstract is only used as a last resort, with the strict cues
    "accepted"/"to appear"/"camera-ready" and a venue in the same sentence.

github_url, has_code
    First ``github.com/<owner>[/<repo>]`` link (comments first, then abstract),
    normalised to ``https://github.com/...``. `has_code` is also True for gitlab.com
    and bitbucket.org links.

n_pages, n_figures
    ``12 pages``, ``8+2 pages`` (summed), ``10pp``, ``3 figures``, ``one figure``
    (number words up to twenty). Only parsed from `comments`; first match wins.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

# ------------------------------------------------------------------------ venues


def _acronym(name: str) -> str:
    return rf"(?<![A-Za-z]){name}(?![A-Za-z])"


_ACL_FAMILY = r"of\s+the\s+Association\s+for\s+Computational\s+Linguistics"

VENUE_PATTERNS: dict[str, list[str]] = {
    "ACL": [_acronym("ACL"), rf"Annual\s+Meeting\s+{_ACL_FAMILY}"],
    "EMNLP": [_acronym("EMNLP"), r"Empirical\s+Methods\s+in\s+Natural\s+Language\s+Processing"],
    "NAACL": [_acronym("NAACL"), rf"North\s+American\s+Chapter\s+{_ACL_FAMILY}"],
    "EACL": [_acronym("EACL"), rf"European\s+Chapter\s+{_ACL_FAMILY}"],
    "AACL": [_acronym("AACL"), rf"Asia-Pacific\s+Chapter\s+{_ACL_FAMILY}"],
    "COLING": [
        _acronym("COLING"),
        r"International\s+Conference\s+on\s+Computational\s+Linguistics",
    ],
    "LREC": [_acronym("LREC"), r"Language\s+Resources\s+and\s+Evaluation\s+Conference"],
    "CoNLL": [_acronym("CoNLL"), r"Computational\s+Natural\s+Language\s+Learning"],
    "TACL": [_acronym("TACL"), r"Transactions\s+of\s+the\s+Association\s+for\s+Computational"],
    "NeurIPS": [
        _acronym("NeurIPS"),
        _acronym("NIPS"),
        r"Neural\s+Information\s+Processing\s+Systems",
    ],
    "ICML": [_acronym("ICML"), r"International\s+Conference\s+on\s+Machine\s+Learning(?!\s+and)"],
    "ICLR": [_acronym("ICLR"), r"International\s+Conference\s+on\s+Learning\s+Representations"],
    "AAAI": [_acronym("AAAI")],
    "IJCAI": [
        _acronym("IJCAI"),
        r"International\s+Joint\s+Conference\s+on\s+Artificial\s+Intelligence",
    ],
    "CVPR": [
        _acronym("CVPR"),
        r"Conference\s+on\s+Computer\s+Vision\s+and\s+Pattern\s+Recognition",
    ],
    "ICCV": [_acronym("ICCV"), r"International\s+Conference\s+on\s+Computer\s+Vision"],
    "ECCV": [_acronym("ECCV"), r"European\s+Conference\s+on\s+Computer\s+Vision"],
    "KDD": [_acronym("KDD"), r"Knowledge\s+Discovery\s+and\s+Data\s+Mining"],
    "SIGIR": [_acronym("SIGIR"), r"Research\s+and\s+Development\s+in\s+Information\s+Retrieval"],
    # "WWW" is case-sensitive so that leftover "www." host names never match.
    "WWW": [r"(?-i:(?<![A-Za-z])WWW(?![A-Za-z]))", r"The\s*Web\s*Conf(?:erence)?"],
    "WSDM": [_acronym("WSDM")],
    "CIKM": [_acronym("CIKM")],
    "ECIR": [_acronym("ECIR")],
    "Interspeech": [_acronym("Interspeech")],
    "ICASSP": [_acronym("ICASSP")],
    "JMLR": [_acronym("JMLR"), r"Journal\s+of\s+Machine\s+Learning\s+Research"],
    "TMLR": [_acronym("TMLR"), r"Transactions\s+on\s+Machine\s+Learning\s+Research"],
}

_VENUE_RES: list[tuple[str, re.Pattern[str]]] = [
    (venue, re.compile(p, re.IGNORECASE)) for venue, pats in VENUE_PATTERNS.items() for p in pats
]

ACCEPT_RE = re.compile(
    r"\b(accepted|to\s+appear|appear(?:s|ed|ing)?\s+(?:in|at)|published|presented|"
    r"camera[- ]ready|proceedings|main\s+conference|oral|spotlight|best\s+paper)\b",
    re.IGNORECASE,
)
STRICT_ACCEPT_RE = re.compile(r"\b(accepted|to\s+appear|camera[- ]ready)\b", re.IGNORECASE)
NEGATIVE_RE = re.compile(
    r"\b(submitted|under\s+review|in\s+review|submission|rejected|withdrawn)\b", re.IGNORECASE
)
_FINDINGS_RE = re.compile(r"\bfindings\b", re.IGNORECASE)
_WORKSHOP_RE = re.compile(r"\bworkshops?\b", re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|org|io|net)/\S*", re.IGNORECASE)
_CLAUSE_SEP_RE = re.compile(r"[;\n|]")
_SENTENCE_SEP_RE = re.compile(r"(?<=[.!?])\s+")

_YEAR4_AFTER_RE = re.compile(r"\s*[-–]?\s*((?:19|20)\d{2})(?!\d)")
_YEAR2_AFTER_RE = re.compile(r"(?:\s*['’]|[-–_]|)(\d{2})(?!\d)")
_YEAR4_BEFORE_RE = re.compile(r"((?:19|20)\d{2})(?!\d)")

# ------------------------------------------------------------------ code / sizes

GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)(?:/([A-Za-z0-9_.-]+))?",
    re.IGNORECASE,
)
OTHER_CODE_RE = re.compile(r"\b(?:gitlab\.com|bitbucket\.org)/[\w.-]+", re.IGNORECASE)

NUMBER_WORDS: dict[str, int] = {
    w: i + 1
    for i, w in enumerate(
        [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
_NUM = "(" + "|".join([r"\d{1,4}", *sorted(NUMBER_WORDS, key=len, reverse=True)]) + ")"
PAGES_RE = re.compile(
    rf"(?<![\w.]){_NUM}(?:\s*\+\s*{_NUM})?\s*(?:pages?|pp|pgs?)(?![a-z])", re.IGNORECASE
)
FIGURES_RE = re.compile(rf"(?<![\w.]){_NUM}\s*(?:figures?|figs?)(?![a-z])", re.IGNORECASE)


@dataclass(frozen=True)
class VenueInfo:
    venue: str | None = None
    venue_year: int | None = None
    is_accepted: bool = False


@dataclass(frozen=True)
class Enrichment:
    venue: str | None
    venue_year: int | None
    is_accepted: bool
    github_url: str | None
    has_code: bool
    n_pages: int | None
    n_figures: int | None


# ---------------------------------------------------------------------- helpers


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Bounds of the clause (between `;`, `|` or newlines) containing [start, end)."""
    left = max((m.end() for m in _CLAUSE_SEP_RE.finditer(text, 0, start)), default=0)
    right_m = _CLAUSE_SEP_RE.search(text, end)
    return left, right_m.start() if right_m else len(text)


def _plausible_year(year: int) -> int | None:
    return year if 1990 <= year <= 2035 else None


def _venue_year(text: str, start: int, end: int) -> int | None:
    """Year right after the venue (`ACL 2024`, `NeurIPS'23`, `AAAI-24`) or just before it."""
    after = text[end : end + 12]
    if m := _YEAR4_AFTER_RE.match(after):
        return _plausible_year(int(m.group(1)))
    if m := _YEAR2_AFTER_RE.match(after):
        return _plausible_year(2000 + int(m.group(1)))
    before = _YEAR4_BEFORE_RE.findall(text[max(0, start - 40) : start])
    return _plausible_year(int(before[-1])) if before else None


def _to_int(token: str | None) -> int | None:
    if not token:
        return None
    return int(token) if token.isdigit() else NUMBER_WORDS.get(token.lower())


# ------------------------------------------------------------------- detection


def detect_venue(text: str | None) -> VenueInfo:
    """Venue, year and acceptance status from one free-text field."""
    if not text:
        return VenueInfo()
    clean = _URL_RE.sub(" ", text)
    matches = sorted(
        ((m.start(), m.end(), venue) for venue, rx in _VENUE_RES for m in rx.finditer(clean)),
        key=lambda t: (t[0], -t[1]),
    )
    if not matches:
        accepted = bool(ACCEPT_RE.search(clean)) and not NEGATIVE_RE.search(clean)
        return VenueInfo(is_accepted=accepted)

    def clause_of(match: tuple[int, int, str]) -> str:
        lo, hi = _clause_bounds(clean, match[0], match[1])
        return clean[lo:hi]

    chosen = next((m for m in matches if ACCEPT_RE.search(clause_of(m))), matches[0])
    start, end, venue = chosen
    clause = clause_of(chosen)
    if _FINDINGS_RE.search(clause):
        venue = f"{venue} Findings"
    elif _WORKSHOP_RE.search(clause):
        venue = f"{venue} Workshop"
    year = _venue_year(clean, start, end)
    accepted = bool(ACCEPT_RE.search(clause)) or (
        year is not None and not NEGATIVE_RE.search(clause)
    )
    return VenueInfo(venue=venue, venue_year=year, is_accepted=accepted)


def detect_venue_in_abstract(abstract: str | None) -> VenueInfo:
    """Venue only from sentences with a strict acceptance cue ("accepted at ...")."""
    if not abstract:
        return VenueInfo()
    for sentence in _SENTENCE_SEP_RE.split(abstract):
        if STRICT_ACCEPT_RE.search(sentence):
            info = detect_venue(sentence)
            if info.venue:
                return VenueInfo(info.venue, info.venue_year, True)
    return VenueInfo()


def extract_venue(
    comments: str | None, journal_ref: str | None, abstract: str | None = None
) -> VenueInfo:
    """Combine comments, then journal_ref, then abstract (see module docstring)."""
    info = detect_venue(comments)
    if journal_ref:
        jr = detect_venue(journal_ref)
        if info.venue is None:
            return VenueInfo(jr.venue, jr.venue_year, True)
        same = jr.venue == info.venue
        return VenueInfo(info.venue, info.venue_year or (jr.venue_year if same else None), True)
    if info.venue is None:
        from_abstract = detect_venue_in_abstract(abstract)
        if from_abstract.venue:
            return from_abstract
    return info


def extract_github_url(*texts: str | None) -> str | None:
    """First github.com link found in `texts` (in order), as `https://github.com/...`."""
    for text in texts:
        if not text:
            continue
        for m in GITHUB_RE.finditer(text):
            owner, repo = m.group(1).rstrip("."), (m.group(2) or "").rstrip(".")
            repo = repo.removesuffix(".git")
            if owner:
                return f"https://github.com/{owner}" + (f"/{repo}" if repo else "")
    return None


def extract_n_pages(comments: str | None) -> int | None:
    """`12 pages` -> 12, `8+2 pages` -> 10."""
    if not comments or not (m := PAGES_RE.search(comments)):
        return None
    total = (_to_int(m.group(1)) or 0) + (_to_int(m.group(2)) or 0)
    return total if 0 < total <= 2000 else None


def extract_n_figures(comments: str | None) -> int | None:
    """`3 figures` -> 3, `one figure` -> 1."""
    if not comments or not (m := FIGURES_RE.search(comments)):
        return None
    n = _to_int(m.group(1))
    return n if n is not None and n <= 500 else None


def enrich_record(
    comments: str | None, journal_ref: str | None, abstract: str | None
) -> Enrichment:
    """All enrichment fields for one paper."""
    venue = extract_venue(comments, journal_ref, abstract)
    github = extract_github_url(comments, abstract)
    other_code = any(t and OTHER_CODE_RE.search(t) for t in (comments, abstract))
    return Enrichment(
        venue=venue.venue,
        venue_year=venue.venue_year,
        is_accepted=venue.is_accepted,
        github_url=github,
        has_code=github is not None or bool(other_code),
        n_pages=extract_n_pages(comments),
        n_figures=extract_n_figures(comments),
    )


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the enrichment columns appended."""

    def text(value: object) -> str | None:  # pandas stores missing strings as NaN
        return value if isinstance(value, str) else None

    rows = [
        asdict(enrich_record(text(c), text(j), text(a)))
        for c, j, a in zip(df["comments"], df["journal_ref"], df["abstract"], strict=True)
    ]
    extra = pd.DataFrame(rows, index=df.index, columns=list(Enrichment.__dataclass_fields__))
    for col in ("venue_year", "n_pages", "n_figures"):
        extra[col] = extra[col].astype("Int64")
    for col in ("is_accepted", "has_code"):
        extra[col] = extra[col].astype(bool)
    return pd.concat([df, extra], axis=1)
