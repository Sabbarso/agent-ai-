"""Search filters shared by BM25 (pandas mask) and dense stores (Chroma `where`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from rpe.data.store import as_list


def venue_base(venue: str | None) -> str:
    """`"EMNLP Findings"` -> `"EMNLP"`; `None` -> `""`."""
    return venue.split()[0] if venue else ""


class SearchFilters(BaseModel):
    """Optional filters; a field left to `None` does not filter."""

    year_min: int | None = None
    year_max: int | None = None
    categories: list[str] | None = None  # any-of, on all categories of the paper
    venue: str | None = None  # "EMNLP" matches "EMNLP Findings"; "EMNLP Findings" is exact
    has_code: bool | None = None
    is_accepted: bool | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean mask over a papers DataFrame (row order preserved)."""
        m = np.ones(len(df), dtype=bool)
        year = df["year"].astype("float64").to_numpy()
        if self.year_min is not None:
            m &= np.nan_to_num(year, nan=-1) >= self.year_min
        if self.year_max is not None:
            m &= np.nan_to_num(year, nan=1e9) <= self.year_max
        if self.categories:
            wanted = set(self.categories)
            m &= df["categories"].map(lambda cats: bool(wanted.intersection(cats))).to_numpy()
        if self.venue:
            col = df["venue"].fillna("")
            if " " in self.venue.strip():
                m &= (col.str.lower() == self.venue.strip().lower()).to_numpy()
            else:
                m &= (col.map(venue_base).str.lower() == self.venue.strip().lower()).to_numpy()
        if self.has_code is not None:
            m &= df["has_code"].astype(bool).to_numpy() == self.has_code
        if self.is_accepted is not None:
            m &= df["is_accepted"].astype(bool).to_numpy() == self.is_accepted
        return m

    def to_chroma_where(self) -> dict[str, Any] | None:
        """Equivalent Chroma metadata filter (see `paper_metadata`)."""
        clauses: list[dict[str, Any]] = []
        if self.year_min is not None:
            clauses.append({"year": {"$gte": self.year_min}})
        if self.year_max is not None:
            clauses.append({"year": {"$lte": self.year_max}})
        if self.categories:
            cats = [{f"cat_{c}": True} for c in self.categories]
            clauses.append(cats[0] if len(cats) == 1 else {"$or": cats})
        if self.venue:
            v = self.venue.strip()
            clauses.append({"venue": v} if " " in v else {"venue_base": v})
        if self.has_code is not None:
            clauses.append({"has_code": self.has_code})
        if self.is_accepted is not None:
            clauses.append({"is_accepted": self.is_accepted})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def paper_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata for a paper row (no `None` values allowed by Chroma)."""
    year = row.get("year")
    venue = row.get("venue") if isinstance(row.get("venue"), str) else ""
    meta: dict[str, Any] = {
        "year": int(year) if year is not None and not pd.isna(year) else -1,
        "primary_category": row.get("primary_category") or "",
        "venue": venue,
        "venue_base": venue_base(venue),
        "has_code": bool(row.get("has_code")),
        "is_accepted": bool(row.get("is_accepted")),
    }
    for c in as_list(row.get("categories")):
        meta[f"cat_{c}"] = True
    return meta
