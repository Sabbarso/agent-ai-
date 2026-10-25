"""Pydantic schemas for every structured LLM output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _clean_list(value: Any) -> list[str]:
    """None -> [], strip strings, drop empties and case-insensitive duplicates."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        s = str(item).strip() if item is not None else ""
        if s and s.lower() not in seen and s.lower() not in {"none", "n/a", "null"}:
            seen.add(s.lower())
            out.append(s)
    return out


class _Lists(BaseModel):
    """Base class: every `list[str]` field is cleaned by `_clean_list`."""

    @field_validator("*", mode="before")
    @classmethod
    def _lists(cls, v: Any, info: Any) -> Any:
        field = cls.model_fields.get(info.field_name)
        if field is not None and field.annotation == list[str]:
            return _clean_list(v)
        return v


# ------------------------------------------------------------- RAG


class TLDR(BaseModel):
    tldr: str


class ComparisonItem(_Lists):
    """Comparison fields of one paper, extracted from its text only (empty if absent)."""

    arxiv_id: str
    problem: str = ""
    method: str = ""
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    results: str = ""
    limitations: list[str] = Field(default_factory=list)

    @field_validator("problem", "method", "results", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        return "" if v is None else str(v).strip()


class ComparisonBatch(BaseModel):
    papers: list[ComparisonItem] = Field(default_factory=list)
