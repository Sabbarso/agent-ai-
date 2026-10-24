"""Pydantic schemas for every structured LLM output."""

from __future__ import annotations

import re
from typing import Any, Literal

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


# ----------------------------------------------------------- phase 3 (RAG)


class ReadingSheet(_Lists):
    summary: str
    contributions: list[str] = Field(default_factory=list)
    method: str = ""
    results: str = ""
    limitations: list[str] = Field(default_factory=list)
    comprehension_questions: list[str] = Field(min_length=3, description="exactly 3 questions")

    @field_validator("comprehension_questions", mode="after")
    @classmethod
    def _three(cls, v: list[str]) -> list[str]:
        return v[:3]


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


# -------------------------------------------------------- phase 4 (topics)


class TopicLabel(BaseModel):
    label: str = Field(description="3 to 6 words")

    @field_validator("label")
    @classmethod
    def _short(cls, v: str) -> str:
        v = v.strip().strip('"')
        if " " not in v:  # small models sometimes answer "CamelCaseLabel"
            v = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", v)
        return " ".join(v.split()[:8])


# --------------------------------------------------- phase 5 (extraction)

ContributionType = Literal[
    "new_method", "new_dataset", "benchmark", "survey", "analysis", "application", "other"
]


class ClaimedResult(BaseModel):
    metric: str = ""
    dataset: str = ""
    value: str = ""
    context: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _str(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()


class PaperExtraction(_Lists):
    """Entities explicitly stated in an abstract (empty lists otherwise)."""

    tasks: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    claimed_results: list[ClaimedResult] = Field(default_factory=list)
    contribution_type: ContributionType = "other"

    @field_validator("contribution_type", mode="before")
    @classmethod
    def _ctype(cls, v: Any) -> str:
        s = str(v or "other").strip().lower().replace(" ", "_").replace("-", "_")
        allowed = {"new_method", "new_dataset", "benchmark", "survey", "analysis", "application"}
        return s if s in allowed else "other"

    @field_validator("claimed_results", mode="before")
    @classmethod
    def _results(cls, v: Any) -> Any:
        return [r for r in (v or []) if isinstance(r, dict | ClaimedResult)]


class ExtractionItem(PaperExtraction):
    arxiv_id: str


class BatchExtraction(BaseModel):
    papers: list[ExtractionItem] = Field(default_factory=list)


# ------------------------------------------------------ phase 5 (GraphRAG)

GraphRelation = Literal[
    "WRITTEN_BY",
    "ADDRESSES",
    "PROPOSES_OR_USES",
    "EVALUATED_ON",
    "MEASURED_BY",
    "IN_CATEGORY",
    "CO_AUTHOR",
    "CITES",
]
EntityType = Literal["Paper", "Author", "Task", "Method", "Dataset", "Metric", "Category"]


class GraphQuery(BaseModel):
    entity: str
    entity_type: EntityType
    relation: GraphRelation
    target_type: EntityType = "Paper"


# --------------------------------------------------------- phase 6 (agent)


class Explanation(BaseModel):
    arxiv_id: str
    explanation: str


class Explanations(BaseModel):
    items: list[Explanation] = Field(default_factory=list)


class GapJustification(BaseModel):
    hypothesis: str
    rationale: str


class SotaPlan(_Lists):
    subquestions: list[str] = Field(min_length=1)


class ApproachGroup(_Lists):
    name: str
    description: str = ""
    arxiv_ids: list[str] = Field(default_factory=list)


class SotaGrouping(BaseModel):
    groups: list[ApproachGroup] = Field(default_factory=list)


class DigestTheme(_Lists):
    theme: str
    summary: str
    arxiv_ids: list[str] = Field(default_factory=list)


class DigestGrouping(BaseModel):
    themes: list[DigestTheme] = Field(default_factory=list)


class AgentDecision(BaseModel):
    """One ReAct step: call a tool, or give the final answer."""

    thought: str = ""
    action: str = Field(description="tool name, or 'final_answer'")
    args: dict[str, Any] = Field(default_factory=dict)
    answer: str = ""


# ---------------------------------------------------------- phase 9 (eval)


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = ""


class RelevanceVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str = ""


class GeneratedQuery(BaseModel):
    query: str
