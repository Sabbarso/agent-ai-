"""All LLM prompts (English), with strict output instructions.

Templates use `str.format` placeholders. Context blocks are built by the callers with the
`format_*` helpers below so that every prompt shows papers in the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# ------------------------------------------------------------------ common

JSON_OUTPUT_INSTRUCTIONS = """\
OUTPUT FORMAT: reply with ONE valid JSON object and nothing else (no markdown fences, \
no commentary). It must validate against this JSON schema:
{schema}"""

FIX_JSON = """\
Your previous reply was invalid: {error}
Return ONLY the corrected JSON object, matching the schema exactly."""

CITATION_RULES = """\
CITATION RULES (mandatory):
- Use ONLY the papers in the CONTEXT. Never use outside knowledge.
- After every factual claim, cite its source as [arXiv:ID] using the exact IDs shown in the \
CONTEXT (e.g. [arXiv:2301.00001]). Several sources: [arXiv:ID1, arXiv:ID2].
- Never invent an ID. Never cite a paper that is not in the CONTEXT."""

INSUFFICIENT = "INSUFFICIENT_CONTEXT"


def language_instruction(lang: str) -> str:
    return "Write the answer in French." if lang == "fr" else "Write the answer in English."


def format_paper(p: Mapping[str, Any], idx: int | None = None, max_chars: int = 1200) -> str:
    """One paper as a context block: `[i] arXiv:ID | Title (year, venue)` + abstract."""
    head = f"[{idx}] " if idx is not None else ""
    meta = ", ".join(str(x) for x in (p.get("year"), p.get("venue")) if x)
    text = (p.get("text") or p.get("abstract") or "")[:max_chars]
    ref = p.get("ref") or f"arXiv:{p['arxiv_id']}"
    return f"{head}{ref} | {p.get('title', '')}" + (f" ({meta})" if meta else "") + f"\n{text}"


def format_papers(papers: Iterable[Mapping[str, Any]], max_chars: int = 1200) -> str:
    return "\n\n".join(format_paper(p, i, max_chars) for i, p in enumerate(papers, 1))


# ------------------------------------------------------------- RAG

RAG_SYSTEM = f"""\
You are a careful research assistant answering questions about scientific papers.
{CITATION_RULES}
- If the CONTEXT does not contain enough information to answer, reply exactly: {INSUFFICIENT}
- Be concise (at most ~200 words), factual, and do not speculate."""

RAG_USER = """\
CONTEXT:
{context}

QUESTION: {question}

{language}
Answer with citations in the form [arXiv:ID] (or [arXiv:ID §Section] when the context \
block shows a section)."""

TLDR_USER = """\
Write a one-sentence TL;DR (max 30 words) of this paper, based only on its abstract.

{paper}"""

COMPARE_EXTRACT_SYSTEM = """\
You extract comparison facts about scientific papers from the TEXT given for each paper.
STRICT RULES:
- Use ONLY what is explicitly written in that paper's TEXT. Never use outside knowledge.
- If a piece of information is not written in the TEXT, return an empty string or an \
empty list. Do NOT guess. This is frequent for results and limitations.
- problem: one sentence. method: main approach in 1-2 sentences.
- datasets / metrics: names exactly as written.
- results: stated results only; include numbers ONLY if they appear in the TEXT.
- limitations: only limitations stated in the TEXT."""

COMPARE_EXTRACT_USER = """\
Extract the comparison facts of EACH paper below. Return {{"papers": [...]}} with one item \
per paper, each with its "arxiv_id".

{papers}"""

COMPARE_SYNTHESIS_SYSTEM = f"""\
You write short comparative syntheses of scientific papers for a researcher.
{CITATION_RULES}
- Use only the COMPARISON TABLE below; if a field says it is not mentioned, do not invent it."""

COMPARE_SYNTHESIS_USER = """\
QUESTION: {question}

COMPARISON TABLE:
{table}

Write 4 to 6 sentences: which approaches are similar, the main differences of method, \
shared datasets/benchmarks, which paper seems best suited to the question, and common \
limitations. Cite every paper you mention as [arXiv:ID].
{language}"""
