"""RAG question answering with verified `[arXiv:ID]` citations.

answer_question(question, scope="corpus" | arxiv_id):
hybrid retrieval -> numbered context -> LLM answer with citations -> validate_citations()
-> "I don't know" when the context is insufficient or no valid citation remains.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from rpe.index.hybrid import HybridSearcher
from rpe.llm import prompts
from rpe.llm.client import LLMClient
from rpe.rag.citations import validate_citations

IDK_MESSAGE = {
    "en": "I don't know: the retrieved papers do not contain enough information to answer.",
    "fr": "Je ne sais pas : les papiers récupérés ne contiennent pas assez d'information.",
}


class ContextItem(BaseModel):
    ref: str  # "arXiv:ID" or "arXiv:ID §Section"
    arxiv_id: str
    title: str
    year: int | None = None
    venue: str | None = None
    text: str


class RAGAnswer(BaseModel):
    question: str
    scope: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    contexts: list[ContextItem] = Field(default_factory=list)
    insufficient: bool = False


class ChunkRetriever(Protocol):
    """Full-text chunks of one paper (implemented in phase 7)."""

    def has_chunks(self, arxiv_id: str) -> bool: ...

    def retrieve(self, arxiv_id: str, query: str, k: int) -> list[ContextItem]: ...


def paper_context(paper: dict) -> ContextItem:
    return ContextItem(
        ref=f"arXiv:{paper['arxiv_id']}",
        arxiv_id=paper["arxiv_id"],
        title=paper.get("title", ""),
        year=paper.get("year"),
        venue=paper.get("venue"),
        text=paper.get("abstract") or "",
    )


def retrieve_context(
    question: str,
    searcher: HybridSearcher,
    scope: str = "corpus",
    top_k: int = 5,
    chunks: ChunkRetriever | None = None,
) -> list[ContextItem]:
    """Context items for the corpus (hybrid search) or a single paper (chunks or abstract)."""
    if scope == "corpus":
        hits = searcher.search(question, top_k=top_k)
        return [paper_context(p) for p in searcher.papers.many(h.arxiv_id for h in hits)]
    paper = searcher.papers.get(scope)
    if paper is None:
        raise KeyError(f"Unknown arxiv_id: {scope}")
    items = [paper_context(paper)]
    if chunks is not None and chunks.has_chunks(scope):
        items += chunks.retrieve(scope, question, top_k)
    return items


def generate_answer(
    question: str,
    contexts: list[ContextItem],
    llm: LLMClient,
    scope: str = "corpus",
    lang: str = "en",
) -> RAGAnswer:
    """LLM answer over `contexts`, with citation validation and IDK fallback."""
    idk = IDK_MESSAGE.get(lang, IDK_MESSAGE["en"])
    if not contexts:
        return RAGAnswer(question=question, scope=scope, answer=idk, insufficient=True)
    context = prompts.format_papers([c.model_dump() for c in contexts], max_chars=1500)
    raw = llm.chat(
        prompts.RAG_USER.format(
            context=context, question=question, language=prompts.language_instruction(lang)
        ),
        system=prompts.RAG_SYSTEM,
    )
    base = {"question": question, "scope": scope, "contexts": contexts}
    if prompts.INSUFFICIENT in raw:
        return RAGAnswer(**base, answer=idk, insufficient=True)
    sections: dict[str, set[str]] = {}
    for c in contexts:
        sections.setdefault(c.arxiv_id, set())
        if "§" in c.ref:
            sections[c.arxiv_id].add(c.ref.split("§", 1)[1].strip())
    report = validate_citations(raw, sections, allowed_sections=sections)
    if not report.valid:
        return RAGAnswer(**base, answer=idk, invalid_citations=report.invalid, insufficient=True)
    return RAGAnswer(
        **base,
        answer=report.text.strip(),
        citations=report.valid,
        invalid_citations=report.invalid,
    )


def answer_question(
    question: str,
    searcher: HybridSearcher,
    llm: LLMClient,
    scope: str = "corpus",
    top_k: int = 5,
    chunks: ChunkRetriever | None = None,
    lang: str = "en",
) -> RAGAnswer:
    """Retrieve then answer (see module docstring)."""
    contexts = retrieve_context(question, searcher, scope, top_k, chunks)
    return generate_answer(question, contexts, llm, scope, lang)


def tldr(paper: dict, llm: LLMClient) -> str:
    """One-sentence TL;DR of a paper, from its abstract (followed by its citation)."""
    from rpe.llm.schemas import TLDR

    out = llm.structured(
        prompts.TLDR_USER.format(paper=prompts.format_paper(paper)), TLDR
    ).tldr.strip()
    return f"{out} [arXiv:{paper['arxiv_id']}]"
