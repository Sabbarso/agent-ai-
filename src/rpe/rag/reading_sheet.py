"""Structured reading sheet of a paper (abstract + full text when available)."""

from __future__ import annotations

from rpe.llm import prompts
from rpe.llm.client import LLMClient
from rpe.llm.schemas import ReadingSheet

MAX_FULL_TEXT_CHARS = 12000

LABELS = {
    "en": {
        "summary": "Summary",
        "contributions": "Contributions",
        "method": "Method",
        "results": "Results",
        "limitations": "Limitations",
        "questions": "Comprehension questions",
        "source": "Source",
        "fulltext": "full text",
        "abstract_only": "abstract only",
    },
    "fr": {
        "summary": "Résumé",
        "contributions": "Contributions",
        "method": "Méthode",
        "results": "Résultats",
        "limitations": "Limites",
        "questions": "Questions de compréhension",
        "source": "Source",
        "fulltext": "texte complet",
        "abstract_only": "résumé uniquement",
    },
}


def generate_reading_sheet(
    paper: dict, llm: LLMClient, lang: str = "en", full_text: str | None = None
) -> ReadingSheet:
    """Ask the LLM for a `ReadingSheet` of `paper` (dict from PaperStore)."""
    excerpt = ""
    if full_text:
        excerpt = "FULL-TEXT EXCERPTS:\n" + full_text[:MAX_FULL_TEXT_CHARS]
    prompt = prompts.READING_SHEET_USER.format(
        arxiv_id=paper["arxiv_id"],
        title=paper.get("title", ""),
        abstract=paper.get("abstract", ""),
        full_text=excerpt,
        language=prompts.language_instruction(lang),
    )
    return llm.structured(prompt, ReadingSheet, system=prompts.READING_SHEET_SYSTEM)


def reading_sheet_markdown(
    sheet: ReadingSheet, paper: dict, lang: str = "en", used_full_text: bool = False
) -> str:
    """Render the sheet as Markdown (the paper is cited as [arXiv:ID])."""
    lab = LABELS.get(lang, LABELS["en"])
    aid = paper["arxiv_id"]
    authors = ", ".join((paper.get("authors_list") or [])[:6])
    meta = " · ".join(str(x) for x in (paper.get("year"), paper.get("venue"), authors) if x)
    source = lab["fulltext"] if used_full_text else lab["abstract_only"]
    lines = [
        f"# {paper.get('title', aid)}",
        f"[arXiv:{aid}](https://arxiv.org/abs/{aid}) · {meta}",
        f"*{lab['source']} : {source}*",
        f"\n## {lab['summary']}\n{sheet.summary}",
        f"\n## {lab['contributions']}",
        *[f"- {c}" for c in sheet.contributions],
        f"\n## {lab['method']}\n{sheet.method}",
        f"\n## {lab['results']}\n{sheet.results}",
        f"\n## {lab['limitations']}",
        *[f"- {x}" for x in sheet.limitations],
        f"\n## {lab['questions']}",
        *[f"{i}. {q}" for i, q in enumerate(sheet.comprehension_questions, 1)],
    ]
    return "\n".join(lines) + "\n"
