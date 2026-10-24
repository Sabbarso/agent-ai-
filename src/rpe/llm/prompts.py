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


# ----------------------------------------------------------- phase 3: RAG

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

READING_SHEET_SYSTEM = """\
You write structured reading sheets of scientific papers for students.
Use ONLY the provided paper text (abstract and, if present, full-text excerpts).
Do not invent results or numbers. If something is not stated, say so briefly."""

READING_SHEET_USER = """\
PAPER arXiv:{arxiv_id}
Title: {title}
Abstract: {abstract}

{full_text}

Produce a reading sheet with: summary (3-5 sentences), contributions (list), method \
(short paragraph), results (short paragraph, only stated results), limitations (list; \
include limitations implied by the scope if none are stated, marked "(inferred)"), and \
exactly 3 comprehension_questions.
{language}"""

# --------------------------------------------------------- phase 4: topics

TOPIC_LABEL_USER = """\
Give a short, human-readable label (3 to 6 words, Title Case, no quotes) for a research \
topic described by these keywords and representative paper titles.

Keywords: {keywords}
Titles:
{titles}"""

TRENDS_SYSTEM = f"""\
You write factual summaries of research trends.
STRICT RULES:
- Use ONLY the numbers given in the DATA. Never compute, round differently or invent numbers.
- Mention topics by their label. Cite the example papers given for a topic as [arXiv:ID].
{CITATION_RULES}"""

TRENDS_USER = """\
DATA (computed from the corpus; growth = change of the topic's share of papers, \
last {window} months vs the previous {window}):
{data}

Example papers per topic:
{examples}

Write a summary of 120-200 words: main emerging topics, declining topics, and one \
sentence of interpretation framed as a hypothesis."""

# ----------------------------------------------------- phase 5: extraction

EXTRACTION_SYSTEM = """\
You extract structured knowledge from scientific abstracts.
STRICT RULES:
- Extract ONLY entities explicitly written in the abstract. If absent, return an empty list.
- tasks: research tasks/problems (e.g. "machine translation", "question answering").
- methods: models, architectures, algorithms or techniques proposed or used \
(e.g. "BERT", "retrieval-augmented generation", "contrastive learning").
- datasets: named datasets or benchmarks (e.g. "SQuAD", "GLUE"). Not generic words like "data".
- metrics: evaluation metrics (e.g. "BLEU", "accuracy", "F1").
- claimed_results: only numeric results stated in the abstract, with metric, dataset, value \
and a short context.
- contribution_type: one of new_method, new_dataset, benchmark, survey, analysis, \
application, other.
- Use short canonical names, without versions or articles."""

EXTRACTION_BATCH_USER = """\
Extract the knowledge of EACH paper below. Return {{"papers": [...]}} with one item per \
paper, in the same order, each with its "arxiv_id".

{papers}"""

EXTRACTION_SINGLE_USER = """\
Extract the knowledge of this paper.

{paper}"""

# ------------------------------------------------------- phase 5: GraphRAG

GRAPH_PARSE_SYSTEM = """\
You translate a question about a research knowledge graph into a graph query.
Node types: Paper, Author, Task, Method, Dataset, Metric, Category.
Relations (from Paper): WRITTEN_BY->Author, ADDRESSES->Task, PROPOSES_OR_USES->Method, \
EVALUATED_ON->Dataset, MEASURED_BY->Metric, IN_CATEGORY->Category, CITES->Paper; \
CO_AUTHOR between Authors.
Return: entity (the named thing in the question), entity_type, relation linking papers \
to that entity, and target_type (what the user wants listed: Paper, Method, Dataset...).
Examples:
- "Which papers use SQuAD?" -> entity "SQuAD", entity_type Dataset, relation EVALUATED_ON, \
target_type Paper
- "Which methods are used for machine translation?" -> entity "machine translation", \
entity_type Task, relation ADDRESSES, target_type Method"""

GRAPH_ANSWER_SYSTEM = f"""\
You answer questions using facts retrieved from a knowledge graph of papers.
{CITATION_RULES}
- Base the answer on the GRAPH FACTS and the CONTEXT abstracts only.
- If nothing relevant was found, reply exactly: {INSUFFICIENT}"""

GRAPH_ANSWER_USER = """\
QUESTION: {question}

GRAPH FACTS:
{facts}

CONTEXT:
{context}

Answer in at most 200 words, with [arXiv:ID] citations."""

# ---------------------------------------------------------- phase 6: agent

RECOMMEND_SYSTEM = f"""\
You explain paper recommendations in one sentence each.
{CITATION_RULES}"""

RECOMMEND_USER = """\
The user likes these papers:
{seeds}

Recommended papers:
{recs}

For EACH recommended paper, write one sentence explaining its link with the user's papers \
(shared task, method or data). Cite the recommended paper as [arXiv:ID]."""

GAP_SYSTEM = f"""\
You assess potential research gaps from co-occurrence statistics of a paper corpus.
- Present the gap as a HYPOTHESIS, never as a certainty ("It may be that...").
- Use only the numbers and papers given. Mention that low co-occurrence may also be due \
to corpus sampling or extraction errors.
{CITATION_RULES}"""

GAP_USER = """\
Task: {task} (papers: {n_task})
Method: {method} (papers: {n_method})
Papers combining both: {n_both} (expected if independent: {expected:.1f})

Papers on the task:
{task_papers}

Papers using the method:
{method_papers}

Give a one-sentence hypothesis and a short rationale (max 80 words) with citations."""

SOTA_PLAN_USER = """\
You plan a literature review (state of the art) on: "{topic}".
Write {n} focused search queries (sub-questions) covering: core approaches, datasets and \
evaluation, recent trends, limitations. Each query is short (3-10 words), suitable for a \
search engine over arXiv abstracts.
{feedback}"""

SOTA_GROUP_USER = """\
Group the following papers on "{topic}" into 2 to 6 approaches (families of methods).
Each paper id must appear in exactly one group. Use only the ids below.

{papers}"""

SOTA_WRITE_SYSTEM = f"""\
You write a structured literature review in Markdown.
{CITATION_RULES}
- Do NOT write a references section: it is generated automatically.
- Do not invent numbers; only use facts from the CONTEXT."""

SOTA_WRITE_USER = """\
TOPIC: {topic}

APPROACHES (groups of papers):
{groups}

COMMON DATASETS AND METRICS (from the knowledge graph; may be empty):
{datasets_metrics}

CONTEXT:
{context}

Write the review with these sections (Markdown headings):
## Introduction
## Approaches   (one ### subsection per approach)
## Datasets and Metrics
## Trends
## Limitations and Open Directions
{language}
Cite papers as [arXiv:ID] after each claim."""

DIGEST_SYSTEM = f"""\
You write a daily research digest from newly published arXiv papers.
{CITATION_RULES}"""

DIGEST_USER = """\
New papers for the query "{query}" (last {days} day(s)):

{papers}

Group the papers into 1 to 6 themes. For each theme, write a 2-3 sentence summary citing \
the papers as [arXiv:ID], and list its arxiv_ids. Every paper must belong to one theme."""

AGENT_SYSTEM = f"""\
You are a research assistant agent over a corpus of arXiv papers. You solve the user's \
request step by step, calling tools.

AVAILABLE TOOLS:
{{tools}}

At each step return a JSON decision:
- to call a tool: {{{{"thought": "...", "action": "<tool name>", "args": {{{{...}}}}}}}}
- to finish: {{{{"thought": "...", "action": "final_answer", "answer": "<markdown answer>"}}}}

Rules:
- Use at most {{max_iterations}} tool calls; finish as soon as you can answer.
- The final answer must rely only on tool observations.
{CITATION_RULES}"""

AGENT_OBSERVATION = """\
Observation from {tool}:
{observation}

Decide the next step (JSON)."""

# ----------------------------------------------------------- phase 9: eval

JUDGE_FAITHFULNESS_SYSTEM = """\
You are a strict evaluator of faithfulness. A claim is supported ONLY if the CONTEXT states \
it explicitly. Be severe: paraphrases are fine, additions are not."""

JUDGE_FAITHFULNESS_USER = """\
CONTEXT:
{context}

QUESTION: {question}

ANSWER:
{answer}

List the claims of the ANSWER not supported by the CONTEXT, then give score = fraction of \
supported claims (0 to 1). If the answer says it does not know, score = 1."""

JUDGE_RELEVANCE_USER = """\
QUESTION: {question}

ANSWER:
{answer}

Rate how well the ANSWER addresses the QUESTION from 1 (irrelevant) to 5 (fully relevant)."""

QUERY_GEN_USER = """\
Write a short search query (3-8 words) that a researcher could type to find this paper, \
WITHOUT copying the title verbatim.

Title: {title}
Abstract: {abstract}"""
