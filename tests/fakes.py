"""Test doubles: deterministic encoder, scripted LLM, tiny corpus."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from rpe.index.bm25 import tokenize


class HashEncoder:
    """Bag-of-words hashed into `dim` dimensions, L2-normalised."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 32, **kwargs: object) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class ScriptedLLM:
    """Fake completion function: returns answers in order, or from a router callable."""

    def __init__(self, responses: list[Any] | Callable[[list[dict]], Any]) -> None:
        self.responses = responses
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict], **kwargs: Any) -> str:
        self.calls.append(messages)
        if callable(self.responses):
            out = self.responses(messages)
        else:
            out = self.responses[min(len(self.calls), len(self.responses)) - 1]
        return out if isinstance(out, str) else json.dumps(out)


CORPUS = [
    (
        "2301.00001",
        "Retrieval-augmented generation for open-domain question answering",
        "We combine a dense retriever with a generator. Evaluated on SQuAD and Natural Questions.",
        2023,
        "ACL",
        True,
        ["cs.CL"],
    ),
    (
        "2302.00002",
        "Neural machine translation for low-resource languages",
        "We propose back-translation for low-resource machine translation, measured by BLEU.",
        2023,
        "EMNLP Findings",
        False,
        ["cs.CL", "cs.LG"],
    ),
    (
        "2403.00003",
        "LLM agents with tool use",
        "Large language model agents call external tools. Code at github.",
        2024,
        "NeurIPS",
        True,
        ["cs.AI", "cs.CL"],
    ),
    (
        "2404.00004",
        "Hallucination detection in large language models",
        "We detect hallucination in generation with retrieval and question answering checks.",
        2024,
        None,
        False,
        ["cs.CL"],
    ),
    (
        "2405.00005",
        "A survey of machine translation evaluation metrics",
        "This survey reviews BLEU, COMET and human evaluation for translation.",
        2024,
        "TACL",
        False,
        ["cs.CL"],
    ),
    (
        "2506.00006",
        "Dense passage retrieval revisited",
        "Dense retrieval with contrastive training improves question answering on SQuAD.",
        2025,
        "SIGIR",
        True,
        ["cs.IR", "cs.CL"],
    ),
]


def make_papers_df() -> pd.DataFrame:
    rows = []
    for i, (aid, title, abstract, year, venue, code, cats) in enumerate(CORPUS):
        rows.append(
            {
                "arxiv_id": aid,
                "title": title,
                "abstract": abstract,
                "authors_list": ["Ada Lovelace", f"Author {i}"]
                + (["Alan Turing"] if i % 2 else []),
                "n_authors": 2 + i % 2,
                "categories": cats,
                "primary_category": cats[0],
                "published_date": date(year, (i % 12) + 1, 1),
                "year": year,
                "month": (i % 12) + 1,
                "update_date": date(year, (i % 12) + 1, 2),
                "n_versions": 1,
                "comments": None,
                "journal_ref": None,
                "doi": None,
                "venue": venue,
                "venue_year": year if venue else None,
                "is_accepted": venue is not None,
                "github_url": "https://github.com/x/y" if code else None,
                "has_code": code,
                "n_pages": None,
                "n_figures": None,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("year", "month", "venue_year", "n_pages", "n_figures"):
        df[col] = df[col].astype("Int64")
    return df
