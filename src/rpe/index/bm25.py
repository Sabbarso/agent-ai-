"""BM25 index over "title + abstract" (rank_bm25), serialised with pickle."""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up us
    very was we were what when where which while who whom why will with would you your yours
    yourself yourselves via within without however thus using use used based show shows shown
    propose proposed paper work approach method results new one two may et al""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Very light plural stripping ("models" -> "model"), applied to docs and queries."""
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without English stopwords."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """BM25Okapi with the document ids kept alongside."""

    def __init__(self, ids: Sequence[str], tokenized: Sequence[list[str]]) -> None:
        self.ids = list(ids)
        # rank_bm25 divides by the corpus size: guard against empty corpora.
        self.bm25 = BM25Okapi([t or ["<empty>"] for t in tokenized]) if self.ids else None

    @classmethod
    def build(cls, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        return cls(ids, [tokenize(t) for t in texts])

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return np.zeros(len(self.ids))
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)

    def search(
        self, query: str, top_k: int, mask: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        """Top-k `(id, score)` with a positive score, ties broken by id order."""
        scores = self.scores(query)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> BM25Index:
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}. Run: rpe build-index")
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - local file produced by build-index
