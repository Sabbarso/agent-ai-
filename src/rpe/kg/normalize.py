"""Entity normalisation: cleaning, editable alias table, fuzzy merging.

Pipeline for a mention of a given kind (task / method / dataset / metric):
1. `clean_surface`: strip quotes/punctuation, versions ("SQuAD v1.1" -> "SQuAD"), and
   generic suffixes after a proper name ("SQuAD dataset" -> "SQuAD", "BERT model" -> "BERT").
2. `comparison_key`: casefold, alphanumerics only, light plural stripping
   ("Question-Answering" == "question answering").
3. Alias table (`data/processed/aliases.yaml`, editable): `{kind: {Canonical: [aliases]}}`.
4. Fuzzy merge (difflib ratio >= threshold) of keys sharing their first 4 characters;
   the most frequent surface form becomes the canonical name.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

KINDS = ("task", "method", "dataset", "metric")

# "SQuAD v1.1", "SQuAD (v2)", "X version 3" -> removed; "GPT-3.5", "YOLOv5" are kept.
_VERSION_RE = re.compile(
    r"(?:\s+|\s*[\(\[]\s*)(?:v|version\s*)\d+(?:\.\d+)*\s*[\)\]]?$", re.IGNORECASE
)
_SUFFIXES = {
    "dataset": ("dataset", "datasets", "benchmark", "corpus", "data set"),
    "method": ("model", "models", "method", "framework", "architecture", "approach"),
    "metric": ("score", "scores", "metric"),
    "task": ("task", "tasks"),
}
_STRIP_CHARS = " \t\"'`“”‘’.,;:()[]{}"
_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_ALIASES: dict[str, dict[str, list[str]]] = {
    "dataset": {
        "SQuAD": ["Stanford Question Answering Dataset", "SQuAD1.1", "SQuAD 1.1"],
        "GLUE": ["General Language Understanding Evaluation"],
        "MMLU": ["Massive Multitask Language Understanding"],
        "GSM8K": ["GSM-8K", "Grade School Math 8K"],
        "Natural Questions": ["NQ", "NaturalQuestions"],
    },
    "task": {
        "Machine Translation": ["MT", "translation", "neural machine translation"],
        "Question Answering": ["QA"],
        "Named Entity Recognition": ["NER"],
        "Natural Language Inference": ["NLI", "textual entailment"],
        "Sentiment Analysis": ["sentiment classification"],
        "Summarization": ["text summarization", "summarisation"],
    },
    "method": {
        "Retrieval-Augmented Generation": ["RAG", "retrieval augmented generation"],
        "Large Language Models": ["LLM", "LLMs", "large language model"],
        "Chain-of-Thought": ["CoT", "chain-of-thought prompting", "chain of thought"],
        "RLHF": ["reinforcement learning from human feedback"],
        "LoRA": ["low-rank adaptation"],
        "In-Context Learning": ["ICL"],
    },
    "metric": {
        "F1": ["F1 score", "F1-score", "F-score", "F-measure"],
        "BLEU": ["BLEU score"],
        "ROUGE": ["ROUGE score"],
        "Accuracy": ["acc"],
        "Exact Match": ["EM"],
    },
}


def _depluralize(token: str) -> str:
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def comparison_key(name: str) -> str:
    """Case/punctuation/plural-insensitive key used for matching."""
    return "".join(_depluralize(t) for t in _TOKEN_RE.findall(name.casefold()))


def clean_surface(name: str, kind: str = "") -> str:
    """Display-form cleaning (see module docstring)."""
    s = re.sub(r"\s+", " ", str(name)).strip(_STRIP_CHARS)
    s = _VERSION_RE.sub("", s).strip(_STRIP_CHARS)
    for suffix in _SUFFIXES.get(kind, ()):
        if s.lower().endswith(" " + suffix):
            head = s[: -len(suffix) - 1].strip()
            # Only after a proper name (uppercase or digit): "language model" is kept.
            if head and re.search(r"[A-Z0-9]", head):
                s = head
            break
    return s


class EntityNormalizer:
    """Maps mentions to canonical names per kind."""

    def __init__(
        self,
        aliases: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
        threshold: float = 0.92,
    ) -> None:
        self.threshold = threshold
        self.alias_to_canonical: dict[str, dict[str, str]] = defaultdict(dict)
        for kind, table in (aliases or {}).items():
            for canonical, alts in table.items():
                for a in [canonical, *alts]:
                    self.alias_to_canonical[kind][comparison_key(clean_surface(a, kind))] = (
                        canonical
                    )
        self.mapping: dict[str, dict[str, str]] = defaultdict(dict)  # kind -> key -> canonical

    @classmethod
    def load(cls, path: Path, threshold: float = 0.92) -> EntityNormalizer:
        """Load the alias YAML, creating it with defaults when missing."""
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Alias table: kind -> Canonical name -> [aliases]. Edit freely.\n"
                + yaml.safe_dump(DEFAULT_ALIASES, sort_keys=True, allow_unicode=True),
                encoding="utf-8",
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(data, threshold)

    def _alias(self, kind: str, key: str) -> str | None:
        return self.alias_to_canonical.get(kind, {}).get(key)

    def fit(self, mentions: Mapping[str, Iterable[str]]) -> EntityNormalizer:
        """Learn canonical names from all mentions of each kind."""
        for kind, names in mentions.items():
            surfaces: dict[str, Counter[str]] = defaultdict(Counter)
            for n in names:
                s = clean_surface(n, kind)
                if s:
                    surfaces[comparison_key(s) or s.casefold()][s] += 1
            totals = {k: sum(c.values()) for k, c in surfaces.items()}
            parent = {k: k for k in surfaces}

            # Fuzzy merge within blocks of keys sharing a 4-char prefix.
            blocks: dict[str, list[str]] = defaultdict(list)
            for k in surfaces:
                blocks[k[:4]].append(k)
            for keys in blocks.values():
                keys.sort(key=lambda k: (-totals[k], k))
                for i, k in enumerate(keys):
                    if parent[k] != k or self._alias(kind, k):
                        continue
                    for big in keys[:i]:
                        if parent[big] == big and (
                            difflib.SequenceMatcher(None, k, big).ratio() >= self.threshold
                        ):
                            parent[k] = big
                            break

            groups: dict[str, list[str]] = defaultdict(list)
            for k in surfaces:
                groups[parent[k]].append(k)
            for root, members in groups.items():
                merged: Counter[str] = Counter()
                for m in members:
                    merged.update(surfaces[m])
                canonical = (
                    next((c for m in members if (c := self._alias(kind, m))), None)
                    or merged.most_common(1)[0][0]
                )
                for m in members:
                    self.mapping[kind][m] = self._alias(kind, m) or canonical
                self.mapping[kind].setdefault(root, canonical)
        return self

    def normalize(self, name: str, kind: str) -> str | None:
        """Canonical display name, or `None` for empty mentions."""
        s = clean_surface(name, kind)
        if not s:
            return None
        key = comparison_key(s) or s.casefold()
        return self._alias(kind, key) or self.mapping.get(kind, {}).get(key) or s

    def key(self, name: str, kind: str) -> str | None:
        """Stable node key of the canonical form."""
        canonical = self.normalize(name, kind)
        if canonical is None:
            return None
        return comparison_key(canonical) or canonical.casefold()
