"""Dense vector stores: persistent Chroma (default) and an exact in-memory NumPy store."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from rpe.index.filters import SearchFilters

logger = logging.getLogger(__name__)


class DenseIndex(Protocol):
    def query(
        self, vector: np.ndarray, top_k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        """Top-k `(id, cosine similarity)`."""
        ...


class NumpyVectorStore:
    """Exact cosine search on a normalised matrix; `meta` is aligned with `ids`."""

    def __init__(self, matrix: np.ndarray, ids: Sequence[str], meta: pd.DataFrame) -> None:
        self.matrix = matrix
        self.ids = list(ids)
        self.meta = meta.reset_index(drop=True)
        self._pos = {i: k for k, i in enumerate(self.ids)}

    def vector(self, arxiv_id: str) -> np.ndarray | None:
        k = self._pos.get(arxiv_id)
        return None if k is None else self.matrix[k]

    def query(
        self, vector: np.ndarray, top_k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.matrix @ vector.astype(np.float32)
        if filters is not None and not filters.is_empty():
            sims = np.where(filters.mask(self.meta), sims, -np.inf)
        order = np.argsort(-sims, kind="stable")[:top_k]
        return [(self.ids[i], float(sims[i])) for i in order if np.isfinite(sims[i])]


class ChromaVectorStore:
    """Persistent Chroma collection with cosine distance and paper metadata."""

    def __init__(self, path: Path, collection: str = "papers") -> None:
        import chromadb
        from chromadb.config import Settings

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def existing_ids(self) -> set[str]:
        return set(self.collection.get(include=[])["ids"])

    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
        documents: Sequence[str] | None = None,
        batch_size: int = 2000,
    ) -> int:
        """Add the vectors whose id is not stored yet (resumable); returns #added."""
        existing = self.existing_ids()
        todo = [k for k, i in enumerate(ids) if i not in existing]
        for start in range(0, len(todo), batch_size):
            sel = todo[start : start + batch_size]
            self.collection.add(
                ids=[ids[k] for k in sel],
                embeddings=embeddings[sel].tolist(),
                metadatas=[metadatas[k] for k in sel],
                documents=[documents[k] for k in sel] if documents is not None else None,
            )
        return len(todo)

    def query(
        self,
        vector: np.ndarray,
        top_k: int,
        filters: SearchFilters | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if where is None and filters is not None:
            where = filters.to_chroma_where()
        n = min(top_k, self.count())
        if n == 0:
            return []
        res = self.collection.query(
            query_embeddings=[vector.tolist()], n_results=n, where=where, include=["distances"]
        )
        return [
            (i, 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0], strict=True)
        ]
