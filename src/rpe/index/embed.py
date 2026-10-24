"""Dense embeddings of "title. abstract" (sentence-transformers), cached on disk.

Output: `<index_dir>/embeddings.npy` (float32, L2-normalised) + `ids.json`.
Resumable: embeddings are written in checkpoint files (`emb_parts/part_XXXXX.npy`)
and existing parts are skipped on restart.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Encoder(Protocol):
    def encode(self, texts: list[str], batch_size: int = ..., **kwargs: object) -> np.ndarray: ...


def embedding_text(title: str, abstract: str) -> str:
    """Text embedded for a paper."""
    return f"{title.strip()}. {abstract.strip()}".strip()


class Embedder:
    """Lazy sentence-transformers wrapper producing L2-normalised float32 vectors."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        query_instruction: str = "",
        encoder: Encoder | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self._encoder = encoder

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def embed_documents(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        vecs = self.encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=progress,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([self.query_instruction + query])[0]


def _part_path(parts_dir: Path, i: int) -> Path:
    return parts_dir / f"part_{i:05d}.npy"


def build_embeddings(
    ids: Sequence[str],
    texts: Sequence[str],
    embedder: Embedder,
    out_dir: Path,
    checkpoint_every: int = 2048,
    force: bool = False,
) -> np.ndarray:
    """Embed `texts` (aligned with `ids`) with checkpointing; returns the full matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path, ids_path = out_dir / "embeddings.npy", out_dir / "ids.json"
    if not force and emb_path.exists() and ids_path.exists():
        if json.loads(ids_path.read_text(encoding="utf-8")) == list(ids):
            logger.info("Embeddings up to date in %s", out_dir)
            return np.load(emb_path)
        logger.info("Ids changed: recomputing embeddings")

    parts_dir = out_dir / "emb_parts"
    parts_dir.mkdir(exist_ok=True)
    meta_path = parts_dir / "meta.json"
    meta = {"model": embedder.model_name, "n": len(ids), "chunk": checkpoint_every}
    if force or not meta_path.exists() or json.loads(meta_path.read_text()) != meta:
        for p in parts_dir.glob("part_*.npy"):
            p.unlink()
        meta_path.write_text(json.dumps(meta))

    n_parts = (len(ids) + checkpoint_every - 1) // checkpoint_every
    for i in tqdm(range(n_parts), desc="Embedding (parts)", unit="part"):
        part = _part_path(parts_dir, i)
        if part.exists():
            continue
        chunk = texts[i * checkpoint_every : (i + 1) * checkpoint_every]
        tmp = part.with_suffix(".tmp.npy")
        np.save(tmp, embedder.embed_documents(chunk))
        tmp.replace(part)

    matrix = (
        np.concatenate([np.load(_part_path(parts_dir, i)) for i in range(n_parts)])
        if n_parts
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.save(emb_path, matrix)
    ids_path.write_text(json.dumps(list(ids)), encoding="utf-8")
    return matrix


def load_embeddings(index_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load `(matrix, ids)`; raises FileNotFoundError with the command to run."""
    emb_path, ids_path = index_dir / "embeddings.npy", index_dir / "ids.json"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Embeddings not found in {index_dir}. Run: rpe build-index")
    return np.load(emb_path), json.loads(ids_path.read_text(encoding="utf-8"))
