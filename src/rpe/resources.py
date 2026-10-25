"""Lazily-loaded shared resources (papers, search indexes, LLM, PDF chunks)."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

from rpe.config import Config
from rpe.data.store import PaperStore
from rpe.llm.client import LLMClient, get_llm

logger = logging.getLogger(__name__)


class Resources:
    """Loaded on first use and shared by the CLI and the Streamlit app."""

    def __init__(self, cfg: Config, dev: bool = False, llm: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.dev = dev
        self._llm = llm

    @cached_property
    def llm(self) -> LLMClient | None:
        return self._llm or get_llm(self.cfg)

    @cached_property
    def papers(self) -> PaperStore:
        return PaperStore.load(self.cfg, self.dev)

    @cached_property
    def searcher(self) -> Any:
        from rpe.index.hybrid import load_searcher

        return load_searcher(self.cfg, self.dev)

    @cached_property
    def chunks(self) -> Any:
        """Full-text chunk store, or None if no PDF has been indexed."""
        try:
            from rpe.pdf.chunks import ChunkStore

            store = ChunkStore(self.cfg, self.dev)
            return store if store.count() else None
        except Exception as exc:  # noqa: BLE001 - optional component
            logger.debug("No chunk store: %s", exc)
            return None
