"""Read-only access to the prepared papers (Parquet) by `arxiv_id`."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from rpe.config import Config


def as_list(value: Any) -> list[Any]:
    """List columns read from Parquet are numpy arrays: normalise to a Python list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return list(value)


def _plain(value: Any) -> Any:
    """Convert numpy / pandas scalars and arrays to plain Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


class PaperStore:
    """Thin wrapper around the papers DataFrame, indexed by `arxiv_id`."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.set_index("arxiv_id", drop=False)
        self.df.index.name = None  # keep "arxiv_id" unambiguous as a column

    @classmethod
    def load(cls, cfg: Config, dev: bool = False) -> PaperStore:
        path = cfg.papers_path(dev)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m rpe.cli prepare-data" + (" --dev" if dev else "")
            )
        return cls(pd.read_parquet(path))

    @classmethod
    def from_path(cls, path: Path) -> PaperStore:
        return cls(pd.read_parquet(path))

    def __len__(self) -> int:
        return len(self.df)

    def __contains__(self, arxiv_id: object) -> bool:
        return arxiv_id in self.df.index

    @property
    def ids(self) -> list[str]:
        return self.df["arxiv_id"].tolist()

    def get(self, arxiv_id: str) -> dict[str, Any] | None:
        """One paper as a plain dict, or `None` if unknown."""
        if arxiv_id not in self.df.index:
            return None
        return {k: _plain(v) for k, v in self.df.loc[arxiv_id].items()}

    def many(self, arxiv_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Known papers among `arxiv_ids`, in the given order."""
        return [p for i in arxiv_ids if (p := self.get(i)) is not None]
