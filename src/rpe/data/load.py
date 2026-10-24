"""Streaming reader for the arXiv metadata snapshot (JSON Lines, ~5.6 GB).

The file is read line by line in binary mode and never loaded into memory.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Counters filled while streaming."""

    n_lines: int = 0
    n_invalid: int = 0
    n_prefiltered: int = 0

    @property
    def n_parsed(self) -> int:
        return self.n_lines - self.n_invalid - self.n_prefiltered


def stream_records(
    path: str | Path,
    stats: StreamStats | None = None,
    raw_filter: Callable[[bytes], bool] | None = None,
    progress: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield one dict per valid JSON line; invalid lines are skipped and counted.

    `raw_filter` is an optional cheap test on the raw bytes: lines it rejects are
    skipped without JSON parsing (counted in `stats.n_prefiltered`, not validated).
    """
    path = Path(path)
    stats = stats if stats is not None else StreamStats()
    with (
        path.open("rb") as fh,
        tqdm(
            total=path.stat().st_size,
            unit="B",
            unit_scale=True,
            desc=f"Reading {path.name}",
            disable=not progress,
            mininterval=0.5,
        ) as bar,
    ):
        for raw in fh:
            bar.update(len(raw))
            stats.n_lines += 1
            line = raw.strip()
            if raw_filter is not None and line and not raw_filter(line):
                stats.n_prefiltered += 1
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError:
                record = None
            if not isinstance(record, dict):
                stats.n_invalid += 1
                logger.debug("Invalid JSON at line %d", stats.n_lines)
                continue
            yield record
    if stats.n_invalid:
        logger.warning("%d invalid lines skipped in %s", stats.n_invalid, path)
