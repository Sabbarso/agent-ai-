"""Knowledge-graph construction from papers + LLM extractions."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rpe.config import Config
from rpe.data.store import PaperStore, as_list
from rpe.kg.extract import cache_path, load_cache
from rpe.kg.graph_store import NetworkXGraphStore, node_id
from rpe.kg.normalize import EntityNormalizer, comparison_key
from rpe.llm.schemas import PaperExtraction

logger = logging.getLogger(__name__)

# extraction field -> (node type, normaliser kind, relation)
ENTITY_FIELDS = {
    "tasks": ("Task", "task", "ADDRESSES"),
    "methods": ("Method", "method", "PROPOSES_OR_USES"),
    "datasets": ("Dataset", "dataset", "EVALUATED_ON"),
    "metrics": ("Metric", "metric", "MEASURED_BY"),
}


@dataclass
class GraphReport:
    nodes: dict[str, int]
    edges: dict[str, int]
    n_extracted: int


def author_key(name: str) -> str:
    return comparison_key(name) or name.casefold()


def fit_normalizer(
    extractions: Mapping[str, PaperExtraction], normalizer: EntityNormalizer
) -> EntityNormalizer:
    mentions = {
        kind: [m for e in extractions.values() for m in getattr(e, field)]
        for field, (_, kind, _) in ENTITY_FIELDS.items()
    }
    return normalizer.fit(mentions)


def build_graph(
    papers: pd.DataFrame,
    extractions: Mapping[str, PaperExtraction],
    normalizer: EntityNormalizer,
    max_coauthors: int = 30,
    progress: bool = False,
) -> NetworkXGraphStore:
    """Paper/Author/Category nodes for all papers; entity nodes for extracted ones."""
    fit_normalizer(extractions, normalizer)
    store = NetworkXGraphStore()
    for row in tqdm(papers.to_dict("records"), desc="Graph", disable=not progress):
        aid = row["arxiv_id"]
        pid = node_id("Paper", aid)
        year = row.get("year")
        store.add_node(
            pid,
            "Paper",
            row["title"],
            arxiv_id=aid,
            year=int(year) if year is not None and not pd.isna(year) else -1,
            venue=row.get("venue") if isinstance(row.get("venue"), str) else "",
            has_code=bool(row.get("has_code")),
            is_accepted=bool(row.get("is_accepted")),
        )
        for cat in as_list(row.get("categories")):
            cid = node_id("Category", cat)
            store.add_node(cid, "Category", cat)
            store.add_edge(pid, cid, "IN_CATEGORY")

        authors = []
        for name in as_list(row.get("authors_list")):
            aid_node = node_id("Author", author_key(name))
            store.add_node(aid_node, "Author", name)
            store.add_edge(pid, aid_node, "WRITTEN_BY")
            authors.append(aid_node)
        if 1 < len(authors) <= max_coauthors:
            for a, b in itertools.combinations(sorted(set(authors)), 2):
                store.add_edge(a, b, "CO_AUTHOR")

        ext = extractions.get(aid)
        if ext is None:
            continue
        store.g.nodes[pid]["contribution_type"] = ext.contribution_type
        store.g.nodes[pid]["extracted"] = True
        for field, (ntype, kind, relation) in ENTITY_FIELDS.items():
            for mention in getattr(ext, field):
                canonical = normalizer.normalize(mention, kind)
                key = normalizer.key(mention, kind)
                if not canonical or not key:
                    continue
                eid = node_id(ntype, key)
                store.add_node(eid, ntype, canonical)
                store.add_edge(pid, eid, relation)
    return store


def graph_path(cfg: Config, dev: bool = False) -> Path:
    return cfg.artifact(cfg.graph.path, dev)


def load_normalizer(cfg: Config) -> EntityNormalizer:
    return EntityNormalizer.load(cfg.path(cfg.graph.aliases_path), cfg.graph.fuzzy_threshold)


def run_build_graph(cfg: Config, dev: bool = False) -> tuple[NetworkXGraphStore, GraphReport]:
    papers = PaperStore.load(cfg, dev)
    extractions = load_cache(cache_path(cfg, dev))
    if not extractions:
        logger.warning("No extractions found: graph will only contain papers/authors/categories")
    store = build_graph(
        papers.df, extractions, load_normalizer(cfg), cfg.graph.max_coauthors_per_paper, True
    )
    store.save(graph_path(cfg, dev))
    return store, GraphReport(store.count_by_type(), store.count_by_relation(), len(extractions))


def load_graph(cfg: Config, dev: bool = False) -> NetworkXGraphStore:
    return NetworkXGraphStore.load(graph_path(cfg, dev))
