"""Graph storage behind a `GraphStore` interface.

`NetworkXGraphStore` (default) keeps a `MultiDiGraph` whose edge keys are relation names;
it is saved as GraphML (`kg.graphml`, interoperable) plus a pickle (fast reload).
`Neo4jGraphStore` is optional: it needs the `neo4j` package and a running server, and is
never required by the rest of the code.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import networkx as nx

logger = logging.getLogger(__name__)

NODE_TYPES = ("Paper", "Author", "Task", "Method", "Dataset", "Metric", "Category")
RELATIONS: dict[str, tuple[str, str]] = {
    "WRITTEN_BY": ("Paper", "Author"),
    "ADDRESSES": ("Paper", "Task"),
    "PROPOSES_OR_USES": ("Paper", "Method"),
    "EVALUATED_ON": ("Paper", "Dataset"),
    "MEASURED_BY": ("Paper", "Metric"),
    "IN_CATEGORY": ("Paper", "Category"),
    "CO_AUTHOR": ("Author", "Author"),
    "CITES": ("Paper", "Paper"),
}
Direction = Literal["out", "in", "both"]


def node_id(node_type: str, key: str) -> str:
    return f"{node_type.lower()}:{key}"


class GraphStore(ABC):
    """Minimal property-graph interface used by GraphRAG, gaps and the UI."""

    @abstractmethod
    def add_node(self, nid: str, node_type: str, name: str, **attrs: Any) -> None: ...

    @abstractmethod
    def add_edge(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None: ...

    @abstractmethod
    def has_node(self, nid: str) -> bool: ...

    @abstractmethod
    def node(self, nid: str) -> dict[str, Any]: ...

    @abstractmethod
    def nodes(self, node_type: str | None = None) -> list[str]: ...

    @abstractmethod
    def neighbors(
        self, nid: str, relation: str | None = None, direction: Direction = "out"
    ) -> list[str]: ...

    @abstractmethod
    def edge_weight(self, src: str, dst: str, relation: str) -> float: ...


class NetworkXGraphStore(GraphStore):
    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self.g = graph if graph is not None else nx.MultiDiGraph()

    # ------------------------------------------------------------- writes

    def add_node(self, nid: str, node_type: str, name: str, **attrs: Any) -> None:
        clean = {k: v for k, v in attrs.items() if v is not None}
        if nid in self.g:
            self.g.nodes[nid].update(clean)
        else:
            self.g.add_node(nid, type=node_type, name=name, **clean)

    def add_edge(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None:
        if self.g.has_edge(src, dst, key=relation):
            self.g.edges[src, dst, relation]["weight"] += weight
        else:
            self.g.add_edge(src, dst, key=relation, relation=relation, weight=weight)

    # -------------------------------------------------------------- reads

    def has_node(self, nid: str) -> bool:
        return nid in self.g

    def node(self, nid: str) -> dict[str, Any]:
        return {"id": nid, **self.g.nodes[nid]} if nid in self.g else {}

    def nodes(self, node_type: str | None = None) -> list[str]:
        if node_type is None:
            return list(self.g.nodes)
        return [n for n, t in self.g.nodes(data="type") if t == node_type]

    def neighbors(
        self, nid: str, relation: str | None = None, direction: Direction = "out"
    ) -> list[str]:
        if nid not in self.g:
            return []
        out: dict[str, None] = {}
        if direction in ("out", "both"):
            for _, v, k in self.g.out_edges(nid, keys=True):
                if relation is None or k == relation:
                    out.setdefault(v, None)
        if direction in ("in", "both"):
            for u, _, k in self.g.in_edges(nid, keys=True):
                if relation is None or k == relation:
                    out.setdefault(u, None)
        return list(out)

    def edge_weight(self, src: str, dst: str, relation: str) -> float:
        for a, b in ((src, dst), (dst, src)):
            if self.g.has_edge(a, b, key=relation):
                return float(self.g.edges[a, b, relation].get("weight", 1.0))
        return 0.0

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, t in self.g.nodes(data="type"):
            counts[t] = counts.get(t, 0) + 1
        return counts

    def count_by_relation(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for *_, k in self.g.edges(keys=True):
            counts[k] = counts.get(k, 0) + 1
        return counts

    def subgraph(self, nids: Iterable[str]) -> nx.MultiDiGraph:
        return self.g.subgraph(list(nids)).copy()

    # ---------------------------------------------------------- persistence

    def save(self, graphml_path: Path) -> None:
        graphml_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            nx.write_graphml(self.g, graphml_path)
        except (ValueError, TypeError) as exc:  # e.g. control characters in a title
            logger.warning("GraphML export failed (%s); the pickle is still written", exc)
        with graphml_path.with_suffix(".pkl").open("wb") as fh:
            pickle.dump(self.g, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, graphml_path: Path) -> NetworkXGraphStore:
        pkl = graphml_path.with_suffix(".pkl")
        if pkl.exists() and (
            not graphml_path.exists() or pkl.stat().st_mtime >= graphml_path.stat().st_mtime
        ):
            with pkl.open("rb") as fh:
                return cls(pickle.load(fh))  # noqa: S301 - local file written by save()
        if graphml_path.exists():
            return cls(nx.read_graphml(graphml_path, force_multigraph=True))
        raise FileNotFoundError(f"Graph not found: {graphml_path}. Run: rpe build-graph")


class Neo4jGraphStore(GraphStore):  # pragma: no cover - optional backend
    """Optional Neo4j backend (`pip install neo4j`); same interface, Cypher underneath."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def _run(self, query: str, **params: Any) -> list[Any]:
        with self.driver.session() as s:
            return list(s.run(query, **params))

    def add_node(self, nid: str, node_type: str, name: str, **attrs: Any) -> None:
        self._run(
            f"MERGE (n:{node_type} {{id: $id}}) SET n.name = $name, n += $attrs",
            id=nid,
            name=name,
            attrs={k: v for k, v in attrs.items() if v is not None},
        )

    def add_edge(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None:
        self._run(
            f"MATCH (a {{id: $s}}), (b {{id: $d}}) MERGE (a)-[r:{relation}]->(b) "
            "ON CREATE SET r.weight = $w ON MATCH SET r.weight = r.weight + $w",
            s=src,
            d=dst,
            w=weight,
        )

    def has_node(self, nid: str) -> bool:
        return bool(self._run("MATCH (n {id: $id}) RETURN n LIMIT 1", id=nid))

    def node(self, nid: str) -> dict[str, Any]:
        rows = self._run("MATCH (n {id: $id}) RETURN n, labels(n)[0] AS t", id=nid)
        return {**dict(rows[0]["n"]), "type": rows[0]["t"]} if rows else {}

    def nodes(self, node_type: str | None = None) -> list[str]:
        label = f":{node_type}" if node_type else ""
        return [r["id"] for r in self._run(f"MATCH (n{label}) RETURN n.id AS id")]

    def neighbors(
        self, nid: str, relation: str | None = None, direction: Direction = "out"
    ) -> list[str]:
        rel = f":{relation}" if relation else ""
        pattern = {"out": f"-[{rel}]->", "in": f"<-[{rel}]-", "both": f"-[{rel}]-"}[direction]
        q = f"MATCH (a {{id: $id}}){pattern}(b) RETURN DISTINCT b.id AS id"
        return [r["id"] for r in self._run(q, id=nid)]

    def edge_weight(self, src: str, dst: str, relation: str) -> float:
        rows = self._run(
            f"MATCH (a {{id: $s}})-[r:{relation}]-(b {{id: $d}}) RETURN r.weight AS w",
            s=src,
            d=dst,
        )
        return float(rows[0]["w"]) if rows else 0.0

    @classmethod
    def from_networkx(
        cls, store: NetworkXGraphStore, uri: str, user: str, pw: str
    ) -> Neo4jGraphStore:
        neo = cls(uri, user, pw)
        for n, data in store.g.nodes(data=True):
            attrs = {k: v for k, v in data.items() if k not in ("type", "name")}
            neo.add_node(n, data["type"], data.get("name", n), **attrs)
        for u, v, k, data in store.g.edges(keys=True, data=True):
            neo.add_edge(u, v, k, data.get("weight", 1.0))
        return neo
