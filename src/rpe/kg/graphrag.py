"""GraphRAG: structured graph queries + LLM answers grounded in linked abstracts.

graph_query("SQuAD", "EVALUATED_ON")                         -> papers evaluated on SQuAD
graph_query("machine translation", "ADDRESSES", target_type="Method")
                                                             -> methods used for the task
answer_graph_question(question): LLM parses the question into a `GraphQuery` (rule-based
fallback), the graph is queried, linked abstracts are retrieved and the LLM answers with
citations checked by `validate_citations()`.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from rpe.data.store import PaperStore
from rpe.kg.graph_store import RELATIONS, GraphStore, NetworkXGraphStore, node_id
from rpe.kg.normalize import EntityNormalizer, comparison_key
from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import GraphQuery
from rpe.rag.citations import validate_citations
from rpe.rag.qa import IDK_MESSAGE

logger = logging.getLogger(__name__)

TYPE_TO_RELATION = {dst: rel for rel, (src, dst) in RELATIONS.items() if src == "Paper"}
TYPE_TO_RELATION.pop("Paper", None)
TYPE_TO_KIND = {"Task": "task", "Method": "method", "Dataset": "dataset", "Metric": "metric"}


class GraphQueryResult(BaseModel):
    query: str
    entity_type: str
    relation: str
    target_type: str = "Paper"
    node_id: str | None = None
    entity_name: str | None = None
    papers: list[str] = Field(default_factory=list)
    entities: list[tuple[str, int]] = Field(default_factory=list)  # (name, count)

    @property
    def matched(self) -> bool:
        return self.node_id is not None


class GraphAnswer(BaseModel):
    question: str
    parsed: GraphQuery | None = None
    result: GraphQueryResult | None = None
    answer: str
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------- matching


def find_entity(
    store: GraphStore,
    name: str,
    entity_type: str,
    normalizer: EntityNormalizer | None = None,
) -> str | None:
    """Node id of `name` among nodes of `entity_type`: exact key, alias, fuzzy, substring."""
    if entity_type == "Paper":
        nid = node_id("Paper", name.replace("arXiv:", "").strip())
        return nid if store.has_node(nid) else None
    kind = TYPE_TO_KIND.get(entity_type)
    keys = [comparison_key(name)]
    if normalizer is not None and kind and (k := normalizer.key(name, kind)):
        keys.insert(0, k)
    if entity_type == "Category":
        keys.insert(0, name.strip())
    for k in keys:
        if k and store.has_node(nid := node_id(entity_type, k)):
            return nid
    prefix = node_id(entity_type, "")
    candidates = {n[len(prefix) :]: n for n in store.nodes(entity_type)}
    key = keys[-1]
    if not key or not candidates:
        return None
    close = difflib.get_close_matches(key, list(candidates), n=1, cutoff=0.85)
    if close:
        return candidates[close[0]]
    contains = [c for c in candidates if len(key) >= 3 and key in c]
    if contains:
        return candidates[min(contains, key=lambda c: (len(c), c))]
    return None


def _paper_sort_key(store: GraphStore, pid: str) -> tuple:
    n = store.node(pid)
    return (-(n.get("year") or -1), not n.get("venue"), not n.get("has_code"), pid)


# ---------------------------------------------------------------- queries


def graph_query(
    store: GraphStore,
    entity: str,
    relation: str | None = None,
    entity_type: str | None = None,
    target_type: str = "Paper",
    limit: int = 50,
    normalizer: EntityNormalizer | None = None,
) -> GraphQueryResult:
    """Papers (or aggregated entities of `target_type`) linked to `entity` via `relation`."""
    if entity_type is None:
        entity_type = RELATIONS[relation][1] if relation else "Task"
    relation = relation or TYPE_TO_RELATION.get(entity_type, "ADDRESSES")
    res = GraphQueryResult(
        query=entity, entity_type=entity_type, relation=relation, target_type=target_type
    )
    nid = find_entity(store, entity, entity_type, normalizer)
    if nid is None:  # fall back on the other entity types
        for t, rel in TYPE_TO_RELATION.items():
            if t != entity_type and (nid := find_entity(store, entity, t, normalizer)):
                res.entity_type, res.relation = t, rel
                break
    if nid is None:
        return res
    res.node_id, res.entity_name = nid, store.node(nid).get("name")

    if res.relation == "CO_AUTHOR":
        co = store.neighbors(nid, "CO_AUTHOR", "both")
        ranked = sorted(co, key=lambda a: (-store.edge_weight(nid, a, "CO_AUTHOR"), a))
        res.entities = [
            (store.node(a)["name"], int(store.edge_weight(nid, a, "CO_AUTHOR")))
            for a in ranked[:limit]
        ]
        res.papers = [p.split(":", 1)[1] for p in store.neighbors(nid, "WRITTEN_BY", "in")][:limit]
        return res
    if res.entity_type == "Paper":
        targets = store.neighbors(nid, res.relation, "out")
        res.entities = [(store.node(t).get("name", t), 1) for t in targets[:limit]]
        res.papers = [nid.split(":", 1)[1]]
        return res

    papers = sorted(
        store.neighbors(nid, res.relation, "in"), key=lambda p: _paper_sort_key(store, p)
    )
    res.papers = [p.split(":", 1)[1] for p in papers]
    if target_type != "Paper" and (t_rel := TYPE_TO_RELATION.get(target_type)):
        counts: Counter[str] = Counter()
        for p in papers:
            counts.update(t for t in store.neighbors(p, t_rel, "out") if t != nid)
        res.entities = [
            (store.node(t)["name"], c)
            for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        ]
    res.papers = res.papers[:limit]
    return res


_RULES: list[tuple[re.Pattern[str], str, str, str]] = [
    (re.compile(p, re.IGNORECASE), et, rel, tt)
    for p, et, rel, tt in [
        (r"(?:co-?authors?|collaborators?) of (?P<e>.+)", "Author", "CO_AUTHOR", "Author"),
        (r"papers? (?:written )?by (?P<e>.+)", "Author", "WRITTEN_BY", "Paper"),
        (
            r"methods? (?:are )?(?:used |applied |proposed )?(?:for|to|in) (?:the )?(?P<e>.+)",
            "Task",
            "ADDRESSES",
            "Method",
        ),
        (
            r"datasets? (?:are )?(?:used )?(?:for|in) (?:the )?(?P<e>.+)",
            "Task",
            "ADDRESSES",
            "Dataset",
        ),
        (
            r"metrics? (?:are )?(?:used )?(?:for|in) (?:the )?(?P<e>.+)",
            "Task",
            "ADDRESSES",
            "Metric",
        ),
        (
            r"papers? (?:that |which )?(?:use|uses|using|evaluated on|evaluate on|report on) "
            r"(?:the )?(?P<e>.+)",
            "Dataset",
            "EVALUATED_ON",
            "Paper",
        ),
        (r"papers? (?:on|about|addressing) (?:the )?(?P<e>.+)", "Task", "ADDRESSES", "Paper"),
        (
            r"m[ée]thodes? (?:utilis[ée]es? )?pour (?:la |le |les |l')?(?P<e>.+)",
            "Task",
            "ADDRESSES",
            "Method",
        ),
        (
            r"papiers? (?:qui )?(?:utilisent|utilise|évalués sur) (?P<e>.+)",
            "Dataset",
            "EVALUATED_ON",
            "Paper",
        ),
    ]
]


def parse_question_rules(question: str) -> GraphQuery | None:
    """Rule-based question parsing (fallback when no LLM)."""
    q = question.strip().rstrip("?").strip()
    for rx, et, rel, tt in _RULES:
        if m := rx.search(q):
            entity = m.group("e").strip(" ?.\"'")
            if entity:
                return GraphQuery(entity=entity, entity_type=et, relation=rel, target_type=tt)
    return None


def parse_question(question: str, llm: LLMClient | None) -> GraphQuery | None:
    if llm is not None:
        try:
            return llm.structured(question, GraphQuery, system=prompts.GRAPH_PARSE_SYSTEM)
        except LLMError as exc:
            logger.warning("LLM graph parsing failed (%s); using rules", exc)
    return parse_question_rules(question)


def describe_result(res: GraphQueryResult) -> str:
    if not res.matched:
        return f'No node matched "{res.query}".'
    lines = [
        f'Matched {res.entity_type} "{res.entity_name}" ({res.node_id}); relation {res.relation}; '
        f"{len(res.papers)} linked papers (showing up to the limit)."
    ]
    if res.entities:
        lines.append(
            f"Top {res.target_type} entities (name: number of papers): "
            + "; ".join(f"{n}: {c}" for n, c in res.entities[:20])
        )
    return "\n".join(lines)


def answer_graph_question(
    question: str,
    store: GraphStore,
    papers: PaperStore,
    llm: LLMClient | None,
    normalizer: EntityNormalizer | None = None,
    max_context: int = 8,
) -> GraphAnswer:
    """Parse -> query the graph -> fetch abstracts -> LLM answer with verified citations."""
    parsed = parse_question(question, llm)
    if parsed is None:
        return GraphAnswer(question=question, answer=IDK_MESSAGE["en"])
    res = graph_query(
        store,
        parsed.entity,
        parsed.relation,
        parsed.entity_type,
        parsed.target_type,
        normalizer=normalizer,
    )
    context = papers.many(res.papers[:max_context])
    if not res.matched or not context:
        return GraphAnswer(question=question, parsed=parsed, result=res, answer=IDK_MESSAGE["en"])
    if llm is None:  # deterministic answer without LLM: facts + cited papers
        lines = [describe_result(res)] + [
            f"- {p['title']} [arXiv:{p['arxiv_id']}]" for p in context
        ]
        return GraphAnswer(
            question=question,
            parsed=parsed,
            result=res,
            answer="\n".join(lines),
            citations=[p["arxiv_id"] for p in context],
        )
    raw = llm.chat(
        prompts.GRAPH_ANSWER_USER.format(
            question=question,
            facts=describe_result(res),
            context=prompts.format_papers(context, max_chars=900),
        ),
        system=prompts.GRAPH_ANSWER_SYSTEM,
    )
    if prompts.INSUFFICIENT in raw:
        return GraphAnswer(question=question, parsed=parsed, result=res, answer=IDK_MESSAGE["en"])
    rep = validate_citations(raw, [p["arxiv_id"] for p in context])
    return GraphAnswer(
        question=question,
        parsed=parsed,
        result=res,
        answer=rep.text.strip(),
        citations=rep.valid,
        invalid_citations=rep.invalid,
    )


# ------------------------------------------------------------- statistics


def coauthor_graph(store: NetworkXGraphStore) -> nx.Graph:
    g = nx.Graph()
    for u, v, k, d in store.g.edges(keys=True, data=True):
        if k == "CO_AUTHOR":
            g.add_edge(u, v, weight=float(d.get("weight", 1.0)))
    return g


def central_authors(
    store: NetworkXGraphStore, top_n: int = 10, method: str = "pagerank"
) -> list[dict[str, Any]]:
    """Most central authors of the co-author graph (PageRank or weighted degree)."""
    g = coauthor_graph(store)
    if g.number_of_nodes() == 0:
        return []
    scores = (
        nx.pagerank(g, weight="weight") if method == "pagerank" else dict(g.degree(weight="weight"))
    )
    top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    return [
        {
            "author": store.node(a)["name"],
            "score": round(float(s), 6),
            "n_papers": len(store.neighbors(a, "WRITTEN_BY", "in")),
            "n_coauthors": g.degree(a),
        }
        for a, s in top
    ]


def author_communities(
    store: NetworkXGraphStore, top_n: int = 10, min_size: int = 3, seed: int = 42
) -> list[dict[str, Any]]:
    """Louvain communities of the co-author graph (largest first)."""
    g = coauthor_graph(store)
    if g.number_of_nodes() == 0:
        return []
    comms = nx.community.louvain_communities(g, weight="weight", seed=seed)
    comms = sorted((c for c in comms if len(c) >= min_size), key=lambda c: (-len(c), min(c)))
    out = []
    for i, c in enumerate(comms[:top_n]):
        members = sorted(c, key=lambda a: (-g.degree(a, weight="weight"), a))
        out.append(
            {
                "community": i,
                "size": len(c),
                "members": [store.node(a)["name"] for a in members[:10]],
            }
        )
    return out
