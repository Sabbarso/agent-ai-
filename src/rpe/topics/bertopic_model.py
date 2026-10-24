"""BERTopic over abstracts, reusing the phase-2 embeddings (never recomputed).

Outputs (with `_dev` suffix in --dev mode):
- data/processed/topics.parquet     arxiv_id, topic_id, topic_label, x, y (2D UMAP map)
- data/processed/topics_info.json   per topic: label, keywords, size, representative ids
- data/processed/bertopic_model/    the fitted model (safetensors)
- data/processed/topic_map.html     interactive Plotly map
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from rpe.config import Config
from rpe.data.store import PaperStore
from rpe.index.embed import load_embeddings
from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import TopicLabel

logger = logging.getLogger(__name__)
OUTLIER_LABEL = "Outliers"


@dataclass
class TopicPaths:
    topics: Path
    info: Path
    model: Path
    map_html: Path


def topic_paths(cfg: Config, dev: bool = False) -> TopicPaths:
    d = cfg.processed_dir
    return TopicPaths(
        cfg.artifact(d / "topics.parquet", dev),
        cfg.artifact(d / "topics_info.json", dev),
        cfg.artifact(d / "bertopic_model", dev),
        cfg.artifact(d / "topic_map.html", dev),
    )


def representative_indices(members: np.ndarray, matrix: np.ndarray, k: int = 5) -> list[int]:
    """Indices of the `k` members closest to the topic centroid (cosine)."""
    if len(members) == 0:
        return []
    vecs = matrix[members]
    centroid = vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    order = np.argsort(-(vecs @ centroid), kind="stable")[:k]
    return [int(members[i]) for i in order]


def fallback_label(keywords: list[str]) -> str:
    return ", ".join(keywords[:3]) if keywords else "Unlabelled topic"


def label_topic(keywords: list[str], titles: list[str], llm: LLMClient | None) -> str:
    """LLM label (3-6 words) from keywords + representative titles; keyword fallback."""
    if llm is None:
        return fallback_label(keywords)
    prompt = prompts.TOPIC_LABEL_USER.format(
        keywords=", ".join(keywords[:10]), titles="\n".join(f"- {t}" for t in titles)
    )
    try:
        return llm.structured(prompt, TopicLabel).label
    except LLMError as exc:
        logger.warning("Topic labelling failed (%s); using keywords", exc)
        return fallback_label(keywords)


def build_topic_model(cfg: Config, n_docs: int, dev: bool) -> Any:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    t = cfg.topics
    min_cluster = t.dev_min_cluster_size if dev else t.min_cluster_size
    return BERTopic(
        umap_model=UMAP(
            n_neighbors=t.n_neighbors,
            n_components=t.n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=cfg.data.seed,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=min(t.min_samples, min_cluster),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2 if n_docs < 5000 else 5, max_df=0.9
        ),
        calculate_probabilities=False,
        verbose=False,
    )


def map_coordinates(matrix: np.ndarray, cfg: Config) -> np.ndarray:
    from umap import UMAP

    return UMAP(
        n_components=2,
        n_neighbors=cfg.topics.n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=cfg.data.seed,
    ).fit_transform(matrix)


def topic_map_figure(topics: pd.DataFrame, papers: pd.DataFrame) -> Any:
    """Interactive 2D map: colour = topic, hover = title."""
    import plotly.express as px

    df = topics.merge(papers[["arxiv_id", "title", "year"]], on="arxiv_id", how="left")
    df = df.sort_values("topic_id")
    df["topic"] = df["topic_id"].astype(str) + " · " + df["topic_label"]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="topic",
        hover_name="title",
        hover_data={"arxiv_id": True, "year": True, "x": False, "y": False, "topic": False},
        opacity=0.7,
        height=700,
    )
    fig.update_traces(marker={"size": 4})
    fig.update_layout(legend_title_text="Topic", xaxis_visible=False, yaxis_visible=False)
    return fig


def fit_topics(
    cfg: Config, dev: bool = False, llm: LLMClient | None = None, force: bool = False
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit BERTopic on cached embeddings, label topics, save outputs (skip if present)."""
    paths = topic_paths(cfg, dev)
    if paths.topics.exists() and paths.info.exists() and not force:
        logger.info("Topics already built (%s); use --force to refit", paths.topics)
        return load_topics(cfg, dev)

    papers = PaperStore.load(cfg, dev)
    matrix, ids = load_embeddings(cfg.index_dir(dev))
    df = papers.df.loc[ids].reset_index(drop=True)
    docs = (df["title"] + ". " + df["abstract"]).tolist()

    model = build_topic_model(cfg, len(docs), dev)
    logger.info("Fitting BERTopic on %d documents", len(docs))
    topic_ids, _ = model.fit_transform(docs, embeddings=matrix)
    topic_ids = np.asarray(topic_ids)

    info: list[dict[str, Any]] = []
    labels: dict[int, str] = {}
    for tid in tqdm(sorted(set(topic_ids.tolist())), desc="Labelling topics"):
        members = np.flatnonzero(topic_ids == tid)
        reps = representative_indices(members, matrix, 5)
        keywords = [w for w, _ in (model.get_topic(tid) or [])][:10]
        titles = df.loc[reps, "title"].tolist()
        label = OUTLIER_LABEL if tid == -1 else label_topic(keywords, titles, llm)
        labels[tid] = label
        info.append(
            {
                "topic_id": int(tid),
                "label": label,
                "keywords": keywords,
                "size": int(len(members)),
                "representative_ids": df.loc[reps, "arxiv_id"].tolist(),
            }
        )
    model.set_topic_labels(labels)

    coords = map_coordinates(matrix, cfg)
    topics = pd.DataFrame(
        {
            "arxiv_id": df["arxiv_id"],
            "topic_id": topic_ids.astype(int),
            "topic_label": [labels[t] for t in topic_ids.tolist()],
            "x": coords[:, 0].astype(float),
            "y": coords[:, 1].astype(float),
        }
    )
    paths.topics.parent.mkdir(parents=True, exist_ok=True)
    topics.to_parquet(paths.topics, index=False)
    paths.info.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        model.save(str(paths.model), serialization="safetensors", save_ctfidf=True)
    except Exception as exc:  # noqa: BLE001 - saving the model is best-effort
        logger.warning("Could not save BERTopic model: %s", exc)
    topic_map_figure(topics, df).write_html(paths.map_html, include_plotlyjs="cdn")
    return topics, info


def load_topics(cfg: Config, dev: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = topic_paths(cfg, dev)
    if not paths.topics.exists() or not paths.info.exists():
        raise FileNotFoundError(
            f"Topics not built ({paths.topics}). Run: rpe build-topics" + (" --dev" if dev else "")
        )
    return pd.read_parquet(paths.topics), json.loads(paths.info.read_text(encoding="utf-8"))
