"""Command-line interface: `python -m rpe.cli <command>`."""

from __future__ import annotations

import logging
from typing import Annotated, Any

import typer

from rpe.config import get_config

app = typer.Typer(help="Research Paper Explorer CLI.", no_args_is_help=True)

DevOpt = Annotated[bool, typer.Option("--dev", help="Petit échantillon (config.data.sample_dev).")]
ForceOpt = Annotated[bool, typer.Option("--force", help="Recalculer même si la sortie existe.")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Research Paper Explorer."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_report(report: dict[str, Any]) -> None:
    scan, summary = report.get("scan"), report["summary"]
    if scan:
        typer.echo("\n== Lecture")
        typer.echo(f"  lignes lues          : {scan['n_lines']:,}")
        typer.echo(f"  lignes invalides     : {scan['n_invalid']:,}")
        typer.echo(f"  catégorie OK         : {scan['n_category_match']:,}")
        typer.echo(f"  catégorie + dates OK : {scan['n_in_date_range']:,}")
        typer.echo(f"  gardés (échantillon) : {scan['n_kept']:,}")
    typer.echo(f"\n== Corpus : {summary['n_papers']:,} papiers")
    typer.echo("  par année : " + ", ".join(f"{y}: {c}" for y, c in summary["by_year"].items()))
    typer.echo(f"  avec code : {summary['pct_has_code']} %")
    typer.echo(f"  acceptés  : {summary['pct_accepted']} %")
    typer.echo(f"  avec venue: {summary['pct_with_venue']} %")
    typer.echo("  top venues:")
    for venue, count in summary["top_venues"].items():
        typer.echo(f"    {venue:<22} {count}")


@app.command("prepare-data")
def prepare_data(dev: DevOpt = False, force: ForceOpt = False) -> None:
    """Lecture streaming du dump arXiv -> data/processed/papers.parquet."""
    from rpe.data.prepare import run_prepare

    cfg = get_config()
    try:
        out, _, report = run_prepare(cfg, dev=dev, force=force)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    if report.get("reused"):
        typer.echo(f"Sortie existante réutilisée : {out} (--force pour reconstruire)")
    _print_report(report)
    typer.echo(f"\nÉcrit : {out}")


def _fail(exc: Exception) -> typer.Exit:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    return typer.Exit(1)


# ----------------------------------------------------------------- phase 2


@app.command("build-index")
def build_index_cmd(dev: DevOpt = False, force: ForceOpt = False) -> None:
    """Embeddings (.npy) + collection Chroma + index BM25."""
    from rpe.index.build import build_index

    try:
        rep = build_index(get_config(), dev=dev, force=force)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    typer.echo(
        f"{rep.n_papers} papiers | dim={rep.embedding_dim} | Chroma +{rep.n_added_chroma} "
        f"(total {rep.chroma_count}) | BM25 {'reconstruit' if rep.bm25_rebuilt else 'à jour'}"
    )


@app.command("search")
def search_cmd(
    query: str,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    year_min: Annotated[int | None, typer.Option("--year-min")] = None,
    year_max: Annotated[int | None, typer.Option("--year-max")] = None,
    category: Annotated[list[str] | None, typer.Option("--category")] = None,
    venue: Annotated[str | None, typer.Option("--venue")] = None,
    has_code: Annotated[bool, typer.Option("--has-code")] = False,
    accepted: Annotated[bool, typer.Option("--accepted")] = False,
    mode: Annotated[str, typer.Option("--mode", help="hybrid | bm25 | dense")] = "hybrid",
    dev: DevOpt = False,
) -> None:
    """Recherche hybride BM25 + dense (RRF) avec filtres."""
    from rpe.index.filters import SearchFilters
    from rpe.index.hybrid import load_searcher

    filters = SearchFilters(
        year_min=year_min,
        year_max=year_max,
        categories=category or None,
        venue=venue,
        has_code=True if has_code else None,
        is_accepted=True if accepted else None,
    )
    try:
        searcher = load_searcher(get_config(), dev=dev)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    for rank, r in enumerate(searcher.search(query, top_k, filters, mode=mode), 1):  # type: ignore[arg-type]
        badges = " ".join(
            b for b in (r.venue or "", "[code]" if r.has_code else "", str(r.year or "")) if b
        )
        typer.echo(f"{rank:>2}. [arXiv:{r.arxiv_id}] {r.title}  ({badges})  score={r.score:.4f}")
        typer.echo(f"    {r.snippet}")


# ----------------------------------------------------------------- helpers


def _llm(required: bool = True) -> Any:
    from rpe.llm.client import get_llm

    llm = get_llm(get_config())
    if llm is None and required:
        typer.secho(
            "LLM non configuré : renseigner LLM_BASE_URL, LLM_MODEL (et LLM_API_KEY) dans .env",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    return llm


def _resources(dev: bool, llm: Any = None) -> Any:
    from rpe.agent.tools import Resources

    return Resources(get_config(), dev=dev, llm=llm)


def _stats(llm: Any) -> None:
    if llm is not None:
        typer.secho(str(llm.stats), fg=typer.colors.BRIGHT_BLACK, err=True)


def _write_out(text: str, out: str | None) -> None:
    if out:
        from pathlib import Path

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        typer.echo(f"\nÉcrit : {out}")


OutOpt = Annotated[str | None, typer.Option("--out", "-o", help="Fichier Markdown de sortie")]
LangOpt = Annotated[str, typer.Option("--lang", help="en | fr")]
PdfOpt = Annotated[str | None, typer.Option("--pdf", help="Exporter aussi un rapport PDF")]


def _write_pdf(data: bytes, path: str) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    typer.echo(f"PDF écrit : {path}")


# ----------------------------------------------------------------- phase 3


@app.command("ask")
def ask_cmd(
    question: str,
    paper: Annotated[str | None, typer.Option("--paper", help="arxiv_id (sinon : corpus)")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 8,
    lang: LangOpt = "en",
    pdf: PdfOpt = None,
    dev: DevOpt = False,
) -> None:
    """RAG : réponse avec citations [arXiv:ID] vérifiées (--pdf : rapport comparatif)."""
    from rpe.rag.qa import answer_question

    llm, res = _llm(), _resources(dev)
    try:
        ans = answer_question(
            question,
            res.searcher,
            llm,
            scope=paper or "corpus",
            top_k=top_k,
            chunks=res.chunks,
            lang=lang,
        )
    except (FileNotFoundError, KeyError) as exc:
        raise _fail(exc) from exc
    typer.echo(ans.answer)
    typer.echo("\nSources : " + ", ".join(f"arXiv:{c}" for c in ans.citations))
    if ans.invalid_citations:
        typer.secho(
            f"Citations retirées (hors contexte) : {ans.invalid_citations}",
            fg=typer.colors.YELLOW,
        )
    if pdf and ans.citations:
        from rpe.ui.report_view import make_report

        report = make_report(
            res,
            {"question": question, "answer": ans.answer, "citations": ans.citations, "lang": lang},
        )
        for w in report.warnings:
            typer.secho(w, fg=typer.colors.YELLOW)
        _write_pdf(report.pdf, pdf)
    _stats(llm)


@app.command("reading-sheet")
def reading_sheet_cmd(
    arxiv_id: str, lang: LangOpt = "en", out: OutOpt = None, dev: DevOpt = False
) -> None:
    """Fiche de lecture structurée (résumé + texte PDF si disponible)."""
    from rpe.pdf.reader import full_text_for_sheet
    from rpe.rag.reading_sheet import generate_reading_sheet, reading_sheet_markdown

    llm, res = _llm(), _resources(dev)
    try:
        paper = res.papers.get(arxiv_id)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    if paper is None:
        raise _fail(KeyError(f"arxiv_id inconnu dans le corpus : {arxiv_id}"))
    full = full_text_for_sheet(get_config(), arxiv_id)
    sheet = generate_reading_sheet(paper, llm, lang=lang, full_text=full)
    md = reading_sheet_markdown(sheet, paper, lang=lang, used_full_text=full is not None)
    typer.echo(md)
    _write_out(md, out)
    _stats(llm)


# ----------------------------------------------------------------- phase 4


@app.command("build-topics")
def build_topics_cmd(
    dev: DevOpt = False,
    force: ForceOpt = False,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Étiquettes par mots-clés")] = False,
) -> None:
    """BERTopic sur les embeddings existants + étiquettes LLM + carte 2D."""
    from rpe.topics.bertopic_model import fit_topics, topic_paths

    cfg = get_config()
    llm = None if no_llm or not cfg.topics.label_with_llm else _llm(required=False)
    try:
        topics, info = fit_topics(cfg, dev=dev, llm=llm, force=force)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    n_out = int((topics["topic_id"] == -1).sum())
    typer.echo(f"{len(info) - (n_out > 0)} topics, {n_out} outliers sur {len(topics)} papiers")
    for t in sorted(info, key=lambda t: -t["size"])[:15]:
        typer.echo(f"  [{t['topic_id']:>3}] {t['label']:<45} {t['size']:>5}")
    typer.echo(f"Carte : {topic_paths(cfg, dev).map_html}")
    _stats(llm)


@app.command("trends")
def trends_cmd(
    topic: Annotated[str | None, typer.Option("--topic")] = None,
    summary: Annotated[bool, typer.Option("--summary/--no-summary")] = True,
    dev: DevOpt = False,
) -> None:
    """Sujets émergents / en déclin (+ synthèse LLM factuelle)."""
    from rpe.topics.trends import (
        build_trend_report,
        match_topics,
        summarize_trends,
        trends_config,
    )

    res = _resources(dev)
    try:
        topics, info = res.topics
        papers = res.papers
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    ids = match_topics(info, topic) if topic else None
    if topic and not ids:
        raise _fail(KeyError(f"Aucun topic ne correspond à {topic!r}"))
    rep = build_trend_report(topics, papers.df, trends_config(get_config().trends, dev), ids)
    window = f"{rep.recent_months[0]} → {rep.recent_months[-1]}" if rep.recent_months else "?"
    typer.echo(f"Fenêtre récente : {window}")
    sections = (("Émergents", rep.emerging), ("En déclin", rep.declining), ("Topics", rep.topics))
    for title, rows in sections:
        typer.echo(f"\n== {title}")
        for t in rows[:10]:
            typer.echo(
                f"  {t.topic_label:<45} récents={t.recent:<4} précédents={t.previous:<4} "
                f"croissance={t.growth:+.1f}%"
            )
    llm = _llm(required=False) if summary else None
    if llm is not None:
        examples = {t["topic_id"]: papers.many(t["representative_ids"][:2]) for t in info}
        s = summarize_trends(rep, llm, examples)
        typer.echo("\n== Synthèse\n" + s.text)
        if s.unknown_numbers:
            typer.secho(
                f"Nombres absents des données : {s.unknown_numbers}", fg=typer.colors.YELLOW
            )
        _stats(llm)


# ----------------------------------------------------------------- phase 5


@app.command("extract")
def extract_cmd(
    dev: DevOpt = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Nombre max de papiers")] = None,
) -> None:
    """Extraction LLM (tâches, méthodes, datasets, métriques...) avec cache JSONL."""
    from rpe.kg.extract import cache_path, run_extraction

    cfg, llm = get_config(), _llm()
    try:
        rep = run_extraction(cfg, llm, dev=dev, limit=limit)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    typer.echo(
        f"Sélection {rep.n_selected} | déjà en cache {rep.n_cached} | nouveaux {rep.n_new} | "
        f"erreurs {rep.n_errors} -> {cache_path(cfg, dev)}"
    )
    _stats(llm)


@app.command("build-graph")
def build_graph_cmd(dev: DevOpt = False) -> None:
    """Graphe de connaissances NetworkX -> data/graph/kg.graphml."""
    from rpe.kg.build_graph import graph_path, run_build_graph

    cfg = get_config()
    try:
        _, rep = run_build_graph(cfg, dev)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Papiers extraits : {rep.n_extracted}")
    typer.echo("Nœuds : " + ", ".join(f"{k}={v}" for k, v in sorted(rep.nodes.items())))
    typer.echo("Arêtes : " + ", ".join(f"{k}={v}" for k, v in sorted(rep.edges.items())))
    typer.echo(f"Écrit : {graph_path(cfg, dev)}")


@app.command("graph-ask")
def graph_ask_cmd(question: str, dev: DevOpt = False) -> None:
    """GraphRAG : question -> requête de graphe -> réponse citée."""
    from rpe.kg.graphrag import answer_graph_question, describe_result

    res = _resources(dev)
    llm = _llm(required=False)
    try:
        ans = answer_graph_question(question, res.graph, res.papers, llm, res.normalizer)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    if ans.parsed:
        typer.secho(f"Requête : {ans.parsed.model_dump()}", fg=typer.colors.BRIGHT_BLACK)
    if ans.result:
        typer.secho(describe_result(ans.result), fg=typer.colors.BRIGHT_BLACK)
    typer.echo("\n" + ans.answer)
    _stats(llm)


@app.command("graph-stats")
def graph_stats_cmd(dev: DevOpt = False, top: int = 10) -> None:
    """Auteurs centraux (PageRank) et communautés (Louvain)."""
    from rpe.kg.graphrag import author_communities, central_authors

    try:
        g = _resources(dev).graph
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    typer.echo("== Auteurs centraux (PageRank co-auteurs)")
    for a in central_authors(g, top):
        typer.echo(f"  {a['author']:<30} score={a['score']:.5f} papiers={a['n_papers']}")
    typer.echo("\n== Communautés (Louvain)")
    for c in author_communities(g, top):
        typer.echo(f"  #{c['community']} ({c['size']}) : {', '.join(c['members'][:6])}")


# ----------------------------------------------------------------- phase 6


@app.command("agent")
def agent_cmd(
    question: str,
    dev: DevOpt = False,
    max_iterations: Annotated[int | None, typer.Option("--max-iterations")] = None,
) -> None:
    """Agent ReAct (LangGraph) avec les 8 outils."""
    from rpe.agent.graph import run_agent
    from rpe.agent.tools import build_tools

    llm = _llm()
    res = _resources(dev, llm)
    n = max_iterations or get_config().agent.max_iterations
    result = run_agent(question, build_tools(res), llm, n)
    for s in result.steps:
        typer.secho(
            f"[{s.iteration}] {s.tool}({s.args}) -> {s.result_summary}",
            fg=typer.colors.BRIGHT_BLACK,
        )
    typer.echo("\n" + result.answer)
    if result.invalid_citations:
        typer.secho(f"Citations retirées : {result.invalid_citations}", fg=typer.colors.YELLOW)
    _stats(llm)


@app.command("sota")
def sota_cmd(
    topic: str, lang: LangOpt = "en", out: OutOpt = None, pdf: PdfOpt = None, dev: DevOpt = False
) -> None:
    """Génère un état de l'art Markdown avec références [arXiv:ID]."""
    from rpe.agent.sota import build_sota_graph, stream_sota

    cfg = get_config()
    llm = _llm()
    res = _resources(dev, llm)
    try:
        searcher = res.searcher
        vectors = {i: res.dense.matrix[k] for k, i in enumerate(res.dense.ids)}
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    try:
        graph = res.graph
    except FileNotFoundError:
        graph = None
    a = cfg.agent
    wf = build_sota_graph(
        searcher,
        llm,
        a.sota_max_papers,
        a.sota_min_papers,
        a.sota_max_iterations,
        vectors=vectors,
        graph=graph,
    )
    final: dict[str, Any] = {}
    for node, update in stream_sota(wf, topic, lang):
        detail = (update.get("trace") or [{}])[-1].get("detail", "")
        typer.secho(f"[{node}] {detail}", fg=typer.colors.BRIGHT_BLACK)
        final.update(update)
    md = final.get("markdown", "")
    typer.echo("\n" + md)
    if final.get("invalid_citations"):
        typer.secho(f"Citations retirées : {final['invalid_citations']}", fg=typer.colors.YELLOW)
    _write_out(md, out)
    if pdf and md:
        from rpe.reports.pdf_report import build_markdown_pdf

        _write_pdf(build_markdown_pdf(f"État de l'art — {topic}", md), pdf)
    _stats(llm)


@app.command("gaps")
def gaps_cmd(
    task: Annotated[str | None, typer.Option("--task")] = None,
    dev: DevOpt = False,
    top: Annotated[int | None, typer.Option("--top")] = None,
) -> None:
    """Research gaps : combinaisons Tâche x Méthode rares (hypothèses)."""
    from rpe.agent.gaps import find_gaps, justify_gaps

    cfg = get_config()
    res = _resources(dev)
    llm = _llm(required=False)
    g = cfg.gaps
    try:
        gaps = find_gaps(
            res.graph,
            g.min_task_papers,
            g.min_method_papers,
            g.max_cooccurrence,
            top or g.top_n,
            task,
            res.normalizer,
        )
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    if not gaps:
        typer.echo("Aucun gap trouvé (seuils trop stricts ou peu d'extractions).")
    for x in justify_gaps(gaps, res.papers, llm):
        typer.echo(
            f"\n* {x.task} × {x.method}  (score {x.score}; tâche={x.n_task}, "
            f"méthode={x.n_method}, ensemble={x.n_both}, attendu={x.expected})"
        )
        typer.echo(f"  {x.hypothesis}\n  {x.rationale}")
    _stats(llm)


@app.command("recommend")
def recommend_cmd(
    arxiv_ids: list[str], k: Annotated[int, typer.Option("-k")] = 5, dev: DevOpt = False
) -> None:
    """Recommandations par centroïde d'embeddings (1 à 5 ids)."""
    from rpe.agent.recommend import recommend_similar

    llm = _llm(required=False)
    res = _resources(dev)
    try:
        recs = recommend_similar(arxiv_ids, res.dense, res.papers, k, llm)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(exc) from exc
    for r in recs:
        typer.echo(
            f"- [arXiv:{r.arxiv_id}] {r.title} ({r.year}, {r.venue or '-'}) sim={r.score:.3f}\n"
            f"    {r.explanation}"
        )
    _stats(llm)


@app.command("digest")
def digest_cmd(
    query: Annotated[str, typer.Option("--query", "-q")],
    days: Annotated[int, typer.Option("--days")] = 1,
    max_results: Annotated[int | None, typer.Option("--max-results")] = None,
) -> None:
    """Veille : nouveaux papiers arXiv (API) -> digest Markdown dans data/digests/."""
    from rpe.agent.digest import fetch_latest_arxiv, make_digest, save_digest

    cfg = get_config()
    llm = _llm(required=False)
    papers = fetch_latest_arxiv(
        query, days, max_results or cfg.digest.max_results, cfg.digest.categories
    )
    d = make_digest(query, days, papers, llm)
    typer.echo(d.markdown)
    typer.echo(f"Écrit : {save_digest(d, cfg.path(cfg.digest.output_dir))}")
    _stats(llm)


# ----------------------------------------------------------------- phase 7


@app.command("fetch-pdfs")
def fetch_pdfs_cmd(
    dev: DevOpt = False, max_pdfs: Annotated[int | None, typer.Option("--max")] = None
) -> None:
    """Télécharge au plus pdfs.max_pdfs PDF depuis arxiv.org (délai entre requêtes)."""
    from rpe.pdf.download import fetch_pdfs, select_pdf_candidates

    cfg = get_config()
    n = max_pdfs or (min(20, cfg.pdfs.max_pdfs) if dev else cfg.pdfs.max_pdfs)
    try:
        ids = select_pdf_candidates(cfg, _resources(dev).searcher, n)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    typer.echo(f"{len(ids)} PDF candidats (délai {cfg.pdfs.delay_seconds}s)")
    typer.echo(str(fetch_pdfs(cfg, ids)))


@app.command("index-pdfs")
def index_pdfs_cmd(dev: DevOpt = False) -> None:
    """Sections + chunks -> Chroma 'chunks' ; arêtes CITES ajoutées au graphe."""
    from rpe.kg.build_graph import graph_path
    from rpe.pdf.chunks import index_pdfs

    cfg = get_config()
    res = _resources(dev)
    try:
        graph = res.graph
    except FileNotFoundError:
        graph = None
    try:
        stats = index_pdfs(cfg, dev, graph)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    if graph is not None and stats["cites"]:
        graph.save(graph_path(cfg, dev))
    typer.echo(str(stats))


# ----------------------------------------------------------------- phase 9


@app.command("make-eval-sets")
def make_eval_sets_cmd(dev: DevOpt = False, force: ForceOpt = False) -> None:
    """Génère les brouillons des jeux d'évaluation (à valider à la main)."""
    from rpe.eval.datasets_gen import make_eval_sets

    res = _resources(dev, _llm(required=False))
    for p in make_eval_sets(get_config(), res, force=force):
        typer.echo(f"Écrit : {p}")


@app.command("evaluate")
def evaluate_cmd(
    all_: Annotated[bool, typer.Option("--all", help="retrieval + extraction + RAG")] = False,
    retrieval: Annotated[bool, typer.Option("--retrieval")] = False,
    extraction: Annotated[bool, typer.Option("--extraction")] = False,
    rag: Annotated[bool, typer.Option("--rag")] = False,
    dev: DevOpt = False,
) -> None:
    """Évaluation -> eval/report.md."""
    from rpe.eval.report import run_evaluation

    if not (all_ or retrieval or extraction or rag):
        all_ = True
    res = _resources(dev, _llm(required=False))
    path = run_evaluation(
        get_config(),
        res,
        retrieval=all_ or retrieval,
        extraction=all_ or extraction,
        rag=all_ or rag,
    )
    typer.echo(path.read_text(encoding="utf-8"))
    typer.echo(f"Écrit : {path}")


if __name__ == "__main__":
    app()
