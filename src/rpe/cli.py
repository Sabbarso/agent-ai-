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


# ------------------------------------------------------------------ RAG


def _llm() -> Any:
    from rpe.llm.client import get_llm

    llm = get_llm(get_config())
    if llm is None:
        typer.secho(
            "LLM non configuré : renseigner LLM_BASE_URL, LLM_MODEL (et LLM_API_KEY) dans .env",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    return llm


def _resources(dev: bool, llm: Any = None) -> Any:
    from rpe.resources import Resources

    return Resources(get_config(), dev=dev, llm=llm)


@app.command("ask")
def ask_cmd(
    question: str,
    paper: Annotated[str | None, typer.Option("--paper", help="arxiv_id (sinon : corpus)")] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", "-k", help="papiers du contexte")] = None,
    lang: Annotated[str, typer.Option("--lang", help="en | fr")] = "en",
    pdf: Annotated[str | None, typer.Option("--pdf", help="Exporter un rapport PDF")] = None,
    dev: DevOpt = False,
) -> None:
    """Question RAG : réponse avec citations arXiv vérifiées (--pdf : rapport comparatif)."""
    from rpe.rag.qa import answer_question

    llm = _llm()
    res = _resources(dev, llm)
    try:
        ans = answer_question(
            question,
            res.searcher,
            llm,
            scope=paper or "corpus",
            top_k=top_k or get_config().retrieval.rag_top_k,
            chunks=res.chunks,
            lang=lang,
        )
    except (FileNotFoundError, KeyError) as exc:
        raise _fail(exc) from exc
    typer.echo(ans.answer)
    typer.echo("\nSources : " + ", ".join(f"arXiv:{c}" for c in ans.citations))
    if ans.invalid_citations:
        typer.secho(
            f"Citations retirées (hors contexte) : {ans.invalid_citations}", fg=typer.colors.YELLOW
        )
    if pdf and ans.citations:
        from pathlib import Path

        from rpe.ui.report_view import make_report

        msg = {"question": question, "answer": ans.answer, "citations": ans.citations, "lang": lang}
        report = make_report(res, msg)
        for w in report.warnings:
            typer.secho(w, fg=typer.colors.YELLOW)
        Path(pdf).parent.mkdir(parents=True, exist_ok=True)
        Path(pdf).write_bytes(report.pdf)
        typer.echo(f"PDF écrit : {pdf}")
    typer.secho(str(llm.stats), fg=typer.colors.BRIGHT_BLACK, err=True)


# ----------------------------------------------------------- texte intégral


@app.command("fetch-pdfs")
def fetch_pdfs_cmd(
    dev: DevOpt = False, max_pdfs: Annotated[int | None, typer.Option("--max")] = None
) -> None:
    """Télécharge des PDF depuis arxiv.org (pause entre les requêtes, reprise)."""
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
    """Découpe les PDF téléchargés en sections et indexe les passages (Chroma 'chunks')."""
    from rpe.pdf.chunks import index_pdfs

    try:
        stats = index_pdfs(get_config(), dev)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    typer.echo(str(stats))


if __name__ == "__main__":
    app()
