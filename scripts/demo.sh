#!/usr/bin/env bash
# Quick end-to-end demo in --dev mode (1 000 papers, 50 extractions, 20 PDFs).
# Prerequisites: venv with requirements installed, dump in data/raw/, .env configured.
# Every step is resumable: re-running skips what already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
RPE="$PY -m rpe.cli"

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

step "1. Préparation des données"
$RPE prepare-data --dev

step "2. Index (embeddings + Chroma + BM25) et recherche"
$RPE build-index --dev
$RPE search "retrieval-augmented generation" --dev -k 5
$RPE search "machine translation" --year-min 2024 --has-code --dev -k 5

step "3. RAG avec citations"
$RPE ask "How can retrieval reduce hallucinations in LLMs?" --dev

step "4. Topics et tendances"
$RPE build-topics --dev
$RPE trends --dev

step "5. Extraction et graphe de connaissances"
$RPE extract --dev
$RPE build-graph --dev
$RPE graph-ask "Which papers use SQuAD?" --dev
$RPE graph-stats --dev

step "6. Agent"
$RPE gaps --dev
FIRST_ID=$($PY -c "import pandas as pd; print(pd.read_parquet('data/processed/papers_dev.parquet').arxiv_id.iloc[0])")
$RPE recommend "$FIRST_ID" --dev -k 3
$RPE sota "retrieval-augmented generation" --dev --out data/digests/sota_rag_dev.md
$RPE agent "Which datasets are used to evaluate RAG systems?" --dev
$RPE digest --query "LLM agents" --days 2 --max-results 15

step "7. PDF (20 PDF, 3 s entre requêtes)"
$RPE fetch-pdfs --dev
$RPE index-pdfs --dev
$RPE reading-sheet "$FIRST_ID" --lang fr --dev

step "9. Évaluation"
$RPE make-eval-sets --dev
$RPE evaluate --all --dev

step "Terminé. Interface : streamlit run app/streamlit_app.py"
