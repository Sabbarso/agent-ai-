#!/usr/bin/env bash
# Démo rapide (corpus de 1 000 papiers) puis lancement de l'application.
# Prérequis : dépendances installées, dump dans data/raw/, .env configuré.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
RPE="$PY -m rpe.cli"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

step "1. Préparation des données (1 000 papiers)"
$RPE prepare-data --dev
step "2. Index : embeddings + ChromaDB + BM25"
$RPE build-index --dev
step "3. Recherche hybride"
$RPE search "retrieval-augmented generation" --dev -k 5
step "4. Texte intégral (20 PDF, 3 s entre requêtes)"
$RPE fetch-pdfs --dev
$RPE index-pdfs --dev
step "5. Question RAG + rapport PDF"
$RPE ask "How can retrieval reduce hallucinations in LLMs?" --lang fr --pdf rapport_demo.pdf --dev
step "6. Application : http://localhost:8501"
$PY -m streamlit run app/streamlit_app.py
