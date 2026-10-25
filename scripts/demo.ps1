# Démo rapide (corpus de 1 000 papiers) puis lancement de l'application.
# Prérequis : dépendances installées, dump dans data\raw\, .env configuré.
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PYTHON) { $env:PYTHON } else { ".venv\Scripts\python.exe" }

function Rpe { & $py -m rpe.cli @args; if ($LASTEXITCODE -ne 0) { throw "rpe $args a échoué" } }
function Step($t) { Write-Host "`n== $t" -ForegroundColor Cyan }

Step "1. Préparation des données (1 000 papiers)";   Rpe prepare-data --dev
Step "2. Index : embeddings + ChromaDB + BM25";       Rpe build-index --dev
Step "3. Recherche hybride";                           Rpe search "retrieval-augmented generation" --dev -k 5
Step "4. Texte intégral (20 PDF, 3 s entre requêtes)"; Rpe fetch-pdfs --dev; Rpe index-pdfs --dev
Step "5. Question RAG + rapport PDF"
Rpe ask "How can retrieval reduce hallucinations in LLMs?" --lang fr --pdf rapport_demo.pdf --dev
Step "6. Application : http://localhost:8501"
& $py -m streamlit run app/streamlit_app.py
