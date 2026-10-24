# Windows equivalent of scripts/demo.sh (PowerShell 5.1+). Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PYTHON) { $env:PYTHON } else { ".venv\Scripts\python.exe" }

function Rpe { & $py -m rpe.cli @args; if ($LASTEXITCODE -ne 0) { throw "rpe $args failed" } }
function Step($t) { Write-Host "`n== $t" -ForegroundColor Cyan }

Step "1. Preparation des donnees";      Rpe prepare-data --dev
Step "2. Index et recherche";           Rpe build-index --dev
Rpe search "retrieval-augmented generation" --dev -k 5
Step "3. RAG";                          Rpe ask "How can retrieval reduce hallucinations in LLMs?" --dev
Step "4. Topics et tendances";          Rpe build-topics --dev; Rpe trends --dev
Step "5. Extraction et graphe";         Rpe extract --dev; Rpe build-graph --dev
Rpe graph-ask "Which papers use SQuAD?" --dev
Step "6. Agent";                        Rpe gaps --dev
$first = & $py -c "import pandas as pd; print(pd.read_parquet('data/processed/papers_dev.parquet').arxiv_id.iloc[0])"
Rpe recommend $first --dev -k 3
Rpe sota "retrieval-augmented generation" --dev --out data/digests/sota_rag_dev.md
Rpe digest --query "LLM agents" --days 2 --max-results 15
Step "7. PDF";                          Rpe fetch-pdfs --dev; Rpe index-pdfs --dev
Rpe reading-sheet $first --lang fr --dev
Step "9. Evaluation";                   Rpe make-eval-sets --dev; Rpe evaluate --all --dev
Step "Termine. Interface : streamlit run app/streamlit_app.py"
