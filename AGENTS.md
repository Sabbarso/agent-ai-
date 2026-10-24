# AGENTS.md — Research Paper Explorer (projet Generative AI, ENSIAS)

> Ce fichier est le cahier des charges du projet. Lis-le entièrement avant toute modification
> et respecte-le à chaque phase.

## 1. Contexte et objectif

Nous construisons **Research Paper Explorer** : une plateforme d'IA générative qui aide un
étudiant ou un chercheur à explorer la littérature scientifique en informatique.
Elle combine recherche hybride, RAG avec citations, graphe de connaissances (GraphRAG),
analyse de tendances et un agent (LangGraph) capable de produire un état de l'art.

**Source de données unique : le dataset arXiv de Kaggle (Cornell University).**
- Lien : https://www.kaggle.com/datasets/Cornell-University/arxiv
- Fichier : `data/raw/arxiv-metadata-oai-snapshot.json` (~5,6 Go, **JSON Lines** : un objet JSON par ligne).
- Le fichier est téléchargé manuellement par l'équipe. **Ne jamais tenter de le télécharger, ne jamais le charger entièrement en mémoire.**
- En complément (même source arXiv) : l'API publique arXiv (package Python `arxiv`) pour la veille, et `https://arxiv.org/pdf/{id}` pour quelques centaines de PDF.

### Schéma d'un enregistrement
```json
{
  "id": "0704.0001",
  "submitter": "Pavel Nadolsky",
  "authors": "C. Bal\\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan",
  "title": "Calculation of prompt diphoton production ...",
  "comments": "37 pages, 15 figures; published version",
  "journal-ref": "Phys.Rev.D76:013009,2007",
  "doi": "10.1103/PhysRevD.76.013009",
  "report-no": "ANL-HEP-PR-07-12",
  "categories": "hep-ph",            // chaîne séparée par des espaces, ex. "cs.CL cs.LG"
  "license": null,
  "abstract": " A fully differential calculation ...",
  "versions": [{"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"}, ...],
  "update_date": "2008-11-26",
  "authors_parsed": [["Balázs", "C.", ""], ["Berger", "E. L.", ""], ...]
}
```
Notes de nettoyage :
- Date de publication = `versions[0].created` (format RFC 2822), et non `update_date`.
- Titres et résumés : retirer les retours à la ligne et les espaces multiples. Les accents LaTeX (`\'a`) peuvent apparaître dans les auteurs : utiliser de préférence `authors_parsed`.
- Champs pouvant être `null` : `comments`, `journal-ref`, `doi`, `report-no`, `license`.

## 2. Périmètre fonctionnel

1. **Préparation des données** : lecture en streaming, filtrage (catégories, dates), nettoyage, export Parquet.
2. **Enrichissement depuis `comments` / `journal-ref` / `abstract`** :
   - `venue` : conférence ou revue détectée (ACL, EMNLP, NAACL, EACL, COLING, NeurIPS, ICML, ICLR, AAAI, IJCAI, CVPR, ICCV, ECCV, KDD, SIGIR, WWW, TACL, JMLR, TMLR…) et `is_accepted` (bool), à partir de motifs comme « accepted at/to/by », « to appear in », « published in », « camera-ready ».
   - `github_url` : premier lien github.com trouvé ; `has_code` (bool).
   - `n_pages`, `n_figures` lorsqu'ils sont indiqués.
3. **Recherche hybride** : BM25 + embeddings denses, fusion RRF, filtres (année, catégorie, venue, has_code).
4. **RAG avec citations** : questions-réponses sur un papier ou sur le corpus. Chaque affirmation cite un `arxiv_id` présent dans le contexte récupéré.
5. **Topic modeling et tendances** : BERTopic sur les résumés, carte 2D (UMAP), évolution mensuelle des thèmes, détection de sujets émergents, synthèse rédigée par le LLM.
6. **Extraction de connaissances par le LLM** : pour chaque résumé, extraire en sortie structurée (Pydantic) `tasks`, `methods`, `datasets`, `metrics`, `claimed_results`, `contribution_type`. Normaliser les noms (par exemple « SQuAD v1.1 » → « SQuAD »).
7. **Graphe de connaissances + GraphRAG** : nœuds Paper, Author, Task, Method, Dataset, Metric, Category ; arêtes `WRITTEN_BY`, `ADDRESSES`, `PROPOSES_OR_USES`, `EVALUATED_ON`, `MEASURED_BY`, `IN_CATEGORY`, `CO_AUTHOR`. Il doit répondre à « Quels papiers utilisent SQuAD ? » ou « Quelles méthodes pour la traduction automatique ? ».
8. **Agent (LangGraph)** avec outils :
   - `search_papers(query, filters)`
   - `get_paper(arxiv_id)`
   - `graph_query(entity, relation)`
   - `recommend_similar(arxiv_ids)`
   - `get_trends(topic | None)`
   - `find_research_gaps(task | None)`
   - `read_pdf(arxiv_id)`
   - `fetch_latest_arxiv(query, days)`
9. **Fonctionnalités de l'agent** :
   - **Générateur d'état de l'art** : sujet → plan → recherche → sélection (pertinence, récence, venue, code) → regroupement par approche → revue structurée en Markdown avec références `[arXiv:XXXX.XXXXX]`.
   - **Détecteur de research gaps** : matrice Tâche × Méthode. Repérer les combinaisons rares alors que la tâche et la méthode sont chacune fréquentes. Le LLM explique en quoi c'est un gap, avec les papiers proches comme preuves, et le présente comme une **hypothèse** et non une certitude.
   - **Veille** : nouveaux papiers sur un thème sur les N derniers jours via l'API arXiv, puis digest résumé. Lancement en CLI (`python -m rpe.cli digest --query "LLM agents" --days 1`).
   - **Fiche de lecture** : résumé, contributions, méthode, résultats, limites, 3 questions de compréhension. Option `--lang fr`.
   - **Recommandation** : l'utilisateur donne 1 à 5 `arxiv_id`, l'outil calcule le centroïde de leurs embeddings et renvoie les plus proches voisins, en excluant les entrées et en expliquant brièvement le lien.
10. **Interface Streamlit** : pages Recherche, Chat (RAG), État de l'art (agent), Tendances (carte + courbes), Graphe (visualisation pyvis + requêtes), Research gaps, Veille, Fiche de lecture.
11. **Évaluation** : retrieval (Recall@k, MRR, nDCG sur un jeu de requêtes annoté), qualité de l'extraction (précision et rappel sur 50 résumés annotés à la main), RAG (fidélité et taux de citations valides).

## 3. Stack technique

- Python 3.11, gestion de dépendances avec `uv` ou `pip` + `requirements.txt`.
- Données : `orjson` (lecture rapide), `polars` ou `pandas`, Parquet (`pyarrow`).
- Embeddings : `sentence-transformers`, modèle par défaut `BAAI/bge-small-en-v1.5` (configurable). Calcul par batch, cache sur disque (`.npy`).
- Index vectoriel : `chromadb` (persistant dans `data/index/`). BM25 : `rank_bm25` ou `bm25s`.
- Topics : `bertopic`, `umap-learn`, `hdbscan`, `plotly`.
- Graphe : `networkx` (sérialisé en `.graphml` ou pickle), visualisation `pyvis`. **Neo4j optionnel** derrière une interface `GraphStore` (ne pas le rendre obligatoire).
- LLM : couche d'abstraction `LLMClient` compatible API OpenAI, configurée par variables d'environnement (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). Elle doit fonctionner avec Groq, Gemini (endpoint compatible OpenAI), OpenAI ou Ollama local. Sorties structurées validées par Pydantic, avec retry si le JSON est invalide.
- Agent : `langgraph` + `langchain-core`.
- PDF : `arxiv` (API), `pymupdf`.
- UI : `streamlit`.
- CLI : `typer`.
- Tests : `pytest`. Qualité : `ruff`.

## 4. Structure du dépôt

```
research-paper-explorer/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env.example            # LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, EMBEDDING_MODEL
├── config.yaml             # catégories, dates, tailles, chemins, hyperparamètres
├── data/                   # ignoré par git (sauf .gitkeep)
│   ├── raw/                # arxiv-metadata-oai-snapshot.json
│   ├── processed/          # papers.parquet, extractions.jsonl, topics.parquet
│   ├── index/              # chroma, bm25, embeddings.npy
│   ├── graph/              # kg.graphml
│   └── pdfs/
├── src/rpe/
│   ├── config.py
│   ├── cli.py
│   ├── data/      (load.py, clean.py, enrich_comments.py)
│   ├── index/     (embed.py, vector_store.py, bm25.py, hybrid.py)
│   ├── llm/       (client.py, prompts.py, schemas.py)
│   ├── rag/       (qa.py, reading_sheet.py)
│   ├── topics/    (bertopic_model.py, trends.py)
│   ├── kg/        (extract.py, normalize.py, build_graph.py, graph_store.py, graphrag.py)
│   ├── agent/     (tools.py, graph.py, sota.py, gaps.py, digest.py, recommend.py)
│   └── eval/      (retrieval_eval.py, extraction_eval.py, rag_eval.py, datasets/)
├── app/streamlit_app.py  (+ pages/)
├── tests/
└── notebooks/              # exploration uniquement
```

## 5. Paramètres par défaut (`config.yaml`)

```yaml
data:
  raw_path: data/raw/arxiv-metadata-oai-snapshot.json
  categories: ["cs.CL"]        # un papier est gardé si au moins une de ses catégories y figure
  start_date: "2023-01-01"
  end_date: null
  max_papers: 20000            # échantillon reproductible (seed 42) si dépassé
  sample_dev: 1000             # mode --dev pour itérer vite
extraction:
  max_papers: 2000             # l'extraction LLM est coûteuse : limiter
  batch_size: 10
  cache_path: data/processed/extractions.jsonl
retrieval:
  top_k: 10
  rrf_k: 60
pdfs:
  max_pdfs: 300
  delay_seconds: 3             # respecter les serveurs arXiv
```

## 6. Règles de développement (obligatoires)

- **Travailler phase par phase** (voir `PROMPTS_CODEX.md`). À la fin de chaque phase : lancer les tests et `ruff`, résumer ce qui a été fait, indiquer la commande pour vérifier, puis **s'arrêter** et attendre la validation.
- Tout script long doit avoir une barre de progression (`tqdm`), être **reprenable** (cache, checkpoint, sauter ce qui existe déjà) et avoir un mode `--dev` sur un petit échantillon.
- **Jamais** de clé API dans le code ni dans git. Lire depuis `.env` (`python-dotenv`). `data/` et `.env` sont dans `.gitignore`.
- Appels LLM : timeout, retries exponentiels, journalisation du nombre d'appels, cache par hash d'entrée (pas d'appel redondant).
- **Anti-hallucination** : toute réponse générée cite des `arxiv_id`. Une fonction `validate_citations()` vérifie que chaque ID cité appartient au contexte récupéré et retire ou signale les autres.
- Prompts LLM centralisés dans `src/rpe/llm/prompts.py`, en anglais, avec consignes de sortie strictes.
- Code typé (type hints), docstrings courtes, fonctions petites et testables.
- Ne pas ajouter de dépendance hors de la stack sans l'expliquer.
- README à jour : installation, configuration, commandes de chaque phase, captures de l'UI.
