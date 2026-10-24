# Prompts Codex — Research Paper Explorer

## Mode d'emploi
1. Créez le dossier du projet, ouvrez-le dans VS Code et placez-y **`AGENTS.md`** à la racine (Codex le lit comme contexte).
2. Téléchargez le dataset Kaggle et mettez `arxiv-metadata-oai-snapshot.json` dans `data/raw/`.
3. Envoyez à Codex **le prompt maître**, puis **un prompt de phase à la fois**. Testez et vérifiez avant de passer à la phase suivante.
4. Si Codex dévie : « Relis AGENTS.md, section X, et corrige. »

---

## PROMPT MAÎTRE (à envoyer en premier)

```
Tu es un ingénieur senior en IA générative et agents (RAG, GraphRAG, LangGraph) et en data engineering Python.
Nous construisons le projet "Research Paper Explorer" décrit dans AGENTS.md à la racine du dépôt.

Avant d'écrire du code :
1. Lis AGENTS.md en entier.
2. Résume en 10 lignes maximum ce que tu as compris (objectif, données, modules, contraintes).
3. Liste les questions ou ambiguïtés éventuelles.
4. Propose le plan des 9 phases (je te les enverrai une par une) et attends ma validation.

Contraintes permanentes :
- Une phase à la fois. À la fin de chaque phase : tests pytest + ruff, résumé des fichiers créés,
  commandes exactes pour vérifier, puis tu t'arrêtes.
- Le fichier data/raw/arxiv-metadata-oai-snapshot.json fait 5,6 Go en JSON Lines :
  lecture ligne par ligne uniquement, jamais en mémoire d'un coup.
- Aucun secret dans le code. Config dans config.yaml et .env.
- Scripts reprenables, avec cache, tqdm et mode --dev.
- Toute sortie générée par LLM cite des arxiv_id, vérifiés par validate_citations().
```

---

## PHASE 1 — Squelette du projet et préparation des données

```
Phase 1 : squelette et pipeline de données.

1. Crée la structure de dépôt de AGENTS.md (section 4) : requirements.txt, .env.example,
   config.yaml (section 5), .gitignore (data/, .env, __pycache__, .venv), src/rpe/config.py
   (chargement config + .env), src/rpe/cli.py (Typer).
2. src/rpe/data/load.py : générateur stream_records(path) qui lit le JSON Lines ligne par ligne
   avec orjson, ignore et compte les lignes invalides.
3. src/rpe/data/clean.py :
   - garder un papier si au moins une de ses catégories (champ "categories" séparé par des espaces)
     est dans config.data.categories ;
   - published_date = versions[0].created (RFC 2822 → date), filtre start_date/end_date ;
   - nettoyer titre et résumé (espaces, retours ligne) ;
   - authors_list depuis authors_parsed ("Prénom Nom") ;
   - colonnes finales : arxiv_id, title, abstract, authors_list, n_authors, categories (liste),
     primary_category, published_date, year, month, update_date, n_versions, comments,
     journal_ref, doi ;
   - si plus de max_papers : échantillon reproductible (seed 42) ;
   - export data/processed/papers.parquet.
4. src/rpe/data/enrich_comments.py : à partir de comments, journal_ref et abstract, ajoute
   venue, is_accepted, venue_year, github_url, has_code, n_pages, n_figures
   (regex documentées, liste des venues dans AGENTS.md).
5. Commande CLI : `python -m rpe.cli prepare-data [--dev]`, qui affiche des statistiques
   (nombre lu, gardé, par année, % avec code, top venues).
6. Tests : clean (dates, catégories), enrich_comments (au moins 12 cas réels de comments :
   "Accepted at ACL 2024", "To appear in NeurIPS 2023", "Code: https://github.com/x/y",
   "12 pages, 3 figures", etc.).

Arrête-toi après cette phase.
```

---

## PHASE 2 — Embeddings et recherche hybride

```
Phase 2 : indexation et recherche hybride.

1. src/rpe/index/embed.py : embeddings de "title. abstract" avec sentence-transformers
   (modèle dans .env/config), par batch, normalisés, sauvegardés dans data/index/embeddings.npy
   + ids.json. Reprenable.
2. src/rpe/index/vector_store.py : collection Chroma persistante avec métadonnées
   (year, primary_category, venue, has_code, is_accepted).
3. src/rpe/index/bm25.py : index BM25 sur title + abstract (tokenisation simple + stopwords anglais),
   sérialisé.
4. src/rpe/index/hybrid.py : search(query, top_k, filters) qui fusionne BM25 et dense en
   Reciprocal Rank Fusion (k configurable). Retourne des objets Pydantic SearchResult
   (arxiv_id, title, score, snippet, year, venue, has_code).
5. CLI : `build-index [--dev]` et `search "query" --year-min 2024 --has-code`.
6. Tests : filtres, fusion RRF sur un mini-corpus factice, déterminisme.

Arrête-toi après cette phase.
```

---

## PHASE 3 — Couche LLM, RAG avec citations et fiche de lecture

```
Phase 3 : client LLM, RAG et fiche de lecture.

1. src/rpe/llm/client.py : LLMClient compatible API OpenAI (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL),
   méthodes chat() et structured(schema: type[BaseModel]) avec validation Pydantic,
   retry si JSON invalide, timeout, backoff, cache disque par hash(prompt+model), compteur d'appels.
2. src/rpe/llm/prompts.py : tous les prompts (en anglais).
3. src/rpe/rag/qa.py : answer(question, scope = corpus | arxiv_id) :
   récupération hybride → contexte numéroté avec arxiv_id → réponse avec citations [arXiv:ID]
   → validate_citations() (retire ou signale les IDs absents du contexte)
   → si le contexte est insuffisant, répondre "je ne sais pas" au lieu d'inventer.
4. src/rpe/rag/reading_sheet.py : fiche de lecture structurée (Pydantic) : summary, contributions,
   method, results, limitations, 3 comprehension_questions ; option lang="fr".
   Utilise le résumé et, si disponible, le texte PDF (phase 7).
5. CLI : `ask "question"` et `reading-sheet 2310.xxxxx --lang fr`.
6. Tests : validate_citations, parsing structuré avec un LLM mocké.

Arrête-toi après cette phase.
```

---

## PHASE 4 — Topics et tendances

```
Phase 4 : topic modeling et tendances.

1. src/rpe/topics/bertopic_model.py : BERTopic sur les résumés en réutilisant les embeddings
   de la phase 2 (ne pas recalculer). UMAP + HDBSCAN paramétrables, seed fixe.
   Étiquettes de topics lisibles générées par le LLM (3 à 6 mots) à partir des mots-clés
   et de 5 titres représentatifs. Sauvegarde : data/processed/topics.parquet
   (arxiv_id, topic_id, topic_label) + le modèle.
2. Carte 2D : UMAP 2D, figure Plotly interactive (couleur = topic, survol = titre).
3. src/rpe/topics/trends.py :
   - nombre de papiers par topic et par mois, et part relative ;
   - croissance = comparaison des 6 derniers mois vs les 6 précédents ;
   - "sujets émergents" = forte croissance relative et volume minimum (seuil configurable) ;
   - "sujets en déclin" ;
   - summarize_trends() : le LLM rédige une synthèse factuelle à partir des chiffres calculés
     (il ne doit inventer aucun chiffre, seulement ceux fournis dans le prompt).
4. CLI : `build-topics [--dev]`, `trends [--topic "agents"]`.
5. Tests : calcul de croissance et détection d'émergence sur des données factices.

Arrête-toi après cette phase.
```

---

## PHASE 5 — Extraction de connaissances et graphe (GraphRAG)

```
Phase 5 : extraction par LLM et graphe de connaissances.

1. src/rpe/llm/schemas.py : PaperExtraction(tasks, methods, datasets, metrics,
   claimed_results: list[{metric, dataset, value, context}], contribution_type:
   new_method | new_dataset | benchmark | survey | analysis | application | other).
   Consigne : n'extraire que ce qui est explicitement dans le résumé, listes vides sinon.
2. src/rpe/kg/extract.py : extraction par batch (config.extraction.max_papers, en priorité les papiers
   acceptés ou avec code, puis aléatoire), cache JSONL reprenable, log des erreurs.
3. src/rpe/kg/normalize.py : normalisation des entités (minuscules pour la comparaison, alias,
   suppression des versions "v1.1", fusion par similarité de chaînes ou d'embeddings au-dessus
   d'un seuil, table d'alias éditable data/processed/aliases.yaml).
4. src/rpe/kg/build_graph.py + graph_store.py : graphe NetworkX selon AGENTS.md (nœuds et arêtes),
   incluant CO_AUTHOR (poids = nombre de papiers communs) et IN_CATEGORY. Interface GraphStore
   (implémentation NetworkX ; Neo4j optionnel et non requis). Export kg.graphml.
5. src/rpe/kg/graphrag.py :
   - graph_query(entity, relation) : par exemple papiers qui utilisent le dataset X, méthodes
     appliquées à la tâche Y ;
   - answer_graph_question(question) : le LLM identifie les entités et la relation, on interroge
     le graphe, puis on récupère les résumés liés et on répond avec citations ;
   - statistiques : auteurs centraux (degree / PageRank), communautés (Louvain).
6. CLI : `extract [--dev]`, `build-graph`, `graph-ask "Which papers use SQuAD?"`.
7. Tests : normalisation, construction du graphe, graph_query sur un mini-graphe.

Arrête-toi après cette phase.
```

---

## PHASE 6 — Agent LangGraph

```
Phase 6 : agent.

1. src/rpe/agent/tools.py : outils LangChain typés et documentés :
   search_papers, get_paper, graph_query, recommend_similar, get_trends,
   find_research_gaps, read_pdf, fetch_latest_arxiv.
2. src/rpe/agent/recommend.py : centroïde des embeddings de 1 à 5 arxiv_id, puis k plus proches
   voisins (en excluant les entrées), avec une phrase d'explication par recommandation.
3. src/rpe/agent/gaps.py : matrice Tâche × Méthode depuis le graphe ; gap = combinaison rare
   (≤ seuil) alors que tâche et méthode sont fréquentes (≥ seuil). Score, top N, et justification
   LLM présentée comme hypothèse, avec papiers proches en preuve.
4. src/rpe/agent/sota.py : workflow LangGraph de génération d'état de l'art :
   plan (sous-questions) → recherche par sous-question → sélection (pertinence, récence, venue,
   has_code, diversité) → regroupement par approche → rédaction Markdown (Introduction,
   approches, datasets et métriques courants, tendances, limites et pistes, références)
   → validate_citations. Limite de N papiers et d'itérations.
5. src/rpe/agent/digest.py : fetch_latest_arxiv(query, days) via le package `arxiv`,
   puis résumé groupé par thème, export Markdown dans data/digests/.
6. src/rpe/agent/graph.py : agent général (ReAct LangGraph) qui choisit ses outils, avec
   max_iterations et une trace des étapes (outil appelé, arguments, résumé du résultat)
   affichable dans l'UI.
7. CLI : `agent "question"`, `sota "retrieval-augmented generation"`, `gaps [--task ...]`,
   `recommend ID1 ID2 ID3`, `digest --query "LLM agents" --days 1`.
8. Tests avec LLM mocké : enchaînement des nœuds, respect de max_iterations, citations validées.

Arrête-toi après cette phase.
```

---

## PHASE 7 — PDF (texte complet)

```
Phase 7 : texte complet pour un sous-ensemble.

1. Télécharger au maximum config.pdfs.max_pdfs PDF (priorité : papiers les plus pertinents
   pour les requêtes de démo, puis acceptés ou avec code) depuis https://arxiv.org/pdf/{id},
   avec délai entre requêtes, reprise, et skip si déjà présent.
2. Extraction PyMuPDF, découpage par sections (Abstract, Introduction, Method, Experiments,
   Results, Conclusion, References) avec heuristiques sur les titres, chunks de taille bornée
   avec chevauchement.
3. Collection Chroma "chunks" (métadonnées arxiv_id, section). Le RAG sur un papier utilise
   ces chunks quand ils existent et cite [arXiv:ID §Section].
4. Bonus : extraire de la section References les IDs arXiv cités, et ajouter des arêtes CITES
   dans le graphe entre papiers du corpus.
5. CLI : `fetch-pdfs`, `index-pdfs`.

Arrête-toi après cette phase.
```

---

## PHASE 8 — Interface Streamlit

```
Phase 8 : interface Streamlit (app/streamlit_app.py + pages/).

Pages :
1. Recherche : barre de requête, filtres (années, catégorie, venue, has_code), cartes de résultats
   (titre, auteurs, année, venue, badge code, lien arXiv, TLDR via LLM à la demande).
2. Chat RAG : choix corpus ou papier, réponses avec citations cliquables.
3. État de l'art : saisie d'un sujet, trace de l'agent en direct (st.status), revue Markdown
   téléchargeable.
4. Tendances : carte 2D des topics, courbes par mois, liste des sujets émergents et en déclin,
   synthèse LLM.
5. Graphe : visualisation pyvis d'un sous-graphe autour d'une entité, questions GraphRAG,
   auteurs centraux.
6. Research gaps : tableau des gaps avec score et justification, heatmap Tâche × Méthode.
7. Veille : requête + nombre de jours, digest.
8. Fiche de lecture : arxiv_id, puis fiche (FR/EN), export Markdown.

Exigences : st.cache_resource pour les index et modèles, gestion d'erreurs propre (message clair
si l'index n'est pas construit), design sobre et cohérent.
Commande : `streamlit run app/streamlit_app.py`.

Arrête-toi après cette phase.
```

---

## PHASE 9 — Évaluation, README et démo

```
Phase 9 : évaluation et finalisation.

1. src/rpe/eval/retrieval_eval.py : jeu eval/datasets/queries.jsonl (30 requêtes avec arxiv_id
   pertinents, généré semi-automatiquement puis à valider à la main). Comparaison BM25 vs dense
   vs hybride : Recall@5/10, MRR, nDCG@10, tableau + graphique.
2. src/rpe/eval/extraction_eval.py : 50 résumés annotés à la main (gold JSONL) → précision, rappel,
   F1 par champ (tasks, methods, datasets, metrics), avec correspondance après normalisation.
3. src/rpe/eval/rag_eval.py : 20 questions → taux de citations valides, fidélité (LLM-as-judge
   avec prompt strict) et pertinence ; RAGAS en option.
4. CLI : `evaluate --all`, qui génère eval/report.md.
5. README complet : présentation, architecture (diagramme Mermaid du pipeline et de l'agent),
   installation, configuration .env, commandes dans l'ordre, résultats d'évaluation, limites,
   pistes d'amélioration, captures d'écran.
6. scripts/demo.sh : enchaîne toutes les étapes en mode --dev pour une démo rapide.

Arrête-toi et donne la liste finale des commandes pour reproduire le projet de zéro.
```

---

## Prompts utiles en cas de problème
- « Le script plante sur la ligne X du dataset : ajoute une gestion d'erreur, log l'ID, continue. »
- « C'est trop lent : profile, propose 2 optimisations, applique la meilleure. »
- « Le LLM invente des IDs : renforce le prompt et validate_citations, ajoute un test. »
- « Explique-moi ce module comme pour une soutenance : rôle, choix techniques, alternatives. »
