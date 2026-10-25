"""Generate docs/rapport_projet.pdf: detailed project report (French), with diagrams.

Run: python scripts/make_project_report.py
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

from rpe.reports.pdf_report import ACCENT, GREY, MARGIN, fonts, styles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "rapport_projet.pdf"
WIDTH = A4[0] - 2 * MARGIN

REG, BOLD, ITAL = fonts()
ST = styles()
CODE = ParagraphStyle(
    "code",
    fontName=REG,
    fontSize=8,
    leading=10.5,
    backColor=colors.HexColor("#f3f5f8"),
    borderPadding=6,
    leftIndent=6,
    rightIndent=6,
    spaceBefore=4,
    spaceAfter=8,
)
CAPTION = ParagraphStyle("caption", parent=ST["meta"], alignment=1, spaceAfter=10)

C_DATA = colors.HexColor("#dbeafe")
C_PROC = colors.HexColor("#e0f2e9")
C_LLM = colors.HexColor("#fde9d9")
C_UI = colors.HexColor("#ede9fe")
C_STORE = colors.HexColor("#fef9c3")
ARROW = colors.HexColor("#334155")


# ------------------------------------------------------------- helpers


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, ST[style])


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(t, ST["body"]), leftIndent=12) for t in items],
        bulletType="bullet",
        start="•",
        bulletFontName=REG,
        leftIndent=12,
    )


def table(rows: list[list[str]], widths: list[float], font_size: float = 8.5) -> Table:
    cell = ParagraphStyle("c", parent=ST["cell"], fontSize=font_size, leading=font_size + 2.5)
    head = ParagraphStyle("h", parent=ST["head"], fontSize=font_size, leading=font_size + 2.5)
    data = [[Paragraph(c, head) for c in rows[0]]]
    data += [[Paragraph(c, cell) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d1dc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), CODE)


class Diagram:
    """Boxes + arrows drawn with reportlab.graphics (coordinates in points)."""

    def __init__(self, height: float) -> None:
        self.d = Drawing(WIDTH, height)
        self.boxes: dict[str, tuple[float, float, float, float]] = {}

    def box(
        self,
        key: str,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        fill: colors.Color = C_PROC,
        bold_first: bool = True,
    ) -> None:
        self.boxes[key] = (x, y, w, h)
        self.d.add(
            Rect(
                x,
                y,
                w,
                h,
                rx=5,
                ry=5,
                fillColor=fill,
                strokeColor=colors.HexColor("#64748b"),
                strokeWidth=0.7,
            )
        )
        lines = label.split("\n")
        lh = 9
        top = y + h / 2 + (len(lines) - 1) * lh / 2 - 3
        for i, ln in enumerate(lines):
            self.d.add(
                String(
                    x + w / 2,
                    top - i * lh,
                    ln,
                    textAnchor="middle",
                    fontName=BOLD if (i == 0 and bold_first) else REG,
                    fontSize=7.5,
                )
            )

    def _anchor(self, key: str, side: str) -> tuple[float, float]:
        x, y, w, h = self.boxes[key]
        return {
            "r": (x + w, y + h / 2),
            "l": (x, y + h / 2),
            "t": (x + w / 2, y + h),
            "b": (x + w / 2, y),
        }[side]

    def arrow(
        self,
        a: str,
        b: str,
        sa: str = "r",
        sb: str = "l",
        label: str = "",
        color: colors.Color = ARROW,
        dashed: bool = False,
    ) -> None:
        (x1, y1), (x2, y2) = self._anchor(a, sa), self._anchor(b, sb)
        line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.9)
        if dashed:
            line.strokeDashArray = [3, 2]
        self.d.add(line)
        import math

        ang = math.atan2(y2 - y1, x2 - x1)
        s = 5
        p1 = (x2 - s * math.cos(ang - 0.4), y2 - s * math.sin(ang - 0.4))
        p2 = (x2 - s * math.cos(ang + 0.4), y2 - s * math.sin(ang + 0.4))
        self.d.add(Polygon([x2, y2, *p1, *p2], fillColor=color, strokeColor=color))
        if label:
            self.d.add(
                String(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2 + 3,
                    label,
                    textAnchor="middle",
                    fontName=ITAL,
                    fontSize=6.5,
                    fillColor=colors.HexColor("#475569"),
                )
            )


def architecture_diagram() -> Drawing:
    g = Diagram(250)
    bw, bh = 88, 36
    # column 1: data
    g.box("raw", 0, 200, bw, bh, "Dump arXiv Kaggle\nJSONL 5,6 Go", C_DATA)
    g.box("prep", 0, 130, bw, bh, "prepare-data\nfiltre + nettoyage", C_PROC)
    g.box("parquet", 0, 60, bw, bh, "papers.parquet\n20 000 papiers", C_STORE)
    g.arrow("raw", "prep", "b", "t", "streaming")
    g.arrow("prep", "parquet", "b", "t")
    # column 2: indexes
    x2 = 118
    g.box("chroma", x2, 160, bw, bh, "ChromaDB\nembeddings bge-small", C_STORE)
    g.box("bm25", x2, 95, bw, bh, "BM25\nbm25.pkl", C_STORE)
    g.box("pdfs", x2, 10, bw, bh, "PDF arXiv\npassages par section", C_STORE)
    g.arrow("parquet", "chroma", "r", "l", "build-index")
    g.arrow("parquet", "bm25", "r", "l")
    # column 3: services
    x3 = 236
    g.box("hybrid", x3, 130, bw, bh, "Recherche hybride\nfusion RRF + filtres", C_PROC)
    g.box("rag", x3, 60, bw, bh, "RAG\nvalidate_citations", C_LLM)
    g.box("cmp", x3, 0, bw, bh, "Comparaison\n+ rapport PDF", C_PROC)
    g.arrow("chroma", "hybrid", "r", "l")
    g.arrow("bm25", "hybrid", "r", "l")
    g.arrow("hybrid", "rag", "b", "t")
    g.arrow("pdfs", "rag", "r", "l", dashed=True)
    g.arrow("rag", "cmp", "b", "t")
    # column 4: LLM + UI
    x4 = 354
    g.box("llm", x4, 175, WIDTH - x4, 40, "LLMClient\nOllama · Groq · Gemini", C_LLM)
    g.box("ui", x4, 40, WIDTH - x4, 90, "Streamlit\nChat (RAG)\nRecherche\nHistorique", C_UI)
    g.arrow("llm", "rag", "b", "r", dashed=True)
    g.arrow("hybrid", "ui", "r", "l")
    g.arrow("rag", "ui", "r", "l")
    g.arrow("cmp", "ui", "r", "l")
    return g.d


def rag_sequence_diagram() -> Drawing:
    g = Diagram(250)
    bw, bh = 100, 34
    g.box("user", 0, 200, bw, bh, "Utilisateur\nquestion (page Chat)", C_UI)
    g.box("bm25", 140, 225, bw, 28, "BM25\n100 candidats", C_STORE)
    g.box("dense", 140, 180, bw, 28, "Chroma (dense)\n100 candidats", C_STORE)
    g.box("rrf", 280, 200, bw, bh, "Fusion RRF\nk = 60 → top 5", C_PROC)
    g.box("ctx", 280, 120, bw, bh, "Contexte numéroté\n[1] arXiv:ID | titre", C_PROC)
    g.box("llm", 140, 120, bw, bh, "LLMClient\ncache → LLM → retry", C_LLM)
    g.box("valid", 0, 120, bw, bh, "validate_citations\nretire IDs inventés", C_PROC)
    g.box("answer", 0, 40, bw, bh, "Réponse affichée\n[arXiv:ID] cliquables", C_UI)
    g.box("cmp", 140, 40, bw, bh, "Comparaison\npapiers cités", C_PROC)
    g.box("pdf", 280, 40, bw, bh, "Rapport PDF\n(option, ReportLab)", C_UI)
    g.arrow("user", "bm25", "r", "l", "1")
    g.arrow("user", "dense", "r", "l")
    g.arrow("bm25", "rrf", "r", "l", "2")
    g.arrow("dense", "rrf", "r", "l")
    g.arrow("rrf", "ctx", "b", "t", "3")
    g.arrow("ctx", "llm", "l", "r", "4 prompt")
    g.arrow("llm", "valid", "l", "r", "5 texte")
    g.arrow("valid", "answer", "b", "t", "6")
    g.arrow("answer", "cmp", "r", "l", "7")
    g.arrow("cmp", "pdf", "r", "l", "8")
    g.box("idk", 410, 120, 66, bh, "« Je ne sais\npas »", colors.HexColor("#fee2e2"), False)
    g.arrow("ctx", "idk", "r", "l", "insuffisant", dashed=True)
    return g.d


def llm_client_diagram() -> Drawing:
    g = Diagram(120)
    bw, bh = 82, 36
    g.box("call", 0, 60, bw, bh, "structured()\nschéma Pydantic", C_PROC)
    g.box("cache", 100, 60, bw, bh, "Cache disque\nsha256(entrée)", C_STORE)
    g.box("api", 200, 60, bw, bh, "Endpoint OpenAI\ntimeout 300 s", C_LLM)
    g.box("parse", 300, 60, bw, bh, "parse_json +\nvalidation Pydantic", C_PROC)
    g.box("ok", 400, 60, 76, bh, "Objet typé\nretourné", C_UI)
    g.box("retry", 200, 0, bw, bh, "Retry exponentiel\n2 s, 4 s, 8 s…", colors.HexColor("#fee2e2"))
    g.box("fix", 300, 0, bw, bh, "Message correctif\n+ erreur JSON", colors.HexColor("#fee2e2"))
    g.arrow("call", "cache", "r", "l")
    g.arrow("cache", "api", "r", "l", "miss")
    g.arrow("api", "parse", "r", "l")
    g.arrow("parse", "ok", "r", "l", "valide")
    g.arrow("api", "retry", "b", "t", "erreur réseau", dashed=True)
    g.arrow("parse", "fix", "b", "t", "invalide", dashed=True)
    g.arrow("fix", "api", "l", "b", dashed=True)
    return g.d


# ------------------------------------------------------------- content


def story() -> list:
    s: list = []
    now = datetime.now()

    # ---- title page
    s += [
        Spacer(1, 4 * cm),
        Paragraph("Research Paper Explorer", ST["title"]),
        Paragraph("Rapport de projet — Generative AI (ENSIAS)", ST["subtitle"]),
        Spacer(1, 0.8 * cm),
        Paragraph(
            "Assistant de recherche bibliographique fondé sur l'IA générative : recherche "
            "hybride et RAG avec citations vérifiées sur le dataset arXiv de Kaggle, "
            "comparaison des papiers cités et rapport PDF.",
            ST["subtitle"],
        ),
        Spacer(1, 1.2 * cm),
        Paragraph(f"Version du {now:%d/%m/%Y}", ST["subtitle"]),
        Paragraph("Dépôt : github.com/Sabbarso/agent-ai-", ST["subtitle"]),
        PageBreak(),
    ]

    # ---- contents
    s += [
        P("Sommaire", "h1"),
        bullets(
            [
                "1. Objectif et vue d'ensemble",
                "2. Les données : le dataset arXiv",
                "3. Architecture générale",
                "4. Les outils et technologies utilisés",
                "5. Le pipeline, étape par étape",
                "6. Communication entre les composants",
                "7. Garde-fous contre les hallucinations",
                "8. L'interface utilisateur",
                "9. Tests et qualité",
                "10. Limites et pistes d'amélioration",
                "11. Commandes pour reproduire le projet",
                "12. Glossaire",
            ]
        ),
        PageBreak(),
    ]

    # ---- 1
    s += [
        P("1. Objectif et vue d'ensemble", "h1"),
        P(
            "Un étudiant ou un chercheur qui commence un sujet se heurte à deux problèmes : le "
            "volume (des dizaines de milliers de papiers par an rien qu'en traitement "
            "automatique du langage) et la fiabilité des assistants IA, qui inventent "
            "facilement des références. <b>Research Paper Explorer</b> répond aux deux : il "
            "permet de <b>trouver</b> les bons papiers, de <b>poser des questions</b> dont "
            "chaque affirmation est rattachée à un papier réel et vérifié, de <b>comparer</b> "
            "les papiers cités et d'en <b>exporter un rapport PDF</b>."
        ),
        table(
            [
                ["Étape", "Contenu", "Résultat"],
                [
                    "Données",
                    "lecture streaming, filtre, nettoyage, enrichissement",
                    "papers.parquet (20 000 papiers)",
                ],
                ["Index", "embeddings + ChromaDB, BM25", "recherche hybride avec filtres"],
                [
                    "LLM et RAG",
                    "client LLM unique, réponses citées et vérifiées",
                    "Chat avec citations [arXiv:ID]",
                ],
                [
                    "Comparaison",
                    "extraction structurée des papiers cités",
                    "tableau, synthèse, rapport PDF",
                ],
                [
                    "Texte intégral",
                    "PDF arXiv découpés par section",
                    "citations [arXiv:ID §Section]",
                ],
                [
                    "Interface",
                    "Streamlit : Chat et Recherche",
                    "historique des conversations enregistré",
                ],
            ],
            [0.18, 0.47, 0.35],
        ),
        Spacer(1, 6),
        P(
            "Le code représente 38 modules Python (environ 4 200 lignes) dans "
            "<i>src/rpe/</i>, couverts par 113 tests automatisés."
        ),
    ]

    # ---- 2
    s += [
        P("2. Les données : le dataset arXiv", "h1"),
        P(
            "La source unique est le dataset arXiv publié par Cornell University sur Kaggle : "
            "un fichier <b>JSON Lines</b> de 5,6 Go, un objet JSON par ligne et par papier "
            "(identifiant, titre, auteurs, catégories, résumé, commentaires, versions…). Le "
            "fichier est trop gros pour être chargé en mémoire : il est <b>lu ligne par "
            "ligne</b> (streaming) avec la bibliothèque orjson."
        ),
        P("Chiffres réels obtenus sur le dump", "h2"),
        table(
            [
                ["Étape", "Nombre", "Commentaire"],
                ["Lignes lues", "3 215 335", "0 ligne invalide"],
                [
                    "Écartées sans décodage JSON",
                    "3 092 613",
                    "pré-filtre sur les octets « cs.CL » (96 %)",
                ],
                ["Papiers de la catégorie cs.CL", "122 720", "au moins une catégorie cs.CL"],
                ["… publiés depuis le 01/01/2023", "81 503", "date = versions[0].created"],
                ["Échantillon conservé", "20 000", "tirage aléatoire reproductible (seed 42)"],
            ],
            [0.38, 0.17, 0.45],
        ),
        Spacer(1, 6),
        P(
            "Répartition : 2023 : 3 304 · 2024 : 5 063 · 2025 : 5 916 · 2026 : 5 717 papiers. "
            "16,8 % des papiers ont un lien vers du code, 36,3 % sont acceptés dans une "
            "conférence ou une revue, et 28,7 % ont une venue identifiée (EMNLP 829, ACL 769, "
            "ACL Findings 425, EMNLP Findings 411, ICLR 351, NeurIPS 310…)."
        ),
        P("Nettoyage et enrichissement", "h2"),
        bullets(
            [
                "<b>Date de publication</b> : première version (<i>versions[0].created</i>, "
                "format RFC 2822) et non la date de mise à jour.",
                "<b>Auteurs</b> : reconstruits depuis <i>authors_parsed</i>, accents LaTeX "
                "(\\'e → é) convertis en Unicode.",
                "<b>Venue et acceptation</b> : expressions régulières sur les commentaires et "
                "la référence de revue (« Accepted at ACL 2024 », « To appear in NeurIPS'23 », "
                "« Findings of EMNLP »…), 27 venues reconnues, formulations négatives "
                "(« under review ») détectées.",
                "<b>Code</b> : premier lien github.com ; <b>pages / figures</b> : « 8+2 pages, "
                "3 figures ».",
                "<b>Échantillonnage par réservoir</b> : tirage uniforme en une seule passe, avec "
                "une mémoire bornée à 20 000 papiers.",
            ]
        ),
        PageBreak(),
    ]

    # ---- 3
    s += [
        P("3. Architecture générale", "h1"),
        P(
            "Le schéma montre les blocs et le sens des données. En bleu les données brutes, en "
            "jaune les fichiers stockés sur disque, en vert les traitements, en orange ce qui "
            "fait appel au LLM et en violet l'interface."
        ),
        architecture_diagram(),
        Paragraph("Figure 1 — Architecture : des données brutes à l'interface.", CAPTION),
        P(
            "Chaque étape produit des <b>fichiers</b> que l'étape suivante relit : chacune est "
            "indépendante, reprenable après une interruption et testable séparément. Un mode "
            "<b>--dev</b> travaille sur 1 000 papiers avec des fichiers séparés, pour itérer "
            "vite sans écraser le corpus complet."
        ),
        P("Organisation du code", "h2"),
        table(
            [
                ["Dossier", "Rôle"],
                ["src/rpe/config.py", "lecture de config.yaml (paramètres) et .env (secrets)"],
                ["src/rpe/data/", "streaming, nettoyage, enrichissement, accès aux papiers"],
                ["src/rpe/index/", "embeddings, ChromaDB, BM25, filtres, recherche hybride"],
                ["src/rpe/llm/", "client LLM, prompts centralisés, schémas Pydantic"],
                ["src/rpe/rag/", "RAG, vérification des citations, comparaison des papiers"],
                ["src/rpe/pdf/", "téléchargement des PDF, découpage en sections, passages"],
                ["src/rpe/reports/", "génération des rapports PDF"],
                ["src/rpe/ui/, app/", "interface Streamlit et historique des conversations"],
                ["src/rpe/resources.py", "chargement partagé des index et du LLM"],
                ["src/rpe/cli.py", "ligne de commande (Typer)"],
            ],
            [0.3, 0.7],
        ),
        PageBreak(),
    ]

    # ---- 4
    s += [
        P("4. Les outils et technologies utilisés", "h1"),
        P("Chaque outil a un rôle précis ; le tableau indique pourquoi il est là et où."),
        table(
            [
                ["Outil", "Rôle dans le projet", "Où"],
                ["Python 3.12", "langage principal (typé, testé)", "partout"],
                [
                    "orjson",
                    "décodage JSON très rapide, ligne par ligne, du dump de 5,6 Go",
                    "data/load.py",
                ],
                ["pandas + pyarrow", "tableaux de papiers, stockage Parquet compressé", "data/"],
                [
                    "sentence-transformers",
                    "modèle BAAI/bge-small-en-v1.5 : chaque « titre. résumé » devient un vecteur de 384 nombres",
                    "index/embed.py",
                ],
                [
                    "ChromaDB",
                    "base vectorielle persistante : vecteurs les plus proches, filtres sur l'année, la venue, le code…",
                    "index/vector_store.py",
                ],
                [
                    "rank-bm25",
                    "recherche lexicale classique (mots exacts, pondération BM25)",
                    "index/bm25.py",
                ],
                [
                    "openai (client)",
                    "protocole standard pour parler à tout LLM compatible : Ollama local, Groq, Gemini, OpenAI",
                    "llm/client.py",
                ],
                ["Ollama + qwen2.5:3b", "LLM exécuté localement, sans clé API ni coût", ".env"],
                [
                    "Pydantic",
                    "schémas des sorties du LLM : JSON validé, sinon corrigé par le LLM",
                    "llm/schemas.py",
                ],
                ["PyMuPDF", "extraction du texte des PDF arXiv", "pdf/parse.py"],
                [
                    "httpx",
                    "téléchargement des PDF, avec pause entre les requêtes",
                    "pdf/download.py",
                ],
                ["ReportLab", "rapports PDF (comparaison, ce document)", "reports/"],
                ["Streamlit", "application web : Chat et Recherche", "app/"],
                ["Typer", "ligne de commande python -m rpe.cli …", "cli.py"],
                ["pytest + ruff", "tests automatisés et contrôle de qualité du code", "tests/"],
                ["Git / GitHub", "versionnement et partage du code", "dépôt"],
            ],
            [0.22, 0.58, 0.20],
            font_size=8,
        ),
        PageBreak(),
    ]

    # ---- 5
    s += [P("5. Le pipeline, étape par étape", "h1")]
    steps = [
        (
            "Préparation des données",
            "Lecture en streaming, pré-filtre sur les octets (une ligne qui ne contient pas "
            "« cs.CL » est ignorée sans être décodée), filtre exact sur les catégories et les "
            "dates, nettoyage, enrichissement par expressions régulières, échantillonnage par "
            "réservoir et écriture atomique du Parquet. Une passe sur le dump prend environ "
            "1 min 15 s.",
        ),
        (
            "Index et recherche hybride",
            "Deux moteurs complémentaires : <b>BM25</b> retrouve les mots exacts (noms de "
            "datasets, acronymes) et la recherche <b>dense</b> retrouve le sens (« reduce "
            "hallucination » ≈ « improve factuality »). Chacun propose 100 candidats ; la "
            "<b>Reciprocal Rank Fusion</b> les combine : score = Σ 1/(60 + rang). Les "
            "embeddings sont calculés par parties de 2 048 papiers pour pouvoir reprendre "
            "après une interruption.",
        ),
        (
            "Couche LLM et RAG",
            "Le <b>LLMClient</b> encapsule tous les appels (section 6). Le <b>RAG</b> récupère "
            "les 5 papiers les plus pertinents, les numérote dans le prompt avec leur "
            "identifiant, demande une réponse citée, puis vérifie chaque citation.",
        ),
        (
            "Comparaison des papiers cités et rapport PDF",
            "Les papiers cités dans la réponse (5 au maximum) sont décrits en un seul appel "
            "LLM groupé (problème, méthode, datasets, métriques, résultats, limites), mis en "
            "cache par papier. Une synthèse comparative citée est rédigée. Le rapport PDF "
            "reprend la réponse, une fiche par papier, un tableau comparatif en paysage et les "
            "références.",
        ),
        (
            "Texte intégral (PDF)",
            "Jusqu'à 300 PDF téléchargés depuis arxiv.org, avec 3 s de pause entre deux "
            "requêtes. Le texte est découpé en sections (Introduction, Method, Experiments…) "
            "puis en passages de 1 500 caractères avec chevauchement, indexés dans ChromaDB. "
            "Le RAG sur un papier cite alors la section ([arXiv:ID §Method]) et la comparaison "
            "utilise le texte complet.",
        ),
        (
            "Interface",
            "Application Streamlit qui s'ouvre sur le Chat, avec la Recherche en seconde page "
            "et l'historique des conversations enregistré (section 8).",
        ),
    ]
    for title, text in steps:
        s += [KeepTogether([P(title, "h2"), P(text)])]
    s.append(PageBreak())

    # ---- 6
    s += [
        P("6. Communication entre les composants", "h1"),
        P(
            "Les composants ne s'appellent pas en réseau : ce sont des modules Python qui "
            "échangent des <b>objets typés</b> (Pydantic) en mémoire et des <b>fichiers</b> sur "
            "disque. Seuls deux échanges sortent de la machine : les requêtes vers le LLM (en "
            "local avec Ollama sur http://localhost:11434) et le téléchargement des PDF sur "
            "arxiv.org."
        ),
        P("6.1 Les fichiers échangés", "h2"),
        table(
            [
                ["Fichier", "Produit par", "Lu par", "Format"],
                ["papers.parquet", "prepare-data", "index, RAG, interface", "Parquet"],
                ["embeddings.npy + ids.json", "build-index", "build-index (reprise)", "NumPy"],
                ["chroma/", "build-index, index-pdfs", "recherche dense, RAG", "base vectorielle"],
                ["bm25.pkl", "build-index", "recherche lexicale", "objet sérialisé"],
                ["data/pdfs/", "fetch-pdfs", "index-pdfs, comparaison", "PDF + JSON"],
                ["data/cache/llm/", "LLMClient", "LLMClient", "JSON par empreinte"],
                ["data/cache/compare/", "comparaison", "comparaison", "JSON par papier"],
                ["data/conversations/", "page Chat", "page Chat", "JSON par conversation"],
            ],
            [0.28, 0.22, 0.28, 0.22],
            font_size=8,
        ),
        Spacer(1, 8),
        P("6.2 Le LLMClient : un point d'entrée unique vers le modèle", "h2"),
        P(
            "Tous les modules passent par la même classe. Elle vérifie d'abord le cache disque "
            "(clé = empreinte sha256 du modèle et des messages) : une question déjà posée ne "
            "coûte aucun appel. Sinon elle appelle l'endpoint compatible OpenAI avec un délai "
            "maximal, relance avec une attente croissante en cas d'erreur réseau, puis valide "
            "le JSON reçu contre le schéma Pydantic attendu. Si le JSON est invalide, le LLM "
            "reçoit sa propre réponse et l'erreur, et doit la corriger."
        ),
        llm_client_diagram(),
        Paragraph("Figure 2 — Cycle d'un appel structuré au LLM.", CAPTION),
        code(
            """
# Exemple : la comparaison demande un objet typé, pas du texte libre
lot = llm.structured(prompt, ComparisonBatch, system=COMPARE_EXTRACT_SYSTEM)
lot.papers[0].datasets   # -> ["SQuAD", "Natural Questions"]  (liste validée)
llm.stats                # -> LLM calls=1 cache_hits=0 retries=0 parse_failures=0
"""
        ),
        PageBreak(),
        P("6.3 Le trajet d'une question dans le Chat", "h2"),
        P("Voici ce qui se passe, étape par étape, quand l'utilisateur pose une question :"),
        rag_sequence_diagram(),
        Paragraph(
            "Figure 3 — Séquence d'une question RAG (numéros = ordre des échanges).", CAPTION
        ),
        bullets(
            [
                "<b>1–2.</b> La question part vers les deux moteurs ; BM25 et ChromaDB "
                "renvoient chacun une liste classée d'identifiants arXiv, fusionnée par RRF.",
                "<b>3.</b> Les 5 meilleurs papiers forment un contexte numéroté : « [1] "
                "arXiv:2503.10677 | titre (année, venue) » suivi du résumé (et des passages du "
                "PDF s'il a été indexé).",
                "<b>4–5.</b> Le prompt impose de citer chaque affirmation avec un identifiant du "
                "contexte, ou de répondre INSUFFICIENT_CONTEXT ; le LLM renvoie un texte.",
                "<b>6.</b> <i>validate_citations()</i> retire tout identifiant absent du "
                "contexte (et toute section inventée). S'il ne reste aucune citation valide, la "
                "réponse devient « Je ne sais pas ».",
                "<b>7–8.</b> Les papiers cités sont comparés (un appel groupé, mis en cache) ; le "
                "rapport PDF réutilise cette comparaison, sans nouvel appel au LLM.",
            ]
        ),
        P("6.4 L'interface et le reste du code", "h2"),
        P(
            "Les pages Streamlit ne contiennent pas de logique métier : elles appellent les "
            "mêmes fonctions que la ligne de commande, via l'objet <i>Resources</i> qui charge "
            "à la demande (et une seule fois, grâce à <i>st.cache_resource</i>) les papiers, "
            "les index et le client LLM. Chaque conversation est enregistrée dans un fichier "
            "JSON (questions, réponses, comparaisons) et rechargée quand on la sélectionne. La "
            "page Recherche communique avec le Chat en déposant dans la session le papier "
            "choisi ou la question, puis en changeant de page (<i>st.switch_page</i>)."
        ),
        PageBreak(),
    ]

    # ---- 7
    s += [
        P("7. Garde-fous contre les hallucinations", "h1"),
        P(
            "Un LLM peut inventer une référence plausible. Le projet ne fait jamais confiance "
            "au texte généré sans vérification :"
        ),
        table(
            [
                ["Risque", "Garde-fou"],
                [
                    "Citation d'un papier inexistant ou hors contexte",
                    "validate_citations() retire tout [arXiv:ID] absent du contexte",
                ],
                [
                    "Section inventée ([arXiv:ID §3.3])",
                    "la section est supprimée si elle ne figure pas dans le contexte",
                ],
                ["Réponse sans source", "aucune citation valide → « Je ne sais pas »"],
                [
                    "Résultats inventés (comparaison)",
                    "un résultat contenant un chiffre absent du texte source est écarté",
                ],
                [
                    "Information absente",
                    "« Non mentionné dans le résumé » au lieu d'une valeur devinée",
                ],
                [
                    "Références du rapport PDF",
                    "générées par le code à partir des seuls papiers cités et validés",
                ],
                ["JSON mal formé", "validation Pydantic + nouvel essai avec le message d'erreur"],
            ],
            [0.38, 0.62],
        ),
    ]

    # ---- 8
    s += [
        P("8. L'interface utilisateur", "h1"),
        P(
            "L'application (<i>streamlit run app/streamlit_app.py</i>, puis "
            "http://localhost:8501) s'ouvre directement sur le Chat et comporte deux pages :"
        ),
        bullets(
            [
                "<b>Chat (RAG)</b> : réponse avec citations cliquables et « Contexte récupéré » ; "
                "dessous, la <b>comparaison des papiers cités</b> (tableau, onglets Méthodes / "
                "Datasets & métriques / Résultats / Limitations, synthèse, export Markdown / "
                "CSV) et l'option <b>Générer le rapport PDF</b>. Le menu <i>Options</i> règle "
                "la portée (corpus ou un papier), la langue et l'affichage de la comparaison.",
                "<b>Historique</b> : dans la barre latérale, « Nouvelle conversation », la liste "
                "des conversations enregistrées (un clic rouvre la conversation), le "
                "téléchargement en Markdown et la suppression.",
                "<b>Recherche</b> : requête, mode (hybride, BM25, dense), filtres (années, "
                "catégories, venue, code, accepté). Chaque résultat propose « Interroger avec "
                "le RAG » (ouvre le Chat sur ce papier) et un TL;DR.",
            ]
        ),
        P(
            "Aucun réglage technique n'est affiché : l'application choisit seule le corpus "
            "disponible (complet, sinon échantillon)."
        ),
    ]

    # ---- 9
    s += [
        P("9. Tests et qualité", "h1"),
        P(
            "113 tests automatisés passent (pytest) et le code respecte les règles de ruff. Les "
            "tests n'appellent ni le réseau ni un vrai LLM : un faux LLM scripté renvoie des "
            "réponses prévues, ce qui rend les tests rapides (environ 20 s) et reproductibles."
        ),
        table(
            [
                ["Fichier de tests", "Tests", "Ce qui est vérifié"],
                [
                    "test_enrich_comments",
                    "36",
                    "venues, années, acceptation, GitHub, pages / figures",
                ],
                [
                    "test_llm_rag",
                    "16",
                    "citations, cache, retries, JSON invalide, « Je ne sais pas »",
                ],
                [
                    "test_clean / test_index",
                    "15 / 15",
                    "dates, catégories, RRF, filtres, déterminisme",
                ],
                [
                    "test_compare / test_report",
                    "8 / 8",
                    "papiers cités, « Non mentionné », PDF valide",
                ],
                [
                    "test_pdf / test_ui / test_conversations",
                    "6 / 5 / 4",
                    "sections, rendu des pages, historique",
                ],
            ],
            [0.34, 0.12, 0.54],
        ),
        PageBreak(),
    ]

    # ---- 10
    s += [
        P("10. Limites et pistes d'amélioration", "h1"),
        P("Limites", "h2"),
        bullets(
            [
                "Un LLM local de 3 milliards de paramètres sur CPU est lent (1 à 4 minutes par "
                "réponse) et moins précis qu'un modèle hébergé. Un fournisseur hébergé (Groq, "
                "Gemini) l'accélère fortement, sans changer le code (seulement le fichier .env).",
                "Les venues hors liste (FAccT, WMT…) ne sont pas reconnues.",
                "Le découpage des PDF en sections repose sur des titres usuels.",
                "Le corpus est limité à la catégorie cs.CL depuis 2023 (paramétrable dans "
                "config.yaml).",
            ]
        ),
        P("Pistes d'amélioration", "h2"),
        bullets(
            [
                "Re-classement des résultats par un cross-encoder après la fusion RRF.",
                "Évaluation chiffrée (Recall@k, MRR, fidélité) sur un jeu annoté à la main.",
                "Analyse du domaine : thèmes et tendances, graphe de connaissances, agent.",
                "Mise à jour incrémentale du corpus avec les nouveaux papiers de l'API arXiv.",
            ]
        ),
    ]

    # ---- 11
    s += [
        P("11. Commandes pour reproduire le projet", "h1"),
        code(
            """
py -3.12 -m venv .venv ; .venv\\Scripts\\activate
pip install -r requirements.txt ; pip install -e . --no-deps
copy .env.example .env          # LLM_BASE_URL, LLM_MODEL (ex. Ollama local)

python -m rpe.cli prepare-data          # 1. corpus     (--dev : 1 000 papiers)
python -m rpe.cli build-index           # 2. index
python -m rpe.cli fetch-pdfs            # 3. optionnel : PDF arXiv
python -m rpe.cli index-pdfs
streamlit run app/streamlit_app.py      # 4. application : http://localhost:8501

python -m rpe.cli search "retrieval-augmented generation" --year-min 2024
python -m rpe.cli ask "How can retrieval reduce hallucinations?" --lang fr --pdf rapport.pdf

pytest ; ruff check .                   # tests et qualité
"""
        ),
    ]

    # ---- 12
    s += [
        P("12. Glossaire", "h1"),
        table(
            [
                ["Terme", "Définition"],
                [
                    "Embedding",
                    "vecteur de nombres qui représente le sens d'un texte ; deux textes proches ont des vecteurs proches",
                ],
                [
                    "BM25",
                    "score classique de recherche par mots-clés, qui favorise les mots rares et présents",
                ],
                [
                    "RRF",
                    "Reciprocal Rank Fusion : combine plusieurs classements, score = Σ 1/(k + rang)",
                ],
                [
                    "RAG",
                    "Retrieval-Augmented Generation : le LLM répond à partir de documents récupérés et non de sa mémoire",
                ],
                ["Pydantic", "bibliothèque de validation de données typées en Python"],
                [
                    "Ollama",
                    "logiciel qui exécute des LLM en local et expose une API compatible OpenAI",
                ],
                [
                    "Streaming",
                    "lecture d'un fichier ligne par ligne, sans le charger entièrement en mémoire",
                ],
            ],
            [0.22, 0.78],
        ),
    ]
    return s


def footer(canvas, doc) -> None:  # noqa: ANN001
    canvas.saveState()
    canvas.setFont(REG, 7.5)
    canvas.setFillColor(GREY)
    w = canvas._pagesize[0]
    canvas.drawString(MARGIN, 1.1 * cm, "Research Paper Explorer — rapport de projet")
    canvas.drawRightString(w - MARGIN, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> bytes:
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Research Paper Explorer — Rapport de projet",
        author="Research Paper Explorer",
    )
    frame = Frame(MARGIN, MARGIN, WIDTH, A4[1] - 2 * MARGIN, id="f")
    doc.addPageTemplates([PageTemplate("p", [frame], onPage=footer)])
    doc.build(story())
    return buf.getvalue()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = build()
    OUT.write_bytes(pdf)
    print(f"Written: {OUT} ({len(pdf) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
