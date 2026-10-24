"""PDF reports (ReportLab): Chat RAG answer + paper descriptions + comparison table.

Layout: portrait A4 (2 cm margins) for the text, landscape A4 for the comparison table
(BaseDocTemplate + NextPageTemplate), page number and disclaimer in every footer.
Text is cleaned of simple LaTeX and escaped for ReportLab's mini-markup; `[arXiv:ID]`
citations become clickable links. DejaVuSans (assets/fonts) handles accents.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from rpe.data.clean import strip_latex_accents
from rpe.rag.citations import ARXIV_ID_PATTERN
from rpe.rag.compare import NOT_MENTIONED, RELEVANCE_LABELS, PaperComparison

FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
MARGIN = 2 * cm
FOOTER_NOTE = (
    "Généré automatiquement à partir de résumés arXiv ; les cellules 'Non mentionné' "
    "indiquent une information absente du texte."
)
MAX_CELL_CHARS = 450  # keeps every table row shorter than a page (no LayoutError)
ACCENT = colors.HexColor("#1f4e79")
GREY = colors.HexColor("#8a8f98")

_CITE_RE = re.compile(rf"\[arXiv:\s*({ARXIV_ID_PATTERN})((?:\s*§[^\]]+)?)\]")
_LATEX_CMD_ARG_RE = re.compile(
    r"\\(?:emph|textbf|textit|texttt|mathrm|mathbf|text|textsc)\{([^{}]*)\}"
)
_LATEX_SYMBOLS = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\sigma": "σ",
    r"\theta": "θ",
    r"\pi": "π",
    r"\times": "×",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\approx": "≈",
    r"\sim": "~",
    r"\infty": "∞",
    r"\rightarrow": "→",
    r"\to": "→",
    r"\pm": "±",
    r"\%": "%",
    r"\&": "&",
}


# ------------------------------------------------------------------- text


def clean_latex(text: str) -> str:
    """Simple LaTeX cleanup: \\emph{x} -> x, $\\alpha$ -> α, \\'e -> é, braces removed."""
    if not text:
        return ""
    s = _LATEX_CMD_ARG_RE.sub(r"\1", text)
    for cmd, sym in sorted(_LATEX_SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(cmd, sym)
    s = strip_latex_accents(s)  # also drops remaining { }
    s = s.replace("$", "")
    s = re.sub(r"\\[A-Za-z]+\s*", "", s)  # unknown commands
    return re.sub(r"\s+", " ", s).strip()


def rl_text(text: str, links: bool = True) -> str:
    """LaTeX-cleaned, XML-escaped text; `[arXiv:ID]` turned into clickable links."""
    s = escape(clean_latex(text))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    if links:
        s = _CITE_RE.sub(
            lambda m: (
                f'<link href="https://arxiv.org/abs/{m.group(1)}" color="#1a5fb4">'
                f"[arXiv:{m.group(1)}{m.group(2)}]</link>"
            ),
            s,
        )
    return s


def short(text: str, n: int = MAX_CELL_CHARS) -> str:
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def authors_short(authors: Sequence[str], n: int = 3) -> str:
    if not authors:
        return NOT_MENTIONED
    return ", ".join(authors[:n]) + (" et al." if len(authors) > n else "")


# ------------------------------------------------------------------ styles


@lru_cache(maxsize=1)
def fonts() -> tuple[str, str, str]:
    """(regular, bold, italic) font names; DejaVu if available, else Helvetica."""
    files = {
        "DejaVuSans": "DejaVuSans.ttf",
        "DejaVuSans-Bold": "DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
    }
    if all((FONT_DIR / f).exists() for f in files.values()):
        for name, f in files.items():
            pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f)))
        pdfmetrics.registerFontFamily(
            "DejaVuSans",
            normal="DejaVuSans",
            bold="DejaVuSans-Bold",
            italic="DejaVuSans-Oblique",
            boldItalic="DejaVuSans-Bold",
        )
        return "DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


@lru_cache(maxsize=1)
def styles() -> dict[str, ParagraphStyle]:
    regular, bold, italic = fonts()
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "body", parent=base["BodyText"], fontName=regular, fontSize=10, leading=14, spaceAfter=6
    )
    return {
        "title": ParagraphStyle(
            "title",
            parent=body,
            fontName=bold,
            fontSize=22,
            leading=28,
            textColor=ACCENT,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=body, alignment=TA_CENTER, fontSize=12, leading=17
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=body,
            fontName=bold,
            fontSize=16,
            leading=20,
            textColor=ACCENT,
            spaceBefore=6,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2", parent=body, fontName=bold, fontSize=12, leading=15, spaceBefore=10, spaceAfter=4
        ),
        "h3": ParagraphStyle(
            "h3", parent=body, fontName=bold, fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=2
        ),
        "body": body,
        "meta": ParagraphStyle(
            "meta",
            parent=body,
            fontName=italic,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#555555"),
        ),
        "field": ParagraphStyle("field", parent=body, fontSize=9.5, leading=13, spaceAfter=3),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0),
        "cell_grey": ParagraphStyle(
            "cell_grey", parent=body, fontSize=8, leading=10, spaceAfter=0, textColor=GREY
        ),
        "head": ParagraphStyle(
            "head",
            parent=body,
            fontName=bold,
            fontSize=8.5,
            leading=10,
            textColor=colors.white,
            spaceAfter=0,
        ),
        "ref": ParagraphStyle("ref", parent=body, fontSize=9, leading=12),
    }


# --------------------------------------------------------------- document


def _footer(canvas, doc) -> None:  # noqa: ANN001 - ReportLab callback signature
    regular = fonts()[0]
    w = canvas._pagesize[0]  # current page size (portrait or landscape template)
    canvas.saveState()
    canvas.setFont(regular, 7)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, 1.1 * cm, FOOTER_NOTE)
    canvas.drawRightString(w - MARGIN, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _doc(buf: io.BytesIO, title: str) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=title,
        author="Research Paper Explorer",
    )
    pw, ph = A4
    lw, lh = landscape(A4)
    doc.addPageTemplates(
        [
            PageTemplate(
                "portrait",
                [Frame(MARGIN, MARGIN, pw - 2 * MARGIN, ph - 2 * MARGIN, id="p")],
                onPage=_footer,
                pagesize=A4,
            ),
            PageTemplate(
                "landscape",
                [Frame(MARGIN, MARGIN, lw - 2 * MARGIN, lh - 2 * MARGIN, id="l")],
                onPage=_footer,
                pagesize=landscape(A4),
            ),
        ]
    )
    return doc


def _title_page(title: str, question: str, n_papers: int, when: datetime) -> list:
    st = styles()
    return [
        Spacer(1, 5 * cm),
        Paragraph(escape(title), st["title"]),
        Paragraph(f"<b>Question :</b> {rl_text(question, links=False)}", st["subtitle"]),
        Spacer(1, 0.6 * cm),
        Paragraph(f"Généré le {when:%d/%m/%Y à %H:%M}", st["subtitle"]),
        Paragraph(f"{n_papers} papier(s) analysé(s)", st["subtitle"]),
        PageBreak(),
    ]


def _paper_card(c: PaperComparison) -> list:
    st = styles()
    meta = " · ".join(x for x in (authors_short(c.authors), str(c.year or ""), c.venue or "") if x)
    url = f"https://arxiv.org/abs/{c.arxiv_id}"
    fields = [
        ("Problème", c.problem),
        ("Méthode", c.method),
        ("Datasets & métriques", f"{', '.join(c.datasets)} — métriques : {', '.join(c.metrics)}"),
        ("Résultats", c.results),
        ("Limitations", "; ".join(c.limitations)),
    ]
    head = [
        Paragraph(rl_text(c.title, links=False), st["h2"]),
        Paragraph(
            f'{escape(meta)} · <link href="{url}" color="#1a5fb4">arXiv:{c.arxiv_id}</link>'
            + (f" · pertinence : {RELEVANCE_LABELS[c.relevance]}" if c.relevance else ""),
            st["meta"],
        ),
    ]
    body = [Paragraph(f"<b>{k} :</b> {rl_text(short(v, 1500))}", st["field"]) for k, v in fields]
    return [KeepTogether(head + body[:1]), *body[1:], Spacer(1, 0.3 * cm)]


def comparison_table(comparisons: Sequence[PaperComparison], width: float) -> Table:
    st = styles()
    headers = ["Papier", "Méthode", "Datasets", "Résultats", "Limitations", "Pertinence"]
    fractions = [0.20, 0.22, 0.14, 0.19, 0.16, 0.09]
    rows = [[Paragraph(h, st["head"]) for h in headers]]
    for c in comparisons:
        style = st["cell_grey"] if c.relevance == "low" else st["cell"]
        url = f"https://arxiv.org/abs/{c.arxiv_id}"
        paper = (
            f'<link href="{url}" color="#1a5fb4">arXiv:{c.arxiv_id}</link><br/>'
            f"{rl_text(short(c.title, 90), links=False)}" + (f" ({c.year})" if c.year else "")
        )
        cells = [
            c.method,
            ", ".join(c.datasets),
            c.results,
            "; ".join(c.limitations),
            RELEVANCE_LABELS[c.relevance],
        ]
        rows.append(
            [Paragraph(paper, style)] + [Paragraph(rl_text(short(x)), style) for x in cells]
        )
    table = Table(rows, colWidths=[f * width for f in fractions], repeatRows=1)
    table.setStyle(
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
    return table


def references(comparisons: Sequence[PaperComparison]) -> ListFlowable:
    st = styles()
    items = []
    for c in comparisons:
        url = f"https://arxiv.org/abs/{c.arxiv_id}"
        text = (
            f"[arXiv:{c.arxiv_id}] {rl_text(c.title, links=False)}. "
            f"{escape(authors_short(c.authors))}"
            + (f", {c.year}" if c.year else "")
            + f'. <link href="{url}" color="#1a5fb4">{url}</link>'
        )
        items.append(ListItem(Paragraph(text, st["ref"]), leftIndent=14))
    return ListFlowable(items, bulletType="1", bulletFontName=fonts()[0], bulletFontSize=9)


def _paragraphs(text: str, style: ParagraphStyle) -> list[Paragraph]:
    return [Paragraph(rl_text(p), style) for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def build_pdf(
    question: str,
    answer: str,
    comparisons: Sequence[PaperComparison],
    synthesis: str = "",
    generated_at: datetime | None = None,
) -> bytes:
    """Full Chat report; low-relevance papers are listed last (and grey in the table)."""
    st = styles()
    comps = sorted(comparisons, key=lambda c: c.relevance == "low")
    buf = io.BytesIO()
    doc = _doc(buf, "Research Paper Explorer — Rapport")
    story: list = _title_page(
        "Research Paper Explorer — Rapport", question, len(comps), generated_at or datetime.now()
    )
    story += [Paragraph("Réponse", st["h1"]), *_paragraphs(answer, st["body"])]
    story += [PageBreak(), Paragraph("Description des papiers", st["h1"])]
    for c in comps:
        story += _paper_card(c)
    if not comps:
        story.append(Paragraph("Aucun papier cité.", st["body"]))
    lw = landscape(A4)[0] - 2 * MARGIN
    story += [NextPageTemplate("landscape"), PageBreak(), Paragraph("Tableau comparatif", st["h1"])]
    story.append(comparison_table(comps, lw) if comps else Paragraph("—", st["body"]))
    story += [
        NextPageTemplate("portrait"),
        PageBreak(),
        Paragraph("Synthèse comparative", st["h1"]),
    ]
    story += _paragraphs(synthesis, st["body"]) or [Paragraph("Synthèse indisponible.", st["meta"])]
    story += [Spacer(1, 0.4 * cm), Paragraph("Références", st["h1"])]
    if comps:
        story.append(references(comps))
    doc.build(story)
    return buf.getvalue()


# ------------------------------------------------------- Markdown -> PDF


def markdown_flowables(markdown: str) -> list:
    """Minimal Markdown (#, ##, ###, -, paragraphs, **bold**) to ReportLab flowables."""
    st = styles()
    out: list = []
    para: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        if para:
            out.append(Paragraph(rl_text(" ".join(para)), st["body"]))
            para.clear()
        if bullets:
            out.append(
                ListFlowable(
                    [ListItem(Paragraph(rl_text(b), st["body"]), leftIndent=12) for b in bullets],
                    bulletType="bullet",
                    start="•",
                    bulletFontName=fonts()[0],
                )
            )
            bullets.clear()

    for line in markdown.splitlines():
        s = line.strip()
        heading = re.match(r"^(#{1,3})\s+(.*)$", s)
        if heading:
            flush()
            level = {1: "h1", 2: "h2", 3: "h3"}[len(heading.group(1))]
            out.append(Paragraph(rl_text(heading.group(2), links=False), st[level]))
        elif re.match(r"^[-*]\s+", s):
            if para:
                flush()
            bullets.append(re.sub(r"^[-*]\s+", "", s))
        elif not s:
            flush()
        else:
            if bullets:
                flush()
            para.append(s)
    flush()
    return out


def build_markdown_pdf(title: str, markdown: str, generated_at: datetime | None = None) -> bytes:
    """Same layout for a Markdown document (e.g. the state-of-the-art review + References)."""
    st = styles()
    buf = io.BytesIO()
    doc = _doc(buf, title)
    when = generated_at or datetime.now()
    story: list = [
        Spacer(1, 5 * cm),
        Paragraph(escape(title), st["title"]),
        Paragraph(f"Généré le {when:%d/%m/%Y à %H:%M}", st["subtitle"]),
        PageBreak(),
        *markdown_flowables(markdown),
    ]
    doc.build(story)
    return buf.getvalue()


def report_filename(question: str, when: datetime | None = None, n_words: int = 4) -> str:
    """rapport_<question-keywords>_<date>.pdf"""
    from rpe.index.bm25 import tokenize

    words = "-".join(tokenize(question)[:n_words]) or "question"
    return f"rapport_{words}_{(when or datetime.now()):%Y-%m-%d}.pdf"
