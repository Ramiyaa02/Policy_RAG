"""Generate Policy_RAG_Presentation.pptx — project presentation.

Structure follows the GeoThink_INT450.docx report format:
  Title/Author -> Abstract -> Introduction -> Related Work ->
  Dataset/Benchmarks/Prompt Design -> System Architecture and Methods ->
  Experiments and Results -> Discussion -> Conclusion and Future Work ->
  References

Contents are entirely the Policy_RAG project (no GeoThink content):
architecture flow diagrams drawn from shapes, annotated code snippets,
and the real Phase 6 evaluation numbers from reports/phase6_report.md.

Run:  .venv/Scripts/python make_ppt.py
Out:  Policy_RAG_Presentation.pptx
"""

from __future__ import annotations

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ---------------------------------------------------------------- theme

BG = RGBColor(0x0E, 0x11, 0x16)
PANEL = RGBColor(0x16, 0x1B, 0x23)
PANEL2 = RGBColor(0x1D, 0x23, 0x2E)
LINE = RGBColor(0x2A, 0x31, 0x40)
TEXT = RGBColor(0xE6, 0xE9, 0xEF)
MUTED = RGBColor(0x9A, 0xA4, 0xB2)
ACCENT = RGBColor(0x4F, 0x8C, 0xFF)
OK = RGBColor(0x35, 0xC2, 0x6E)
WARN = RGBColor(0xE0, 0xA6, 0x3E)
BAD = RGBColor(0xE0, 0x52, 0x52)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def add_slide(prs: Presentation) -> Presentation:  # noqa: F841 helper naming
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


def fill_bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG


def box(slide, x, y, w, h, text="", size=14, color=None, bold=False, font="Segoe UI",
        align=PP_ALIGN.LEFT, panel=False, panel2=False, line=None, anchor=MSO_ANCHOR.TOP) -> object:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = PANEL2 if panel2 else (PANEL if panel else BG)
    if line is not None:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.margin_top = tf.margin_bottom = Inches(0.08)
    tf.vertical_anchor = anchor
    body = tf.paragraphs[0]
    body.alignment = align
    run = body.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color or TEXT
    run.font.bold = bold
    run.font.name = font
    return shape


def bullets(slide, x, y, w, h, items, size=15, color=None, gap=6, bullet="\u2022  "):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        indent, text = (item if isinstance(item, tuple) else ("", item))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        run = p.add_run()
        run.text = f"{indent}{bullet}{text}"
        run.font.size = Pt(size)
        run.font.color.rgb = color or TEXT
        run.font.name = "Segoe UI"
        if indent:
            for r in p.runs:
                r.font.size = Pt(size - 1)
                r.font.color.rgb = MUTED
    return tb


def header(slide, kicker, title):
    box(slide, Inches(0.45), Inches(0.28), Inches(8), Inches(0.4), kicker, 11, ACCENT, bold=True)
    box(slide, Inches(0.42), Inches(0.62), Inches(12.5), Inches(0.7), title, 28, TEXT, bold=True)
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(1.42), Inches(12.45), Pt(1.2))
    ln.fill.solid()
    ln.fill.fore_color.rgb = LINE
    ln.line.fill.background()
    ln.shadow.inherit = False


def code(slide, x, y, w, h, snippet, size=10.5):
    # syntax-ish highlight: comments in muted, rest in light
    panel = box(slide, x, y, w, h, "", panel2=True, line=LINE)
    tf = panel.text_frame
    tf.word_wrap = True
    first = True
    for raw in snippet.strip("\n").split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        run = p.add_run()
        run.text = raw.rstrip()
        run.font.name = "Consolas"
        run.font.size = Pt(size)
        stripped = raw.strip()
        if stripped.startswith("#") or stripped.startswith("\"\"\""):
            run.font.color.rgb = MUTED
        else:
            run.font.color.rgb = RGBColor(0xCF, 0xE3, 0xFF)
    return panel


def results_table(slide, x, y, w, rows, col_w=None, size=12, header_size=12):
    n_rows, n_cols = len(rows), len(rows[0])
    tbl = slide.shapes.add_table(n_rows, n_cols, x, y, w, Inches(0.32 * n_rows)).table
    if col_w:
        total = sum(col_w)
        for i, frac in enumerate(col_w):
            tbl.columns[i].width = Emu(int(w * frac / total))
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if r else PANEL2
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(header_size if r == 0 else size)
            run.font.bold = r == 0
            run.font.name = "Segoe UI"
            run.font.color.rgb = TEXT if r == 0 else (MUTED if c else TEXT)
            if r > 0 and (str(row[c]).startswith(("+", "\u2191")) and c > 0):
                run.font.color.rgb = OK
            if r > 0 and str(row[c]).startswith("\u2193") and c > 0:
                run.font.color.rgb = BAD
    return tbl


def arrow(slide, x1, y1, x2, y2, color=None, width=2.2):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    conn.line.color.rgb = color or ACCENT
    conn.line.width = Pt(width)
    # arrow head
    ln = conn.line._get_or_add_ln()
    try:
        from lxml import etree
        tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
        tail.set("type", "triangle")
        tail.set("w", "med")
        tail.set("len", "med")
    except Exception:
        pass
    return conn


def diagram_node(slide, x, y, w, h, title, sub="", fill=None, accent=ACCENT, tsize=11.5, ssize=9):
    shape = box(slide, x, y, w, h, "", panel=True, line=accent)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = title
    run.font.size = Pt(tsize)
    run.font.bold = True
    run.font.color.rgb = TEXT
    run.font.name = "Segoe UI"
    if sub:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        run2 = p2.add_run()
        run2.text = sub
        run2.font.size = Pt(ssize)
        run2.font.color.rgb = MUTED
        run2.font.name = "Segoe UI"
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill or PANEL
    return shape


def footer(slide, n, total=0):
    box(slide, Inches(12.55), Inches(7.05), Inches(0.7), Inches(0.3), str(n), 10, MUTED, align=PP_ALIGN.RIGHT)


# ---------------------------------------------------------------- deck

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
n = 0

# 1 — Title and Author Information ------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
box(s, Inches(0.8), Inches(2.0), Inches(11.7), Inches(1.2), "Policy_RAG", 54, TEXT, bold=True)
box(s, Inches(0.85), Inches(3.15), Inches(11.7), Inches(0.6),
    "Insurance Policy Intelligence — Verified, Citation-Backed Question Answering over Indian Policy Wordings",
    19, MUTED)
box(s, Inches(0.85), Inches(4.05), Inches(2.2), Inches(0.5), "Phase 1 – 7 complete", 13, OK, bold=True)
box(s, Inches(3.2), Inches(4.05), Inches(6.5), Inches(0.5),
    "RAG · Hybrid retrieval · Agentic planning · Grounding verification · FastAPI", 13, ACCENT)
box(s, Inches(0.85), Inches(5.6), Inches(11.5), Inches(0.5), "Authors", 12, MUTED)
box(s, Inches(0.85), Inches(6.0), Inches(11.5), Inches(0.5),
    "Ramiyaa02  —  Policy_RAG  ·  built against the SRS (bachelor's capstone project)", 15, TEXT)

# 2 — Abstract ---------------------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "2  ·  ABSTRACT", "Abstract")
bullets(s, Inches(0.6), Inches(1.75), Inches(12.2), Inches(5.4), [
    ("", "Insurance policy wordings are long, cross-referencing documents; naive LLM chatbots quote them confidently and wrongly."),
    ("", "Policy_RAG answers questions over 10 public insurer wordings (553 pages) with a retrieval-augmented pipeline that refuses to answer what it cannot verify."),
    ("", "Documents are chunked at clause boundaries with full trace metadata; retrieval fuses in-house BM25 with dense embeddings and is reranked by a cross-encoder (P@5 0.34 → 0.43, nDCG@5 0.41 → 0.55)."),
    ("", "An agentic layer classifies each query and plans one retrieval step per product × criterion; the answer must cite the evidence blocks it was shown as [n]."),
    ("", "A grounding layer extracts every fact claim, verifies it against the cited excerpts (token containment + number consistency), strips unsupported sentences, discloses contradictions, and abstains with a targeted clarification question when nothing can be verified."),
    ("", "Phase 6 evaluates baseline vs full pipelines on a frozen 45-query set with a failure taxonomy (F1–F16); Phase 7 ships the whole system as a FastAPI service with a bundled web UI."),
    ("", "Verification catches real model errors live: a wrong \u201c2 years\u201d cataract waiting-period claim was stripped and replaced by an abstention + clarification."),
], size=15, gap=10)

# 3 — Introduction -----------------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "3  ·  INTRODUCTION", "Introduction & Problem")
box(s, Inches(0.6), Inches(1.8), Inches(6.0), Inches(4.9), "", panel=True, line=LINE)
box(s, Inches(0.75), Inches(1.95), Inches(5.6), Inches(0.4), "The problem", 14, ACCENT, bold=True)
bullets(s, Inches(0.75), Inches(2.4), Inches(5.7), Inches(4.2), [
    "Customers ask policy questions; answers exist but are buried across tables, waiting-period lists and exclusion clauses",
    "Plain LLM answers hallucinate numbers and mix clauses from different policies",
    "\"What is the cataract waiting period?\" — one document answers 30 days (initial), another lists it under 2-year specified-procedure waits",
    "Wordings differ per insurer and per product; a single index blends incompatible clause sets",
], size=12.5, gap=7)

box(s, Inches(6.85), Inches(1.8), Inches(6.0), Inches(4.9), "", panel=True, line=LINE)
box(s, Inches(7.0), Inches(1.95), Inches(5.6), Inches(0.4), "This project's answer", 14, OK, bold=True)
bullets(s, Inches(7.0), Inches(2.4), Inches(5.7), Inches(4.2), [
    "Traceability first: every chunk keeps document, product, section, clause ids, page span — every claim can be traced to the source PDF",
    "Retrieval earns the context: clause-aware chunking + hybrid retrieval + cross-encoder reranking",
    "Generation is constrained: cite [n] markers from the shown evidence or abstain",
    "Verification is enforced downstream: three-way §8 verdicts (supported / contradicted / insufficient), never “absence = false”",
    "Failure modes measured with a taxonomy (§15), not vibes",
], size=12.5, gap=7)

# 4 — Related Work -----------------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "4  ·  RELATED WORK", "Related Work")
bullets(s, Inches(0.6), Inches(1.8), Inches(12.2), Inches(5.2), [
    ("", "RAG (Lewis et al., 2020) grounds generation in retrieved passages — the baseline for this project; we extend it with verification and abstention."),
    ("", "BM25 (Robertson & Zaragoza, 2009) is the lexical workhorse; clause ids, procedure names and defined terms that blur under embeddings still surface exactly (implemented in-house, zero extra deps)."),
    ("", "Dense retrieval (Karpukhin et al., 2020; sentence-BERT embeddings, Reimers & Gurevych, 2019) provides semantic recall — fused with BM25 by Reciprocal Rank Fusion."),
    ("", "Cross-encoder reranking (Nogueira & Cho, 2019; ms-marco MiniLM) rescores the fused pool — the single biggest retrieval win in our ablations."),
    ("", "ReAct-style agent decomposition (Yao et al., 2022) shapes the agentic layer: classify → plan → retrieve per step, with plans inspectable as data."),
    ("", "Attribution / grounding verification draws on token-containment and number-consistency scoring; a three-way verdict (supported / contradicted / insufficient) is enforced rather than only reported."),
    ("", "Compared with existing systems, Policy_RAG uniquely combines clause-preserving chunk traceability, product-aware multi-step retrieval, and enforced claim-level verification with abstention in one local-first stack (Ollama, ChromaDB, FastAPI)."),
], size=15, gap=10)

# 5 — Dataset, Benchmarks, Prompt Design -------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "5  ·  DATASET, BENCHMARKS & PROMPT DESIGN", "Dataset, Benchmarks & Prompt Design")
box(s, Inches(0.6), Inches(1.8), Inches(3.9), Inches(4.9), "", panel=True, line=LINE)
box(s, Inches(0.75), Inches(1.95), Inches(3.5), Inches(0.4), "Corpus", 14, ACCENT, bold=True)
bullets(s, Inches(0.75), Inches(2.4), Inches(3.6), Inches(4.1), [
    "10 public Indian insurer policy wordings",
    "553 pages · 551 native-text, 1 OCR page",
    "1,504 structure-aware chunks",
    "clause / heading boundaries, tables kept separate as table chunks",
    "every chunk: product, section, clause ids, page span, portable source path",
], size=12, gap=6)
box(s, Inches(4.75), Inches(1.8), Inches(3.9), Inches(4.9), "", panel=True, line=LINE)
box(s, Inches(4.9), Inches(1.95), Inches(3.5), Inches(0.4), "Frozen benchmark", 14, ACCENT, bold=True)
bullets(s, Inches(4.9), Inches(2.4), Inches(3.6), Inches(4.1), [
    "45 queries: 43 answerable + 2 out-of-domain probes",
    "6 categories mirroring the SRS test taxonomy (waiting periods, exclusions, eligibility, claims, comparisons, OOD)",
    "chunk-level qrels (ground truth), single-annotator with a relevance audit note",
    "fixed label set across every compared system → like-for-like baseline/ablation runs",
], size=12, gap=6)
box(s, Inches(8.9), Inches(1.8), Inches(3.95), Inches(4.9), "", panel=True, line=LINE)
box(s, Inches(9.05), Inches(1.95), Inches(3.6), Inches(0.4), "Prompt design", 14, ACCENT, bold=True)
bullets(s, Inches(9.05), Inches(2.4), Inches(3.7), Inches(4.1), [
    "numbered evidence blocks, one per retrieved excerpt",
    "model must cite blocks as [n] — unmarked facts are unverifiable by construction",
    "explicit abstain instruction when the corpus lacks the answer",
    "comparison prompts: per-policy sections, facts separated from conclusions",
    "low temperature (0.2) for factual consistency",
], size=12, gap=6)

# 6 — Architecture Overview (flow diagram) -----------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "6  ·  SYSTEM ARCHITECTURE AND METHODS", "Architecture Overview")
box(s, Inches(0.5), Inches(1.6), Inches(2.6), Inches(0.35), "OFFLINE BUILD (once per corpus)", 11, OK, bold=True)
row1 = [
    ("PDFs", "rag_policy_dataset/", ACCENT),
    ("Phase 1", "normalize → DOC-0XX.json", OK),
    ("Phase 2", "clause-preserving chunks", OK),
    ("Embed", "sentence-transformers", ACCENT),
    ("Indexes", "ChromaDB + BM25", ACCENT),
]
x = 0.5
for i, (t, sub, col) in enumerate(row1):
    w = 2.32
    y = 2.05
    diagram_node(s, Inches(x), Inches(y), Inches(w), Inches(0.95), t, sub, accent=col)
    if i < len(row1) - 1:
        arrow(s, Inches(x + w + 0.04), Inches(y + 0.475), Inches(x + w + 0.38), Inches(y + 0.475))
    x += w + 0.42

box(s, Inches(0.5), Inches(3.35), Inches(4.6), Inches(0.35), "QUERY TIME — AGENTIC VERIFIED PIPELINE", 11, ACCENT, bold=True)
row2 = [
    ("Agent 1", "QueryAnalyzer classify → type / products / criteria", ACCENT),
    ("Agent 2", "RetrievalPlanner → steps", ACCENT),
    ("Agent 3", "PlanExecutor hybrid + rerank (per step)", ACCENT),
    ("Agent 4/6", "cite-as-[n] generation", ACCENT),
    ("Agent 5", "verify claims S / C / I", WARN),
    ("Handler", "strip · note · abstain", BAD),
]
x = 0.5
for i, (t, sub, col) in enumerate(row2):
    w = 1.94
    y = 3.8
    diagram_node(s, Inches(x), Inches(y), Inches(w), Inches(1.15), t, sub, accent=col, tsize=11, ssize=8.5)
    if i < len(row2) - 1:
        arrow(s, Inches(x + w + 0.03), Inches(y + 0.575), Inches(x + w + 0.31), Inches(y + 0.575))
    x += w + 0.345

row3 = [
    ("verified answer + citations", OK),
    ("abstention + targeted clarification question", WARN),
]
x = 0.5
for t, col in row3:
    diagram_node(s, Inches(x), Inches(5.35), Inches(4.1), Inches(0.7), t, "", accent=col, tsize=12)
    x += 4.45
arrow(s, Inches(6.7), Inches(4.95), Inches(2.6), Inches(5.3), color=OK)
arrow(s, Inches(11.6), Inches(4.95), Inches(9.2), Inches(5.3), color=WARN)

box(s, Inches(0.5), Inches(6.35), Inches(12.3), Inches(0.75),
    "One code path: the same VerifiedAgent.run() executes on the CLI (run_phase5.py), in the evaluator (run_phase6.py) and over HTTP (/api/query) — answers shown in the UI are exactly what the evaluator scores.",
    12.5, MUTED, panel=True, line=LINE)

# 7 — LLM and Model Selection ------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "6.1  ·  LLM AND MODEL SELECTION", "LLM and Model Selection")
bullets(s, Inches(0.6), Inches(1.8), Inches(7.6), Inches(5.0), [
    ("", "Generation: llama3.2:3b served locally by Ollama — no API key, no data leaves the machine, $0/query (tokens recorded as the cost proxy)."),
    ("", "Why local + small: SRS privacy constraints (policy data stays local), env-friendly, and every phase must rerun unchanged offline with --no-llm."),
    ("", "Extractive fallback (AnswerGenerator vs ExtractiveGenerator): when Ollama is down the system still answers by quoting retrieved excerpts — evaluation never blocks."),
    ("", "Configuration is a chain: phase5 → phase4 → phase3 → phase2 JSON; model, temperature (0.2), num_predict, timeout all live there."),
    ("", "Verification second pass runs a separate zero-temperature fast call (num_predict=64) so it cannot inherit verbosity."),
    ("", "Embeddings: all-MiniLM-L6-v2, L2-normalized so cosine = dot product; swappable per FR-005."),
    ("", "Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 with a model-free lexical fallback (FR / NFR-008 replaceable-reranker interface)."),
], size=14.5, gap=9)
code(s, Inches(8.5), Inches(1.8), Inches(4.35), Inches(2.1), """
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0.2,
    num_predict=512,
    timeout_seconds=180,
)
llm_ready = llm.is_available()
generator = (AnswerGenerator(llm)
             if llm_ready else ExtractiveGenerator())
""", size=10)
box(s, Inches(8.5), Inches(4.15), Inches(4.35), Inches(2.5),
    "The Ollama client records per-call + cumulative token counts (NFR-005), which the Phase 6 evaluator snapshots per query — so token/cost records exist even where no API cost applies.",
    12, MUTED, panel=True, line=LINE)

# 8 — Agentic Design ---------------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "6.3  ·  AGENTIC DESIGN", "Agentic Design — six specialized agents")
results_table(s, Inches(0.55), Inches(1.75), Inches(7.6), [
    ["Agent", "Responsibility (SRS FR)"],
    ["1 · QueryAnalyzer", "rules-first classification (FR-007) + requirement slots (FR-013); LLM only at low confidence"],
    ["2 · RetrievalPlanner", "plan of steps; comparison = one step per product × criterion (FR-014)"],
    ["3 · PlanExecutor", "runs Phase 3 hybrid retriever per step; global evidence budget (FR-008/009)"],
    ["4/6 · Generator", "cite-as-[n] grounded answers; per-policy comparison sections (FR-012)"],
    ["5 · Validator", "three-way §8 verification of every claim (FR-015–018)"],
    ["Handler", "strip unsupported, disclose contradictions, abstain + clarify (O8)"],
], col_w=[26, 74], size=11.5)
box(s, Inches(8.5), Inches(1.75), Inches(4.35), Inches(2.0), "", panel=True, line=LINE)
bullets(s, Inches(8.65), Inches(1.9), Inches(4.05), Inches(1.8), [
    "Short-circuits: out-of-domain (FR-019) → scope statement, no retrieval",
    "Underspecified (FR-018) → one targeted clarification question",
], size=11.5, gap=6)
code(s, Inches(8.5), Inches(3.95), Inches(4.35), Inches(2.75), """
# comparison plan (src/agent/planner.py)
for product in products:
    for criterion in criteria:
        steps.append(PlanStep(
            question=..., product=product,
            criteria=[criterion], top_k=k))

# execute: per-step hybrid retrieval,
# de-duped under a global chunk budget
bundles = executor.execute(plan)
chunks = executor.flatten(bundles)
""", size=9.5)

# 9 — Retrieval core (code + explanation) ------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "6.5  ·  RAG PIPELINE — RETRIEVAL CORE", "Hybrid retrieval & fusion (the code that earns the context)")
code(s, Inches(0.55), Inches(1.7), Inches(7.4), Inches(3.1), """
# src/retrieval/hybrid.py — Reciprocal Rank Fusion
doc_scores: dict[str, float] = defaultdict(float)
pool: dict[str, dict] = {}

for hits in (dense_hits, bm25_hits):
    for rank, chunk in enumerate(hits, start=1):
        doc_scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)   # rrf_k=60
        pool[chunk.chunk_id] = chunk

return [pool[cid] for cid in sorted(
    doc_scores, key=doc_scores.get, reverse=True)[:top_k]]
""", size=10.5)
code(s, Inches(0.55), Inches(5.0), Inches(7.4), Inches(2.0), """
# metadata filter is applied to BOTH paths (Phase 3, FR-004)
hits = store.search(vector, k, filter={"product": product})
lex  = bm25.search(terms,   k, filter={"product": product})
""", size=10.5)
bullets(s, Inches(8.25), Inches(1.7), Inches(4.6), Inches(5.2), [
    ("", "Rank-based fusion: no score scaling needed across two incomparable scorers; rank r contributes 1/(60+r)."),
    ("", "BM25 is in-house (src/retrieval/bm25.py): exact terms — clause ids \u201cXIII\u201d, \u201ccataract\u201d, defined terms — rank where embeddings blur."),
    ("", "Dense pass: cosine over L2-normalized embeddings in a persistent ChromaDB collection (idempotent upserts, FR-006)."),
    ("", "The fused pool (candidate_k=20) is rescored by the cross-encoder, or by the lexical feature-blend fallback when the model can't load — never a silent no-op."),
    ("", "Result (frozen set): P@5 0.34 → 0.43, nDCG@5 0.41 → 0.55, MRR 0.59 → 0.78 over the Phase 2 semantic baseline."),
], size=12.5, gap=8)

# 10 — Verification agent (code + explanation) -------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "6.4  ·  VALIDATION AGENT", "Claim verification — the guardrail, in code")
code(s, Inches(0.55), Inches(1.7), Inches(7.4), Inches(4.4), """
# src/grounding/verifier.py — Agent 5 rules pass
claim_tokens = tokenize(claim.text, stopwords)
best = 0.0
for chunk in cited:                      # excerpts its [n] markers point to
    score = containment(claim_tokens,
                        tokenize(chunk_text(chunk), stopwords))
    best = max(best, score)

claim_numbers    = set(NUMBER_RE.findall(claim.text))
evidence_numbers = union of numbers across cited evidence

if claim_numbers and evidence_numbers and not (claim_numbers & evidence_numbers):
    return CONTRADICTED  if number_strict else flag the mismatch

if best >= support_threshold:            # 0.35
    return SUPPORTED
return INSUFFICIENT                      # absence of evidence != evidence of absence
""", size=10.5)
code(s, Inches(0.55), Inches(6.25), Inches(7.4), Inches(0.95), """
# handler: insufficient fact claims are CUT OUT of the answer;
# if no fact claim survives -> abstain + one targeted clarification question
""", size=10.5)
bullets(s, Inches(8.25), Inches(1.7), Inches(4.6), Inches(5.4), [
    ("", "Three-way verdict (§8): supported / contradicted / insufficient — a claim with no evidence is never silently dropped or called false."),
    ("", "Number consistency: a claim saying \u201c2 years\u201d cannot be supported by evidence that only contains \u201c30 days\u201d — this stripped a real hallucination live."),
    ("", "LLM second pass (optional) can only lift CONTRADICTED when it says supported — a parser error cannot paper over a contradiction."),
    ("", "Conclusions are labelled, not verified as fact (FR-012 distinction)."),
    ("", "Verified live: \u201cwaiting period ... 2 years [XXIII]\u201d → [MISS] → abstain + \u201cCould you name the specific procedure…?\u201d"),
], size=12.5, gap=8)

# 11 — App layer (code + explanation) ----------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "PHASE 7  ·  APPLICATION LAYER", "FastAPI service + bundled UI")
code(s, Inches(0.55), Inches(1.7), Inches(7.6), Inches(2.4), """
# src/api/service.py
@app.post("/api/query")
def api_query(body: QueryRequest) -> dict:
    agent = state.agent()          # built lazily, cached
    response = agent.run(body.query, k=body.k)
    return _serialize(response, state.llm_mode)
    # -> answer, normalized citations, grounding,
    #    abstention/clarification, latency breakdown
""", size=10.5)
code(s, Inches(0.55), Inches(4.25), Inches(7.6), Inches(1.6), """
# SSE variant consumed by the bundled chat UI
@app.get("/api/stream")
def api_stream(q: str, k: int | None = None):
    def events():
        yield sse(working="planning retrieval…")
        yield sse(result=agent.run(q, k=k))
""", size=10.5)
code(s, Inches(0.55), Inches(6.0), Inches(7.6), Inches(1.0), """
# optional access control (SRS Phase 7: "authentication where required")
key_dep = Depends(check_api_key)   # enables X-API-Key when
                                   # POLICY_RAG_API_KEY is set
""", size=10)
bullets(s, Inches(8.45), Inches(1.7), Inches(4.4), Inches(5.2), [
    ("", "One code path: /api/query, the CLI and the evaluator all call VerifiedAgent.run — the UI shows exactly what Phase 6 scores."),
    ("", "UI: dependency-free single static page (src/api/static/index.html) — chat box, sample questions, no-LLM toggle, grounding badges, citation list, latency grid, retrieval-step trace."),
    ("", "/api/reports serves the phase 1–6 evaluation artifacts (traversal blocked)."),
    ("", "Every request logged as one JSON line → reports/phase7_api_log.jsonl."),
    ("", "Verified live: 10/10 supported claims on the cataract query, OOD abstained via stream, UI + /docs serve 200."),
], size=12.5, gap=8)

# 12 — Hyperparameters -------------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "6.6  ·  HYPERPARAMETERS & IMPLEMENTATION", "Hyperparameters & implementation details")
results_table(s, Inches(0.55), Inches(1.8), Inches(7.5), [
    ["Setting", "Value", "Where"],
    ["chunking", "\u2248 700 chars / 60 overlap, clause-aware", "configs/phase2.json"],
    ["embedding model", "all-MiniLM-L6-v2 (384-d, L2-normalized)", "configs/phase2.json"],
    ["retrieval top-k", "5 (candidate pool 20)", "configs/phase3.json"],
    ["RRF constant", "rrf_k = 60", "configs/phase3.json"],
    ["reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2", "configs/phase3.json"],
    ["LLM", "llama3.2:3b · temp 0.2 · num_predict 512", "configs/phase2.json"],
    ["support_threshold", "0.35, number_strict = true", "configs/phase5.json"],
    ["evidence budget", "max 80 chunks / prompt 6000 chars", "configs/phase4.json"],
], col_w=[26, 44, 30], size=11.5)
bullets(s, Inches(8.4), Inches(1.85), Inches(4.4), Inches(4.6), [
    ("", "No fine-tuning; behaviour comes from prompts + deterministic verification."),
    ("", "Deterministic ids: chunk_id = DOC::cNNNN → idempotent rebuilds (FR-006)."),
    ("", "215 tests across 7 phases run offline (no server) in ~30 s; phase-1 OCR suite is machine-bound."),
    ("", "Everything regenerable: vector + BM25 indexes are git-ignored build artifacts."),
], size=12.5, gap=8)

# 13 — Experiments: setup + quantitative -------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "7  ·  EXPERIMENTS AND RESULTS", "Experiments and Results — setup & headline numbers")
box(s, Inches(0.55), Inches(1.7), Inches(12.3), Inches(0.6),
    "Setup: frozen 45-query set · Agent 5 validator scores BOTH pipelines offline like-for-like · full-LLM and offline (--no-llm) records · tokens + latency per component (§14.7/14.8)",
    12.5, MUTED, panel=True, line=LINE)
box(s, Inches(0.55), Inches(2.5), Inches(3.6), Inches(0.35), "AC-010 — baseline vs full", 13, ACCENT, bold=True)
results_table(s, Inches(0.55), Inches(2.9), Inches(6.2), [
    ["metric", "baseline", "full", "\u0394"],
    ["answer correctness (proxy) \u2191", "0.791", "0.809", "+0.019"],
    ["unsupported claim rate \u2193", "0.066", "0.061", "-0.005"],
    ["mean grounding support \u2191", "0.943", "0.949", "+0.006"],
    ["OOD probes abstained", "0/2", "1/2", "+1"],
    ["ms / query", "70", "5 894", "\u00d784"],
], col_w=[40, 20, 20, 20], size=11.5)
box(s, Inches(7.0), Inches(2.5), Inches(5.8), Inches(0.35), "Retrieval quality (same frozen set)", 13, ACCENT, bold=True)
results_table(s, Inches(7.0), Inches(2.9), Inches(5.85), [
    ["mode", "P@5", "R@5", "nDCG@5", "MRR"],
    ["semantic (baseline)", "0.340", "0.367", "0.410", "0.586"],
    ["keyword BM25", "0.312", "0.350", "0.386", "0.538"],
    ["hybrid RRF", "0.391", "0.417", "0.481", "0.685"],
    ["hybrid + rerank", "0.428", "0.481", "0.552", "0.776"],
], size=11.5)
box(s, Inches(0.55), Inches(6.05), Inches(12.3), Inches(1.1),
    "Reading it: the agentic+verified pipeline answers the same share of questions while cutting unsupported claims; reranking is the retrieval gain (+0.14 nDCG@5 over baseline); the cost is latency, dominated by cross-encoder reranking on CPU.",
    12.5, TEXT, panel=True, line=LINE)

# 14 — Ablations -------------------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "7.4  ·  COMPARISON WITH BASELINES (ABLATIONS)", "Ablation experiments")
results_table(s, Inches(0.55), Inches(1.75), Inches(12.2), [
    ["variant", "correct \u2191", "unsupported \u2193", "grounding \u2191", "abstentions", "mean ms"],
    ["default (hybrid + rerank + grounding)", "0.809", "0.061", "0.949", "1/2", "2 251"],
    ["no_rerank", "0.857", "0.088", "0.924", "1/2", "38"],
    ["keyword_only (BM25)", "0.833", "0.045", "0.959", "1/2", "2 312"],
    ["semantic_only", "0.833", "0.072", "0.940", "1/2", "2 621"],
    ["no_strip (handler off)", "0.809", "0.061", "0.949", "1/2", "2 599"],
    ["rules_only_verify", "0.809", "0.061", "0.949", "1/2", "2 486"],
], size=12)
bullets(s, Inches(0.6), Inches(5.0), Inches(12.2), Inches(2.2), [
    ("", "no_rerank: 80× faster but its answers lean on weaker evidence — grounding support drops 0.949 → 0.924 and unsupported claims rise 0.061 → 0.088; the proxy-correctness metric can't see the evidence-quality drop, grounding can."),
    ("", "keyword_only quietly wins on unsupported-claims (0.045) — exact term matching matters for clause ids; a hybrid of BM25 + dense remains the balanced default."),
    ("", "no_strip / rules_only_verify ≈ default on this set: the offline eval set is too clean to always expose the handler; the live run (cataract \u201c2 years\u201d hallucination stripped) shows where it fires."),
    ("", "Ablations run single-change on the same frozen set → every delta is attributable."),
], size=13, gap=8)

# 15 — Failure taxonomy + discussion -----------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "7.3  ·  QUALITATIVE ANALYSIS / §15", "Failure taxonomy (AC-012)")
results_table(s, Inches(0.55), Inches(1.75), Inches(7.7), [
    ["code", "failure", "queries"],
    ["F1", "retrieval failure — no relevant chunk retrieved", "9"],
    ["F2", "missing relevant evidence — k=5 budget can't cover multi-chunk qrels", "30"],
    ["F4", "query classification failure", "1"],
    ["F7", "unsupported generation (caught by Agent 5)", "11"],
    ["F10", "incorrect abstention", "1"],
    ["F12", "product comparison failure", "1"],
    ["F14 / F16 / F15", "OOD detection miss / correct abstention / grounding corrected", "1 / 1 / 11"],
], col_w=[12, 70, 18], size=11.5)
box(s, Inches(8.6), Inches(1.75), Inches(4.2), Inches(0.4), "F15/F16 = positive detections", 12, OK, bold=True)
bullets(s, Inches(8.6), Inches(2.3), Inches(4.2), Inches(4.2), [
    ("", "F2 dominates (30/45): a k=5 evidence budget structurally can't cover multi-chunk ground truth — a corpus problem, not a model one; raising evidence budget is the lever."),
    ("", "F7 (11×) is the model emitting unverifiable claims; all 11 were corrected by the grounding handler (F15) — the guardrail's real work."),
    ("", "1 out-of-domain probe answered instead of abstained (F14) — the rules classifier scores cued words, and this probe's phrasing dodged them (LLM fallback engages only at low rule-confidence)."),
    ("", "Failure codes are recorded per query in reports/phase6_report.json — inspection, not vibes."),
], size=12, gap=8)

# 16 — Discussion (limitations) ----------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "8  ·  DISCUSSION", "Discussion — known limitations")
bullets(s, Inches(0.6), Inches(1.8), Inches(12.2), Inches(5.2), [
    ("", "PDF table extraction can interleave columns — the original cataract fusion (30-day rule + a 2-year list entry squeezed from a scrambled two-column table) came from this; retrieval + verification now catch the symptom, extraction noise remains."),
    ("", "Full-pipeline latency is CPU-bound: \u22486 s/query offline (reranker) and +tens of seconds per Ollama call; levers: --reranker lexical, lower candidate_k, --no-llm."),
    ("", "Answer-format reliability depends on the model emitting [n]; llama3.2:3b occasionally copies a clause's own label (\u201c[XXIII]\u201d) — markers don't resolve and the citation list shortens."),
    ("", "Single-annotator frozen set: qrels are fixed across all compared systems so comparisons are like-for-like, but absolute answer-correctness is a chunk-qrels proxy, not a human answer grade."),
    ("", "Fallback reranker weights were tuned on the same frozen set (optimistic for it); the cross-encoder needs no such tuning and is the configured default."),
    ("", "Docker packaging of the app layer is the remaining Phase 8 item."),
], size=14, gap=10)

# 17 — Conclusion ------------------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "9  ·  CONCLUSION AND FUTURE WORK", "Conclusion and Future Work")
box(s, Inches(0.55), Inches(1.8), Inches(6.0), Inches(4.9), "", panel=True, line=OK)
box(s, Inches(0.7), Inches(1.95), Inches(5.6), Inches(0.4), "Delivered", 14, OK, bold=True)
bullets(s, Inches(0.7), Inches(2.4), Inches(5.7), Inches(4.2), [
    "Phases 1–7 complete: ingestion → verified RAG → application layer",
    "Retrieval: P@5 0.34 → 0.43 · nDCG@5 0.41 → 0.55 vs baseline",
    "Grounding: every answer claim verified 3-way; hallucination (2 years cataract) caught live",
    "Evaluation: baseline vs full + 6 ablations + F1–F16 taxonomy on a frozen 45-query set",
    "Ship: FastAPI + dependency-free chat UI, 215 offline tests, reproducible reports",
], size=12.5, gap=7)
box(s, Inches(6.85), Inches(1.8), Inches(6.0), Inches(4.9), "", panel=True, line=ACCENT)
box(s, Inches(7.0), Inches(1.95), Inches(5.6), Inches(0.4), "Future work", 14, ACCENT, bold=True)
bullets(s, Inches(7.0), Inches(2.4), Inches(5.7), Inches(4.2), [
    "Phase 8 — Docker packaging + deployment configuration",
    "Fix the root cause of inter-column table extraction (layout-aware OCR/table parser)",
    "Multi-annotator qrels + human answer grading beyond the chunk-proxy metric",
    "Second-pass page-level evidence budget to attack F2 (30 queries missing evidence)",
    "Streaming token generation for the UI; optional hosted LLM with true cost accounting",
], size=12.5, gap=7)

# 18 — References ------------------------------------------------------------
s = add_slide(prs); fill_bg(s); n += 1
header(s, "REFERENCES", "References")
bullets(s, Inches(0.6), Inches(1.85), Inches(12.2), Inches(5.3), [
    ("", "Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 33."),
    ("", "Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR."),
    ("", "Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP."),
    ("", "Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP."),
    ("", "Nogueira, R. & Cho, K. (2019). Passage Re-ranking with BERT."),
    ("", "Yao, S. et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023."),
    ("", "Insurance Policy Intelligence — Software Requirements Specification, V1 (project SRS)."),
    ("", "Public insurer policy wordings corpus (rag_policy_dataset/), included for research use."),
], size=14.5, gap=10)

prs.save("Policy_RAG_Presentation.pptx")
print(f"saved Policy_RAG_Presentation.pptx — {n} slides")
