# Policy_RAG — Insurance Policy Intelligence

A retrieval-augmented question-answering system over a corpus of Indian insurance
policy wordings. Built in phases against the SRS
(`srs_document/Insurance Policy Intelligence_V1.pdf`).

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Document ingestion: inspect → extract → OCR → normalize to canonical JSON | ✅ Complete |
| **Phase 2** | Baseline RAG: chunk → embed → index → retrieve → generate grounded answers with citations | ✅ Complete |
| Phase 3+ | Hybrid retrieval, reranking, agentic planning, claim verification, comparisons | ⬜ Not started |

---

## Pipeline

```
PDFs (rag_policy_dataset/)
  │
  ▼  Phase 1  run_phase1.py
inspect → native text (PyMuPDF) → OCR only where flagged → normalize
  │                                       │
  ▼                                       ▼
reports/phase1_extraction_report.{json,md}   data/normalized/DOC-0XX.json
  │
  ▼  Phase 2  run_phase2.py build
structure-aware chunking → sentence-transformers embeddings → ChromaDB index
  │
  ├──► data/chunks/chunks.jsonl        data/chroma/ (rebuildable index)
  │
  ▼  Phase 2  run_phase2.py query "..."
top-k retrieval (+ metadata filters) → Ollama LLM → answer with [n] citations
```

---

## Setup

```bash
python -m venv .venv                     # tested on Python 3.11.0
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

`requirements.txt` covers both phases: PyMuPDF, numpy, Pillow, sentence-transformers,
chromadb, pytest. OCR is optional — install one engine from `requirements-ocr.txt`
(EasyOCR *or* pytesseract); without it, OCR pages are skipped and the run still succeeds.

Generation needs a local [Ollama](https://ollama.com) server:

```bash
ollama serve
ollama pull llama3.2:3b
```

## Phase 1 — Ingestion

```bash
python run_phase1.py                 # discovers PDFs, writes data/normalized/
python run_phase1.py --verbose       # per-document logging
python run_phase1.py --corpus-dir path/to/pdfs    # extra corpus directories
```

Last verified run: **10 documents, 553 pages — 551 native-text, 1 OCR page,
0 errors, 0 warnings.**

## Phase 2 — Baseline RAG

```bash
python run_phase2.py build            # chunk → embed → index (idempotent)
python run_phase2.py build --reset    # rebuild from scratch
python run_phase2.py stats            # index summary
python run_phase2.py query "What is the waiting period for cataract surgery?"
python run_phase2.py query "..." --k 8 --product "MediClaim Policy"   # metadata filters
python run_phase2.py query "..." --no-llm --out reports/answer.json   # retrieval only
```

Tuning lives in `configs/phase2.json` (chunk sizes, embedding model, LLM model,
temperature, `default_k`).

### Example (verified end-to-end)

```
QUERY: What is the waiting period for cataract surgery?

ANSWER (llama3.2:3b, 37.05s):
The waiting period for cataract surgery is 30 days [1].

CITATIONS:
  [1] MediCare (p. 13) — Tata AIG MediCare Policy Wording  [DOC-006 DOC-006::c0028]

RETRIEVED 5 chunks in 1933.1 ms:
  [1] sim=0.501 DOC-006 DOC-006::c0028 pp.13-13 MediCare | ...
```

Every `[n]` marker is resolved into a citation carrying document id, product, page
span, section and clause ids, so any claim can be traced back to the source PDF.

---

## Design notes

- **Traceability first** (SRS FR-004): phase 1 marks headers/footers as
  `include_in_chunk_text=False`, and every phase 2 chunk records document, product,
  section/subsection, clause ids and page span.
- **Clause-preserving chunking**: new chunks are cut at clause/heading boundaries,
  not mid-clause; oversized blocks split at paragraph then sentence boundaries; tables
  become their own `chunk_type="table"` chunks because tab-separated text embeds
  poorly when mixed with prose.
- **Configurable embedding model** (FR-005): default `all-MiniLM-L6-v2`, L2-normalized
  so cosine similarity reduces to a dot product.
- **Idempotent indexing** (FR-006): deterministic `chunk_id` upserts into a ChromaDB
  persistent collection; rebuilds never duplicate.
- **Grounded generation** (FR-008): the prompt contains numbered evidence blocks, the
  model must cite them as `[n]` and is instructed to abstain when the corpus does not
  contain the answer.

## Repository layout

```
run_phase1.py / run_phase2.py     Phase CLIs
configs/                         documents.json (phase 1), phase2.json (phase 2)
src/ingestion/                   phase 1: inspection, extraction, OCR, normalization
src/chunking/                    phase 2: structure-aware chunker
src/embeddings/                  phase 2: sentence-transformers wrapper
src/retrieval/                   phase 2: ChromaDB store + top-k retriever
src/generation/                  phase 2: Ollama client + grounded answer generator
data/normalized/                 canonical per-document JSON (phase 1 output)
data/chunks/chunks.jsonl         chunk corpus (phase 2 build artifact)
data/chroma/                     vector index — rebuildable, git-ignored
reports/                         extraction report, quality audit, query log
tests/                           62 tests (34 phase 1, 28 phase 2)
```

## Tests

```bash
python -m pytest tests/test_phase2_rag.py -q     # 28 passed (~4s)
python -m pytest tests/test_phase1_ingestion.py -q   # 34 passed (~3m42s, OCR-heavy)
```

## Known limitations

- **Answer faithfulness is baseline-level.** In the verified run above, the model fused
  two clauses from a *scrambled two-column table* (a general 30-day initial waiting
  period with a cataract entry from a separate two-year list) and answered "30 days"
  where the cited page lists cataract under the **two-year** waiting-period list. The
  citation mechanics are correct; the semantic gap is what Phases 3–5 (hybrid
  retrieval, reranking, claim verification) address. Related: PDF table layout
  extraction can interleave columns.
- **`data/chunks/chunks.jsonl` embeds an absolute `source_path`** from the machine that
  built it. Rebuild with `python run_phase2.py build --reset` on another machine to
  refresh it.
- **Generation requires a running Ollama server**; use `--no-llm` for retrieval-only runs
  (no server needed).
- Phase 2 retrieval is **semantic only** — no keyword/hybrid search or reranking yet.

## License / data

The PDFs under `rag_policy_dataset/` are public insurer policy documents, included for
research use.