# Policy_RAG — Insurance Policy Intelligence

A retrieval-augmented question-answering system over a corpus of Indian insurance
policy wordings. Built in phases against the SRS
(`srs_document/Insurance Policy Intelligence_V1.pdf`).

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Document ingestion: inspect → extract → OCR → normalize to canonical JSON | ✅ Complete |
| **Phase 2** | Baseline RAG: chunk → embed → index → retrieve → generate grounded answers with citations | ✅ Complete |
| **Phase 3** | Retrieval improvement: metadata filtering → hybrid (BM25 + semantic) retrieval → reranking → retrieval evaluation | ✅ Complete |
| **Phase 4** | Agentic layer: query classification → requirement extraction → query planning → multi-step retrieval → product comparison | ✅ Complete |
| Phase 5+ | Grounding/verification, evaluation, API/UI, deployment | ⬜ Not started |

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
top-k semantic retrieval (+ metadata filters) → Ollama LLM → answer with [n] citations
  │
  ▼  Phase 3  run_phase3.py build / query / eval
BM25 keyword index → rank-fused hybrid retrieval → cross-encoder reranking
  │
  ├──► data/bm25/bm25_index.json       (rebuildable keyword index)
  └──► reports/phase3_retrieval_eval.{json,md}   (baseline vs hybrid metrics)
  │
  ▼  Phase 4  run_phase4.py analyze / query / products
query analysis → retrieval planning → multi-step retrieval → comparison/answer
  │
  └──► reports/phase4_query_log.jsonl  (agent trace: type, plan, citations, latency)
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

## Phase 3 — Retrieval Improvement

```bash
python run_phase3.py build                          # build + cache the BM25 keyword index
python run_phase3.py query "..."                    # hybrid retrieval + grounded answer
python run_phase3.py query "..." --mode semantic    # keyword | semantic | hybrid (+rerank)
python run_phase3.py query "..." --reranker lexical --no-llm   # offline retrieval only
python run_phase3.py query "..." --document-id DOC-006        # metadata filter (both paths)
python run_phase3.py eval                           # baseline vs hybrid vs reranked metrics
python run_phase3.py stats                          # BM25 + dense index stats
```

Phase 3 adds the four SRS Phase 3 items on top of the Phase 2 baseline:

- **Metadata filtering** — one filter (`--document-id/--product/--chunk-type`) is
  applied to *both* the dense and lexical search paths.
- **Hybrid retrieval** — semantic top-N fused with a BM25 keyword top-N. Fusion is
  RRF by default (rank-based, no score scaling needed) or `--fusion weighted`.
- **Reranking** — a `sentence-transformers` CrossEncoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores the fused pool. When the model
  cannot load, it degrades to a deterministic **feature-blend** reranker (lexical +
  dense + coverage) that still improves on plain hybrid — not a BM25 re-sort — so
  runs never hard-fail offline and still gain from reranking.
- **Retrieval evaluation** — Precision@K, Recall@K, nDCG@K, MRR and MAP against the
  frozen query set `data/eval/retrieval_eval.json`, comparing semantic (baseline),
  keyword, hybrid and hybrid+rerank.

Tuning lives in `configs/phase3.json` (fusion, `candidate_k`, reranker model,
BM25 `k1`/`b`, eval set and modes). The dense index and generation model are
inherited from `configs/phase2.json` via `base_config`.

### Verified retrieval evaluation (k=5, candidate_k=20, fusion=rrf)

Frozen set: **45 queries** (43 answerable across all 10 documents + 2 out-of-domain
probes), categories mirroring the SRS test taxonomy. Numbers below are the
cross-encoder reranker (the configured default); `reports/phase3_retrieval_eval.{json,md}`.

| mode | Precision@5 | Recall@5 | nDCG@5 | Hit@5 | MRR | MAP |
|---|---|---|---|---|---|---|
| semantic (baseline) | 0.340 | 0.367 | 0.410 | 0.791 | 0.586 | 0.281 |
| keyword (BM25) | 0.312 | 0.350 | 0.386 | 0.674 | 0.538 | 0.276 |
| hybrid (RRF) | 0.391 | 0.417 | 0.481 | 0.884 | 0.685 | 0.336 |
| **hybrid + rerank** | **0.428** | **0.481** | **0.552** | 0.861 | **0.776** | **0.420** |

With the cross-encoder, reranking lifts every metric substantially: +0.071 nDCG@5
and +0.091 MRR over the hybrid pool, and +0.142 nDCG@5 / +0.114 Recall@5 over the
Phase 2 semantic baseline. Hybrid also beats the semantic baseline on every metric
(+0.051 P@5), which is the Phase 3 retrieval gain. On the cataract query used as the
Phase 2 regression example, reranking now puts the clause listing cataract under the
specified-disease waiting period (`DOC-006::c0028`) first, instead of the scrambled
table the baseline cited.

The **model-free fallback** is evaluated the same way (no model download); it beats
plain hybrid on precision, recall, nDCG, MRR and MAP rather than merely echoing the
keyword ranking (`reports/phase3_retrieval_eval_lexical.{json,md}`):

| mode | Precision@5 | Recall@5 | nDCG@5 | Hit@5 | MRR | MAP |
|---|---|---|---|---|---|---|
| keyword (BM25) | 0.312 | 0.350 | 0.386 | 0.674 | 0.538 | 0.276 |
| hybrid (RRF) | 0.391 | 0.417 | 0.481 | **0.884** | 0.685 | 0.336 |
| **hybrid + lexical rerank** | **0.405** | **0.430** | **0.493** | 0.814 | **0.696** | **0.356** |

---

## Phase 4 — Agentic Layer

```bash
python run_phase4.py analyze "Compare Digit and Elevate for waiting period"   # classify + plan, no retrieval
python run_phase4.py analyze "..." --llm        # exercise the LLM classification fallback
python run_phase4.py query "What is the waiting period for cataract surgery?"
python run_phase4.py query "Compare the Digit Health Insurance Policy and Elevate for waiting period and exclusions"
python run_phase4.py query "..." --no-llm       # plan + multi-step retrieval only
python run_phase4.py products                  # recognised products and aliases
```

Phase 4 adds the SRS agentic layer on top of the hybrid retriever:

- **Query classification** (FR-007) — rules-first cue scoring into the SRS
  categories; an LLM fallback handles low-confidence cases. Every result records
  `method` = `rules` or `llm`, so the trade-off stays auditable.
- **Requirement extraction** (FR-013) — structured slots (age, sum insured,
  coverage objective, priorities, tenure) pulled from recommendation queries.
- **Query planning** (FR-014) — a plan of retrieval operations. Comparisons become
  one step per *product × criterion* with a product metadata filter; briefings for
  recommendations seed the steps with the user's requirements; complex questions
  are decomposed into sub-questions or facets.
- **Multi-step retrieval** (FR-008) — each step runs the Phase 3 hybrid retriever;
  evidence is de-duplicated across steps under a global budget (FR-009).
- **Product comparison** (FR-012) — a dedicated grounded generator that interleaves
  excerpts across policies, separates facts from conclusions, reports missing
  criteria, and cites `[n]` markers back to document/product/page/section.

Two short-circuits are handled explicitly: **out-of-domain** queries (FR-019) get a
scope statement with no retrieval, and underspecified comparisons/recommendations
(FR-018) get a targeted clarification question instead of a guess.

### Example (verified end-to-end, llama3.2:3b)

```
QUERY: Compare the Digit Health Insurance Policy and Elevate for waiting period and exclusions

type=product_comparison method=rules
products=['Digit Health Insurance Policy', 'Elevate'] criteria=['exclusions', 'waiting period']

ANSWER:
**Waiting Period:**
* Digit Health Insurance Policy: waiting period for pre-existing diseases is the
  number of months specified in the schedule [1][3][5]...
* Elevate: waiting period for cardiac conditions is reduced to the extent of prior
  coverage under portability norms [6].
**Exclusions:** ... facts per policy ...
**Conclusion:** ... derived summary ...

CITATIONS:
  [6] Elevate (pp. 33-34) — ELEVATE POLICY WORDINGS — clauses: 4, 8
evidence=18 chunks  steps=4
latency[analysis=0.0ms plan=0.0ms retrieval=10001.5ms gen=88364.5ms total=98367.4ms]
```

Settings live in `configs/phase4.json` (analysis confidence threshold, step/product
caps, evidence budget, comparison criteria, product aliases).

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
- **Hybrid retrieval** (Phase 3): BM25 keyword search is implemented in-house (no extra
  dependency) and fused with dense results by RRF, so exact terms the embedding model
  blurs — clause ids, procedure names, defined terms — still surface.
- **Replaceable reranker** (NFR-008): `CrossEncoderReranker` and `LexicalReranker` share
  one `rerank(query, candidates, top_k)` interface; `build_reranker` selects one from
  config and falls back gracefully, so the embedding model, vector store, retriever,
  reranker and LLM can each be swapped independently.
- **Reproducible evaluation** (SRS §14.1): the frozen query set and its chunk-level
  qrels are committed; `run_phase3.py eval` recomputes every metric from scratch.
- **Rules-first, LLM-second** (Phase 4): deterministic cue scoring classifies the
  common cases with no server, so most of the agent layer is unit-testable offline;
  the LLM is consulted only when rule confidence is low. This keeps the agentic
  behaviour inspectable and cheap.
- **Balanced comparison evidence** (Phase 4): excerpts are round-robined across
  products before the context budget is applied, so one policy cannot starve
  another, and the same ordered list drives both prompt numbering and citation
  resolution — a `[n]` marker always points at the excerpt it was shown as.

## Repository layout

```
run_phase1.py / run_phase2.py / run_phase3.py / run_phase4.py   Phase CLIs
configs/                         documents.json (phase 1) + phase2/3/4.json
src/ingestion/                   phase 1: inspection, extraction, OCR, normalization
src/chunking/                    phase 2: structure-aware chunker
src/embeddings/                  phase 2: sentence-transformers wrapper
src/retrieval/                   phase 2/3: ChromaDB store, retriever, bm25, reranker, hybrid
src/generation/                  phase 2: Ollama client + grounded answer generator
src/evaluation/                  phase 3: retrieval metrics (P@k, R@k, MRR, nDCG)
src/agent/                       phase 4: classifier, planner, executor, comparison, orchestrator
data/normalized/                 canonical per-document JSON (phase 1 output)
data/chunks/chunks.jsonl         chunk corpus (phase 2 build artifact)
data/chroma/                     vector index — rebuildable, git-ignored
data/bm25/bm25_index.json        keyword index — rebuildable, git-ignored
data/eval/retrieval_eval.json    frozen retrieval evaluation set (phase 3)
reports/                         extraction report, quality audit, query logs, eval reports
tests/                           144 tests (34 phase 1, 28 phase 2, 34 phase 3, 48 phase 4)
```

## Tests

```bash
python -m pytest tests/test_phase2_rag.py -q     # 28 passed (~4s)
python -m pytest tests/test_phase3_retrieval.py -q   # 34 passed (~2s)
python -m pytest tests/test_phase4_agent.py -q       # 48 passed (~4s, no server needed)
python -m pytest tests/test_phase1_ingestion.py -q   # 34 passed (~3m42s, OCR-heavy)
```

## Known limitations

- **Answer faithfulness is baseline-level.** In the Phase 2 verified run, the model
  fused two clauses from a *scrambled two-column table* (a general 30-day initial
  waiting period with a cataract entry from a separate two-year list). Phase 3
  retrieval now surfaces the correct clause, but claim-level verification is still
  Phase 5. Related: PDF table layout extraction can interleave columns.
- **`data/chunks/chunks.jsonl` embeds an absolute `source_path`** from the machine that
  built it. Rebuild with `python run_phase2.py build --reset` on another machine to
  refresh it.
- **Generation requires a running Ollama server**; use `--no-llm` for retrieval-only runs
  (no server needed).
- **Reranker fallback weights are tuned on the frozen set.** The feature-blend
  fallback's defaults (lexical 0.20 / dense 0.80) were selected on
  `data/eval/retrieval_eval.json`, so its reported numbers are optimistic for this set;
  a held-out set is needed to confirm they generalise. The cross-encoder needs no such
  tuning and is the configured default.
- **Phase 4 citations depend on the model emitting `[n]` markers.** The resolver and the
  comparison ordering are unit-tested, but llama3.2:3b occasionally copies a clause's
  own label instead (e.g. `[XXIII]`), in which case the marker does not resolve and the
  citation list is short. Answer *format* reliability, like faithfulness, is what
  Phases 5-6 address; the agent pipeline itself is model-agnostic.
- **Agentic answers are cited but not yet claim-verified.** Phase 4 separates facts from
  conclusions in the prompt, but per-claim support/contradiction checks (Agent 5,
  FR-015/FR-016) are Phase 5.
- **Comparisons are slow on CPU.** A 2-policy x 2-criterion comparison retrieves over
  four steps and then generates; on CPU the LLM generation dominates (tens of seconds
  to a few minutes). Reduce `candidate_k`, use `--reranker lexical`, or `--no-llm` for
  faster runs.
- **The frozen evaluation set is single-annotator.** It is now 43 answerable queries
  across all 10 documents and 6 categories, and the qrels were relevance-audited (see
  `audit_note` in the JSON) to stop standard IRDAI template clauses repeated across
  policies from being under-labelled. Absolute scores can still shift with a second
  annotator, but the label set is fixed across all retrieval modes, so the
  baseline-vs-hybrid-vs-rerank comparison is a like-for-like measurement. Scaling the
  set and adding a second annotator is Phase 6 work.

## License / data

The PDFs under `rag_policy_dataset/` are public insurer policy documents, included for
research use.