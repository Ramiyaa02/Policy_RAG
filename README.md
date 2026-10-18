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
| **Phase 5** | Grounding: claim extraction → evidence verification → unsupported-claim detection → abstention → clarification | ✅ Complete |
| **Phase 6** | Evaluation vs baseline, ablation experiments, failure taxonomy (§15), accuracy/grounding/latency/cost records (§14) | ✅ Complete |
| Phase 7+ | API/UI, deployment & documentation | ⬜ Not started |

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
  │
  ▼  Phase 5  run_phase5.py verify
claim extraction → evidence verification (Agent 5) → response shaping
  │
  └──► reports/phase5_query_log.jsonl  (grounding: supported/contradicted/insufficient per claim)
  │
  ▼  Phase 6  run_phase6.py evaluate / ablate / report
baseline (Phase 2) vs full (Phase 5 pipeline) on the frozen set → ablations
  │
  ├──► reports/phase6_pipeline_eval.{json,md}    (§14 metrics + token/cost record)
  ├──► reports/phase6_ablations.json             (pipeline ablation grid)
  └──► reports/phase6_report.{json,md}           (AC-010..AC-012 roll-up)
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
```Settings live in `configs/phase4.json` (analysis confidence threshold, step/product
  caps, evidence budget, comparison criteria, product aliases).

---

## Phase 5 — Grounding & Verification

```bash
python run_phase5.py verify "What is the waiting period for cataract surgery?"          # full pipeline + grounding
python run_phase5.py verify "..." --no-llm          # offline: extractive answer, rules-only verification
python run_phase5.py verify "..." --no-verify-llm   # LLM answers, but no LLM second-guess of rule verdicts
python run_phase5.py verify "..." --out reports/verify.json
```

Phase 5 implements the SRS §8 guardrail loop and Agent 5 (FR-015–FR-018) on
  top of the Phase 4 pipeline:

- **Claim extraction (FR-015)** — answers are split into fact claims (headings,
  citation-only fragments and back-matter dropped; `[n]` markers pulled per
  claim, including markers placed after the sentence period). Conclusions under
  a "Conclusion/Note" heading are labelled separately, matching FR-012's
  fact/conclusion distinction: inference is not verified as fact.
- **Evidence verification (Agent 5, FR-015)** — each claim is checked against
  the excerpts its citations resolved to. A deterministic scoring pass does
  content-token containment plus number consistency (waiting periods, sums,
  percentages must appear in the cited evidence) and returns one of the three
  SRS §8 verdicts: **supported / contradicted / insufficient**. Absence of
  evidence is never turned into evidence of absence. An optional LLM batch
  second-guesses only the rules; a contradicted rules-verdict is only lifted
  when the LLM says supported, so contradictions cannot be papered over by a
  parser error.
- **Unsupported-claim handling (FR-016, FR-017)** — insufficient fact claims
  are cut out of the answer rather than presented as policy fact; contradicted
  claims are removed *and* the conflict is disclosed in a grounding note.
- **Abstention (O8)** — when no fact claim survives verification (or the
  supported rate falls below `min_supported_rate`), the response abstains
  instead of shipping unverified sentences.
- **Clarification (FR-018)** — an abstention asks one targeted question keyed
  to the query type (procedure for waiting periods, treatment for exclusions,
  age for eligibility...), never "please provide more information".

### Example (verified end-to-end, llama3.2:3b)

The model repeated its known Phase 4 failure mode — citing a clause's own
label `[XXIII]` instead of an excerpt number. Phase 5 now catches it:

```
QUERY: What is the waiting period for cataract surgery?

CLAIMS:
  [MISS] (insufficient score=0.00, rules) The waiting period for cataract
         surgery is 2 years [XXIII].

GROUNDING: 0 supported / 0 contradicted / 1 insufficient of 1 claims

ABSTAINED (clarification asked)
VERIFIED ANSWER:
I could not verify enough of the answer against the policy documents, so I
will not risk presenting policy facts I cannot support. No retrieved excerpt
supports the key part of this question. Could you name the specific procedure
or illness whose waiting period you need?

Note: the following statements could not be verified against the policy
documents and were removed: "The waiting period for cataract surgery is 2
years [XXIII]".
latency[retrieval=5888ms gen=41049ms verify=6137ms total=53075ms]
```

Offline (`--no-llm`), an extractive answer quoting retrieved excerpts scores
10/10 supported with 9ms verification, so the grounding loop is exercisable
without any LLM server.

A second live run shows the happy path keeping a verified answer and the
handler cleaning it up in place (multi-step maternity question):

```
type=multi_step evidence=10 latency[retrieval=8805ms gen=56664ms verify=19586ms] total=85064ms
GROUNDING: 4 supported / 0 contradicted / 1 insufficient of 5 claims (supported_rate=0.80)

The following products cover maternity:
* SECTION 4. MATERNITY BENEFIT WALLET & NEWBORN COVER [1]
* Optional Cover 10: Infertility Treatment Cover [3]
The waiting period for Infertility Treatment Cover is as opted by You and
mentioned in your Policy Schedule ... [3].

Note: the following statements could not be verified against the policy
documents and were removed: "Optional cover 9.".
```

When the model produces output nothing can be verified in (e.g. only a stray
citation label with no sentences), the handler abstains rather than shipping
it — the same cataract question as above once generated an answer of just
`[XXIII]` and now returns the targeted clarification instead.

Settings live in `configs/phase5.json` (`support_threshold`, `number_strict`,
`strip_unsupported`, `min_supported_rate`); retrieval/agent/generation
settings are inherited through the `phase4 -> phase3 -> phase2` config chain.

---

## Phase 6 — Evaluation

```bash
python run_phase6.py evaluate --no-llm              # baseline vs full pipelines, offline (no Ollama)
python run_phase6.py evaluate                       # with the LLM in the loop (records tokens per query)
python run_phase6.py evaluate --modes full          # only the enhanced pipeline
python run_phase6.py ablate --no-llm                # pipeline ablation grid
python run_phase6.py report                         # roll everything up (AC-010..AC-012)
```

Phase 6 implements the SRS §14/§15 evaluation bundle and DoD items
"evaluation vs a frozen test set", "failure modes documented" and "latency and
token/cost measurements recorded":

- **Baseline comparison (AC-010)** — `evaluate` runs two whole systems over the
  same 45-query frozen set: the **Phase 2 baseline** (semantic-only retrieval,
  one direct grounded generation, no agent/verification) and the **full
  Phase 5 pipeline** (classification → planning → hybrid+reranker retrieval →
  generation → claim verification → response shaping). Agent 5's deterministic
  validator scores BOTH systems offline as a measuring instrument (for the
  baseline it is not part of the system), so the unsupported-claim rate (§14.4)
  is like-for-like.
- **Metrics (§14)** — answer correctness proxy (§14.2: answered AND ≥1 relevant
  chunk among the evidence used — the set is single-annotator, chunk qrels are
  the reference), unsupported claim rate (§14.4), grounding/mean supported rate
  (§14.5), citation resolution failures (§14.3), abstention accuracy reported
  separately for answerable queries vs out-of-domain probes (§14.6),
  per-component latency (§14.7/NFR-004) and token usage (§14.8/NFR-005 — local
  Ollama inference has no API cost, so tokens and wall time are the record).
- **Failure taxonomy (§15, AC-012)** — every evaluated query is classified with
  codes F1–F14 (retrieval failure, missing evidence, classification failure,
  unsupported generation, citation failure, incorrect/unnecessary abstention,
  out-of-domain detection failure, ...); F15 (grounding correction) and F16
  (correct OOD abstention) are recorded as positive detections, excluded from
  the failure rate.
- **Ablation experiments** — `ablate` reruns the full pipeline under single
  changes: no reranker, BM25-only, semantic-only, unsupported-claim keeping,
  rules-only verification — each reported on the same metrics.

### Offline results (--no-llm, 45 frozen queries)

| metric | baseline (Phase 2) | full (Phase 5 pipeline) | delta |
|---|---|---|---|
| answered rate (answerable) | 1.000 | 0.977 | −0.023 |
| answer correctness (proxy) | 0.791 | 0.809 | +0.019 |
| unsupported claim rate ↓ | 0.066 | 0.061 | −0.004 |
| mean grounding supported rate | 0.943 | 0.949 | +0.006 |
| OOD probes abstained | 0/2 | 1/2 | +1 |
| mean ms/query | 70 | 5 894 | +5.8k |

Key taxonomy findings (offline run): 26 of the 45 queries lose at least one
relevant chunk from the evidence actually used (F2: 30–33 — a k=5 evidence
budget cannot cover multi-chunk qrels), 9 queries retrieve no relevant chunk at
all (F1), the baseline answers out-of-domain probes instead of abstaining
(F14: 2) while the full pipeline abstains on one of them (F16: 1) and asks for
clarification on the other, and unsupported-claim correction fired 11× in the
full pipeline (F15).

The full-LLM run (below) is the one that matches the deployed system; the
offline grid is the reproducible fast reference (no Ollama needed).

Settings live in `configs/phase6.json` (`eval_set`, `modes`, `report_dir`).

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
run_phase1.py ... run_phase6.py   Phase CLIs
configs/                         documents.json (phase 1) + phase2..phase6.json
src/ingestion/                   phase 1: inspection, extraction, OCR, normalization
src/chunking/                    phase 2: structure-aware chunker
src/embeddings/                  phase 2: sentence-transformers wrapper
src/retrieval/                   phase 2/3: ChromaDB store, retriever, bm25, reranker, hybrid
src/generation/                  phase 2: Ollama client + grounded answer generator
src/evaluation/                  phase 3: retrieval metrics · phase 6: pipeline eval + failure taxonomy
src/agent/                       phase 4: classifier, planner, executor, comparison, orchestrator
src/grounding/                   phase 5: claim extractor, validator (Agent 5), handler, pipeline
data/normalized/                 canonical per-document JSON (phase 1 output)
data/chunks/chunks.jsonl         chunk corpus (phase 2 build artifact, portable: basename source_path)
data/chroma/                     vector index — rebuildable, git-ignored
data/bm25/bm25_index.json        keyword index — rebuildable, git-ignored
data/eval/retrieval_eval.json    frozen retrieval evaluation set (phase 3)
reports/                         extraction report, quality audit, query logs, eval reports
tests/                           210 tests (34 phase 1, 28 phase 2, 34 phase 3, 48 phase 4, 41 phase 5, 25 phase 6)
```

## Tests

```bash
python -m pytest tests/test_phase2_rag.py -q     # 28 passed (~4s)
python -m pytest tests/test_phase3_retrieval.py -q   # 34 passed (~2s)
python -m pytest tests/test_phase4_agent.py -q       # 48 passed (~4s, no server needed)
python -m pytest tests/test_phase5_grounding.py -q   # 41 passed (~3s, no server needed)
python -m pytest tests/test_phase1_ingestion.py -q   # 34 passed (~9m, OCR-heavy, machine-dependent)
python -m pytest tests/test_phase6_evaluation.py -q  # 25 passed (~1s, no server needed)
```

## Known limitations

- **PDF table layout extraction can interleave columns.** In the Phase 2 verified
  run, the model fused two clauses from a *scrambled two-column table* (a general
  30-day initial waiting period with a cataract entry from a separate two-year list).
  Phase 3 retrieval now surfaces the correct clause and Phase 5 claim verification
  strips the fusion when it happens — but the underlying extraction noise remains.
- **Full-pipeline latency is LLM-bound.** The Phase 6 record shows the offline
  machinery costs ~5.9 s/query (mostly cross-encoder reranking on CPU) against the
  baseline's 70 ms, and each Ollama call adds tens of seconds on CPU. Reduce
  `candidate_k`, use `--reranker lexical`, or `--no-llm` for faster runs.
- **Generation requires a running Ollama server**; use `--no-llm` for retrieval-only runs
  (no server needed).
- **Reranker fallback weights are tuned on the frozen set.** The feature-blend
  fallback's defaults (lexical 0.20 / dense 0.80) were selected on
  `data/eval/retrieval_eval.json`, so its reported numbers are optimistic for this set;
  a held-out set is needed to confirm they generalise. The cross-encoder needs no such
  tuning and is the configured default (its no-reranker ablation is on record in the
  Phase 6 ablation grid).
- **Phase 4 citations depend on the model emitting `[n]` markers.** The resolver and the
  comparison ordering are unit-tested, but llama3.2:3b occasionally copies a clause's
  own label instead (e.g. `[XXIII]`), in which case the marker does not resolve and the
  citation list is short. Answer *format* reliability, like faithfulness, is what
  Phases 5-6 address; the agent pipeline itself is model-agnostic.
- **Comparisons are slow on CPU.** A 2-policy x 2-criterion comparison retrieves over
  four steps and then generates; on CPU the LLM generation dominates (tens of seconds
  to a few minutes). Reduce `candidate_k`, use `--reranker lexical`, or `--no-llm` for
  faster runs.
- **The frozen evaluation set is single-annotator.** It is now 43 answerable queries
  across all 10 documents and 6 categories, and the qrels were relevance-audited (see
  `audit_note` in the JSON) to stop standard IRDAI template clauses repeated across
  policies from being under-labelled. Absolute scores can still shift with a second
  annotator, but the label set is fixed across every compared system, so the
  baseline-vs-full comparison and the Phase 6 ablations are like-for-like.
  Phase 6 answer-correctness is therefore a chunk-qrels *proxy* (answered AND ≥1
  relevant chunk used), not a human answer-grade.

## License / data

The PDFs under `rag_policy_dataset/` are public insurer policy documents, included for
research use.