# Phase 6 — Evaluation report (SRS Phase 6, §14, §15, AC-010..AC-012)

Generated: 2026-10-18T12:15:32.807645+00:00

## AC-010 — Enhanced system vs baseline RAG

| metric | baseline (Phase 2) | full (Phase 5 pipeline) | delta |
|---|---|---|---|
| answered_correct_rate | 0.791 | 0.809 | +0.019 |
| unsupported_claim_rate | 0.066 | 0.061 | -0.004 |
| mean_supported_rate | 0.943 | 0.949 | +0.006 |
| latency ms/query | 70 | 5894 | +5823 |

## AC-011 — Latency and token/cost record

### full

| component | mean ms | max ms |
|---|---|---|
| analysis | 1 | 3 |
| generation | 4 | 59 |
| retrieval | 5884 | 25579 |
| total | 5894 | 25586 |
| verification | 6 | 24 |
| **total** | 5894 | |

Tokens: prompt 0, completion 0, LLM calls 0 — estimated cost **$0.00/query** (local Ollama; NFR-005 records tokens where an API cost does not apply).

### baseline

| component | mean ms | max ms |
|---|---|---|
| generation | 1 | 4 |
| retrieval | 65 | 242 |
| total | 70 | 247 |
| **total** | 70 | |

Tokens: prompt 0, completion 0, LLM calls 0 — estimated cost **$0.00/query** (local Ollama; NFR-005 records tokens where an API cost does not apply).

## Ablation experiments

| variant | correct ↑ | unsupported ↓ | grounding ↑ | abstentions | mean ms |
|---|---|---|---|---|---|
| default | 0.809 | 0.061 | 0.949 | 1 | 2251 |
| no_rerank | 0.857 | 0.088 | 0.924 | 1 | 38 |
| keyword_only | 0.833 | 0.045 | 0.959 | 1 | 2312 |
| semantic_only | 0.833 | 0.072 | 0.940 | 1 | 2621 |
| no_strip | 0.809 | 0.061 | 0.949 | 1 | 2599 |
| rules_only_verify | 0.809 | 0.061 | 0.949 | 1 | 2486 |
- `default`: full pipeline as configured (hybrid + rerank + grounding)
- `no_rerank`: hybrid retrieval without the cross-encoder reranker
- `keyword_only`: drop the dense retriever; BM25 keyword only
- `semantic_only`: drop BM25; semantic retrieval only (≈ baseline retrieval)
- `no_strip`: keep unsupported claims in the answer (grounding handler off)
- `rules_only_verify`: drop the LLM second pass; rules-only verification

## Retrieval quality reference (Phase 3 frozen-set eval)

| mode | P@k | R@k | nDCG@k | MRR |
|---|---|---|---|---|
| semantic | 0.340 | 0.367 | 0.410 | 0.586 |
| keyword | 0.312 | 0.350 | 0.386 | 0.538 |
| hybrid | 0.391 | 0.417 | 0.481 | 0.685 |
| hybrid+rerank | 0.428 | 0.481 | 0.552 | 0.776 |

## AC-012 — Failure taxonomy (full pipeline)

| code | label | queries |
|---|---|---|
| F1 | retrieval_failure | 9 |
| F2 | missing_relevant_evidence | 30 |
| F4 | query_classification_failure | 1 |
| F7 | unsupported_generation | 11 |
| F10 | incorrect_abstention | 1 |
| F12 | product_comparison_failure | 1 |
| F14 | out_of_domain_detection_failure | 1 |
| F15 | grounding_corrected | 11 |
| F16 | abstention_correct | 1 |

F15/F16 are positive detections (unsupported-claim correction, correct OOD abstention) recorded per query but excluded from the failure rate.

