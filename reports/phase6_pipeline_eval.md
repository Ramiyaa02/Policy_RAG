# Phase 6 — Pipeline evaluation (SRS §14)

- Frozen set: data/eval/retrieval_eval.json
- Queries: 45 · generated: 2026-10-18T11:50:40.112902+00:00

| metric | full | baseline |
|---|---|---|
| answer correctness (proxy) ↑ | 0.809 | 0.791 |
| answered (answerable) ↑ | 0.977 | 1.000 |
| abstentions on answerable | 1 | 0 |
| OOD probes abstained ↑ | 1 | 0 |
| unsupported claim rate ↓ | 0.061 | 0.066 |
| mean grounding supported rate ↑ | 0.949 | 0.943 |

## full — latency & cost

- total: mean 5894 ms, median 4500 ms
- analysis: mean 1 ms (max 3)
- generation: mean 4 ms (max 59)
- retrieval: mean 5884 ms (max 25579)
- total: mean 5894 ms (max 25586)
- verification: mean 6 ms (max 24)
- tokens: prompt 0, completion 0, LLM calls 0 — cost ≈ $0 (local Ollama)

## baseline — latency & cost

- total: mean 70 ms, median 60 ms
- generation: mean 1 ms (max 4)
- retrieval: mean 65 ms (max 242)
- total: mean 70 ms (max 247)
- tokens: prompt 0, completion 0, LLM calls 0 — cost ≈ $0 (local Ollama)

## Failure taxonomy

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
