# Phase 3 Retrieval Evaluation

- Frozen set: `data/eval/retrieval_eval.json`
- Cutoff K: 5 · candidate pool: 20 · fusion: rrf
- Answerable queries: 43 of 45

## Summary

| mode | PRECISION@5 | RECALL@5 | NDCG@5 | HIT@5 | MRR | MAP |
|---|---|---|---|---|---|---|
| keyword | 0.312 | 0.350 | 0.386 | 0.674 | 0.538 | 0.276 |
| hybrid | 0.391 | 0.417 | 0.481 | 0.884 | 0.685 | 0.336 |
| hybrid+rerank | 0.405 | 0.430 | 0.493 | 0.814 | 0.696 | 0.356 |

## By category

### keyword

| category | count | PRECISION@5 | RECALL@5 | NDCG@5 | HIT@5 | MRR | MAP |
|---|---|---|---|---|---|---|---|
| comparison | 3 | 0.333 | 0.417 | 0.458 | 0.667 | 0.667 | 0.367 |
| coverage | 19 | 0.221 | 0.306 | 0.328 | 0.526 | 0.465 | 0.249 |
| eligibility | 1 | 0.400 | 0.400 | 0.301 | 1.000 | 0.333 | 0.147 |
| exclusion | 4 | 0.250 | 0.263 | 0.315 | 0.500 | 0.500 | 0.263 |
| factual | 14 | 0.429 | 0.422 | 0.467 | 0.857 | 0.606 | 0.314 |
| waiting_period | 2 | 0.400 | 0.309 | 0.449 | 1.000 | 0.750 | 0.218 |

### hybrid

| category | count | PRECISION@5 | RECALL@5 | NDCG@5 | HIT@5 | MRR | MAP |
|---|---|---|---|---|---|---|---|
| comparison | 3 | 0.600 | 0.639 | 0.688 | 1.000 | 0.778 | 0.542 |
| coverage | 19 | 0.242 | 0.304 | 0.323 | 0.789 | 0.502 | 0.206 |
| eligibility | 1 | 0.400 | 0.400 | 0.384 | 1.000 | 0.500 | 0.233 |
| exclusion | 4 | 0.300 | 0.346 | 0.362 | 0.750 | 0.562 | 0.273 |
| factual | 14 | 0.557 | 0.549 | 0.680 | 1.000 | 0.917 | 0.497 |
| waiting_period | 2 | 0.500 | 0.381 | 0.577 | 1.000 | 1.000 | 0.302 |

### hybrid+rerank

| category | count | PRECISION@5 | RECALL@5 | NDCG@5 | HIT@5 | MRR | MAP |
|---|---|---|---|---|---|---|---|
| comparison | 3 | 0.733 | 0.778 | 0.782 | 1.000 | 0.778 | 0.647 |
| coverage | 19 | 0.221 | 0.283 | 0.320 | 0.684 | 0.539 | 0.213 |
| eligibility | 1 | 0.400 | 0.400 | 0.384 | 1.000 | 0.500 | 0.233 |
| exclusion | 4 | 0.300 | 0.346 | 0.432 | 0.750 | 0.750 | 0.346 |
| factual | 14 | 0.614 | 0.599 | 0.695 | 0.929 | 0.845 | 0.520 |
| waiting_period | 2 | 0.400 | 0.309 | 0.478 | 1.000 | 1.000 | 0.225 |
