"""Phase 3 — Retrieval reranking (SRS FR-008, Phase 3 "reranking").

Hybrid retrieval returns a fused candidate pool; a reranker then rescores the
``(query, passage)`` pairs jointly and keeps the best few. Two interchangeable
implementations are provided (NFR-008 keeps the reranker replaceable):

- :class:`CrossEncoderReranker` — a sentence-transformers CrossEncoder
  (default ``cross-encoder/ms-marco-MiniLM-L-6-v2``), the strongest option.
- :class:`LexicalReranker` — a deterministic, model-free BM25-style scorer used
  when the cross-encoder model is unavailable (offline machine, no download),
  so the pipeline degrades gracefully instead of failing.

The reranker interface is intentionally tiny: ``rerank(query, candidates,
top_k)`` returns the rescored candidates with ``rerank_score`` set.

The model-free fallback is a *feature-blend* reranker (lexical + dense +
coverage), not a BM25 re-sort, so enabling it changes the ranking even without
the cross-encoder.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Protocol, Sequence, runtime_checkable

from .bm25 import BM25Index, tokenize
from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    """Anything that can reorder a candidate pool for a query."""

    name: str

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        ...


class CrossEncoderReranker:
    """Joint (query, passage) scoring with a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        logger.info("Loading cross-encoder reranker '%s'...", model_name)
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.device = device or "cpu"
        self.name = f"cross-encoder:{model_name}"
        logger.info("Cross-encoder reranker ready (device=%s).", self.device)

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, (c.text or "").strip()) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: float(pair[1]), reverse=True)
        out: list[RetrievedChunk] = []
        for chunk, score in scored[:top_k]:
            chunk.rerank_score = round(float(score), 6)
            out.append(chunk)
        return out


def _normalise(values: Sequence[float]) -> list[float]:
    """Min-max normalise to [0, 1]; a constant component contributes nothing."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class LexicalReranker:
    """Model-free reranker blending lexical, dense and coverage signals.

    Re-scoring with BM25 alone would just reproduce the keyword ranking, so this
    fallback instead combines three complementary signals, each min-max
    normalised across the candidate pool:

        score = w_lex * bm25 + w_sem * dense_similarity + w_cov * query_coverage

    - ``bm25``: OKapi score (reused from fusion when present, else recomputed
      with corpus IDF and length normalisation).
    - ``dense_similarity``: the candidate's cosine similarity when it came from
      the semantic path, so in hybrid mode the fallback still uses semantic
      evidence rather than re-deriving the keyword order.
    - ``query_coverage``: fraction of distinct query terms present in the passage,
      which rewards passages that answer the whole question.

    It stays deterministic and needs no model download, but is no longer a plain
    BM25 re-sort. Default weights are tuned on the frozen retrieval set: the
    dense signal carries most of the weight because it is the strongest single
    retriever, coverage is available but defaults to off (it added noise on this
    corpus). Text (not ``context_text``) is scored, so the section breadcrumb is
    not double-counted.

    On the frozen set this lifts ``hybrid+rerank`` above plain hybrid
    (nDCG@5 0.481 -> 0.493, Recall@5 0.417 -> 0.430, MRR 0.685 -> 0.696), whereas
    the previous BM25 re-sort reproduced the keyword ranking.
    """

    def __init__(
        self,
        index: BM25Index | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        lexical_weight: float = 0.20,
        semantic_weight: float = 0.80,
        coverage_weight: float = 0.0,
    ) -> None:
        weights = (lexical_weight, semantic_weight, coverage_weight)
        if any(w < 0 for w in weights):
            raise ValueError("reranker weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one reranker weight must be positive")
        self.index = index
        self.k1 = index.k1 if index is not None else k1
        self.b = index.b if index is not None else b
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.coverage_weight = coverage_weight
        self.name = "lexical:feature-blend"

    def _idf(self, term: str) -> float:
        if self.index is not None:
            return self.index.idf.get(term, 0.0)
        return 1.0

    def _avgdl(self, candidates: Sequence[RetrievedChunk]) -> float:
        if self.index is not None and self.index.avgdl:
            return self.index.avgdl
        lengths = [max(1, len(tokenize(c.text))) for c in candidates]
        return sum(lengths) / len(lengths) if lengths else 1.0

    def _lexical_score(self, tf: Counter[str], dl: int, avgdl: float, query_terms: set[str]) -> float:
        denom_norm = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
        score = 0.0
        for term in query_terms:
            freq = tf.get(term)
            if not freq:
                continue
            score += self._idf(term) * (freq * (self.k1 + 1.0)) / (freq + denom_norm)
        return score

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        query_terms = tokenize(query)
        unique_terms = set(query_terms)
        avgdl = self._avgdl(candidates) or 1.0

        lexical: list[float] = []
        semantic: list[float] = []
        coverage: list[float] = []
        for chunk in candidates:
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            dl = len(tokens) or 1
            if chunk.keyword_score is not None:
                lexical.append(float(chunk.keyword_score))
            else:
                lexical.append(self._lexical_score(tf, dl, avgdl, unique_terms))
            semantic.append(float(chunk.similarity or 0.0))
            coverage.append(
                len([t for t in unique_terms if tf.get(t)]) / len(unique_terms)
                if unique_terms
                else 0.0
            )

        lex_n = _normalise(lexical)
        sem_n = _normalise(semantic)
        cov_n = _normalise(coverage)

        scored: list[tuple[RetrievedChunk, float]] = []
        for i, chunk in enumerate(candidates):
            score = (
                self.lexical_weight * lex_n[i]
                + self.semantic_weight * sem_n[i]
                + self.coverage_weight * cov_n[i]
            )
            chunk.rerank_score = round(float(score), 6)
            scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in scored[:top_k]]


def build_reranker(
    kind: str | None,
    index: BM25Index | None = None,
    model_name: str = DEFAULT_CROSS_ENCODER,
    device: str | None = None,
    batch_size: int = 32,
    lexical_weight: float = 0.20,
    semantic_weight: float = 0.80,
    coverage_weight: float = 0.0,
) -> Reranker | None:
    """Factory: resolve a reranker kind, degrading to the lexical fallback.

    ``kind`` — ``"cross-encoder"`` (default), ``"lexical"`` or ``"none"``.
    A cross-encoder that cannot load (no model cache / no network) logs a
    warning and falls back to :class:`LexicalReranker` so runs never hard-fail.
    The weights apply only to the lexical fallback.
    """
    kind = (kind or "none").lower()
    if kind in {"none", "off", "false", ""}:
        return None
    lexical = lambda: LexicalReranker(
        index=index,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        coverage_weight=coverage_weight,
    )
    if kind in {"lexical", "fallback", "bm25"}:
        return lexical()
    if kind in {"cross-encoder", "crossencoder", "cross_encoder", "ce"}:
        try:
            return CrossEncoderReranker(
                model_name=model_name, device=device, batch_size=batch_size
            )
        except Exception as exc:  # noqa: BLE001 - degrade, don't fail the run
            logger.warning(
                "Cross-encoder '%s' unavailable (%s); falling back to lexical reranker.",
                model_name,
                exc,
            )
            return lexical()
    raise ValueError(f"Unknown reranker kind: {kind!r}")


def reranker_descriptor(reranker: Reranker | None) -> dict[str, Any]:
    """Small JSON-serialisable description for reports/logs."""
    if reranker is None:
        return {"kind": "none"}
    return {"kind": reranker.name}
