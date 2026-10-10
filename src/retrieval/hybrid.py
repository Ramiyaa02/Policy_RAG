"""Phase 3 — Hybrid retrieval and fusion (SRS FR-008, Phase 3 scope).

Combines the Phase 2 dense retriever with the BM25 keyword index and fuses the
two ranked lists, then optionally reranks the fused pool:

    query ──► semantic top-N ──┐
                               ├─► fusion (RRF / weighted) ──► rerank ──► top-k
    query ──► keyword  top-N ──┘

Fusion modes
------------
- ``rrf`` (default): Reciprocal Rank Fusion, ``Σ weight / (rrf_k + rank)``.
  Rank-based, so it needs no score scaling between the two retrievers — the
  robust default when dense similarities and BM25 scores are not comparable.
- ``weighted``: min-max normalise each list then combine as
  ``alpha * semantic + (1 - alpha) * keyword``.

Metadata filtering (document_id / product / chunk_type) is applied to *both*
retrieval paths with one filter dict, satisfying "metadata filtering" as an
explicit Phase 3 item rather than a dense-only extra.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from .bm25 import BM25Index
from .reranker import Reranker
from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)

VALID_MODES = ("semantic", "keyword", "hybrid")
VALID_FUSIONS = ("rrf", "weighted")


def _build_where(
    document_id: str | None = None,
    product: str | None = None,
    chunk_type: str | None = None,
) -> dict[str, Any] | None:
    """Metadata filter understood by both the Chroma store and the BM25 index."""
    clauses: list[dict[str, Any]] = []
    if document_id:
        clauses.append({"document_id": {"$eq": document_id}})
    if product:
        clauses.append({"product": {"$eq": product}})
    if chunk_type:
        clauses.append({"chunk_type": {"$eq": chunk_type}})
    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


class HybridRetriever:
    """Semantic + keyword retrieval with rank fusion and optional reranking."""

    def __init__(
        self,
        semantic: Any | None,
        bm25: BM25Index | None,
        chunks_by_id: dict[str, Any] | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.semantic = semantic
        self.bm25 = bm25
        self.chunks_by_id = chunks_by_id or {}
        self.reranker = reranker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        candidate_k: int = 20,
        fusion: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        rerank: bool = False,
        document_id: str | None = None,
        product: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top ``k`` chunks for ``query``.

        ``candidate_k`` is the per-retriever pool size that gets fused and, when
        ``rerank`` is set, reranked down to ``k``. Reranking triggers whenever a
        reranker is configured and enabled — not only in ``hybrid`` mode — so it
        can be evaluated on top of any single retriever too.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if fusion not in VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {VALID_FUSIONS}, got {fusion!r}")

        where = _build_where(document_id=document_id, product=product, chunk_type=chunk_type)
        pool = max(candidate_k, k)

        semantic_hits: list[RetrievedChunk] = []
        keyword_hits: list[RetrievedChunk] = []

        if mode in ("semantic", "hybrid") and self.semantic is not None:
            semantic_hits = self.semantic.retrieve(
                query, k=pool, document_id=document_id, product=product, chunk_type=chunk_type
            )
        if mode in ("keyword", "hybrid") and self.bm25 is not None:
            keyword_hits = self._keyword_hits(query, pool, where)

        if mode == "semantic":
            ranked = semantic_hits
        elif mode == "keyword":
            ranked = keyword_hits
        else:
            ranked = self._fuse([(semantic_hits, 1.0), (keyword_hits, 1.0)], fusion, alpha, rrf_k)

        ranked = self._dedupe(ranked)[:pool]
        if rerank and self.reranker is not None:
            return self.reranker.rerank(query, ranked, top_k=k)
        return ranked[:k]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _keyword_hits(self, query: str, pool: int, where: dict[str, Any] | None) -> list[RetrievedChunk]:
        """BM25 hits as RetrievedChunks carrying ``keyword_score``."""
        hits: list[RetrievedChunk] = []
        for chunk_id, score in self.bm25.search(query, k=pool, where=where):
            record = self.chunks_by_id.get(chunk_id)
            if record is None:
                continue
            chunk = RetrievedChunk.from_chunk(record, source="keyword")
            chunk.keyword_score = score
            hits.append(chunk)
        return hits

    def _fuse(
        self,
        ranked_lists: Sequence[tuple[list[RetrievedChunk], float]],
        fusion: str,
        alpha: float,
        rrf_k: int,
    ) -> list[RetrievedChunk]:
        by_id: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}

        if fusion == "rrf":
            for hits, weight in ranked_lists:
                for rank, chunk in enumerate(hits, start=1):
                    key = chunk.chunk_id
                    if key not in by_id:
                        by_id[key] = chunk
                    else:
                        by_id[key].merge_scores_from(chunk)
                    scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
        else:  # weighted
            # ranked_lists[0] is semantic (weight alpha); [1] keyword (1 - alpha).
            weighted = [
                (ranked_lists[0][0], alpha) if ranked_lists else ([], 0.0),
                (ranked_lists[1][0] if len(ranked_lists) > 1 else [], 1.0 - alpha),
            ]
            for hits, weight in weighted:
                normalised = _min_max(hits)
                for chunk, norm_score in zip(hits, normalised):
                    key = chunk.chunk_id
                    if key not in by_id:
                        by_id[key] = chunk
                    else:
                        by_id[key].merge_scores_from(chunk)
                    scores[key] = scores.get(key, 0.0) + weight * norm_score

        result: list[RetrievedChunk] = []
        for key, chunk in by_id.items():
            chunk.fused_score = round(scores.get(key, 0.0), 6)
            result.append(chunk)
        result.sort(key=lambda c: c.fused_score or 0.0, reverse=True)
        return result

    @staticmethod
    def _dedupe(ranked: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
        """Collapse duplicate chunk ids, preferring order and merging sources."""
        seen: dict[str, RetrievedChunk] = {}
        ordered: list[RetrievedChunk] = []
        for chunk in ranked:
            if chunk.chunk_id in seen:
                seen[chunk.chunk_id].merge_scores_from(chunk)
                continue
            seen[chunk.chunk_id] = chunk
            ordered.append(chunk)
        return ordered


def _min_max(hits: Sequence[RetrievedChunk]) -> list[float]:
    """Min-max normalise a list's scores (similarity or BM25) to [0, 1]."""
    if not hits:
        return []
    values = [
        h.similarity if h.keyword_score is None else h.keyword_score for h in hits
    ]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
