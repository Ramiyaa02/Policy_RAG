"""Phase 4 — Agent 3: Evidence Retrieval / plan executor (SRS FR-008, FR-014).

Runs each :class:`PlanStep` through the Phase 3 :class:`HybridRetriever`
(metadata filtering + hybrid fusion + optional reranking) and collects the
results into :class:`EvidenceBundle`s, one per step. A global chunk budget and
cross-step de-duplication keep the evidence handed to the LLM bounded (FR-009,
NFR-005).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

from .schema import EvidenceBundle, RetrievalPlan

logger = logging.getLogger(__name__)


class PlanExecutor:
    """Agent 3 — executes a :class:`RetrievalPlan` and returns evidence bundles."""

    def __init__(
        self,
        retriever: Any,
        mode: str = "hybrid",
        fusion: str = "rrf",
        candidate_k: int = 20,
        rerank: bool = True,
        max_chunks: int = 80,
    ) -> None:
        self.retriever = retriever
        self.mode = mode
        self.fusion = fusion
        self.candidate_k = candidate_k
        self.rerank = rerank
        self.max_chunks = max_chunks

    # ------------------------------------------------------------------

    def execute(self, plan: RetrievalPlan) -> list[EvidenceBundle]:
        bundles: list[EvidenceBundle] = []
        seen: set[str] = set()
        retrieved_total = 0

        for step in plan.steps:
            if retrieved_total >= self.max_chunks:
                logger.info("Evidence budget reached (%d chunks).", self.max_chunks)
                break
            remaining = self.max_chunks - retrieved_total
            top_k = min(step.top_k, remaining)
            chunks = self.retriever.retrieve(
                step.query,
                k=top_k,
                mode=self.mode,
                candidate_k=max(self.candidate_k, top_k),
                fusion=self.fusion,
                rerank=self.rerank,
                product=step.product,
                document_id=step.document_id,
                chunk_type=step.chunk_type,
            )
            fresh = []
            for chunk in chunks:
                cid = getattr(chunk, "chunk_id", "")
                if cid in seen:
                    continue
                seen.add(cid)
                fresh.append(chunk)
            retrieved_total += len(fresh)
            bundles.append(EvidenceBundle(step=step, chunks=fresh))
        return bundles

    # ------------------------------------------------------------------

    @staticmethod
    def flatten(bundles: Iterable[EvidenceBundle]) -> list[Any]:
        """All unique chunks across bundles, in step order."""
        ordered: list[Any] = []
        seen: set[str] = set()
        for bundle in bundles:
            for chunk in bundle.chunks:
                cid = getattr(chunk, "chunk_id", "")
                if cid and cid in seen:
                    continue
                seen.add(cid)
                ordered.append(chunk)
        return ordered

    @staticmethod
    def group_by_product(bundles: Sequence[EvidenceBundle]) -> dict[str, list[Any]]:
        """Evidence grouped by the product each bundle's step targeted."""
        grouped: dict[str, list[Any]] = {}
        for bundle in bundles:
            product = bundle.step.product or "(unspecified)"
            bucket = grouped.setdefault(product, [])
            for chunk in bundle.chunks:
                if chunk not in bucket:
                    bucket.append(chunk)
        return grouped


def timed(fn, *args, **kwargs) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_ms)`` (NFR-004)."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    return result, round((time.time() - t0) * 1000, 1)
