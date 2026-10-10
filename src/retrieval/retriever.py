"""Phase 2/3 — Evidence retrieval (SRS FR-008, semantic portion).

Combines the configurable embedder with the ChromaDB store to answer
"give me the top-k policy passages for this query", optionally filtered by
document/product metadata. The semantic path is Phase 2; Phase 3 adds keyword
retrieval, fusion and reranking on top (see src/retrieval/hybrid.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved passage with score and traceability metadata."""

    chunk_id: str
    text: str
    similarity: float
    distance: float
    document_id: str
    filename: str
    product: str
    insurer: str
    uin: str
    document_type: str
    page_start: int
    page_end: int
    section: str
    subsection: str
    clause_ids: list[str] = field(default_factory=list)
    chunk_type: str = "prose"
    # --- Phase 3 retrieval-improvement annotations (all optional) ---
    keyword_score: float | None = None   # BM25 raw score
    fused_score: float | None = None     # score after semantic/keyword fusion
    rerank_score: float | None = None    # cross-encoder / lexical rerank score
    sources: list[str] = field(default_factory=list)  # e.g. ["semantic", "keyword"]

    def citation_label(self) -> str:
        """Human-readable source label used in the answer's citation list."""
        parts = [self.product or self.filename or self.document_id]
        pages = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        parts.append(pages)
        if self.section:
            parts.append(self.section)
        if self.clause_ids:
            parts.append("clauses " + ", ".join(self.clause_ids[:4]))
        return " — ".join(parts)

    @classmethod
    def from_chunk(
        cls,
        chunk: Any,
        similarity: float = 0.0,
        distance: float = 1.0,
        source: str = "keyword",
    ) -> "RetrievedChunk":
        """Build a RetrievedChunk from a chunker.Chunk or a chunks.jsonl dict.

        Used by the Phase 3 keyword (BM25) path, which has full chunk records
        but no vector similarity, so callers supply the score.
        """
        get = chunk.get if isinstance(chunk, dict) else lambda k, d=None: getattr(chunk, k, d)
        clause_ids_raw = get("clause_ids", []) or []
        if isinstance(clause_ids_raw, str):
            clause_ids = [c for c in clause_ids_raw.split(",") if c]
        else:
            clause_ids = list(clause_ids_raw)
        return cls(
            chunk_id=get("chunk_id", "") or "",
            text=get("text", "") or "",
            similarity=similarity,
            distance=distance,
            document_id=get("document_id", "") or "",
            filename=get("filename", "") or "",
            product=get("product", "") or "",
            insurer=get("insurer", "") or "",
            uin=get("uin", "") or "",
            document_type=get("document_type", "") or "",
            page_start=int(get("page_start", 0) or 0),
            page_end=int(get("page_end", 0) or 0),
            section=get("section", "") or "",
            subsection=get("subsection", "") or "",
            clause_ids=clause_ids,
            chunk_type=get("chunk_type", "prose") or "prose",
            sources=[source],
        )

    def merge_scores_from(self, other: "RetrievedChunk") -> None:
        """Fold another copy of the same chunk into this one (dedup helper)."""
        self.similarity = max(self.similarity, other.similarity)
        self.distance = min(self.distance, other.distance)
        if other.keyword_score is not None:
            self.keyword_score = max(self.keyword_score or 0.0, other.keyword_score)
        if other.fused_score is not None:
            self.fused_score = max(self.fused_score or 0.0, other.fused_score)
        for src in other.sources:
            if src not in self.sources:
                self.sources.append(src)


class Retriever:
    """Embeds queries and returns top-k chunks from the vector store."""

    def __init__(self, embedder: Any, store: Any) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        k: int = 5,
        document_id: str | None = None,
        product: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k semantic search with optional metadata filters."""
        where = self._build_where(document_id=document_id, product=product, chunk_type=chunk_type)
        query_vector = self.embedder.embed_query(query)
        hits = self.store.query(query_vector, k=k, where=where)
        return [self._to_retrieved(hit) for hit in hits]

    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(
        document_id: str | None = None,
        product: str | None = None,
        chunk_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Chroma ``where`` filter; equality clauses ANDed via $and."""
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

    @staticmethod
    def _to_retrieved(hit: dict[str, Any]) -> RetrievedChunk:
        meta = hit.get("metadata") or {}
        clause_ids_raw = meta.get("clause_ids") or ""
        distance = float(hit.get("distance", 0.0) or 0.0)
        # Store.query() provides similarity; fall back to 1 - cosine distance
        # for raw result dicts that carry only the distance.
        similarity = float(hit.get("similarity", 1.0 - distance))
        return RetrievedChunk(
            chunk_id=hit.get("chunk_id", ""),
            text=hit.get("text", ""),
            similarity=similarity,
            distance=distance,
            document_id=meta.get("document_id", ""),
            filename=meta.get("filename", ""),
            product=meta.get("product", ""),
            insurer=meta.get("insurer", ""),
            uin=meta.get("uin", ""),
            document_type=meta.get("document_type", ""),
            page_start=int(meta.get("page_start", 0) or 0),
            page_end=int(meta.get("page_end", 0) or 0),
            section=meta.get("section", ""),
            subsection=meta.get("subsection", ""),
            clause_ids=[c for c in clause_ids_raw.split(",") if c],
            chunk_type=meta.get("chunk_type", "prose"),
            sources=["semantic"],
        )
