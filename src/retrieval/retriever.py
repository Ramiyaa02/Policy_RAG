"""Phase 2 — Evidence retrieval (SRS FR-008, semantic portion).

Combines the configurable embedder with the ChromaDB store to answer
"give me the top-k policy passages for this query", optionally filtered by
document/product metadata. Hybrid keyword retrieval and reranking are Phase 3.
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
        )
