"""Phase 2 — Vector storage and search (SRS FR-006).

"The system shall store document embeddings in a vector-search system. The
vector index shall support: similarity search, top-k retrieval, metadata
filtering, document-level identification."

Implemented on ChromaDB's PersistentClient:
- cosine similarity space
- top-k search with optional ``where`` metadata filters
- every entry carries document-level identification (document_id, product,
  pages, clause ids) as scalar metadata
- upsert by deterministic chunk_id makes (re)building idempotent

Chroma metadata values must be scalars, so list-valued fields (clause ids) are
stored as comma-joined strings.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "policy_chunks"


@dataclass
class IndexStats:
    """Basic index descriptors persisted next to the store."""

    collection: str
    count: int
    embedding_model: str
    dimension: int
    documents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyVectorStore:
    """ChromaDB-backed store for policy chunks."""

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection: str = DEFAULT_COLLECTION,
        distance: str = "cosine",
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                "chromadb is required for the vector store. "
                "Install with: pip install chromadb"
            ) from exc

        self.persist_dir = persist_dir
        self.collection_name = collection
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": distance},
        )
        logger.info(
            "Vector store ready: %s (collection '%s', %d entries)",
            persist_dir,
            collection,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Delete and recreate the collection (full rebuild)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=self.collection.metadata,
        )
        logger.info("Collection '%s' reset.", self.collection_name)

    def upsert_chunks(
        self,
        chunks: list[Any],
        embeddings: np.ndarray,
        batch_size: int = 256,
    ) -> int:
        """Upsert chunks with their embedding vectors. Returns number stored.

        ``chunks`` are chunking.chunker.Chunk objects (duck-typed: need
        chunk_id, context_text, vector_store_metadata()).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return 0

        vectors = np.asarray(embeddings, dtype=np.float32)
        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_vecs = vectors[start : start + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.context_text for c in batch],
                embeddings=batch_vecs.tolist(),
                metadatas=[c.vector_store_metadata() for c in batch],
            )
            written += len(batch)
        logger.info("Upserted %d chunks into '%s'.", written, self.collection_name)
        return written

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k similarity search; ``where`` is a Chroma metadata filter."""
        if self.collection.count() == 0:
            return []
        vector = np.asarray(query_embedding, dtype=np.float32).tolist()
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        for i in range(len(docs)):
            distance = float(dists[i]) if i < len(dists) else 1.0
            hits.append(
                {
                    "chunk_id": ids[i] if i < len(ids) else "",
                    "text": docs[i],
                    "distance": distance,
                    # cosine distance in [0, 2]; similarity = 1 - distance
                    "similarity": 1.0 - distance,
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return hits

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self.collection.count()

    def stats(self, embedding_model: str = "", dimension: int = 0) -> IndexStats:
        """Aggregate index statistics (document-level identification)."""
        documents: set[str] = set()
        n = self.count()
        offset = 0
        batch = 256
        while offset < n:
            got = self.collection.get(
                include=["metadatas"], limit=batch, offset=offset
            )
            metas = got.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                if m.get("document_id"):
                    documents.add(m["document_id"])
            offset += len(metas)
        return IndexStats(
            collection=self.collection_name,
            count=n,
            embedding_model=embedding_model,
            dimension=dimension,
            documents=sorted(documents),
        )

    def save_stats(self, path: str, embedding_model: str = "", dimension: int = 0) -> IndexStats:
        """Persist index stats JSON (used by run_phase2.py build)."""
        stats = self.stats(embedding_model=embedding_model, dimension=dimension)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)
        return stats
