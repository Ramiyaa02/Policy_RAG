"""Phase 2 — Embedding generation (SRS FR-005).

"The system shall generate vector representations for searchable document
chunks. The embedding model shall be configurable."

The model name is a constructor argument (and a CLI flag / config value in
run_phase2.py), so the baseline all-MiniLM-L6-v2 can be swapped without
touching the pipeline. Vectors are L2-normalized so cosine similarity reduces
to a dot product.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Sentence-transformers wrapper; model is configurable per FR-005."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        # Imported lazily so tests and non-embedding code paths do not need it.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model '%s'...", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model loaded (device=%s).", self.device)

    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return str(getattr(self.model, "device", "cpu"))

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts -> (n, dim) float32 array."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query -> (dim,) float32 array."""
        return self.embed_texts([text])[0]

    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Model descriptor stored alongside the index for reproducibility."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "normalize": self.normalize,
        }
