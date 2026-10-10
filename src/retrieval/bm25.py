"""Phase 3 — Keyword retrieval via BM25 (SRS FR-008, keyword portion).

Semantic (dense) retrieval misses exact terms users type verbatim — clause
numbers, drug/procedure names, "AYUSH", "PED", defined terms. BM25 gives the
lexical half of hybrid retrieval, so the two rankings can be fused (RRF) or
reranked (see src/retrieval/hybrid.py and reranker.py).

Implemented in-house on pure Python + a small stopword list: no extra
dependency beyond numpy, and it works on any machine that can already build
the Phase 2 index. Metadata filtering accepts the same simple equality form as
the Chroma ``where`` filter, so hybrid retrieval can apply one filter to both
the dense and lexical paths (FR-006 / Phase 3 "metadata filtering").

The index is JSON-serialisable (``to_dict`` / ``from_dict`` / ``save`` /
``load``) so ``run_phase3.py build`` can cache it under data/bm25/.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small English stopword set. Policy wordings are formulaic, so dropping these
# keeps the posting lists short without losing domain signal.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have if in into is it its of on or
    that the their there these this to was were will with you your we our us they
    them he she his her not no any all such than then when where which who whom""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer with a light stopword filter.

    Single characters and stopwords are dropped; everything else (including
    domain terms such as ``ayush``, ``ped``, ``icu``, ``cataract``) is kept.
    """
    return [
        tok
        for tok in _TOKEN_RE.findall((text or "").lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


def _matches_filter(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """True when ``meta`` satisfies a simple equality / ``$eq`` filter."""
    if not where:
        return True
    # Normalise {"$and": [...]} or a bare clause dict into a list of clauses.
    if "$and" in where:
        clauses = where["$and"]
    elif "$or" in where:  # pragma: no cover - not used by phase 3
        clauses = where["$or"]
        return any(_matches_filter(meta, c) for c in clauses)
    else:
        clauses = [where]
    for clause in clauses:
        for key, cond in clause.items():
            expected = cond.get("$eq") if isinstance(cond, dict) else cond
            if str(meta.get(key, "")) != str(expected):
                return False
    return True


class BM25Index:
    """Okapi BM25 over a list of documents with scalar metadata.

    Parameters
    ----------
    k1, b:
        Standard BM25 saturation and length-normalisation constants.
    tokenizer:
        Callable text -> list[str]; defaults to :func:`tokenize`.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer

        self.doc_ids: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.doc_len: list[int] = []
        self.term_freqs: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> "BM25Index":
        """Index ``texts``. ``metadatas[i]`` describes ``doc_ids[i]``."""
        if not (len(doc_ids) == len(texts)):
            raise ValueError("doc_ids/texts length mismatch")
        if metadatas is not None and len(metadatas) != len(doc_ids):
            raise ValueError("metadatas/doc_ids length mismatch")

        self.doc_ids = list(doc_ids)
        self.metadatas = [dict(m) for m in metadatas] if metadatas else [{} for _ in doc_ids]
        self.term_freqs = []
        self.doc_len = []
        df: Counter[str] = Counter()

        for text in texts:
            tokens = self.tokenizer(text)
            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))
            self.doc_len.append(len(tokens))
            df.update(tf.keys())

        self.n_docs = len(self.doc_ids)
        self.df = dict(df)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        # BM25+ style IDF (always positive), so a term in every document still
        # contributes a small amount instead of negative or zero.
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }
        logger.info(
            "BM25 built: %d docs, %d terms, avgdl=%.1f",
            self.n_docs,
            len(self.idf),
            self.avgdl,
        )
        return self

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def score_document(self, query_terms: Sequence[str], index: int) -> float:
        """BM25 score of one document for a tokenised query."""
        if index >= self.n_docs:
            return 0.0
        tf = self.term_freqs[index]
        dl = self.doc_len[index] or 1
        denom_norm = self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
        score = 0.0
        for term in query_terms:
            freq = tf.get(term)
            if not freq:
                continue
            idf = self.idf.get(term, 0.0)
            score += idf * (freq * (self.k1 + 1.0)) / (freq + denom_norm)
        return score

    def search(
        self,
        query: str,
        k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-k ``(chunk_id, score)`` pairs, highest score first.

        Documents whose metadata does not satisfy ``where`` are excluded before
        scoring, so filters narrow the lexical search space exactly like they
        narrow the Chroma query.
        """
        if self.n_docs == 0 or k <= 0:
            return []
        query_terms = self.tokenizer(query)
        if not query_terms:
            return []

        scored: list[tuple[str, float]] = []
        for i, meta in enumerate(self.metadatas):
            if not _matches_filter(meta, where):
                continue
            score = self.score_document(query_terms, i)
            if score > 0.0:
                scored.append((self.doc_ids[i], score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Introspection / persistence
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "documents": self.n_docs,
            "unique_terms": len(self.idf),
            "avg_doc_length": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "metadatas": self.metadatas,
            "doc_len": self.doc_len,
            "term_freqs": self.term_freqs,
            "avgdl": self.avgdl,
            "n_docs": self.n_docs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BM25Index":
        index = cls(k1=float(payload.get("k1", 1.5)), b=float(payload.get("b", 0.75)))
        index.doc_ids = list(payload.get("doc_ids", []))
        index.metadatas = [dict(m) for m in payload.get("metadatas", [])]
        index.doc_len = list(payload.get("doc_len", []))
        index.term_freqs = [dict(tf) for tf in payload.get("term_freqs", [])]
        index.avgdl = float(payload.get("avgdl", 0.0))
        index.n_docs = int(payload.get("n_docs", len(index.doc_ids)))
        df: Counter[str] = Counter()
        for tf in index.term_freqs:
            df.update(tf.keys())
        index.df = dict(df)
        index.idf = {
            term: math.log(1.0 + (index.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in index.df.items()
        }
        return index

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        logger.info("BM25 index saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def load_chunk_corpus(path: str) -> list[dict[str, Any]]:
    """Read the Phase 2 chunk corpus (data/chunks/chunks.jsonl) into dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chunk corpus not found: {path}. "
            "Run 'python run_phase2.py build' first."
        )
    chunks: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_bm25_from_chunks(
    chunks: Iterable[dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """Construct a BM25 index over chunk records (embedded text + metadata).

    The embedded text is ``context_text`` (section breadcrumb + body) to match
    what the dense retriever sees; fall back to ``text`` when absent.
    """
    rows = list(chunks)
    texts = [row.get("context_text") or row.get("text", "") for row in rows]
    doc_ids = [row.get("chunk_id", "") for row in rows]
    metadatas = [
        {
            "document_id": row.get("document_id", ""),
            "product": row.get("product", ""),
            "chunk_type": row.get("chunk_type", "prose"),
        }
        for row in rows
    ]
    return BM25Index(k1=k1, b=b).build(doc_ids, texts, metadatas)
