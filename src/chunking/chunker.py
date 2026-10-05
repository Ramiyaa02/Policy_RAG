"""Phase 2 — Structure-aware chunking (SRS FR-004).

Turns the canonical normalized JSON produced by Phase 1 into retrieval units
(chunks) while preserving semantic and structural boundaries:

- Headers, footers and page numbers are excluded (Phase 1 marks them with
  ``include_in_chunk_text=False``).
- Every chunk records the document, product, section/subsection, clause ids and
  the page span it came from, so answers stay traceable (SRS FR-004).
- Clauses are kept intact where possible: a new chunk is preferred at a clause
  or heading boundary rather than inside a clause (FR-004 "avoid unnecessary
  fragmentation of clauses").
- Oversized blocks (> max_chars, the 243 blocks flagged as deferred in the
  Phase 1 audit) are split at paragraph, then sentence, boundaries.
- Tables become their own chunks (tab-separated text embeds poorly when mixed
  with prose), tagged ``chunk_type="table"``.

Phase 2 STOP boundary — no embeddings, vector store, retrieval, or LLM here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Blocks that carry no body content for retrieval.
_EXCLUDED_TYPES = {"header", "footer"}

# A lettered clause whose title is an all-caps run — "B. DEFINITIONS ...",
# "C. OPERATIVE CLAUSES: ..." — acts as a section marker. The marker is the
# leading "X." plus at least one all-caps word of 2+ letters; lowercase text
# after the caps run (marketing copy) does not extend or invalidate it.
_SECTION_LIKE_CLAUSE = re.compile(r"^[A-Z]\.\s+[A-Z][A-Z&/\-']+")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

DEFAULTS = {
    "target_chars": 900,   # ~220 word-piece tokens, inside MiniLM's 256 window
    "max_chars": 1300,
    "min_chars": 120,
}


@dataclass
class Chunk:
    """One retrieval unit with full source traceability."""

    chunk_id: str
    document_id: str
    text: str
    context_text: str  # what gets embedded: section breadcrumb + text
    page_start: int
    page_end: int
    section: str | None
    subsection: str | None
    clause_ids: list[str] = field(default_factory=list)
    chunk_type: str = "prose"  # prose | table
    char_count: int = 0
    word_count: int = 0
    # Scalars only (Chroma metadata cannot hold lists).
    filename: str | None = None
    insurer: str | None = None
    product: str | None = None
    uin: str | None = None
    document_type: str | None = None
    source_path: str | None = None
    embedding_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def vector_store_metadata(self) -> dict[str, Any]:
        """Flattened scalar metadata suitable for a ChromaDB entry."""
        return {
            "document_id": self.document_id,
            "filename": self.filename or "",
            "insurer": self.insurer or "",
            "product": self.product or "",
            "uin": self.uin or "",
            "document_type": self.document_type or "",
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section or "",
            "subsection": self.subsection or "",
            "clause_ids": ",".join(self.clause_ids),
            "chunk_type": self.chunk_type,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "embedding_model": self.embedding_model or "",
        }


class Chunker:
    """Structure-aware chunker over Phase 1 normalized JSON."""

    def __init__(
        self,
        target_chars: int = DEFAULTS["target_chars"],
        max_chars: int = DEFAULTS["max_chars"],
        min_chars: int = DEFAULTS["min_chars"],
        context_prefix: bool = True,
    ) -> None:
        if not 0 < target_chars <= max_chars:
            raise ValueError("target_chars must be in (0, max_chars]")
        if min_chars < 0 or min_chars > target_chars:
            raise ValueError("min_chars must be within [0, target_chars]")
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.context_prefix = context_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, normalized_doc: dict[str, Any]) -> list[Chunk]:
        """Chunk one normalized document dict from data/normalized/*.json."""
        doc_id = normalized_doc.get("document_id", "DOC-UNKNOWN")
        meta = normalized_doc.get("metadata") or {}

        ordered_blocks = self._ordered_body_blocks(normalized_doc)
        raw_chunks = self._assemble(doc_id, ordered_blocks)
        chunks = self._finalize(doc_id, normalized_doc, meta, raw_chunks)

        logger.info(
            "%s: %d body blocks -> %d chunks",
            doc_id,
            len(ordered_blocks),
            len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Block selection and ordering
    # ------------------------------------------------------------------

    def _ordered_body_blocks(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Reading-order body blocks with headers/footers/page numbers removed."""
        blocks: list[dict[str, Any]] = []
        for page in doc.get("pages", []):
            page_blocks = page.get("blocks", [])
            ordered = sorted(page_blocks, key=lambda b: b.get("reading_order", 0))
            for block in ordered:
                if block.get("type") in _EXCLUDED_TYPES:
                    continue
                if block.get("include_in_chunk_text") is False:
                    continue
                text = (block.get("text") or "").strip()
                if not text:
                    continue
                blocks.append(block)
        return blocks

    # ------------------------------------------------------------------
    # Section / subsection tracking
    # ------------------------------------------------------------------

    def _is_section_marker(self, block: dict[str, Any]) -> bool:
        if block.get("type") == "heading" and block.get("subtype") == "section_heading":
            return True
        if block.get("type") == "clause":
            return bool(_SECTION_LIKE_CLAUSE.match((block.get("text") or "").strip()))
        return False

    def _is_subsection_marker(self, block: dict[str, Any]) -> bool:
        if block.get("type") != "heading":
            return False
        if block.get("subtype") == "subsection_heading_l2":
            return True
        level = block.get("heading_level")
        return isinstance(level, int) and level >= 3

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(
        self, doc_id: str, blocks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Group ordered blocks into raw chunk payloads."""
        raw: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        section: str | None = None
        subsection: str | None = None
        # True while the current chunk holds only heading/marker text — a
        # following marker then updates the state instead of breaking again,
        # so stacked headings lead one chunk instead of leaving 30-char
        # heading-only chunks behind.
        current_marker_only = False

        def flush() -> None:
            nonlocal current
            if current and current["texts"]:
                raw.append(current)
            current = None

        for block in blocks:
            text = (block.get("text") or "").strip()
            btype = block.get("type", "paragraph")
            is_table = btype == "table"
            chunk_type = "table" if is_table else "prose"
            page = int(block.get("page_number") or 0)
            clause_id = block.get("clause_id")
            is_marker = self._is_section_marker(block) or self._is_subsection_marker(block)

            if is_marker:
                # Break only when real body content would be orphaned; stacked
                # markers accumulate into the chunk they will lead.
                if current is not None and not current_marker_only:
                    flush()
                if self._is_section_marker(block):
                    section = self._section_title(text)
                    subsection = None
                else:
                    subsection = text
                current_marker_only = True
            else:
                current_marker_only = False

            # Decide whether this block must start a fresh chunk:
            # - type change (table vs prose)
            # - a clause boundary once the target size is reached
            # - the block would push the chunk past max_chars
            breaks = (
                current is not None
                and (
                    current["chunk_type"] != chunk_type
                    or (
                        btype == "clause"
                        and current["char_count"] >= self.target_chars
                    )
                    or current["char_count"] + len(text) > self.max_chars
                )
            )
            if breaks:
                flush()

            if current is None:
                current = {
                    "chunk_type": chunk_type,
                    "section": section,
                    "subsection": subsection,
                    "texts": [],
                    "page_start": page,
                    "page_end": page,
                    "clause_ids": [],
                    "char_count": 0,
                }

            # An oversized block is split into sentence/paragraph pieces so no
            # single chunk blows past max_chars (Phase 1 deferred issue).
            pieces = (
                self._split_oversized(text) if len(text) > self.max_chars else [text]
            )
            for i, piece in enumerate(pieces):
                if i > 0 or (
                    current is not None
                    and current["char_count"] + len(piece) > self.max_chars
                ):
                    flush()
                    current_marker_only = False
                if current is None:
                    current = {
                        "chunk_type": chunk_type,
                        "section": section,
                        "subsection": subsection,
                        "texts": [],
                        "page_start": page,
                        "page_end": page,
                        "clause_ids": [],
                        "char_count": 0,
                    }
                current["texts"].append(piece)
                current["char_count"] += len(piece)
                current["page_end"] = page
                if clause_id and clause_id not in current["clause_ids"]:
                    current["clause_ids"].append(clause_id)

        flush()
        return raw

    @staticmethod
    def _section_title(text: str) -> str:
        """Short breadcrumb for a section marker.

        For lettered all-caps clauses such as "B. DEFINITIONS Digit
        Simplification: Who says…" only the actual title ("B. DEFINITIONS") is
        kept; headings are capped at 80 chars.
        """
        text = re.sub(r"\s+", " ", text).strip()
        # All-caps words only (2+ letters), so "B. DEFINITIONS Some marketing…"
        # yields "B. DEFINITIONS" and not "B. DEFINITIONS S".
        lettered = re.match(r"^([A-Z]\.\s+)((?:[A-Z][A-Z&/\-']+(?:\s|$))+)", text)
        if lettered:
            title = (lettered.group(1) + lettered.group(2)).strip()
            if len(title) >= 4:
                return title
        return text[:80]

    def _split_oversized(self, text: str) -> list[str]:
        """Split a block longer than max_chars at paragraphs, then sentences."""
        if len(text) <= self.max_chars:
            return [text]

        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) > 1:
            pieces: list[str] = []
            buf = ""
            for para in paragraphs:
                candidate = f"{buf}\n\n{para}".strip() if buf else para
                if len(candidate) > self.max_chars and buf:
                    pieces.append(buf)
                    buf = para
                else:
                    buf = candidate
                # A single paragraph can still be oversized -> fall through to
                # sentence splitting below for the remainder.
                if len(buf) > self.max_chars:
                    pieces.extend(self._split_sentences(buf))
                    buf = ""
            if buf:
                pieces.append(buf)
            return [p for p in pieces if p.strip()]

        return self._split_sentences(text)

    def _split_sentences(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        sentences = _SENTENCE_SPLIT.split(text)
        pieces: list[str] = []
        buf = ""
        for sentence in sentences:
            # A single "sentence" can exceed max_chars (tables, run-ons,
            # lists without punctuation) -> hard-split it at word bounds.
            for word_chunk in self._split_hard(sentence):
                candidate = f"{buf} {word_chunk}".strip() if buf else word_chunk
                if len(candidate) > self.max_chars and buf:
                    pieces.append(buf)
                    buf = word_chunk
                else:
                    buf = candidate
        if buf:
            pieces.append(buf)
        return [p for p in pieces if p.strip()]

    def _split_hard(self, text: str) -> list[str]:
        """Force-split text longer than max_chars at word boundaries.

        A single token longer than max_chars (tables, dotted leaders, glued
        glyphs) is character-sliced; no output piece ever exceeds max_chars.
        """
        if len(text) <= self.max_chars:
            return [text]
        pieces: list[str] = []
        buf = ""
        for word in text.split():
            if len(word) > self.max_chars:
                if buf:
                    pieces.append(buf)
                    buf = ""
                for j in range(0, len(word), self.max_chars):
                    pieces.append(word[j : j + self.max_chars])
                continue
            candidate = f"{buf} {word}".strip() if buf else word
            if len(candidate) > self.max_chars and buf:
                pieces.append(buf)
                buf = word
            else:
                buf = candidate
        if buf:
            pieces.append(buf)
        return pieces or [text[: self.max_chars]]

    # ------------------------------------------------------------------
    # Finalization: build Chunk objects, merge tiny chunks, ids
    # ------------------------------------------------------------------

    def _finalize(
        self,
        doc_id: str,
        doc: dict[str, Any],
        meta: dict[str, Any],
        raw_chunks: list[dict[str, Any]],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for raw in raw_chunks:
            text = "\n".join(raw["texts"]).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id="",  # assigned below, after merging
                    document_id=doc_id,
                    text=text,
                    context_text="",
                    page_start=int(raw["page_start"]),
                    page_end=int(raw["page_end"]),
                    section=raw.get("section"),
                    subsection=raw.get("subsection"),
                    clause_ids=raw.get("clause_ids", []),
                    chunk_type=raw.get("chunk_type", "prose"),
                    char_count=len(text),
                    word_count=len(text.split()),
                    filename=doc.get("filename"),
                    insurer=meta.get("insurer") or doc.get("insurer"),
                    product=meta.get("product") or doc.get("product"),
                    uin=meta.get("uin") or doc.get("uin"),
                    document_type=meta.get("document_type") or doc.get("document_type"),
                    source_path=doc.get("source_path"),
                )
            )

        chunks = self._merge_tiny(chunks)
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{doc_id}::c{i:04d}"
            chunk.context_text = self._context_text(chunk)
        return chunks

    def _merge_tiny(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge undersized chunks into a compatible neighbour.

        Pass 1 merges into the previous chunk (same section/type). Pass 2
        catches leftovers such as a bare clause number followed by its text:
        those are merged forward into the next chunk. A chunk that has no
        compatible neighbour at all (e.g. a lone table) is kept as-is.
        """
        if self.min_chars <= 0 or len(chunks) < 2:
            return chunks

        merged: list[Chunk] = []
        for chunk in chunks:
            prev = merged[-1] if merged else None
            tiny = chunk.char_count < self.min_chars
            compatible = (
                prev is not None
                and prev.chunk_type == chunk.chunk_type
                and prev.section == chunk.section
                # Never bridge across pages for tables; prose may continue.
                and (chunk.chunk_type != "table")
                # Merge tolerance: never exceed max_chars + min_chars.
                and prev.char_count + chunk.char_count
                <= self.max_chars + self.min_chars
            )
            if tiny and compatible:
                self._absorb(prev, chunk)
                continue
            merged.append(chunk)

        # Pass 2: forward-merge remaining tiny chunks with the next one — but
        # never across a section boundary (that would misattribute the chunk's
        # content in citations), and never past max_chars + min_chars.
        result: list[Chunk] = []
        pending: Chunk | None = None
        for chunk in merged:
            if pending is not None:
                same_section = (chunk.section or "") == (pending.section or "")
                fits = chunk.char_count + pending.char_count <= self.max_chars + self.min_chars
                if chunk.chunk_type == pending.chunk_type and same_section and fits:
                    self._absorb(chunk, pending, before=True)
                else:
                    result.append(pending)  # section change / lone table
                pending = None
            tiny = chunk.char_count < self.min_chars
            if tiny and chunk.chunk_type != "table":
                pending = chunk
                continue
            result.append(chunk)
        if pending is not None:
            prev = result[-1] if result else None
            same_section = prev is not None and (prev.section or "") == (pending.section or "")
            if (
                prev is not None
                and same_section
                and prev.char_count + pending.char_count
                <= self.max_chars + self.min_chars
            ):
                self._absorb(prev, pending)
            else:
                result.append(pending)
        return result

    @staticmethod
    def _absorb(target: Chunk, source: Chunk, before: bool = False) -> None:
        """Merge ``source`` into ``target`` in place (after, or before it)."""
        if before:
            target.text = f"{source.text}\n{target.text}".strip()
        else:
            target.text = f"{target.text}\n{source.text}".strip()
        target.char_count = len(target.text)
        target.word_count = len(target.text.split())
        target.page_end = max(target.page_end, source.page_end)
        target.page_start = min(target.page_start, source.page_start)
        for cid in source.clause_ids:
            if cid not in target.clause_ids:
                target.clause_ids.append(cid)
        if source.subsection and not target.subsection:
            target.subsection = source.subsection

    def _context_text(self, chunk: Chunk) -> str:
        """Text handed to the embedding model: breadcrumb prefix + body.

        The section prefix is dropped when the body already opens with it, to
        avoid double-embedding the same words.
        """
        body = chunk.text
        if not self.context_prefix:
            return body
        parts: list[str] = []
        if chunk.product:
            parts.append(chunk.product)
        section_dup = bool(
            chunk.section
            and body.lstrip()[:30].lower().startswith(chunk.section[:30].lower())
        )
        if chunk.section and not section_dup:
            parts.append(chunk.section)
        if chunk.subsection and chunk.subsection != chunk.section:
            parts.append(chunk.subsection)
        prefix = " > ".join(parts)
        return f"{prefix}\n{body}" if prefix else body


# ----------------------------------------------------------------------
# Corpus-level helpers
# ----------------------------------------------------------------------


def chunk_document(normalized_doc: dict[str, Any], **kwargs: Any) -> list[Chunk]:
    """Convenience wrapper: chunk a single normalized document dict."""
    return Chunker(**kwargs).chunk_document(normalized_doc)


def chunk_corpus(
    normalized_dir: str,
    **kwargs: Any,
) -> list[Chunk]:
    """Chunk every *.json document under ``normalized_dir`` (sorted by name)."""
    if not os.path.isdir(normalized_dir):
        raise FileNotFoundError(f"Normalized directory not found: {normalized_dir}")

    chunker = Chunker(**kwargs)
    all_chunks: list[Chunk] = []
    files = sorted(f for f in os.listdir(normalized_dir) if f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"No normalized JSON files in {normalized_dir}")

    for name in files:
        path = os.path.join(normalized_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        all_chunks.extend(chunker.chunk_document(doc))
    return all_chunks


def chunks_to_jsonl(chunks: Iterable[Chunk], path: str) -> int:
    """Write chunks to JSONL for inspection/debugging. Returns count written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
