"""Normalizer — converts native PDF extraction and OCR output into one common structure.

Canonical representation:

Document
  → Page
      → Block

Each block retains:
  - type (heading, paragraph, clause, table, header, footer, list_item, image, etc.)
  - text
  - bbox
  - font_size, font_name
  - source (pymupdf or ocr)
  - page reference
  - structural metadata

Also detects:
  - repeated headers/footers (marked, not deleted)
  - numbered clauses (1., 1.1, 1.2.3, ...)
  - headings (by font size, capitalization, layout)
  - tables (preserved with structure)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

import fitz

from .pymupdf_extractor import TextBlock, PageExtraction, ExtractionResult
from .ocr_extractor import OCRPageResult, OCRDocumentResult
from .pdf_inspector import DocumentInspection


# ---------------------------------------------------------------------------
# Canonical data classes
# ---------------------------------------------------------------------------

@dataclass
class NormalizedBlock:
    """A normalized block — the canonical representation of an extracted element."""
    block_id: str
    type: str = "paragraph"  # heading, subheading, paragraph, clause, table, list_item,
                             # header, footer, image, caption
    subtype: str | None = None  # section_heading, subsection_heading, numbered_clause, bulleted_list, etc.
    text: str = ""
    raw_text: str = ""          # original text before normalization (for debugging)
    normalized_text: str = ""   # text after cleaning (same as text, but explicit)
    bbox: list[float] = field(default_factory=list)     # [x0, y0, x1, y1]
    page_number: int = 0
    font_name: str | None = None
    font_size: float | None = None
    is_bold: bool = False
    is_italic: bool = False
    is_upper: bool = False
    source: str = "pymupdf"
    ocr_used: bool = False
    char_count: int = 0
    word_count: int = 0
    reading_order: int = 0
    heading_level: int | None = None
    clause_number: str | None = None
    clause_id: str | None = None  # structured clause identifier (e.g., "1.2.3")
    confidence: float | None = None                     # from OCR if applicable
    classification_confidence: float = 0.0             # 0.0-1.0 confidence in block type
    is_repeated: bool = False                         # True if header/footer repeated across pages
    include_in_chunk_text: bool = True                # False for headers/footers to exclude from chunks
    has_unicode_issues: bool = False                  # True if text contains replacement chars
    table_info: dict[str, Any] | None = None            # if type == "table"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedPage:
    """Normalized page with blocks and layout info."""
    page_number: int
    blocks: list[NormalizedBlock] = field(default_factory=list)
    page_width: float = 0.0
    page_height: float = 0.0
    char_count: int = 0
    word_count: int = 0
    ocr_used: bool = False
    detected_tables: list[dict[str, Any]] = field(default_factory=list)
    detected_headers: list[int] = field(default_factory=list)   # block indices
    detected_footers: list[int] = field(default_factory=list)
    layout_type: str = "single_column"  # single_column, two_column, mixed
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedDocument:
    """The full canonical document representation."""
    document_id: str
    filename: str
    source_path: str
    insurer: str | None = None
    product: str | None = None
    uin: str | None = None
    document_type: str | None = None
    version: str | None = None
    extraction_method: str = "pymupdf"
    page_count: int = 0
    pages: list[NormalizedPage] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)
    ocr_engine: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    detected_headers: list[dict[str, Any]] = field(default_factory=list)
    detected_footers: list[dict[str, Any]] = field(default_factory=list)
    detected_clauses: list[dict[str, Any]] = field(default_factory=list)
    detected_headings: list[dict[str, Any]] = field(default_factory=list)
    detected_tables: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)
    checksum: str | None = None
    file_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Normalizer:
    """Convert extracted PDF content (native + OCR) into canonical normalized form."""

    # Patterns for numbered clauses
    _CLAUSE_PATTERNS = [
        re.compile(r'^(?:\d{1,3})\.\s+(.+)$'),          # 1.  Text
        re.compile(r'^(?:\d{1,3})\.(\d{1,3})\s+(.+)$'), # 1.1  Text
        re.compile(r'^(?:\d{1,3})\.(\d{1,3})\.(\d{1,3})\s+(.+)$'),  # 1.2.3  Text
        re.compile(r'^([A-Z])\.\s+(.+)$'),             # A.  Text
    ]

    _NUMBERED_REGEX = re.compile(
        r'(?m)^\s*(?:(\d{1,3})\.(\d{1,3})?\.?(\d{1,3})?|([A-Z]))\.\s+'
    )

    # Header/footer detection: text that appears on most pages, near top/bottom
    _HEADER_MARGIN_RATIO = 0.15   # top 15% of page = header area
    _FOOTER_MARGIN_RATIO = 0.10   # bottom 10% of page = footer area

    def __init__(self) -> None:
        self.inspection: DocumentInspection | None = None

    # ------------------------------------------------------------------
    # Main normalization
    # ------------------------------------------------------------------

    def normalize(
        self,
        extraction: ExtractionResult,
        ocr_result: OCRDocumentResult | None = None,
        inspection: DocumentInspection | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NormalizedDocument:
        """Combine native extraction and OCR into a normalized document."""
        self.inspection = inspection

        doc_id = extraction.document_id
        pages: list[NormalizedPage] = []
        ocr_pages: list[int] = []
        ocr_engine = None

        if ocr_result and ocr_result.available and ocr_result.ocr_pages:
            ocr_pages = ocr_result.page_numbers
            ocr_engine = ocr_result.engine

        # Build page-level normalized output
        for page_extraction in extraction.pages:
            ocr_used = page_extraction.page_number in (ocr_pages if ocr_pages else [])

            # Get OCR text for this page if available
            ocr_page_text = ""
            ocr_blocks_by_id: dict[str, OCRPageResult] = {}
            if ocr_result and ocr_used:
                for ocr_page in ocr_result.ocr_pages:
                    if ocr_page.page_number == page_extraction.page_number:
                        ocr_page_text = ocr_page.ocr_text
                        break

            norm_page = self._normalize_page(page_extraction, ocr_used, ocr_page_text)
            pages.append(norm_page)

            if ocr_used:
                norm_page.ocr_used = True

        # Post-process: detect headers/footers across the document
        header_blocks, footer_blocks = self._detect_repeated_headers_footers(pages)

        # Mark header/footer blocks
        for page in pages:
            page.detected_headers = []
            page.detected_footers = []
            for bi, block in enumerate(page.blocks):
                if block.source == "header":
                    page.detected_headers.append(bi)
                    block.type = "header"
                    block.subtype = "repeated_header"
                    block.is_repeated = True
                    block.include_in_chunk_text = False
                elif block.source == "footer":
                    page.detected_footers.append(bi)
                    block.type = "footer"
                    block.subtype = "repeated_footer"
                    block.is_repeated = True
                    block.include_in_chunk_text = False

        # Detect clauses and headings
        detected_clauses, detected_headings = self._detect_structure(pages)

        # Detect tables
        detected_tables = self._collect_tables(pages)

        # Build document metadata
        meta = metadata or {}

        result = NormalizedDocument(
            document_id=doc_id,
            filename=extraction.filename,
            source_path=extraction.source_path,
            insurer=meta.get("insurer"),
            product=meta.get("product"),
            uin=meta.get("uin"),
            document_type=meta.get("document_type"),
            version=meta.get("version"),
            extraction_method="pymupdf" + (" + OCR" if ocr_pages else ""),
            page_count=len(pages),
            pages=pages,
            ocr_pages=ocr_pages,
            ocr_engine=ocr_engine,
            metadata=meta,
            detected_headers=header_blocks,
            detected_footers=footer_blocks,
            detected_clauses=detected_clauses,
            detected_headings=detected_headings,
            detected_tables=detected_tables,
            warnings=extraction.warnings,
            errors=extraction.errors,
            review_flags=list(meta.get("review_flags", [])),
            checksum=meta.get("checksum"),
            file_size=meta.get("file_size", 0),
        )

        # Copy inspection errors if available
        if inspection and inspection.errors:
            result.errors.extend(inspection.errors)
        if inspection and inspection.warnings:
            result.warnings.extend(inspection.warnings)

        return result

    # ------------------------------------------------------------------
    # Text normalization
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """Normalize extracted text: remove control chars, normalize whitespace.
        
        Processing order ensures tables retain their structure:
        - Tables are detected BEFORE normalization in extract_page()
        - This method only cleans paragraph/clause/heading text
        - Tabs are converted to spaces (table structure preserved separately)
        - Bell chars and other control chars removed
        - Unicode replacement chars flagged but not removed
        """
        if not text:
            return text

        # Preserve a copy for raw_text
        normalized = text

        # Replace tabs with space (table structure preserved in detected_tables)
        normalized = normalized.replace('\t', ' ')

        # Remove bell character (PDF indentation artifact) and other unsafe control chars
        # Keep \n and \r for line structure
        normalized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', normalized)

        # Normalize excessive whitespace (but preserve line breaks)
        lines = normalized.split('\n')
        cleaned_lines = []
        for line in lines:
            # Collapse multiple spaces into one
            line = re.sub(r' {2,}', ' ', line)
            # Strip trailing whitespace
            line = line.rstrip()
            cleaned_lines.append(line)
        normalized = '\n'.join(cleaned_lines)

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        return normalized

    # ------------------------------------------------------------------
    # Page normalization
    # ------------------------------------------------------------------

    def _normalize_page(
        self,
        page_extraction: PageExtraction,
        ocr_used: bool,
        ocr_text: str,
    ) -> NormalizedPage:
        """Normalize a single page's extracted blocks."""
        blocks: list[NormalizedBlock] = []
        char_count = 0
        word_count = 0

        # Sort blocks by reading order (top-to-bottom, then left-to-right)
        sorted_blocks = sorted(
            page_extraction.blocks,
            key=lambda b: (round(b.bbox[1]), round(b.bbox[0])),
        )

        for idx, tb in enumerate(sorted_blocks):
            block_type, subtype, confidence = self._classify_block(tb)
                # Detect Unicode issues: replacement chars, non-characters
            has_unicode = (
                '\ufffd' in tb.text or
                any(0xFDD0 <= ord(c) <= 0xFDEF for c in tb.text) or
                '\ufffe' in tb.text or '\uffff' in tb.text
            )

            # Preserve raw text before normalization
            raw_text = tb.text
            normalized_text = self._normalize_text(raw_text)

            nb = NormalizedBlock(
                block_id=tb.block_id,
                type=block_type,
                subtype=subtype,
                text=normalized_text,
                raw_text=raw_text,
                normalized_text=normalized_text,
                bbox=tb.bbox,
                page_number=tb.page_number,
                font_name=tb.font_name,
                font_size=tb.font_size,
                is_bold=tb.is_bold,
                is_italic=tb.is_italic,
                is_upper=tb.is_upper,
                source=tb.source,
                ocr_used=ocr_used,
                char_count=len(normalized_text),
                word_count=len(normalized_text.split()),
                reading_order=idx,
                classification_confidence=confidence,
                has_unicode_issues=has_unicode,
            )

            # Mark header/footer blocks for exclusion from chunk text
            if block_type in ("header", "footer"):
                nb.include_in_chunk_text = False

            # Detect clause number (metadata only, type already set by _classify_block)
            clause_num = self._extract_clause_number(tb.text)
            if clause_num:
                nb.clause_number = clause_num
                nb.clause_id = clause_num

            # Detect heading level (metadata only, type already set by _classify_block)
            heading_level = self._detect_heading_level(tb)
            if heading_level is not None:
                nb.heading_level = heading_level

            # Table info
            if block_type == "table":
                nb.table_info = {
                    "bbox": tb.bbox,
                    "text": normalized_text,
                    "source": tb.source,
                }

            blocks.append(nb)
            char_count += nb.char_count
            word_count += nb.word_count

        # Add OCR text as a fallback block if needed
        if ocr_used and ocr_text and char_count < 50:
            ocr_block = NormalizedBlock(
                block_id=f"p{page_extraction.page_number}_ocr_full",
                type="paragraph",
                text=ocr_text.strip(),
                bbox=[0, 0, page_extraction.page_width, page_extraction.page_height],
                page_number=page_extraction.page_number,
                source="ocr",
                ocr_used=True,
                char_count=len(ocr_text),
                word_count=len(ocr_text.split()),
                reading_order=len(blocks),
            )
            blocks.append(ocr_block)
            char_count += ocr_block.char_count
            word_count += ocr_block.word_count

        # Detect layout type
        layout_type = self._detect_layout_type(page_extraction)

        # Collect table info from extraction
        tables = page_extraction.tables

        # Build warnings for this page
        page_warnings: list[str] = []
        unicode_blocks = sum(1 for b in blocks if b.has_unicode_issues)
        if unicode_blocks > 0:
            page_warnings.append(f"Page contains {unicode_blocks} block(s) with Unicode replacement characters")

        return NormalizedPage(
            page_number=page_extraction.page_number,
            blocks=blocks,
            page_width=page_extraction.page_width,
            page_height=page_extraction.page_height,
            char_count=char_count,
            word_count=word_count,
            ocr_used=ocr_used,
            detected_tables=[t for t in tables],
            layout_type=layout_type,
            warnings=page_warnings,
        )

    # ------------------------------------------------------------------
    # Block classification
    # ------------------------------------------------------------------

    def _classify_block(self, block: TextBlock) -> tuple[str, str | None, float]:
        """Classify a text block into a type, subtype, and confidence.
        
        Returns (type, subtype, confidence) where:
        - type: header, footer, heading, paragraph, clause, list_item, table, unknown
        - subtype: section_heading, subsection_heading, numbered_clause, bulleted_list, etc.
        - confidence: 0.0-1.0 confidence in the classification
        """
        text = block.text.strip()
        if not text:
            return ("paragraph", None, 1.0)

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return ("paragraph", None, 1.0)

        # Clause detection: starts with a number pattern (1., 1.1, A., etc.)
        clause_num = self._extract_clause_number(text)
        if clause_num:
            return ("clause", "numbered_clause", 0.9)

        # Table detection: multiple lines with tab-separated values or aligned columns
        tab_lines = [l for l in lines if "\t" in l]
        if len(tab_lines) >= 2 and all(len(tl.split("\t")) >= 2 for tl in tab_lines):
            return ("table", "tab_separated", 0.8)
        if len(lines) >= 3:
            if self._check_aligned_columns(lines):
                return ("table", "aligned_columns", 0.7)

        # List item detection: starts with bullet character
        first_line = lines[0] if lines else ""
        if re.match(r'^[\*\-\u2022\u25C6\u25C7\u2192\u2219]\s+', first_line):
            return ("list_item", "bulleted_list", 0.8)

        # Header/footer: page numbers (standalone digits or page/x)
        if len(text) < 100 and len(lines) <= 2:
            stripped = text.strip().rstrip('.')
            if stripped.isdigit():
                return ("footer", "page_number", 0.7)
            if re.match(r'^\d+\s*\/\s*\d+$', stripped):
                return ("footer", "page_number", 0.7)

        # Heading detection: uses font size, bold, capitalization
        heading_info = self._is_heading(block, lines)
        if heading_info is not None:
            level, confidence = heading_info
            if level == 1:
                return ("heading", "section_heading", confidence)
            else:
                return ("heading", f"subsection_heading_l{level}", confidence)

        return ("paragraph", None, 1.0)

    def _check_aligned_columns(self, lines: list[str]) -> bool:
        """Check if lines have consistent column alignment (tabular pattern)."""
        if len(lines) < 3:
            return False
        # Check if lines have multiple space-separated segments with consistent gaps
        first_line_spaces = [len(l) - len(l.lstrip()) for l in lines[:3]]
        if len(set(first_line_spaces)) > 1:
            # Different indentation levels might indicate columns
            return False

        # Check for repeated double-space or tab patterns
        space_gaps = []
        for l in lines[:min(5, len(lines))]:
            positions = [i for i, c in enumerate(l) if c == ' ' and i + 1 < len(l) and l[i + 1] == ' ']
            space_gaps.append(len(positions))
        return max(space_gaps) >= 2 if space_gaps else False

    def _is_heading(self, block: TextBlock, lines: list[str]) -> tuple[int, float] | None:
        """Determine if a block is a heading. Returns (level, confidence) or None."""
        text = block.text.strip()
        lines_stripped = [l.strip() for l in lines if l.strip()]

        if not lines_stripped:
            return None

        confidence = 0.0
        level = None

        # All uppercase and short → likely heading (Level 1)
        if text.isupper() and len(text) < 100:
            level = 1
            confidence = 0.7

        # Large font → likely heading
        if block.font_size and block.font_size > 14:
            if len(lines_stripped) <= 3 and len(text) < 200:
                level = 1 if level is None else min(level, 1)
                confidence = max(confidence, 0.6)

        # Bold text, short, single line → subsection heading
        if block.is_bold and len(lines_stripped) == 1 and len(text) < 80:
            level = 2
            confidence = max(confidence, 0.8)

        # Bold text, short, few lines → heading
        if block.is_bold and len(lines_stripped) <= 2 and len(text) < 120:
            level = 2
            confidence = max(confidence, 0.7)

        # Centered heading (bbox width relatively narrow)
        if block.bbox and len(block.bbox) == 4:
            bbox_width = block.bbox[2] - block.bbox[0]
            if bbox_width < 0.6 * 400:
                if len(lines_stripped) <= 2 and block.is_upper and len(text) < 80:
                    level = 1
                    confidence = max(confidence, 0.75)

        if level is not None and confidence >= 0.6:
            return (level, confidence)
        return None

    # ------------------------------------------------------------------
    # Clause number extraction
    # ------------------------------------------------------------------

    def _extract_clause_number(self, text: str) -> str | None:
        """Extract clause number from text (e.g., '1.', '1.1', '1.2.3', 'A.', 'I.')."""
        stripped = text.strip()

        # Pattern: 1.2.3, 1.2, or 1. (the trailing dot is optional, so a clause
        # like "1.1 Standard Definitions" is still matched)
        m = re.match(r'^(\d{1,3})(?:\.(\d{1,3}))?(?:\.(\d{1,3}))?\.?(?=\s|$)', stripped)
        if m:
            parts = [m.group(1)]
            if m.group(2):
                parts.append(m.group(2))
            if m.group(3):
                parts.append(m.group(3))
            return ".".join(parts)

        # Pattern: A. or I.
        m = re.match(r'^([A-Z]|IV|IX|V?I{0,3})\.\s', stripped)
        if m:
            return m.group(1)

        return None

    # ------------------------------------------------------------------
    # Heading level detection
    # ------------------------------------------------------------------

    def _detect_heading_level(self, block: TextBlock) -> int | None:
        """Detect heading level based on font size and style."""
        if not block.font_size:
            return None

        text = block.text.strip()
        if not text:
            return None

        lines = text.split("\n")

        # Level 1: Large, bold, all caps
        if block.font_size > 18 or (block.is_bold and block.is_upper and len(text) < 80):
            return 1

        # Level 2: Bold or slightly smaller
        if block.font_size > 14 or (block.is_bold and len(lines) <= 2):
            return 2

        # Level 3: Medium
        if block.font_size > 12 and block.is_bold:
            return 3

        return None

    # ------------------------------------------------------------------
    # Layout type detection
    # ------------------------------------------------------------------

    def _detect_layout_type(self, page: PageExtraction) -> str:
        """Detect page layout type: single_column, two_column, mixed."""
        if not page.blocks:
            return "single_column"

        # Use bbox x-coordinates to detect columns
        x_positions = []
        for b in page.blocks:
            if b.bbox and len(b.bbox) == 4:
                x_positions.append(b.bbox[0])

        if len(x_positions) < 4:
            return "single_column"

        # Cluster x positions to detect column boundaries
        x_positions.sort()
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i - 1]
            if gap > 50:  # significant gap suggests column break
                gaps.append(gap)

        if len(gaps) >= 2 and max(gaps) > 200:
            # Check if there's a clear bimodal distribution
            left_blocks = sum(1 for x in x_positions if x < page.page_width / 2)
            right_blocks = sum(1 for x in x_positions if x >= page.page_width / 2)

            if left_blocks > 0 and right_blocks > 0:
                if left_blocks > right_blocks:
                    return "two_column" if left_blocks > 5 else "mixed"
                return "two_column" if max(gaps) > 300 else "mixed"

        return "single_column"

    # ------------------------------------------------------------------
    # Header/footer detection across pages
    # ------------------------------------------------------------------

    def _detect_repeated_headers_footers(self, pages: list[NormalizedPage]) -> tuple[list[dict], list[dict]]:
        """Detect repeated header and footer blocks across pages."""
        header_texts: dict[str, list[int]] = defaultdict(list)
        footer_texts: dict[str, list[int]] = defaultdict(list)

        for page in pages:
            page_h = page.page_height
            header_threshold = page_h * self._HEADER_MARGIN_RATIO
            footer_threshold = page_h * (1 - self._FOOTER_MARGIN_RATIO)

            for bi, block in enumerate(page.blocks):
                if not block.bbox or len(block.bbox) < 4:
                    continue

                y0 = block.bbox[1]
                text = block.text.strip()

                if not text:
                    continue

                if y0 < header_threshold:
                    header_texts[text].append(page.page_number)
                elif y0 > footer_threshold:
                    footer_texts[text].append(page.page_number)

        min_repeat = max(2, len(pages) // 3)

        headers = []
        footers = []

        for text, page_nums in header_texts.items():
            if len(page_nums) >= min_repeat:
                headers.append({
                    "text": text,
                    "pages": page_nums,
                    "frequency": len(page_nums),
                    "type": "header",
                })

        for text, page_nums in footer_texts.items():
            if len(page_nums) >= min_repeat:
                footers.append({
                    "text": text,
                    "pages": page_nums,
                    "frequency": len(page_nums),
                    "type": "footer",
                })

        # Mark blocks as header/footer
        for page in pages:
            header_set = set(text.strip() for text, pnums in header_texts.items()
                            if len(pnums) >= min_repeat)
            footer_set = set(text.strip() for text, pnums in footer_texts.items()
                            if len(pnums) >= min_repeat)

            page_h = page.page_height
            header_threshold = page_h * self._HEADER_MARGIN_RATIO
            footer_threshold = page_h * (1 - self._FOOTER_MARGIN_RATIO)

            for bi, block in enumerate(page.blocks):
                if not block.bbox or len(block.bbox) < 4:
                    continue

                y0 = block.bbox[1]
                text = block.text.strip()

                if text in header_set and y0 < header_threshold:
                    block.type = "header"
                    block.source = "header"
                elif text in footer_set and y0 > footer_threshold:
                    block.type = "footer"
                    block.source = "footer"

        return headers, footers

    # ------------------------------------------------------------------
    # Structure detection (clauses and headings)
    # ------------------------------------------------------------------

    def _detect_structure(self, pages: list[NormalizedPage]) -> tuple[list[dict], list[dict]]:
        """Detect numbered clauses and headings across all pages."""
        clauses = []
        headings = []

        for page in pages:
            for block in page.blocks:
                if block.type == "heading" or "heading" in (block.subtype or ""):
                    headings.append({
                        "page": block.page_number,
                        "block_id": block.block_id,
                        "text": block.text[:200],
                        "heading_level": block.heading_level,
                        "bbox": block.bbox,
                    })

                if block.clause_number:
                    clauses.append({
                        "page": block.page_number,
                        "block_id": block.block_id,
                        "clause_number": block.clause_number,
                        "clause_id": block.clause_id,
                        "text": block.text[:200],
                        "bbox": block.bbox,
                    })

        return clauses, headings

    # ------------------------------------------------------------------
    # Table collection
    # ------------------------------------------------------------------

    def _collect_tables(self, pages: list[NormalizedPage]) -> list[dict[str, Any]]:
        """Collect all detected tables from pages, cleaning control characters while preserving tabs."""
        tables = []
        for page in pages:
            for table in page.detected_tables:
                raw_table_text = table.get("text", "")
                # Clean bell chars and other unsafe control chars, but preserve tabs for structure
                cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', raw_table_text)
                # Normalize whitespace but preserve tabs and newlines
                cleaned = re.sub(r'[ \t]+(?=\t|[ \t])', ' ', cleaned)
                has_unicode = '\ufffd' in cleaned
                tables.append({
                    "page": page.page_number,
                    "bbox": table.get("bbox", []),
                    "text": cleaned[:500],
                    "source": table.get("source", "pymupdf"),
                    "detection_method": table.get("detection_method", ""),
                    "has_unicode_issues": has_unicode,
                })
        return tables

    # ------------------------------------------------------------------
    # Document metadata extraction from text
    # ------------------------------------------------------------------

    @staticmethod
    def detect_document_type(text: str) -> str | None:
        """Detect document type from cover page text only (not body boilerplate)."""
        import re

        text_lower = text.lower()
        # Check in order of specificity — cover page labels
        if re.search(r'policy\s*wording[s]?', text_lower):
            return "Policy Wording"
        elif re.search(r'policy\s*clause[s]?', text_lower):
            return "Policy Clause"
        elif re.search(r'prospectus', text_lower):
            return "Prospectus"
        elif re.search(r'policy\s*document[s]?', text_lower):
            return "Policy Document"
        elif re.search(r'policy\s*terms\s*and\s*conditions', text_lower):
            return "Policy Terms and Conditions"
        return None

    @staticmethod
    def extract_metadata_from_text(text: str) -> dict[str, Any]:
        """Extract insurer, product, UIN, and document type from document text."""
        import re

        result: dict[str, Any] = {}

        # --- UIN extraction ---
        # IRDAI UIN format: 3-6 letters + "LIP" + digits + optional letter + digits
        # e.g., SHAHLIP26044V092526, HDFHLIP26058V082526, GODHLIP23073V012223, NBHHLIP26042V022526
        uin_pattern = re.compile(r'([A-Z]{3,6}LIP\d{4,6}[A-Z]?\d{4,6})')
        uins = uin_pattern.findall(text)

        # Also try explicit "UIN:" labels and extract UIN-format values from them
        uin_label_pattern = re.compile(r'UIN\s*[:\-]?\s*([A-Z0-9]+)', re.IGNORECASE)
        for m in uin_label_pattern.finditer(text):
            val = m.group(1).strip()
            # Skip masked UINs (all X's)
            if val and not re.match(r'^X+$', val) and len(val) > 2:
                uins.append(val)

        # Deduplicate while preserving order
        seen = set()
        unique_uins: list[str] = []
        for u in uins:
            if u not in seen:
                seen.add(u)
                unique_uins.append(u)

        # Filter: keep only valid UIN-format strings (skip short/non-standard ones)
        valid_uins = [u for u in unique_uins if uin_pattern.fullmatch(u) or (len(u) >= 12 and re.match(r'^[A-Z]{3,6}LIP', u))]

        if valid_uins:
            result["uin"] = valid_uins[0]
        elif unique_uins:
            result["uin"] = unique_uins[0]
            result["_uin_confidence"] = "low"
        else:
            result["uin"] = None

        # --- Insurer detection ---
        insurers = {
            "HDFC ERGO": "HDFC ERGO General Insurance Company Limited",
            "Niva Bupa": "Niva Bupa Health Insurance Company Limited",
            "Care Health": "Care Health Insurance Limited",
            "ICICI Lombard": "ICICI Lombard General Insurance Company Limited",
            "Star Health": "Star Health and Allied Insurance Company Limited",
            "Go Digit": "Go Digit General Insurance Limited",
            "Tata AIG": "Tata AIG General Insurance Company Limited",
            "Aditya Birla": "Aditya Birla Health Insurance Company Limited",
            "New India": "The New India Assurance Company Limited",
        }

        found_insurers: list[str] = []
        text_lower = text.lower()
        for key, full_name in insurers.items():
            if key.lower() in text_lower:
                found_insurers.append(full_name)

        result["insurer"] = found_insurers[0] if found_insurers else None
        if len(found_insurers) > 1:
            result["_insurer_candidates"] = found_insurers

        # --- Product name extraction ---
        # Method 1: Look for known product names in text (more reliable)
        product = None
        product_keywords = [
            (r'my:?[\s-]*Optima[\s-]*Secure', "my:Optima Secure"),
            (r'ReAssure\s*2\.0', "ReAssure 2.0"),
            (r'Ultimate\s*Care', "Ultimate Care"),
            (r'Elevate\b', "Elevate"),
            (r'Star\s*Comprehensive', "Star Comprehensive"),
            (r'Digit\s*Health\s*Insurance\s*Policy', "Digit Health Insurance Policy"),
            (r'Medi\s*Care', "MediCare"),
            (r'Activ\s*One', "Activ One"),
        ]
        for pattern, name in product_keywords:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                product = name
                break

        # Method 2: Look for "Product Name" or "Product:" label
        if not product:
            pn_match = re.search(
                r'Product\s*(?:Name)?\s*[:\-]?\s*\n?\s*(.+?)(?:\n|$)',
                text, re.IGNORECASE | re.DOTALL,
            )
            if pn_match:
                candidate = pn_match.group(1).strip()
                # Clean up: remove "UIN" suffixes and trailing labels
                candidate = re.split(r'\s*[Uu][Ii][Nn]\s*', candidate)[0].strip()
                candidate = re.split(r'\s*Product\s*', candidate)[0].strip()
                candidate = candidate.rstrip(';,').strip()
                # Skip candidates that are too long or look like boilerplate
                if 2 < len(candidate) < 60 and not candidate.startswith("the"):
                    product = re.sub(r'\s+', ' ', candidate).strip()

        if product:
            result["product"] = product

        # --- Version ---
        # Look for version patterns like "V.1", "V.12" (word-boundary prevents UIN suffix matching)
        version_match = re.search(r'\bV\.?\s*(\d{1,3})\b', text)
        if version_match:
            result["version"] = version_match.group(1)

        return result
