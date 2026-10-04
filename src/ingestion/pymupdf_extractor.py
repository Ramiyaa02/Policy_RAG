"""PyMuPDF Extractor — native text extraction from PDFs using PyMuPDF (fitz).

Preserves page-level information and per-block metadata:
  - text
  - page number
  - bounding box
  - block coordinates
  - font size
  - font name
  - flags / style information

Does NOT flatten everything into one giant string — layout is preserved
so that downstream structural analysis can classify blocks as headings,
clauses, tables, headers, footers, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

import fitz


@dataclass
class TextBlock:
    """A single text block extracted from a PDF page."""
    page_number: int
    block_type: str = "paragraph"     # paragraph, heading, subheading, clause, table, list_item
    text: str = ""
    bbox: list[float] = field(default_factory=list)        # [x0, y0, x1, y1]
    block_id: str = ""
    font_name: str | None = None
    font_size: float | None = None
    flags: int = 0
    is_bold: bool = False
    is_italic: bool = False
    is_upper: bool = False
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    source: str = "pymupdf"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageExtraction:
    """Extracted content for a single page."""
    page_number: int
    blocks: list[TextBlock] = field(default_factory=list)
    block_count: int = 0
    char_count: int = 0
    word_count: int = 0
    page_width: float = 0.0
    page_height: float = 0.0
    images: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    ocr_used: bool = False
    source: str = "pymupdf"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Full extraction result for a document."""
    document_id: str
    filename: str
    source_path: str
    page_count: int
    pages: list[PageExtraction] = field(default_factory=list)
    extraction_method: str = "pymupdf"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PyMuPDFExtractor:
    """Extract text and layout from PDFs using PyMuPDF, preserving structural information."""

    def __init__(self) -> None:
        self.engine_name = "PyMuPDF"
        self.version = fitz.version

    def extract(self, pdf_path: str, document_id: str = "") -> ExtractionResult:
        """Extract all pages from a PDF, preserving layout and block-level information."""
        filename = os.path.basename(pdf_path)
        if not document_id:
            document_id = self._generate_doc_id(filename)

        doc = fitz.open(pdf_path)
        pages: list[PageExtraction] = []
        warnings: list[str] = []
        errors: list[str] = []
        total_chars = 0
        total_blocks = 0

        for page_idx in range(len(doc)):
            try:
                page = doc[page_idx]
                extraction = self.extract_page(page, page_idx + 1)
                pages.append(extraction)
                total_chars += extraction.char_count
                total_blocks += extraction.block_count
            except Exception as e:
                errors.append(f"Page {page_idx + 1}: {str(e)}")
                warnings.append(f"Page {page_idx + 1} extraction partially failed: {str(e)}")

        result = ExtractionResult(
            document_id=document_id,
            filename=filename,
            source_path=pdf_path,
            page_count=len(doc),
            pages=pages,
            extraction_method="pymupdf",
            warnings=warnings,
            errors=errors,
        )

        doc.close()
        return result

    def extract_page(self, page: fitz.Page, page_number: int) -> PageExtraction:
        """Extract a single page, returning blocks with layout info."""
        blocks: list[TextBlock] = []
        rect = page.rect
        page_width = rect.width
        page_height = rect.height

        # --- Extract text blocks with detailed span information ---
        text_dict = page.get_text("dict")

        block_counter = 0
        for block_data in text_dict.get("blocks", []):
            if block_data.get("type", 0) != 0:  # skip non-text blocks (images, etc.)
                continue

            bbox = [
                block_data["bbox"][0],
                block_data["bbox"][1],
                block_data["bbox"][2],
                block_data["bbox"][3],
            ]

            block_text_parts: list[str] = []
            block_lines: list[str] = []
            font_names: list[str] = []
            font_sizes: list[float] = []
            flags_set: set[int] = set()
            is_bold_set = False
            is_italic_set = False

            for line_data in block_data.get("lines", []):
                line_text_parts: list[str] = []
                for span in line_data.get("spans", []):
                    span_text = span.get("text", "")
                    block_text_parts.append(span_text)
                    line_text_parts.append(span_text)

                    font_names.append(span.get("font", ""))
                    font_sizes.append(span.get("size", 0))
                    flags_set.add(span.get("flags", 0))
                    if span.get("flags", 0) & 2:
                        is_bold_set = True
                    if span.get("flags", 0) & 1:
                        is_italic_set = True

                line_text = "".join(line_text_parts)
                block_lines.append(line_text)

            block_text = "".join(block_text_parts)

            # Only create a block if there's text
            if not block_text.strip():
                continue

            block_id = f"p{page_number}_b{block_counter}"
            block_counter += 1

            font_name = font_names[0] if font_names else None
            font_size = max(font_sizes) if font_sizes else None

            # Determine if text is upper case
            stripped = block_text.strip()
            is_upper = stripped.isupper() and len(stripped) > 5

            block = TextBlock(
                page_number=page_number,
                text=block_text,
                bbox=bbox,
                block_id=block_id,
                font_name=font_name,
                font_size=font_size,
                flags=list(flags_set)[0] if flags_set else 0,
                is_bold=is_bold_set,
                is_italic=is_italic_set,
                is_upper=is_upper,
                char_count=len(block_text),
                word_count=len(block_text.split()),
                line_count=len(block_lines),
                source="pymupdf",
            )
            blocks.append(block)

        # --- Image info ---
        images: list[dict[str, Any]] = []
        for img in page.get_images(full=True):
            xref = img[0] if isinstance(img, tuple) else img.get("xref", 0)
            images.append({
                "xref": xref,
                "index": len(images),
                "width": img[2] if isinstance(img, tuple) and len(img) > 2 else 0,
                "height": img[3] if isinstance(img, tuple) and len(img) > 3 else 0,
            })

        # --- Basic table detection ---
        # Use raw blocks (with spacing) for table detection
        raw_blocks = page.get_text("blocks")
        tables = self._detect_tables(raw_blocks=raw_blocks, page_number=page_number)

        total_chars = sum(b.char_count for b in blocks)
        total_words = sum(b.word_count for b in blocks)

        return PageExtraction(
            page_number=page_number,
            blocks=blocks,
            block_count=len(blocks),
            char_count=total_chars,
            word_count=total_words,
            page_width=page_width,
            page_height=page_height,
            images=images,
            tables=tables,
            ocr_used=False,
            source="pymupdf",
        )

    def _detect_tables(self, raw_blocks: list = None, blocks: list[TextBlock] = None, page_number: int = 0) -> list[dict[str, Any]]:
        """Detect table-like structures from raw block text and structured blocks."""
        tables: list[dict[str, Any]] = []

        # Use raw blocks (get_text("blocks")) for table detection — these preserve
        # original spacing and alignment info that's lost in the dict-based extraction
        if raw_blocks:
            for b in raw_blocks:
                text = b[4] if len(b) > 4 else ""
                if not text or not text.strip():
                    continue
                bbox = [b[0], b[1], b[2], b[3]] if len(b) >= 4 else [0, 0, 0, 0]

                lines = [l.strip() for l in text.split("\n") if l.strip()]
                if len(lines) < 2:
                    continue

                # Skip very short blocks (likely headers/footers, not tables)
                if len(text.strip()) < 30:
                    continue

                # Check for tab-separated content: require multiple lines with tabs
                # and each line should have at least 2 tab-separated fields
                tab_lines = [l for l in lines if "\t" in l]
                if len(tab_lines) >= 2:
                    for tl in tab_lines:
                        if len(tl.split("\t")) < 2:
                            tab_lines = []
                            break
                    if len(tab_lines) >= 2:
                        tables.append({
                            "page": page_number,
                            "bbox": bbox,
                            "text": text[:1000],
                            "source": "pymupdf",
                            "detection_method": "tab_separated",
                        })
                        continue

                # Check for aligned columns: require multiple lines with varying
                # leading spaces indicating column structure
                if len(lines) >= 3:
                    leading_spaces = [len(l) - len(l.lstrip()) for l in lines]
                    if len(set(leading_spaces)) > 1:
                        non_zero = [s for s in leading_spaces if s > 0]
                        if len(non_zero) >= 2 and max(non_zero) - min(non_zero) >= 5:
                            tables.append({
                                "page": page_number,
                                "bbox": bbox,
                                "text": text[:1000],
                                "source": "pymupdf",
                                "detection_method": "aligned_columns",
                            })

        return tables

    @staticmethod
    def _generate_doc_id(filename: str) -> str:
        import hashlib
        h = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8]
        return f"DOC-{h}"
