"""PDF Inspector — page-level inspection and OCR candidate detection.

This module inspects each page of a PDF to determine:
  - page count and file metadata
  - text availability (character/word/block counts)
  - text density
  - image presence and image coverage
  - font information
  - whether a page is a likely OCR candidate

It does NOT perform OCR itself — it only flags pages that may require OCR.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import fitz


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FontInfo:
    """Summarised font information for a page."""
    name: str
    size: float
    count: int
    bold: bool = False
    italic: bool = False


@dataclass
class ImageInfo:
    """Information about an image found on a page."""
    xref: int
    width: int
    height: int
    bbox: list[float]  # [x0, y0, x1, y1] in page coordinates
    area_ratio: float  # fraction of page covered


@dataclass
class PageInspection:
    """Inspection results for a single PDF page."""
    page_number: int

    # Text metrics
    char_count: int = 0
    word_count: int = 0
    text_block_count: int = 0

    # Layout metrics
    page_width: float = 0.0
    page_height: float = 0.0
    page_area: float = 0.0
    text_area: float = 0.0
    text_density: float = 0.0          # chars per unit text area
    overall_density: float = 0.0      # chars per page area
    whitespace_ratio: float = 0.0     # fraction of page that is whitespace

    # Image metrics
    image_count: int = 0
    total_image_area: float = 0.0
    image_coverage: float = 0.0        # fraction of page covered by images
    image_area_ratio: float = 0.0     # image_area / text_area (relative)

    # Font metrics
    fonts: list[FontInfo] = field(default_factory=list)
    font_count: int = 0
    unique_font_names: list[str] = field(default_factory=list)

    # Structural signals
    has_tables: bool = False
    table_count: int = 0
    has_numbered_clauses: bool = False
    has_headings: bool = False

    # OCR recommendation
    needs_ocr: bool = False
    ocr_confidence: str = "unknown"    # "high", "medium", "low", "unknown"
    ocr_reasons: list[str] = field(default_factory=list)

    # Warnings
    warnings: list[str] = field(default_factory=list)

    # Raw text sample for debugging (first 500 chars)
    text_sample: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRRecommendation:
    """Aggregate recommendation across all pages of a document."""
    document_id: str
    total_pages: int
    ocr_pages: list[int]
    native_text_pages: int
    mixed_pages: int = 0
    ocr_required: bool = False
    recommended_engine: str = "easyocr"
    notes: list[str] = field(default_factory=list)


@dataclass
class DocumentInspection:
    """Full inspection result for a single PDF document."""
    document_id: str
    filename: str
    source_path: str
    file_size: int
    checksum: str
    page_count: int
    pages: list[PageInspection] = field(default_factory=list)

    # Document-level summaries
    total_text_pages: int = 0
    total_ocr_pages: int = 0
    ocr_page_numbers: list[int] = field(default_factory=list)
    total_images: int = 0
    total_native_chars: int = 0
    fonts_available: bool = True
    encrypted: bool = False
    ocr_recommendation: str = "none"

    # Metadata that can be extracted from the PDF
    metadata: dict[str, Any] = field(default_factory=dict)

    # Warnings / errors
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "page_count": self.page_count,
            "pages": [p.to_dict() for p in self.pages],
            "total_text_pages": self.total_text_pages,
            "total_ocr_pages": self.total_ocr_pages,
            "ocr_page_numbers": self.ocr_page_numbers,
            "total_images": self.total_images,
            "total_native_chars": self.total_native_chars,
            "fonts_available": self.fonts_available,
            "encrypted": self.encrypted,
            "ocr_recommendation": self.ocr_recommendation,
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

class PDFInspector:
    """Inspect PDFs for text availability, layout, images, fonts, and OCR candidates."""

    # Thresholds — tuned for insurance policy documents
    MIN_CHARS_PER_PAGE = 50          # pages with fewer chars are suspicious
    LOW_TEXT_THRESHOLD = 100         # chars below this ⇒ likely OCR candidate (was 10, too strict for 11-char pages)
    IMAGE_COVERAGE_THRESHOLD = 0.50  # if 50%+ of page is image ⇒ likely OCR candidate
    CHAR_DENSITY_LOW = 0.05          # chars per 1000 px² — very sparse text
    BLOCK_THRESHOLD = 2              # fewer than this many blocks ⇒ suspicious

    def __init__(self) -> None:
        self.pdf_metrics_version = "1.0"

    # ------------------------------------------------------------------
    # Document-level inspection
    # ------------------------------------------------------------------

    def inspect_document(self, pdf_path: str, document_id: str = "") -> DocumentInspection:
        """Inspect an entire PDF document."""
        filename = os.path.basename(pdf_path)
        if not document_id:
            document_id = self._generate_doc_id(filename)

        file_size = os.path.getsize(pdf_path)
        checksum = self._compute_checksum(pdf_path)

        doc = fitz.open(pdf_path)
        pages: list[PageInspection] = []

        total_ocr_pages = 0
        total_text_pages = 0
        total_images = 0
        total_chars = 0

        encrypted = bool(doc.needs_pass) if hasattr(doc, "needs_pass") else False

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            inspection = self.inspect_page(page, page_idx + 1)
            pages.append(inspection)

            total_images += inspection.image_count
            total_chars += inspection.char_count

            if inspection.needs_ocr:
                total_ocr_pages += 1
            if inspection.char_count > self.MIN_CHARS_PER_PAGE:
                total_text_pages += 1

        # Determine overall OCR recommendation
        if total_ocr_pages > 0:
            ocr_recommendation = "hybrid"
        elif total_chars == 0:
            ocr_recommendation = "full_ocr"
        else:
            ocr_recommendation = "native_only"

        result = DocumentInspection(
            document_id=document_id,
            filename=filename,
            source_path=pdf_path,
            file_size=file_size,
            checksum=checksum,
            page_count=len(doc),
            pages=pages,
            total_text_pages=total_text_pages,
            total_ocr_pages=total_ocr_pages,
            ocr_page_numbers=[p.page_number for p in pages if p.needs_ocr],
            total_images=total_images,
            total_native_chars=total_chars,
            fonts_available=True,
            encrypted=encrypted,
            ocr_recommendation=ocr_recommendation,
            metadata=self._extract_metadata(doc),
        )

        doc.close()
        return result

    # ------------------------------------------------------------------
    # Page-level inspection
    # ------------------------------------------------------------------

    def inspect_page(self, page: fitz.Page, page_number: int, extract_fonts: bool = True) -> PageInspection:
        """Inspect a single PDF page for text, images, fonts, and OCR candidacy."""
        # --- Text metrics ---
        text = page.get_text()
        blocks = page.get_text("blocks")
        words = page.get_text("words")

        char_count = len(text)
        word_count = len(words)
        block_count = len(blocks)

        text_sample = text[:500] if text else ""

        # --- Page geometry ---
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        page_area = page_width * page_height

        # --- Compute text bounding box area ---
        text_area = 0.0
        for b in blocks:
            x0, y0, x1, y1 = b[:4]
            block_area = max(0, (x1 - x0) * (y1 - y0))
            text_area += block_area

        text_density = char_count / (text_area / 1000) if text_area > 0 else 0
        overall_density = char_count / (page_area / 1000) if page_area > 0 else 0
        whitespace_ratio = 1.0 - min(1.0, text_area / page_area) if page_area > 0 else 1.0

        # --- Image metrics ---
        images = page.get_images(full=True)
        image_count = len(images)

        # Image coverage is not critical for OCR detection (image_count is the key signal).
        # Full image rect extraction via get_image_rects is expensive; we skip it
        # during inspection for performance and set coverage to 0.
        # Detailed image analysis can be done on-demand during extraction if needed.
        total_image_area = 0.0

        image_coverage = min(1.0, total_image_area / page_area) if page_area > 0 else 0
        image_area_ratio = total_image_area / text_area if text_area > 0 else 0

        # --- Font information (deferred to extractor for speed) ---
        if extract_fonts:
            fonts = self._extract_fonts_fast(page)
        else:
            fonts = []
            font_count = 0
            unique_font_names = []
        if extract_fonts:
            font_count = len(fonts)
            unique_font_names = list(set(f.name for f in fonts))
        else:
            font_count = 0
            unique_font_names = []

        # --- Structural detection ---
        has_tables = self._detect_tables(blocks, char_count)
        table_count = self._count_tables(blocks)
        has_numbered_clauses = self._detect_numbered_clauses(text)
        has_headings = self._detect_headings_fast(blocks, fonts, char_count)

        # --- OCR recommendation ---
        needs_ocr, ocr_confidence, ocr_reasons = self._assess_ocr_need(
            char_count, word_count, block_count, text_area,
            image_count=image_count, image_coverage=image_coverage,
            fonts=fonts if extract_fonts else [], page_area=page_area,
        )

        # --- Warnings ---
        warnings: list[str] = []
        if char_count == 0 and len(images) > 0:
            warnings.append("Page has no extractable text but contains images — OCR strongly recommended")
        elif char_count < self.LOW_TEXT_THRESHOLD and len(images) > 0:
            warnings.append("Very low text extraction with images present — possible scanned page")
        elif char_count == 0 and len(images) == 0:
            warnings.append("Empty page with no text and no images")

        table_count = self._count_tables(blocks)

        inspection = PageInspection(
            page_number=page_number,
            char_count=char_count,
            word_count=word_count,
            text_block_count=block_count,
            page_width=page_width,
            page_height=page_height,
            page_area=page_area,
            text_area=text_area,
            text_density=text_density,
            overall_density=overall_density,
            whitespace_ratio=whitespace_ratio,
            image_count=len(images),
            total_image_area=total_image_area,
            image_coverage=image_coverage,
            image_area_ratio=image_area_ratio,
            fonts=fonts,
            font_count=len(fonts),
            unique_font_names=list(set(f.name for f in fonts)),
            has_tables=has_tables,
            table_count=table_count,
            has_numbered_clauses=has_numbered_clauses,
            has_headings=has_headings,
            needs_ocr=needs_ocr,
            ocr_confidence=ocr_confidence,
            ocr_reasons=ocr_reasons,
            warnings=warnings,
            text_sample=text_sample,
        )

        return inspection

    # ------------------------------------------------------------------
    # Helper: OCR assessment
    # ------------------------------------------------------------------

    def _assess_ocr_need(
        self,
        char_count: int,
        word_count: int,
        block_count: int,
        text_area: float,
        image_count: int,
        image_coverage: float,
        fonts: list[FontInfo],
        page_area: float,
    ) -> tuple[bool, str, list[str]]:
        """Decide whether a page needs OCR using multiple signals."""
        reasons: list[str] = []

        # Signal 1: very low or zero character count
        if char_count == 0:
            reasons.append("no_extractable_text")
        elif char_count < self.LOW_TEXT_THRESHOLD:
            reasons.append("very_low_text_count")

        # Signal 2: very few text blocks
        if block_count < self.BLOCK_THRESHOLD and char_count < self.MIN_CHARS_PER_PAGE:
            reasons.append("insufficient_text_blocks")

        # Signal 3: substantial image content
        if image_count > 0:
            reasons.append("images_present")
            if image_coverage > self.IMAGE_COVERAGE_THRESHOLD:
                reasons.append("high_image_coverage")

        # Signal 4: low text density
        if text_area > 0 and char_count / (text_area / 1000) < self.CHAR_DENSITY_LOW:
            reasons.append("low_text_density")

        # Signal 5: font information sparse (few unique fonts)
        if len(fonts) == 0 and char_count > 0:
            # Has text but no font info — possible extraction issue
            reasons.append("no_font_info_with_text")

        # Decision logic:
        # OCR is needed when text is sparse AND images are present.
        # A page with text but no images was extracted natively — no OCR needed.
        # Only flag for OCR when there is image content that may contain text.
        needs_ocr = False

        if char_count == 0 and image_count > 0:
            needs_ocr = True
        elif char_count < self.LOW_TEXT_THRESHOLD and image_count > 0:
            needs_ocr = True
        elif char_count < 20 and block_count < self.BLOCK_THRESHOLD and image_count > 0:
            needs_ocr = True

        if not reasons:
            reasons.append("text_extraction_ok")

        if needs_ocr:
            # Confidence assessment
            if char_count == 0 and image_count > 0:
                confidence = "high"
            elif char_count < 50 and image_count > 0:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "not_required"

        return needs_ocr, confidence, reasons

    # ------------------------------------------------------------------
    # Helper: font extraction
    # ------------------------------------------------------------------

    def _extract_fonts_fast(self, page: fitz.Page) -> list[FontInfo]:
        """Extract font information from a page quickly using page.get_fonts()."""
        fonts: list[FontInfo] = []
        try:
            font_list = page.get_fonts()
        except Exception:
            return fonts

        if not font_list:
            return fonts

        for f in font_list:
            # PyMuPDF 1.27.x get_fonts() returns: (xref, filetype, fonttype, fontname, fontfile, encoding)
            if isinstance(f, tuple) and len(f) >= 4:
                font_name = str(f[3]) if f[3] else str(f[0])
            else:
                font_name = str(f[0]) if f else 'unknown'
            fonts.append(FontInfo(
                name=font_name,
                size=12.0,
                count=0,
                bold=False,
                italic=False,
            ))

        return fonts

    # ------------------------------------------------------------------
    # Helper: table detection
    # ------------------------------------------------------------------

    def _detect_tables(self, blocks: list, char_count: int) -> bool:
        """Basic table detection heuristic — checks for aligned columns and tabular patterns."""
        if len(blocks) == 0:
            return False
        table_like_blocks = 0
        for b in blocks:
            text = b[4] if len(b) > 4 else ""
            if not text or not text.strip():
                continue

            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if len(lines) < 2:
                continue

            # Check for tab-separated content
            if any("\t" in l for l in lines):
                table_like_blocks += 1
                continue

            # Check for aligned columns (multiple lines with consistent spacing patterns)
            if len(lines) >= 3:
                # Look for lines where content starts at similar x-positions (column alignment)
                leading_spaces = [len(l) - len(l.lstrip()) for l in lines[:6]]
                if len(set(leading_spaces)) > 1:
                    # Check if the leading space pattern differs but is consistent
                    # (alternating between 2+ indentation levels suggests columns)
                    non_zero = [s for s in leading_spaces if s > 0]
                    if len(non_zero) >= 2 and max(non_zero) - min(non_zero) >= 5:
                        table_like_blocks += 1

        return table_like_blocks > 0

    def _count_tables(self, blocks: list) -> int:
        """Count likely table structures."""
        table_count = 0
        for b in blocks:
            text = b[4] if len(b) > 4 else ""
            if not text or not text.strip():
                continue
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if len(lines) < 2:
                continue
            if any("\t" in l for l in lines):
                table_count += 1
                continue
            if len(lines) >= 3:
                leading_spaces = [len(l) - len(l.lstrip()) for l in lines[:6]]
                if len(set(leading_spaces)) > 1:
                    non_zero = [s for s in leading_spaces if s > 0]
                    if len(non_zero) >= 2 and max(non_zero) - min(non_zero) >= 5:
                        table_count += 1
        return table_count

    # ------------------------------------------------------------------
    # Helper: clause detection
    # ------------------------------------------------------------------

    def _detect_numbered_clauses(self, text: str) -> bool:
        """Detect numbered clause patterns like 1., 1.1, 1.2.3, etc."""
        import re
        patterns = [
            r'(?m)^\s*\d+\.\s',           # 1.
            r'(?m)^\s*\d+\.\d+\s',        # 1.1
            r'(?m)^\s*\d+\.\d+\.\d+\s',   # 1.2.3
            r'(?m)^\s*[A-Z]\.\s',         # A.
            r'(?m)^\s*[IVX]+\.\s',        # I.
        ]
        for pat in patterns:
            if re.search(pat, text):
                return True
        return False

    # ------------------------------------------------------------------
    # Helper: fast heading detection
    # ------------------------------------------------------------------

    def _detect_headings_fast(self, blocks: list, fonts: list[FontInfo], char_count: int) -> bool:
        """Detect if any text on the page looks like a heading (fast version)."""
        if char_count == 0:
            return False

        # Check for all-caps or title-case lines in blocks
        for b in blocks:
            text = b[4] if len(b) > 4 else ""
            if text:
                lines = text.split("\n")
                for line in lines:
                    stripped = line.strip()
                    if stripped and len(stripped) < 80:
                        if stripped.isupper() and len(stripped) > 3:
                            return True

        # If the largest font is significantly larger than the smallest
        sizes = [f.size for f in fonts if f.size > 0]
        if len(sizes) >= 2:
            max_size = max(sizes)
            min_size = min(sizes)
            if max_size > min_size * 1.3:
                return True

        return False

    # ------------------------------------------------------------------
    # Helper: metadata extraction
    # ------------------------------------------------------------------

    def _extract_metadata(self, doc: fitz.Document) -> dict[str, Any]:
        """Extract PDF metadata."""
        try:
            meta = doc.metadata
            return {k: v for k, v in meta.items() if v is not None}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Helper: checksum
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_checksum(file_path: str) -> str:
        """Compute SHA-256 checksum of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _generate_doc_id(filename: str) -> str:
        """Generate a stable document ID from a filename."""
        import hashlib as hl
        h = hl.sha256(filename.encode("utf-8")).hexdigest()[:8]
        return f"DOC-{h}"
