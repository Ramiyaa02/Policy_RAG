"""OCR Extractor — falls back to OCR for pages flagged by the inspector.

Uses EasyOCR as the primary engine (self-contained, no external binary required).
Falls back to pytesseract if a tesseract binary is available on the system.

Pipeline:
  PDF page → render page image (PyMuPDF pixmap) → OCR → text + confidence

Only pages flagged by the inspector as OCR candidates are processed.
Good native PDF text is never replaced with OCR unnecessarily.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import fitz
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OCRTextBlock:
    """A text block extracted via OCR."""
    page_number: int
    text: str
    bbox: list[float]            # [x0, y0, x1, y1]
    confidence: float | None = None
    block_id: str = ""
    source: str = "ocr"
    engine: str = "easyocr"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRPageResult:
    """OCR result for a single page."""
    page_number: int
    ocr_used: bool = True
    ocr_engine: str = "easyocr"
    ocr_confidence_mean: float | None = None
    ocr_confidence_min: float | None = None
    ocr_text: str = ""
    blocks: list[OCRTextBlock] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    char_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRDocumentResult:
    """OCR results for all pages of a document."""
    document_id: str
    filename: str
    engine: str = "easyocr"
    total_pages: int = 0
    ocr_pages: list[OCRPageResult] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OCRExtractor:
    """OCR extractor with EasyOCR (primary) and pytesseract fallback."""

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._easyocr_reader: Any = None
        self._pytesseract_available: bool = False
        self._detect_engines()

    def _detect_engines(self) -> None:
        """Detect which OCR engines are available."""
        # Try EasyOCR
        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader([self.language], verbose=False)
            logger.info("EasyOCR engine loaded successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")

        # Try pytesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._pytesseract_available = True
            logger.info("pytesseract/tesseract engine available.")
        except Exception:
            self._pytesseract_available = False
            logger.info("pytesseract/tesseract not available — relying on EasyOCR.")

        if not self._easyocr_reader and not self._pytesseract_available:
            logger.error("No OCR engine available! Install tesseract or ensure easyocr is installed.")

    @property
    def engine_name(self) -> str:
        if self._easyocr_reader:
            return "easyocr"
        if self._pytesseract_available:
            return "pytesseract"
        return "none"

    @property
    def available(self) -> bool:
        return self._easyocr_reader is not None or self._pytesseract_available

    # ------------------------------------------------------------------
    # Page-level OCR
    # ------------------------------------------------------------------

    def extract_page(self, page: fitz.Page, page_number: int, dpi: int = 300) -> OCRPageResult:
        """Render a page to an image and run OCR on it.

        Args:
            page: PyMuPDF page object.
            page_number: 1-based page number.
            dpi: Rendering DPI for the image (default 300 for good OCR quality).

        Returns:
            OCRPageResult with extracted text, blocks, and confidence.
        """
        result = OCRPageResult(page_number=page_number)

        if not self.available:
            result.errors.append("No OCR engine available")
            result.ocr_used = False
            return result

        try:
            # Render page to pixmap
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pixmap.tobytes("png")

            # Convert to numpy array / PIL Image
            pil_img = Image.open(io.BytesIO(img_data))

            # Run OCR with the available engine
            if self._easyocr_reader:
                result.ocr_engine = "easyocr"
                self._ocr_easyocr(pil_img, page, page_number, dpi, result)
            elif self._pytesseract_available:
                result.ocr_engine = "pytesseract"
                self._ocr_pytesseract(pil_img, page, page_number, dpi, result)

            result.char_count = len(result.ocr_text)

        except Exception as e:
            logger.error(f"OCR failed for page {page_number}: {e}")
            result.errors.append(str(e))
            result.ocr_used = False

        return result

    # ------------------------------------------------------------------
    # EasyOCR implementation
    # ------------------------------------------------------------------

    def _ocr_easyocr(self, pil_img: Image.Image, page: fitz.Page,
                     page_number: int, dpi: int, result: OCRPageResult) -> None:
        """Run OCR using EasyOCR and extract bounding boxes."""
        if self._easyocr_reader is None:
            return

        results = self._easyocr_reader.readtext(
            np.array(pil_img),
            paragraph=False,
            detail=1,        # return bounding box + confidence
            contrast_ths=0.3,
        )

        scale_x = 72.0 / dpi  # scale from image pixels back to PDF points
        scale_y = 72.0 / dpi

        confidences: list[float] = []
        all_text_parts: list[str] = []
        page_w = page.rect.width
        page_h = page.rect.height

        for detection in results:
            if detection is None:
                continue
            if len(detection) < 2:
                continue

            bbox_points = detection[0]   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = detection[1]
            confidence = detection[2]

            if not text.strip():
                continue

            # Convert image coordinates back to PDF coordinates
            xs = [p[0] * scale_x for p in bbox_points]
            ys = [p[1] * scale_y for p in bbox_points]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)

            # Clamp to page bounds
            x0 = max(0, min(x0, page_w))
            x1 = max(0, min(x1, page_w))
            y0 = max(0, min(y0, page_h))
            y1 = max(0, min(y1, page_h))

            block_id = f"p{page_number}_ocr_b{len(result.blocks)}"

            block = OCRTextBlock(
                page_number=page_number,
                text=text,
                bbox=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                confidence=round(confidence, 4),
                block_id=block_id,
                source="ocr",
                engine="easyocr",
            )
            result.blocks.append(block)
            all_text_parts.append(text)
            confidences.append(confidence)

        result.ocr_text = "\n".join(all_text_parts)
        result.ocr_confidence_mean = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        result.ocr_confidence_min = round(min(confidences), 4) if confidences else 0.0
        result.ocr_used = True

    # ------------------------------------------------------------------
    # pytesseract implementation
    # ------------------------------------------------------------------

    def _ocr_pytesseract(self, pil_img: Image.Image, page: fitz.Page,
                         page_number: int, dpi: int, result: OCRPageResult) -> None:
        """Run OCR using pytesseract."""
        import pytesseract

        # Get data including bounding boxes
        ocr_data = pytesseract.image_to_data(
            pil_img,
            lang=self.language,
            output_type=pytesseract.Output.DICT,
        )

        scale_x = 72.0 / dpi
        scale_y = 72.0 / dpi
        page_w = page.rect.width
        page_h = page.rect.height

        confidences: list[float] = []
        all_text_parts: list[str] = []

        n_boxes = len(ocr_data.get("text", []))
        current_line_blocks: dict[int, dict] = {}

        for i in range(n_boxes):
            text = ocr_data["text"][i].strip()
            conf = ocr_data["conf"][i]

            if not text:
                continue

            x = int(ocr_data["left"][i]) * scale_x
            y = int(ocr_data["top"][i]) * scale_y
            w = int(ocr_data["width"][i]) * scale_x
            h = int(ocr_data["height"][i]) * scale_y

            x0 = max(0, min(x, page_w))
            x1 = max(0, min(x + w, page_w))
            y0 = max(0, min(y, page_h))
            y1 = max(0, min(y + h, page_h))

            block_id = f"p{page_number}_ocr_b{len(result.blocks)}"

            block = OCRTextBlock(
                page_number=page_number,
                text=text,
                bbox=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                confidence=float(conf) / 100.0 if conf >= 0 else None,
                block_id=block_id,
                source="ocr",
                engine="pytesseract",
            )
            result.blocks.append(block)
            all_text_parts.append(text)
            if conf >= 0:
                confidences.append(float(conf) / 100.0)

        result.ocr_text = "\n".join(all_text_parts)
        result.ocr_confidence_mean = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        result.ocr_confidence_min = round(min(confidences), 4) if confidences else 0.0
        result.ocr_used = True

    # ------------------------------------------------------------------
    # Document-level OCR
    # ------------------------------------------------------------------

    def extract_document(self, doc: fitz.Document, document_id: str,
                         ocr_page_numbers: list[int], dpi: int = 300) -> OCRDocumentResult:
        """Run OCR on all pages flagged by the inspector."""
        result = OCRDocumentResult(
            document_id=document_id,
            filename=doc.name if hasattr(doc, 'name') else "",
            total_pages=len(doc),
            engine=self.engine_name,
            page_numbers=ocr_page_numbers,
            available=self.available,
        )

        if not self.available:
            result.warnings.append("No OCR engine available — OCR could not be performed")
            return result

        for page_num in ocr_page_numbers:
            if page_num < 1 or page_num > len(doc):
                result.warnings.append(f"Page {page_num} out of range (1-{len(doc)})")
                continue

            page = doc[page_num - 1]
            try:
                page_result = self.extract_page(page, page_num, dpi)
                result.ocr_pages.append(page_result)
            except Exception as e:
                result.errors.append(f"Page {page_num}: {e}")

        return result
