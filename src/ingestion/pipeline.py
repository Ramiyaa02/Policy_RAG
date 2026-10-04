"""Pipeline Orchestrator — Phase 1 Document Ingestion.

Orchestrates the full Phase 1 pipeline:
  1. Discover PDFs in data/ (or data/raw)
  2. Inspect each PDF (page count, text availability, images, fonts, OCR candidates)
  3. Create/update document registry (configs/documents.json)
  4. Extract native text via PyMuPDF
  5. Identify OCR candidates
  6. OCR only required pages via EasyOCR (or pytesseract if available)
  7. Normalize extraction into canonical JSON
  8. Detect structural information (headings, clauses, tables, headers/footers)
  9. Write normalized JSON to data/normalized/
 10. Produce extraction quality report (reports/phase1_extraction_report.json + .md)

Phase 1 stops here — no chunking, embeddings, vector DB, retrieval, or LLM.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import fitz

from .pdf_inspector import PDFInspector, DocumentInspection, PageInspection
from .pymupdf_extractor import PyMuPDFExtractor, ExtractionResult, PageExtraction
from .ocr_extractor import OCRExtractor, OCRDocumentResult
from .normalizer import Normalizer, NormalizedDocument

logger = logging.getLogger(__name__)

# Set stdout to UTF-8 to handle Unicode characters in policy text
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


@dataclass
class DocumentRegistryEntry:
    """A single entry in configs/documents.json."""
    document_id: str
    filename: str
    insurer: str | None
    product: str | None
    uin: str | None
    document_type: str | None
    source_path: str
    version: str | None = None
    effective_date: str | None = None
    page_count: int = 0
    file_size: int = 0
    checksum: str = ""
    extraction_status: str = "pending"
    review_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IngestionPipeline:
    """Phase 1 document ingestion pipeline."""

    def __init__(
        self,
        data_dir: str = "data",
        configs_dir: str = "configs",
        reports_dir: str = "reports",
        dpi: int = 300,
        corpus_dirs: list[str] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.extracted_dir = os.path.join(data_dir, "extracted")
        self.normalized_dir = os.path.join(data_dir, "normalized")
        self.configs_dir = configs_dir
        self.reports_dir = reports_dir
        self.dpi = dpi
        self.corpus_dirs = corpus_dirs or ["data", "rag_policy_dataset"]

        # Initialize components
        self.inspector = PDFInspector()
        self.extractor = PyMuPDFExtractor()
        self.ocr_extractor = OCRExtractor()
        self.normalizer = Normalizer()

        # Ensure directories exist
        for d in [self.data_dir, self.raw_dir, self.extracted_dir,
                self.normalized_dir, self.configs_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Discover PDFs
    # ------------------------------------------------------------------

    def discover_pdfs(self) -> list[str]:
        """Discover all PDF files recursively under corpus directories, excluding output dirs."""
        pdf_paths: list[str] = []
        searched_dirs = set()

        for search_dir in self.corpus_dirs:
            search_dir = os.path.abspath(search_dir)
            if search_dir in searched_dirs:
                continue
            searched_dirs.add(search_dir)

            if not os.path.isdir(search_dir):
                continue

            for root, dirs, files in os.walk(search_dir):
                # Skip output directories
                dirs[:] = [d for d in dirs if d not in ("normalized", "extracted", "raw")]
                for f in sorted(files):
                    if f.lower().endswith(".pdf"):
                        full_path = os.path.join(root, f)
                        if full_path not in pdf_paths:
                            pdf_paths.append(full_path)

        pdf_paths.sort()
        logger.info(f"Discovered {len(pdf_paths)} PDF files.")
        return pdf_paths

    # ------------------------------------------------------------------
    # Step 2: Document ID generation (stable)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_document_id(filename: str, index: int) -> str:
        """Generate a stable document ID.

        Uses a sequential scheme based on sorted filename order for readability
        (DOC-001, DOC-002, ...) which is deterministic since the file set is fixed.
        """
        return f"DOC-{index:03d}"

    # ------------------------------------------------------------------
    # Step 3: Document metadata extraction
    # ------------------------------------------------------------------

    def _extract_document_metadata(self, pdf_path: str, filename: str) -> dict[str, Any]:
        """Extract insurer, product, UIN, document_type, and version from PDF content."""
        doc = fitz.open(pdf_path)
        first_pages_text: list[str] = []
        full_text_parts: list[str] = []

        # Read first 5 pages — first 2 pages for document type, all 5 + last for UIN/insurer
        for i in range(min(5, len(doc))):
            page_text = doc[i].get_text()
            full_text_parts.append(page_text)
            if i < 2:
                first_pages_text.append(page_text)

        # Also read last page for UIN
        if len(doc) > 5:
            full_text_parts.append(doc[-1].get_text())

        full_text = "\n".join(full_text_parts)
        first_text = "\n".join(first_pages_text)

        # Extract metadata — use first text for document type, full text for UIN/insurer
        metadata = Normalizer.extract_metadata_from_text(full_text)
        # Override document type with first-page-only detection
        metadata["document_type"] = Normalizer.detect_document_type(first_text)
        doc.close()

        return metadata

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

    # ------------------------------------------------------------------
    # Step 4: Document registry creation
    # ------------------------------------------------------------------

    def create_registry(self, pdf_paths: list[str]) -> list[DocumentRegistryEntry]:
        """Create or update configs/documents.json with entries for all discovered PDFs."""
        entries: list[DocumentRegistryEntry] = []

        for idx, pdf_path in enumerate(pdf_paths, 1):
            filename = os.path.basename(pdf_path)
            file_size = os.path.getsize(pdf_path)
            checksum = self._compute_checksum(pdf_path)

            # Extract metadata from content
            try:
                content_meta = self._extract_document_metadata(pdf_path, filename)
            except Exception as e:
                logger.warning(f"Metadata extraction failed for {filename}: {e}")
                content_meta = {}

            insurer = content_meta.get("insurer")
            product = content_meta.get("product")
            uin = content_meta.get("uin")
            doc_type = content_meta.get("document_type")
            version = content_meta.get("version")

            # Filename-based hints
            review_flags: list[str] = []

            # If UIN not found in content, try filename
            if not uin:
                # Try to extract from filename (e.g., PolicyWordings_myOptimaSecure-76673175551_HDFC)
                fn_uin = re.findall(r'([A-Z]{3,6}LIP\d+[A-Z]?\d+)', filename)
                if fn_uin:
                    uin = fn_uin[0]
                    review_flags.append("uin_from_filename_not_content")
                else:
                    uin = None
                    review_flags.append("uin_not_found")

            # If product not found, try filename
            if not product:
                name_lower = filename.lower().replace("policy wordings - ", "").replace(".pdf", "")
                # Heuristic product name from filename
                if "optima" in name_lower:
                    product = "my:Optima Secure"
                elif "reassure" in name_lower:
                    product = "ReAssure 2.0"
                elif "ultimate" in name_lower:
                    product = "Ultimate Care"
                elif "elevate" in name_lower:
                    product = "Elevate"
                elif "star" in name_lower and "comprehensive" in name_lower:
                    product = "Star Comprehensive"
                elif "digit" in name_lower or "godigit" in name_lower:
                    product = "Digit Health Insurance Policy"
                elif "medicare" in name_lower and "tata" in name_lower:
                    product = "MediCare"
                elif "mediclaim" in name_lower and "newindia" in name_lower:
                    product = "MediClaim Policy"
                elif "activ" in name_lower:
                    product = "Activ One"
                else:
                    product = name_lower
                review_flags.append("product_from_filename")

            # If insurer not found, try filename
            if not insurer:
                name_lower = filename.lower()
                if "hdfc" in name_lower:
                    insurer = "HDFC ERGO General Insurance Company Limited"
                elif "niva" in name_lower or "nbhl" in name_lower or "reassure" in name_lower:
                    insurer = "Niva Bupa Health Insurance Company Limited"
                elif "care" in name_lower or "chihlip" in name_lower:
                    insurer = "Care Health Insurance Limited"
                elif "icici" in name_lower:
                    insurer = "ICICI Lombard General Insurance Company Limited"
                elif "star" in name_lower:
                    insurer = "Star Health and Allied Insurance Company Limited"
                elif "digit" in name_lower or "godigit" in name_lower:
                    insurer = "Go Digit General Insurance Limited"
                elif "tata" in name_lower:
                    insurer = "Tata AIG General Insurance Company Limited"
                elif "aditya" in name_lower or "birla" in name_lower:
                    insurer = "Aditya Birla Health Insurance Company Limited"
                elif "newindia" in name_lower or "policyclause" in name_lower:
                    insurer = "The New India Assurance Company Limited"
                review_flags.append("insurer_from_filename")

            # Document type: check filename first (more reliable for classification),
            # then fall back to content-based detection
            name_lower = filename.lower()
            fn_doc_type = None
            if "prospectus" in name_lower:
                fn_doc_type = "Prospectus"
            elif "policy-terms-and-conditions" in name_lower or "terms-and-conditions" in name_lower:
                fn_doc_type = "Policy Terms and Conditions"
            elif "policy-clause" in name_lower or "policyclause" in name_lower:
                fn_doc_type = "Policy Clause"
            elif "policy-document" in name_lower:
                fn_doc_type = "Policy Document"
            elif "policy-wording" in name_lower:
                fn_doc_type = "Policy Wording"

            # If filename gives a specific type, use it; otherwise use content-based
            if fn_doc_type:
                if doc_type and doc_type != fn_doc_type:
                    review_flags.append("document_type_mismatch_content_vs_filename")
                doc_type = fn_doc_type
                if not content_meta.get("document_type_was_inferred"):
                    # Only flag if content-based detection differed
                    if content_meta.get("document_type") and content_meta.get("document_type") != fn_doc_type:
                        review_flags.append("document_type_from_filename")
            elif not doc_type:
                review_flags.append("document_type_inferred")
                if "prospectus" in name_lower:
                    doc_type = "Prospectus"
                else:
                    doc_type = "Policy Wording"
                review_flags.append("document_type_from_filename")

            # Get page count from PDF
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
            except Exception:
                page_count = 0
                review_flags.append("page_count_failed")

            # Determine if this is a duplicate/version variant
            # Compare against existing entries
            for existing in entries:
                if existing.uin == uin and uin is not None:
                    review_flags.append(f"possible_version_variant_of_{existing.document_id}")

            # Check for filename similarity (possible near-duplicate)
            for existing in entries:
                fn_existing = existing.filename
                # Normalize filenames for comparison
                norm_existing = re.sub(r'[\s\-_]', '', fn_existing.lower().replace(".pdf", ""))
                norm_current = re.sub(r'[\s\-_]', '', filename.lower().replace(".pdf", ""))

                # Check if they share the same insurer/product but different UIN
                # (already handled above with UIN comparison)

                # Check for exact checksum match (true duplicate)
                if existing.checksum == checksum:
                    review_flags.append(f"exact_duplicate_of_{existing.document_id}")

            entry = DocumentRegistryEntry(
                document_id=self.generate_document_id(filename, idx),
                filename=filename,
                insurer=insurer,
                product=product,
                uin=uin,
                document_type=doc_type,
                source_path=pdf_path,
                version=version,
                page_count=page_count,
                file_size=file_size,
                checksum=checksum,
                extraction_status="pending",
                review_flags=review_flags,
            )
            entries.append(entry)

        # Write to configs/documents.json
        registry_path = os.path.join(self.configs_dir, "documents.json")
        registry_data = {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents": [e.to_dict() for e in entries],
        }

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Document registry written to {registry_path} with {len(entries)} entries.")
        return entries

    # ------------------------------------------------------------------
    # Step 5-6: Inspect, extract, OCR, normalize for each document
    # ------------------------------------------------------------------

    def process_document(
        self,
        pdf_path: str,
        registry_entry: DocumentRegistryEntry,
        inspection: DocumentInspection | None = None,
    ) -> NormalizedDocument:
        """Process a single document through the full Phase 1 pipeline."""
        doc_id = registry_entry.document_id
        logger.info(f"Processing {registry_entry.filename} ({doc_id})...")

        # Step 2: Inspect (if not already provided)
        if inspection is None:
            inspection = self.inspector.inspect_document(pdf_path, doc_id)

        # Step 4: Extract native text
        extraction = self.extractor.extract(pdf_path, doc_id)

        # Step 5-6: OCR for flagged pages only
        ocr_result = None
        if inspection.ocr_page_numbers:
            if self.ocr_extractor.available:
                doc = fitz.open(pdf_path)
                ocr_result = self.ocr_extractor.extract_document(
                    doc, doc_id, inspection.ocr_page_numbers, dpi=self.dpi,
                )
                doc.close()
                # Write OCR extraction to data/extracted/
                ocr_path = os.path.join(
                    self.extracted_dir, doc_id, "ocr.json"
                )
                os.makedirs(os.path.dirname(ocr_path), exist_ok=True)
                with open(ocr_path, "w", encoding="utf-8") as f:
                    json.dump(ocr_result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            else:
                logger.warning(f"No OCR engine available — cannot process OCR pages for {doc_id}")
                inspection.warnings.append("OCR engine not available")

        # Step 7: Normalize
        # Gather metadata from the registry entry
        metadata = {
            "insurer": registry_entry.insurer,
            "product": registry_entry.product,
            "uin": registry_entry.uin,
            "document_type": registry_entry.document_type,
            "version": registry_entry.version,
            "checksum": registry_entry.checksum,
            "file_size": registry_entry.file_size,
            "review_flags": registry_entry.review_flags,
            "effective_date": registry_entry.effective_date,
        }

        normalized = self.normalizer.normalize(
            extraction=extraction,
            ocr_result=ocr_result,
            inspection=inspection,
            metadata=metadata,
        )

        # Step 9: Write normalized JSON
        output_path = os.path.join(self.normalized_dir, f"{doc_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(normalized.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Normalized output written to {output_path}")

        # Also write raw extraction to data/extracted/
        ext_path = os.path.join(self.extracted_dir, f"{doc_id}.json")
        with open(ext_path, "w", encoding="utf-8") as f:
            json.dump(extraction.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        return normalized

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full Phase 1 pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Phase 1: Document Ingestion Pipeline")
        logger.info("=" * 60)

        # Step 1: Discover PDFs
        pdf_paths = self.discover_pdfs()
        if not pdf_paths:
            logger.error("No PDF files found in data/ directory.")
            return {"status": "error", "message": "No PDFs found"}

        # Step 3: Create/update document registry
        registry_entries = self.create_registry(pdf_paths)

        # Step 2, 4, 5, 6, 7, 8, 9: Process each document
        results: list[NormalizedDocument] = []
        inspection_results: list[DocumentInspection] = []

        for entry in registry_entries:
            try:
                # Step 2: Inspect
                inspection = self.inspector.inspect_document(entry.source_path, entry.document_id)
                inspection_results.append(inspection)

                normalized = self.process_document(entry.source_path, entry, inspection)
                results.append(normalized)

                entry.extraction_status = "completed"
            except Exception as e:
                logger.error(f"Pipeline failed for {entry.filename}: {e}")
                entry.extraction_status = "failed"
                entry.review_flags.append(f"pipeline_error: {str(e)}")

        # Update registry with extraction status
        registry_path = os.path.join(self.configs_dir, "documents.json")
        registry_data = {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents": [e.to_dict() for e in registry_entries],
        }
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)

        # Step 10: Generate extraction quality report
        report = self._generate_report(results, inspection_results, registry_entries)
        report_path_json = os.path.join(self.reports_dir, "phase1_extraction_report.json")
        report_path_md = os.path.join(self.reports_dir, "phase1_extraction_report.md")

        with open(report_path_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        self._generate_markdown_report(report, report_path_md)

        logger.info(f"Extraction report written to {report_path_json}")
        logger.info(f"Extraction report (markdown) written to {report_path_md}")
        logger.info("=" * 60)
        logger.info("Phase 1 pipeline complete.")
        logger.info("=" * 60)

        return report

    # ------------------------------------------------------------------
    # Step 10: Report generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        normalized_docs: list[NormalizedDocument],
        inspections: list[DocumentInspection],
        registry: list[DocumentRegistryEntry],
    ) -> dict[str, Any]:
        """Generate the Phase 1 extraction quality report."""
        import re as _re

        documents_report: list[dict[str, Any]] = []
        duplicate_analysis: list[dict[str, Any]] = []

        # Build checksum map for duplicate detection
        checksum_map: dict[str, list[str]] = {}
        uin_map: dict[str, list[str]] = {}
        text_similarity_map: dict[str, list[str]] = {}

        for i_doc, (norm, insp, entry) in enumerate(zip(normalized_docs, inspections, registry)):
            # Collect key stats
            native_text_pages = 0
            ocr_pages: list[int] = []
            total_images = 0
            suspicious_pages: list[dict[str, Any]] = []
            detected_tables = []

            for page in norm.pages:
                if page.ocr_used:
                    ocr_pages.append(page.page_number)
                else:
                    if page.char_count > 50:
                        native_text_pages += 1

                total_images += insp.pages[page.page_number - 1].image_count if page.page_number <= len(insp.pages) else 0

                # Check for tables
                for t in page.detected_tables:
                    detected_tables.append({
                        "page": page.page_number,
                        "bbox": t.get("bbox", []),
                        "method": t.get("detection_method", ""),
                    })

                # Suspicious pages
                page_insp = insp.pages[page.page_number - 1] if page.page_number <= len(insp.pages) else None
                if page_insp:
                    if page_insp.char_count == 0 and page_insp.image_count > 0:
                        suspicious_pages.append({
                            "page": page.page_number,
                            "reason": "no_text_with_images",
                            "char_count": page_insp.char_count,
                            "image_count": page_insp.image_count,
                        })
                    elif page_insp.char_count < 50 and page_insp.image_count > 0:
                        suspicious_pages.append({
                            "page": page.page_number,
                            "reason": "very_low_text_with_images",
                            "char_count": page_insp.char_count,
                            "image_count": page_insp.image_count,
                        })
                    elif page_insp.warnings:
                        for w in page_insp.warnings:
                            suspicious_pages.append({
                                "page": page.page_number,
                                "reason": w,
                                "char_count": page_insp.char_count,
                            })

            # Checksum-based duplicate detection
            checksum_map.setdefault(entry.checksum, []).append(entry.document_id)
            # UIN-based version detection
            if entry.uin:
                uin_map.setdefault(entry.uin, []).append(entry.document_id)

            # Document-level text similarity (checksum)
            # Already handled by checksum_map

            doc_report = {
                "document_id": entry.document_id,
                "filename": entry.filename,
                "insurer": entry.insurer,
                "product": entry.product,
                "uin": entry.uin,
                "document_type": entry.document_type,
                "version": entry.version,
                "page_count": entry.page_count,
                "file_size": entry.file_size,
                "checksum": entry.checksum,
                "native_text_pages": native_text_pages,
                "ocr_pages": ocr_pages,
                "ocr_page_count": len(ocr_pages),
                "total_images": total_images,
                "detected_tables": len(detected_tables),
                "extraction_method": norm.extraction_method,
                "extraction_errors": norm.errors,
                "warnings": list(set(norm.warnings)),
                "suspicious_pages": suspicious_pages,
                "review_flags": entry.review_flags,
                "ocr_engine": norm.ocr_engine,
            }
            documents_report.append(doc_report)

        # Analyze duplicates/versions
        for checksum, doc_ids in checksum_map.items():
            if len(doc_ids) > 1:
                duplicate_analysis.append({
                    "type": "exact_duplicate",
                    "checksum": checksum,
                    "documents": doc_ids,
                    "action": "none_kept_separate",
                })

        # Check for version conflicts (same UIN should not happen, but same product different UIN)
        # The two Digit PDFs have different UINs but same product
        product_map: dict[str, list[str]] = {}
        for entry in registry:
            if entry.product:
                product_map.setdefault(entry.product, []).append(entry.document_id)

        for product, doc_ids in product_map.items():
            if len(doc_ids) > 1:
                duplicate_analysis.append({
                    "type": "same_product_different_version",
                    "product": product,
                    "documents": doc_ids,
                    "uins": [e.uin for e in registry if e.document_id in doc_ids],
                    "action": "kept_separate_reported_as_related",
                })

        report = {
            "phase": "Phase 1 — Document Ingestion",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "1.0",
            "summary": {
                "total_documents": len(normalized_docs),
                "total_pages": sum(len(n.pages) for n in normalized_docs),
                "total_native_text_pages": sum(
                    sum(1 for p in n.pages if not p.ocr_used and p.char_count > 50)
                    for n in normalized_docs
                ),
                "total_ocr_pages": sum(n.ocr_pages and len(n.ocr_pages) or 0 for n in normalized_docs),
                "documents_with_ocr": sum(1 for n in normalized_docs if n.ocr_pages),
                "total_errors": sum(len(n.errors) for n in normalized_docs),
                "total_warnings": sum(len(n.warnings) for n in normalized_docs),
            },
            "documents": documents_report,
            "duplicate_analysis": duplicate_analysis,
            "manual_review_candidates": self._identify_manual_review_candidates(
                normalized_docs, inspections
            ),
            "assumptions": [
                "Document IDs are assigned sequentially based on sorted filename order",
                "Insurer and product names are inferred from document content first, then filename",
                "UIN values are trusted when explicitly stated in the document",
                "OCR candidates are determined by text density and image coverage thresholds",
                "EasyOCR is used as the primary OCR engine (tesseract not installed in environment)",
                "Two Digit Health Insurance PDFs are different versions of the same product (different UINs)",
                "policy-document.pdf identified as Aditya Birla Activ One based on content extraction",
            ],
        }

        return report

    def _identify_manual_review_candidates(
        self,
        normalized_docs: list[NormalizedDocument],
        inspections: list[DocumentInspection],
    ) -> list[dict[str, Any]]:
        """Identify pages that should be manually inspected."""
        candidates: list[dict[str, Any]] = []

        for doc, insp in zip(normalized_docs, inspections):
            for page in doc.pages:
                page_insp = insp.pages[page.page_number - 1]

                # Scanned/OCR page
                if page.ocr_used:
                    candidates.append({
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "page": page.page_number,
                        "reason": "ocr_processed_page",
                        "char_count": page_insp.char_count,
                        "recommendation": "Verify OCR quality — text may contain recognition errors",
                    })
                    continue

                # Low text page
                if page_insp.char_count < 50:
                    candidates.append({
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "page": page.page_number,
                        "reason": "low_text_content",
                        "char_count": page_insp.char_count,
                        "recommendation": "Review — page may have extraction issues",
                    })
                    continue

                # Page with tables
                if page.detected_tables:
                    candidates.append({
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "page": page.page_number,
                        "reason": "table_page",
                        "char_count": page_insp.char_count,
                        "recommendation": "Review — table structure preservation",
                    })
                    continue

                # First page (typically title/cover)
                if page.page_number == 1:
                    candidates.append({
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "page": page.page_number,
                        "reason": "cover_page",
                        "char_count": page_insp.char_count,
                        "recommendation": "Review — verify metadata extraction accuracy",
                    })

                # First page with headings
                has_heading = any(b.type == "heading" for b in page.blocks)
                if has_heading and page_insp.char_count < 200:
                    candidates.append({
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "page": page.page_number,
                        "reason": "heading_only_page",
                        "char_count": page_insp.char_count,
                        "recommendation": "Review — page may have incomplete extraction",
                    })

        return candidates

    def _generate_markdown_report(self, report: dict[str, Any], path: str) -> None:
        """Generate a markdown version of the extraction report."""
        lines: list[str] = []

        lines.append("# Phase 1 Extraction Report")
        lines.append("")
        lines.append(f"**Generated at:** {report['generated_at']}")
        lines.append(f"**Pipeline version:** {report['pipeline_version']}")
        lines.append("")

        # Summary
        s = report["summary"]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|------|")
        lines.append(f"| Total documents | {s['total_documents']} |")
        lines.append(f"| Total pages | {s['total_pages']} |")
        lines.append(f"| Native text pages | {s['total_native_text_pages']} |")
        lines.append(f"| OCR pages | {s['total_ocr_pages']} |")
        lines.append(f"| Documents with OCR | {s['documents_with_ocr']} |")
        lines.append(f"| Extraction errors | {s['total_errors']} |")
        lines.append(f"| Extraction warnings | {s['total_warnings']} |")
        lines.append("")

        # Per-document
        lines.append("## Per-Document Report")
        lines.append("")
        for doc in report["documents"]:
            lines.append(f"### {doc['document_id']}: {doc['filename']}")
            lines.append("")
            lines.append(f"| Field | Value |")
            lines.append(f"|-------|------|")
            lines.append(f"| Insurer | {doc.get('insurer', 'N/A')} |")
            lines.append(f"| Product | {doc.get('product', 'N/A')} |")
            lines.append(f"| UIN | {doc.get('uin', 'N/A')} |")
            lines.append(f"| Document type | {doc.get('document_type', 'N/A')} |")
            lines.append(f"| Page count | {doc['page_count']} |")
            lines.append(f"| File size | {doc['file_size']:,} bytes |")
            lines.append(f"| Extraction method | {doc['extraction_method']} |")
            lines.append(f"| Native text pages | {doc['native_text_pages']} |")
            lines.append(f"| OCR pages | {doc['ocr_page_count']} |")
            lines.append(f"| OCR page numbers | {doc['ocr_pages'] if doc['ocr_pages'] else 'None'} |")
            lines.append(f"| Total images | {doc['total_images']} |")
            lines.append(f"| Detected tables | {doc['detected_tables']} |")
            lines.append(f"| Checksum | `{doc['checksum'][:16]}...` |")
            lines.append("")

            if doc["extraction_errors"]:
                lines.append(f"**Errors:**")
                for e in doc["extraction_errors"]:
                    lines.append(f"- {e}")
                lines.append("")

            if doc["warnings"]:
                lines.append(f"**Warnings:**")
                for w in doc["warnings"]:
                    lines.append(f"- {w}")
                lines.append("")

            if doc["suspicious_pages"]:
                lines.append(f"**Suspicious pages:**")
                for sp in doc["suspicious_pages"]:
                    lines.append(f"- Page {sp['page']}: {sp['reason']} (chars: {sp['char_count']})")
                lines.append("")

            if doc["review_flags"]:
                lines.append(f"**Review flags:**")
                for rf in doc["review_flags"]:
                    lines.append(f"- {rf}")
                lines.append("")

        # Duplicate analysis
        lines.append("## Duplicate / Version Analysis")
        lines.append("")
        if report["duplicate_analysis"]:
            for da in report["duplicate_analysis"]:
                lines.append(f"### {da['type']}")
                lines.append(f"- Documents: {da['documents']}")
                if "product" in da:
                    lines.append(f"- Product: {da['product']}")
                if "uins" in da:
                    lines.append(f"- UINs: {da['uins']}")
                lines.append(f"- Action: {da['action']}")
                lines.append("")
        else:
            lines.append("No duplicates or version conflicts detected.")
            lines.append("")

        # Manual review candidates
        lines.append("## Manual Review Candidates")
        lines.append("")
        lines.append("| Document | Page | Reason | Recommendation |")
        lines.append("|----------|------|--------|----------------|")
        for c in report["manual_review_candidates"]:
            lines.append(f"| {c['document_id']} | {c['page']} | {c['reason']} | {c['recommendation']} |")
        lines.append("")

        # Assumptions
        lines.append("## Assumptions")
        lines.append("")
        for a in report["assumptions"]:
            lines.append(f"- {a}")
        lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Helper: load registry
    # ------------------------------------------------------------------

    def load_registry(self) -> list[DocumentRegistryEntry]:
        """Load the document registry from configs/documents.json."""
        registry_path = os.path.join(self.configs_dir, "documents.json")
        if not os.path.exists(registry_path):
            return []

        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = []
        for d in data.get("documents", []):
            entries.append(DocumentRegistryEntry(**d))

        return entries
