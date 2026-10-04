#!/usr/bin/env python
"""Phase 1 — Insurance Policy RAG Agent: Document Ingestion.

Usage:
    python run_phase1.py [--data-dir data] [--configs-dir configs] [--reports-dir reports] [--dpi 300]

This script runs the full Phase 1 pipeline:
  1. Discovers all PDFs in data/
  2. Inspects each PDF (text, images, fonts, OCR candidates)
  3. Creates/updates configs/documents.json
  4. Extracts native text via PyMuPDF
  5. OCRs only flagged pages via EasyOCR
  6. Normalizes into canonical JSON under data/normalized/
  7. Produces extraction quality report under reports/

Phase 1 STOP boundary — no chunking, embeddings, vector DB, retrieval, or LLM.
"""

import argparse
import logging
import sys

from src.ingestion.pipeline import IngestionPipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1: Insurance Policy Document Ingestion Pipeline"
    )
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--configs-dir", default="configs", help="Configs directory")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory")
    parser.add_argument("--dpi", type=int, default=300, help="OCR rendering DPI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        configs_dir=args.configs_dir,
        reports_dir=args.reports_dir,
        dpi=args.dpi,
    )

    report = pipeline.run()

    # Print summary to stdout
    summary = report.get("summary", {})
    print("\n" + "=" * 60)
    print("PHASE 1 PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {summary.get('total_documents', 0)}")
    print(f"Total pages: {summary.get('total_pages', 0)}")
    print(f"Native text pages: {summary.get('total_native_text_pages', 0)}")
    print(f"OCR pages: {summary.get('total_ocr_pages', 0)}")
    print(f"Documents with OCR: {summary.get('documents_with_ocr', 0)}")
    print(f"Errors: {summary.get('total_errors', 0)}")
    print(f"Warnings: {summary.get('total_warnings', 0)}")
    print(f"Normalized JSON: {args.data_dir}/normalized/")
    print(f"Reports: {args.reports_dir}/phase1_extraction_report.json, .md")
    print("=" * 60)

    return 0 if report.get("summary", {}).get("total_errors", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
