# Phase 1 Quality Audit Report

## Executive Summary

The Phase 1 pipeline successfully processed all 10 insurance policy PDFs (553 pages, 1 OCR page, 0 errors). However, the quality audit reveals several **critical issues** that must be fixed before chunking experiments:

1. **Block type classification completely broken** — ALL 16,062 blocks across 10 documents are classified as `unknown`
2. **Header/footer text not stripped** from normalized output — contaminates content with repeated branding
3. **Control characters (tabs, bell chars) in text** — artifacts from PyMuPDF extraction
4. **Unicode encoding issues** — garbled characters in DOC-010

---

## A. Phase 1 Quality Assessment

### Overall Grade: C- (functional but with critical structural issues)

**What works well:**
- Text extraction is nearly complete (551/552 native pages extracted)
- Clause detection works — proper clause numbers extracted (1, 2, 1.1, B, I, etc.)
- Percentages preserved correctly (% symbol used 58-118 times per doc)
- Currency references mostly preserved
- Tables detected in DOC-004 (2 legitimate premium rate tables)
- No garbled/missing/duplicate text in 5/10 documents
- Reading order correct in 8/10 documents

**What's broken:**
- Block classification → ALL blocks are `unknown` type
- Headers/footers not stripped → repeated branding text on every page
- Control characters in text → `\t`, `\x07` (bell) in extracted text
- Unicode encoding issues → garbled characters in at least 2 documents

---

## B. Issues That MUST Be Fixed Before Chunking

### Issue 1: Block Type Classification Completely Broken

**What was extracted:**
All blocks across all 10 documents are classified as `block_type: "unknown"`.

**What appears to be wrong:**
The normalizer's `_classify_block` method (normalizer.py:352) is supposed to classify blocks as "heading", "paragraph", "clause", "table", "footer", etc. But NO classification is happening — every single block (975 to 3,529 per document) returns `"unknown"`.

**Why it affects RAG/chunking:**
Without block type classification, the chunker cannot:
- Distinguish headings from body text (critical for hierarchy-aware chunking)
- Identify clause boundaries (clauses are detected at document level but individual blocks aren't labeled)
- Apply heading-based chunk splitting (a standard technique)
- Filter out navigation/header content

**Recommended fix:**
Debug why `_classify_block` returns "unknown" for all blocks. Likely causes:
- The `blocks` list passed to the normalizer contains `NormalizedBlock` objects, but `_classify_block` may be receiving wrong data format
- The classification logic may never match the actual block text patterns
- Check normalizer.py:352-400 `_classify_block` method

---

### Issue 2: Headers/Footers Not Stripped From Normalized Output

**What was extracted:**
Page 1 first blocks include:
- DOC-001: `"Go Digit General Insurance Ltd."`
- DOC-003: `"HDFC ERGO General Insurance Company Limited   Policy Wording"`
- DOC-007: `"ELEVATE POLICY WORDINGS"`
- DOC-009: `"Activ OnePolicy Wording"`

These appear as text blocks in the normalized JSON, not as separate header markers.

**What appears to be wrong:**
The inspector detects headers (DOC-001: 1 header, DOC-003: 1 header, etc.) and footers (DOC-002: 1 footer, DOC-007: 4 footers, etc.), but the normalizer does **not strip** this text from the block content. Headers and footers appear as regular text blocks mixed with body content.

**Why it affects RAG/chunking:**
- Every chunk starting from page 1 will contain branding text that is irrelevant to the query
- Repeated headers create duplicate content issues
- Chunk boundaries will be polluted with non-content text
- Footer page numbers mixed into chunks

**Recommended fix:**
In `normalizer.py`, use the inspector's header/footer detection results to filter out header/footer blocks before creating `NormalizedBlock` objects. Pass the inspector's `PageInspection` to the normalizer and use `detected_headers`/`detected_footers` to exclude those block indices.

---

### Issue 3: Control Characters in Extracted Text

**What was extracted:**
```
DOC-004 Page 2: 'i.\t\nMonthly: 4%\t\nii.\t Quarterly: 3%\t\niii.\t Half Yearly: 2% \t\niv.\t Yearly: 0%\n'
DOC-004 Page 44: '1.\t\x07In-patient Treatment: We will cover...'
```

**What appears to be wrong:**
PyMuPDF's `get_text("blocks")` returns text with tab (`\t`) and bell (`\x07`) characters embedded. These are formatting artifacts from the PDF's internal structure (tab stops, form feed characters). The normalizer does not clean these.

**Why it affects RAG/chunking:**
- Tab characters create artificial whitespace that breaks tokenization
- Bell characters (`\x07`) are non-printable and will cause encoding errors
- Clause numbers are attached to body text with these artifacts (e.g., `"1.\t\x07In-patient"`)
- String matching for clause numbers will fail due to embedded control chars

**Recommended fix:**
Add text cleaning in the normalizer to strip/convert control characters:
- Replace `\t` with single space or remove
- Remove or convert `\x07` (bell) and other non-printable characters
- Normalize whitespace in extracted text

---

### Issue 4: Unicode Encoding Issues in DOC-010

**What was extracted:**
```
DOC-010 Page 1: 'Welcome to the �I Feel Good� policy'
DOC-010: 'clause_number=B, text='B. DEFINITIONS ...' (clause text truncated)
```

**What appears to be wrong:**
DOC-010 (second Digit Health policy) has Unicode replacement characters (`\ufffd` / `�`) in the extracted text. This suggests either:
- The PDF has embedded fonts with encoding issues
- PyMuPDF is not properly decoding the font encoding
- The text is being processed with wrong encoding somewhere

**Why it affects RAG/chunking:**
- Garbles clause definitions and key terms
- Makes semantic search unreliable
- Chunk content will have corrupted text that doesn't match user queries

**Recommended fix:**
- Investigate the PDF's font encoding
- Try `page.get_text("text", sort=False)` with different encoding options
- Consider using `page.get_text("dict")` for text with better font handling
- Add Unicode normalization/cleanup step

---

## C. Issues That Can Be Deferred

### Deferred Issue 1: Paragraph Boundaries Not Split Within Large Blocks

**What was extracted:**
243 blocks exceed 1000 characters (e.g., DOC-009: 30 blocks, DOC-001: 14 blocks). These large blocks likely contain multiple paragraphs that were not split at line breaks or blank lines.

**Why it's deferred:**
The chunker can handle large blocks by splitting on newlines during chunk creation. This doesn't affect data correctness, just chunk granularity. The block classification fix is more critical.

---

### Deferred Issue 2: Currency Format Inconsistency

**What was extracted:**
```
DOC-001: ['rs. 22', 'Rs 1', 'Rs. 15', 'Rs. 1', 'Rs. 1']
DOC-004: ['Rs.5', 'Rs.7', 'Rs.10', 'Rs.15', 'Rs.20']
DOC-009: ['rs   10', 'rs    16', 'rs   23', 'rs    24', 'rs    6']
```

**What appears to be wrong:**
Currency formatting is inconsistent:
- Mixed case (Rs vs rs)
- Inconsistent spacing (Rs.5 vs Rs. 5 vs Rs.     3000)
- Some values split across spans

**Why it's deferred:**
This is a normalization concern, not a structural one. The values are present and extractable. A currency normalization step can be added during chunking or preprocessing.

---

### Deferred Issue 3: "percent" as Text Instead of Symbol

**What was extracted:**
- DOC-001: 2 instances of "percent" as text
- DOC-010: 3 instances of "percent" as text
- (All other docs use `%` symbol correctly)

**Why it's deferred:**
Only 5 instances across 10 documents. Low impact on RAG performance. Can be handled with a simple regex replacement during text preprocessing.

---

### Deferred Issue 4: Exclusion Section False Positives

**What was extracted:**
DOC-001 Page 1: `'Inside: Let's get started! You're already awesome because you decided to protect your most important asset'`

**What appears to be wrong:**
The exclusion section detection flagged marketing text ("Let's get started!") because it contains "exclusion" somewhere in the full page text, but the block text shown is not an actual exclusion clause.

**Why it's deferred:**
This is a precision issue in the section detection, not a data loss issue. The exclusion sections are still captured in the full text. The chunker will benefit from having the full text available even if section labeling is imperfect.

---

### Deferred Issue 5: Multi-Column Reading Order in DOC-007

**What was extracted:**
DOC-007 Pages 1-2: "5 potential multi-column/overlap issues"

**What appears to be wrong:**
Some blocks on pages 1-2 of DOC-007 may be out of reading order. The audit script detected this based on Y-coordinate analysis.

**Why it's deferred:**
Only affects 2 pages of 1 document. The chunker can use bbox information to re-sort blocks if needed. Critical for data correctness but low priority before other fixes.

---

## D. Readiness Assessment for Chunking Experiments

**Status: NOT READY — Critical fixes required first**

The canonical representation has correct data preservation (text, clause numbers, tables, numbers), but three structural issues will severely impact chunking quality:

1. **Block type classification** must be fixed — chunker cannot distinguish headings from paragraphs
2. **Header/footer stripping** must be implemented — repeated text will create duplicate chunks
3. **Text cleaning** must be added — control characters and Unicode issues will cause encoding errors

**Recommended fix order:**
1. Fix `_classify_block` in normalizer (investigate why all blocks return "unknown")
2. Implement header/footer stripping in normalizer
3. Add text cleaning (control chars, Unicode normalization)
4. Then proceed with chunking experiments

---

## Post-Fix Validation

### Block Classification (Before/After)

| Metric | Before | After |
|--------|--------|-------|
| Total blocks | 16,062 | 14,291 |
| Unknown blocks | 16,062 (100%) | 0 (0%) |
| Paragraph | 0 | 11,096 (77.5%) |
| Clause | 0 | 1,296 (9.1%) |
| Heading | 0 | 768 (5.4%) |
| Footer | 0 | 668 (4.7%) |
| Header | 0 | 324 (2.3%) |
| List item | 0 | 139 (1.0%) |

**Subtype support added:** `section_heading`, `subsection_heading_l2`, `numbered_clause`, `bulleted_list`, `repeated_header`, `repeated_footer`, `page_number`

**Classification confidence:** Added `classification_confidence` field (0.0-1.0) calculated from actual rule weights. Average: 0.96 across all blocks.

### Header/Footer Handling (Before/After)

| Metric | Before | After |
|--------|--------|-------|
| `is_repeated` field | Not present | Added — 755 blocks marked `is_repeated=True` |
| `include_in_chunk_text` field | Not present | Added — 992 blocks marked `include_in_chunk_text=False` |
| Headers stripped from text | No | Yes — headers retain text but marked `include_in_chunk_text=False` |
| Raw text preserved | No | Yes — `raw_text` field preserves original |

Example header block:
```json
{
  "type": "header",
  "subtype": "repeated_header",
  "text": "STAR HEALTH AND ALLIED INSURANCE COMPANY LIMITED | PROSPECTUS",
  "raw_text": "STAR HEALTH AND ALLIED INSURANCE COMPANY LIMITED     |     PROSPECTUS",
  "is_repeated": true,
  "include_in_chunk_text": false,
  "source": "header"
}
```

### Control Characters & Unicode (Before/After)

| Metric | Before | After |
|--------|--------|-------|
| Tabs in normalized text | 1,677 | 0 |
| Bell chars (`\\x07`) in normalized text | 1,077 | 0 |
| Tabs in raw text (preserved) | N/A | 3,777 (intentional) |
| Bell chars in table text | Various | 0 (cleaned) |
| Unicode replacement chars (`U+FFFD`) | 0 | 0 |
| `has_unicode_issues` field | Not present | Added (detects U+FFFD, non-characters) |

**Note on Unicode:** The `` characters initially flagged in DOC-010 are valid Unicode (curved quotes U+2018/U+2019, en-dash U+2013, emoji U+1F609), not U+FFFD replacement characters. These are properly handled by JSON serialization with UTF-8 encoding.

### Table Regression Results

| Document | Tables (Before Fix) | Tables (After Fix 1) | Tables (After Fix 2) | Status |
|----------|-------------------|---------------------|---------------------|--------|
| DOC-004 | 0 | 2 | 2 | ✅ Pass |
| DOC-007 | 0 | 53 (false) | 0 | ✅ Pass |

Table text preserves tab-separated structure while removing bell characters:
```json
{
  "detection_method": "tab_separated",
  "text": "i.\t\nMonthly: 4%\t\nii.  Quarterly: 3%\t\niii.  Half Yearly: 2% \t\niv.  Yearly: 0%\n",
  "has_tab": true,
  "has_bell": false
}
```

### Pipeline Summary

| Metric | Value | Expected |
|--------|-------|----------|
| Documents | 10 | 10 ✅ |
| Total pages | 553 | 553 ✅ |
| OCR pages | 1 | 1 ✅ |
| Errors | 0 | 0 ✅ |
| Warnings | 0 | 0 ✅ |

### Representative Block Examples (Post-Fix)

**Paragraph with raw_text preserved:**
```json
{
  "type": "paragraph",
  "subtype": null,
  "text": "Inside: Let's get started! You're already awesome...",
  "raw_text": "Inside: Let's get started! You're already awesome...",
  "classification_confidence": 1.0
}
```

**Clause with control chars cleaned:**
```json
{
  "type": "clause",
  "subtype": "numbered_clause",
  "clause_id": "1",
  "text": "1. In-patient Treatment: We will cover the following Medical Expenses...",
  "raw_text": "1.\t\x07In-patient Treatment: We will cover the following Medical Expenses...",
  "classification_confidence": 0.9
}
```

**Header marked for exclusion:**
```json
{
  "type": "header",
  "subtype": "repeated_header",
  "text": "STAR HEALTH AND ALLIED INSURANCE COMPANY LIMITED | PROSPECTUS",
  "is_repeated": true,
  "include_in_chunk_text": false
}
```

**Footer (page number) marked for exclusion:**
```json
{
  "type": "footer",
  "subtype": "page_number",
  "text": "1",
  "is_repeated": false,
  "include_in_chunk_text": false
}
```

### Remaining Issues (Deferred)

1. **Paragraph splitting for large blocks** (>1000 chars) — 243 blocks still need paragraph-level splitting. Can be handled during chunking.
2. **Currency format normalization** — "Rs.5", "rs. 22", "Rs 1" formats are inconsistent. Low impact on RAG.
3. **"percent" as text** — 5 total instances across DOC-001 and DOC-010. Can be regex-normalized later.
4. **DOC-007 multi-column reading order** — 5 potential issues on pages 1-2. Low impact on 553 total pages.
5. **False-positive exclusion detection** — DOC-001 page 1 marketing text flagged as exclusions. Can be refined later.

### Readiness for Chunking

**Status: READY** — All critical Phase 1 normalization issues have been resolved:

1. ✅ Block classification: 0 unknown blocks, all classified with subtypes and confidence
2. ✅ Header/footer handling: Marked with `is_repeated` and `include_in_chunk_text=False`
3. ✅ Control characters: Removed from normalized text, preserved in `raw_text`
4. ✅ Raw/normalized text: `raw_text` and `normalized_text` fields added alongside `text`
5. ✅ Table regression: DOC-004 → 2 tables, DOC-007 → 0 tables
6. ✅ Unicode safety: No silent character corruption, U+FFFD detection added
7. ✅ Clause numbers preserved: 1,296 clauses detected with `clause_id` field
8. ✅ Legal numbers preserved: Percentages, currency, waiting periods intact
9. ✅ Subtype hierarchy: section_heading, subsection_heading, numbered_clause, bulleted_list
10. ✅ Raw source traceable: `raw_text` preserves original extraction for debugging
