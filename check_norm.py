import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open(r'D:\LLM_PROJECT\data\normalized\DOC-004.json', 'r', encoding='utf-8') as f:
    doc = json.load(f)

print('=== DOC-004 (Star Comprehensive - with OCR) ===')
print(f'document_id: {doc["document_id"]}')
print(f'filename: {doc["filename"]}')
print(f'insurer: {doc["insurer"]}')
print(f'product: {doc["product"]}')
print(f'uin: {doc["uin"]}')
print(f'page_count: {doc["page_count"]}')
print(f'extraction_method: {doc["extraction_method"]}')
print(f'ocr_pages: {doc["ocr_pages"]}')
print(f'ocr_engine: {doc["ocr_engine"]}')
print(f'detected_clauses count: {len(doc["detected_clauses"])}')
print(f'detected_headings count: {len(doc["detected_headings"])}')
print(f'detected_tables count: {len(doc["detected_tables"])}')
print()

page1 = doc['pages'][0]
print(f'Page 1:')
print(f'  ocr_used: {page1["ocr_used"]}')
print(f'  char_count: {page1["char_count"]}')
print(f'  blocks: {len(page1["blocks"])}')
print(f'  detected_headers: {page1["detected_headers"]}')
print(f'  detected_footers: {page1["detected_footers"]}')
if page1['blocks']:
    b = page1['blocks'][0]
    print(f'  first block: type={b["type"]}, text={b["text"][:100]}, source={b["source"]}')
print()

page3 = doc['pages'][2]
print(f'Page 3:')
print(f'  ocr_used: {page3["ocr_used"]}')
print(f'  char_count: {page3["char_count"]}')
print(f'  block_count: {len(page3["blocks"])}')
print(f'  layout_type: {page3["layout_type"]}')
if page3['blocks']:
    print(f'  first block: type={page3["blocks"][0]["type"]}, text={page3["blocks"][0]["text"][:100]}')
    print(f'  font_size: {page3["blocks"][0]["font_size"]}')
print()

if doc['detected_clauses']:
    print(f'First 3 detected clauses:')
    for c in doc['detected_clauses'][:3]:
        print(f'  page={c["page"]}, clause={c["clause_number"]}, text={c["text"][:80]}')

# Check DOC-001 as well
with open(r'D:\LLM_PROJECT\data\normalized\DOC-001.json', 'r', encoding='utf-8') as f:
    doc1 = json.load(f)

print()
print('=== DOC-001 (Digit Health) ===')
page1 = doc1['pages'][0]
print(f'Page 1 blocks ({len(page1["blocks"])}):')
for b in page1['blocks'][:5]:
    print(f'  type={b["type"]}, text={b["text"][:80]}, font_size={b["font_size"]}, bold={b["is_bold"]}')

print(f'\nDetected clauses: {len(doc1["detected_clauses"])}')
print(f'Detected headings: {len(doc1["detected_headings"])}')
print(f'Detected headers: {len(doc1["detected_headers"])}')
print(f'Detected footers: {len(doc1["detected_footers"])}')
if doc1['detected_clauses']:
    for c in doc1['detected_clauses'][:5]:
        print(f'  page={c["page"]}, clause={c["clause_number"]}, text={c["text"][:60]}')
