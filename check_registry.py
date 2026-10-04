import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open(r'D:\LLM_PROJECT\configs\documents.json', 'r', encoding='utf-8') as f:
    reg = json.load(f)

for doc in reg['documents']:
    print(f"{doc['document_id']}: {doc['filename'][:50]}")
    print(f"  insurer={doc['insurer']}")
    print(f"  product={doc['product']}")
    print(f"  uin={doc['uin']}")
    print(f"  type={doc['document_type']}")
    print(f"  version={doc['version']}")
    print(f"  pages={doc['page_count']}")
    print(f"  review_flags={doc['review_flags']}")
    print()
