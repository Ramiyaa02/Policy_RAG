import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import fitz

doc = fitz.open(r'D:\LLM_PROJECT\rag_policy_dataset\Prospectus_Star_Comprehensive_Insurance_Policy_V_12_dc6058c95e.pdf')
page = doc[0]
fonts = page.get_fonts()
print(f"Number of fonts: {len(fonts)}")
for i, f in enumerate(fonts):
    print(f"Font {i}: type={type(f)}, len={len(f)}")
    for j, val in enumerate(f):
        print(f"  [{j}]: type={type(val).__name__}, value={repr(val)[:100]}")
doc.close()
