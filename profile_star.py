import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import fitz

doc = fitz.open(r'D:\LLM_PROJECT\rag_policy_dataset\Prospectus_Star_Comprehensive_Insurance_Policy_V_12_dc6058c95e.pdf')
for i, page in enumerate(doc):
    start = time.time()
    text = page.get_text()
    blocks = page.get_text("blocks")
    words = page.get_text("words")
    fonts = page.get_fonts()
    images = page.get_images(full=True)
    img_count = len(images)

    # Image rects
    if img_count > 0:
        for img in images[:3]:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                pass

    elapsed = time.time() - start
    if elapsed > 0.5:
        print(f"Page {i+1}: slow ({elapsed:.2f}s), chars={len(text)}, images={img_count}")

print(f"\nTotal pages: {len(doc)}")
# Check if get_image_rects is the slow part
page = doc[0]
images = page.get_images(full=True)
print(f"Page 1 images: {len(images)}")
if images:
    start = time.time()
    rects = page.get_image_rects(images[0][0])
    print(f"get_image_rects: {time.time()-start:.3f}s")

doc.close()
