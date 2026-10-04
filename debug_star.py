import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import fitz

doc = fitz.open(r'D:\LLM_PROJECT\rag_policy_dataset\Prospectus_Star_Comprehensive_Insurance_Policy_V_12_dc6058c95e.pdf')

page = doc[0]
text = page.get_text()
print(f"Page 1: chars={len(text)}, words={len(page.get_text('words'))}, blocks={len(page.get_text('blocks'))}")
images = page.get_images(full=True)
print(f"Page 1: images={len(images)}")

start = time.time()
fonts = page.get_fonts()
print(f"Page 1: fonts={len(fonts)}, time={time.time()-start:.3f}s")

start = time.time()
text_dict = page.get_text('dict')
print(f"Page 1: text_dict time={time.time()-start:.3f}s")

if images:
    start = time.time()
    rects = page.get_image_rects(images[0][0])
    print(f"Page 1: get_image_rects time={time.time()-start:.3f}s")

start = time.time()
t = page.get_text()
print(f"Page 1: get_text time={time.time()-start:.3f}s")

# Check page 2
page2 = doc[1]
start = time.time()
t2 = page2.get_text()
print(f"Page 2: get_text time={time.time()-start:.3f}s, chars={len(t2)}")

doc.close()
print("Done")
