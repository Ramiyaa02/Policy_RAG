with open(r'D:\LLM_PROJECT/src/ingestion/pdf_inspector.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_start = '    def _extract_fonts_fast(self, page: fitz.Page) -> list[FontInfo]:'
next_section = '    # ------------------------------------------------------------------\n    # Helper: table detection'

idx_start = content.index(old_start)
idx_end = content.index(next_section, idx_start)

new_block = '''    def _extract_fonts_fast(self, page: fitz.Page) -> list[FontInfo]:
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

'''

content = content[:idx_start] + new_block + content[idx_end:]

with open(r'D:\LLM_PROJECT/src/ingestion/pdf_inspector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done - fixed font extraction')
