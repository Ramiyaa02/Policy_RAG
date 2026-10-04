with open(r'D:\LLM_PROJECT/src/ingestion/pdf_inspector.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_start = '    def _extract_fonts_fast(self, page: fitz.Page) -> list[FontInfo]:'
next_section = '    # ------------------------------------------------------------------\n    # Helper: table detection'

idx_start = content.index(old_start)
idx_end = content.index(next_section, idx_start)

old_block = content[idx_start:idx_end]

new_block = '''    def _extract_fonts_fast(self, page: fitz.Page) -> list[FontInfo]:
        """Extract font information from a page quickly using page.get_fonts() only."""
        fonts: list[FontInfo] = []
        try:
            font_list = page.get_fonts()
        except Exception:
            return fonts

        if not font_list:
            return fonts

        for f in font_list:
            font_name = f[3] if len(f) > 3 else str(f[0])
            flags = f[1] if len(f) > 1 else 0
            fonts.append(FontInfo(
                name=font_name,
                size=12.0,
                count=0,
                bold=bool(flags & 2),
                italic=bool(flags & 1),
            ))

        return fonts

'''

content = content[:idx_start] + new_block + content[idx_end:]

with open(r'D:\LLM_PROJECT/src/ingestion/pdf_inspector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done - replaced font extraction method')
