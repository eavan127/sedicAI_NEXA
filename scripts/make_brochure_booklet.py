"""Build the print variants of the e-brochure from docs/NEXA_eBrochure.pdf.

Run from the repo root after regenerating docs/NEXA_eBrochure.pdf (printed from
web/brochure.html, eight A4 pages: cover, six content pages, back cover).

Writes:
  docs/NEXA_eBrochure_booklet.pdf  folded booklet: two A4 sheets, double-sided,
                                   flipped on the short edge, folded in half
  docs/NEXA_eBrochure_2up.pdf      two pages side by side per sheet, in reading order
"""
import os

import fitz

SRC = "docs/NEXA_eBrochure.pdf"
src = fitz.open(SRC)
assert src.page_count == 8, f"booklet layout assumes an 8-page brochure, found {src.page_count}"

W, H = 841.89, 595.28  # A4 landscape, points


def impose(pairs, path):
    out = fitz.open()
    for left, right in pairs:
        sheet = out.new_page(width=W, height=H)
        for k, n in enumerate((left, right)):
            if n is not None:
                sheet.show_pdf_page(fitz.Rect(k * W / 2, 0, (k + 1) * W / 2, H), src, n - 1)
    out.save(path, deflate=True, garbage=4)
    print(path, out.page_count, "sides", os.path.getsize(path) // 1024, "KB")


# sheet 1 front, sheet 1 back, sheet 2 front, sheet 2 back
impose([(8, 1), (2, 7), (6, 3), (4, 5)], "docs/NEXA_eBrochure_booklet.pdf")
impose([(1, 2), (3, 4), (5, 6), (7, 8)], "docs/NEXA_eBrochure_2up.pdf")
