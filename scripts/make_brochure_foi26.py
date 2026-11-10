"""Build the two PDFs of the SEDIC@FOI2026 brochure from web/brochure-foi26.html.

Run from the repo root (needs Chrome or Edge, internet for the fonts and the QR
library, and PyMuPDF):

    python scripts/make_brochure_foi26.py

Writes:
  docs/NEXA_FOI26_eBrochure.pdf      e-reading: eight A5 pages, one per PDF page
  docs/NEXA_FOI26_eBrochure_2up.pdf  flat 2-up: two pages side by side per A4 landscape
                                     sheet, in reading order (1|2, 3|4, 5|6, 7|8).
                                     Print single-sided at 100% ("actual size"), no fold.
"""
import os
import pathlib
import shutil
import subprocess
import sys

import fitz

SRC = pathlib.Path("web/brochure-foi26.html").resolve()
EREAD = "docs/NEXA_FOI26_eBrochure.pdf"
TWO_UP = "docs/NEXA_FOI26_eBrochure_2up.pdf"
PAGES = 8

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chrome", "msedge",
]


def find_browser():
    for b in BROWSERS:
        path = b if os.path.isfile(b) else shutil.which(b)
        if path:
            return path
    sys.exit("No Chrome or Edge found: print web/brochure-foi26.html to PDF by hand (A5, no margins, background graphics on).")


def print_pdf():
    subprocess.run([
        find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--virtual-time-budget=15000", f"--print-to-pdf={os.path.abspath(EREAD)}", SRC.as_uri(),
    ], check=True, capture_output=True, timeout=180)
    doc = fitz.open(EREAD)
    assert doc.page_count == PAGES, f"expected {PAGES} pages, the browser made {doc.page_count}: a page overflows"
    print(EREAD, doc.page_count, "pages", os.path.getsize(EREAD) // 1024, "KB")
    return doc


def impose(src):
    w, h = 841.89, 595.28  # A4 landscape, points; an A5 page is exactly half of it
    out = fitz.open()
    for left in range(0, PAGES, 2):
        sheet = out.new_page(width=w, height=h)
        for k in (0, 1):
            sheet.show_pdf_page(fitz.Rect(k * w / 2, 0, (k + 1) * w / 2, h), src, left + k)
    out.save(TWO_UP, deflate=True, garbage=4)
    print(TWO_UP, out.page_count, "sheets", os.path.getsize(TWO_UP) // 1024, "KB")


if __name__ == "__main__":
    impose(print_pdf())
