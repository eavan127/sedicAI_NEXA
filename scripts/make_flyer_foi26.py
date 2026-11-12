"""Build the two-sided A5 flyer for SEDIC@FOI2026 from web/brochure-foi26-flyer.html.

Run from the repo root (needs Chrome or Edge, internet for the fonts and the QR
library, and PyMuPDF):

    python scripts/make_flyer_foi26.py

Writes docs/NEXA_FOI26_Flyer_A5.pdf: two A5 pages, front then back. Print
double-sided on A5, flip on the long edge, at 100% ("actual size").
"""
import os
import pathlib
import subprocess
import sys

import fitz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_brochure_foi26 import find_browser  # noqa: E402

SRC = pathlib.Path("web/brochure-foi26-flyer.html").resolve()
OUT = "docs/NEXA_FOI26_Flyer_A5.pdf"
PAGES = 2


def main():
    subprocess.run([
        find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--virtual-time-budget=15000", f"--print-to-pdf={os.path.abspath(OUT)}", SRC.as_uri(),
    ], check=True, capture_output=True, timeout=180)
    doc = fitz.open(OUT)
    assert doc.page_count == PAGES, f"expected {PAGES} pages, the browser made {doc.page_count}: a page overflows"
    print(OUT, doc.page_count, "pages", os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
