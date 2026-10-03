"""Build docs/NEXA_eBrochure_booklet.pdf from docs/NEXA_eBrochure.pdf: a back cover plus booklet imposition.

Run from the repo root after regenerating docs/NEXA_eBrochure.pdf.
"""
import fitz, os
src = fitz.open("docs/NEXA_eBrochure.pdf")
assert src.page_count == 7, "booklet layout assumes a 7-page brochure"
A4W, A4H = 595.28, 841.89
M = 12 / 25.4 * 72
bc = fitz.open(); pg = bc.new_page(width=A4W, height=A4H)
slate = (14/255, 22/255, 34/255); orange = (240/255, 160/255, 32/255); white = (1, 1, 1); dim = (0.60, 0.65, 0.71)
pg.draw_rect(fitz.Rect(M, M, A4W - M, A4H - M), color=None, fill=slate, radius=0.035)
cx = A4W / 2
lw = 300; lh = lw * 264 / 1100
pg.insert_image(fitz.Rect(cx - lw/2, 150, cx + lw/2, 150 + lh), filename="web/img/nexa-logo-light.png")
def centre(text, y, size, color, font="hebo"):
    w = fitz.get_text_length(text, fontname=font, fontsize=size)
    pg.insert_text((cx - w/2, y), text, fontname=font, fontsize=size, color=color)
centre("A Dual-Branch Fusion CNN for", 270, 17, white)
centre("Multi-Label RF Signal Classification", 292, 17, white)
# No QR codes here: they live on the team page (inside back cover), so nothing is repeated.
pg.draw_line((cx - 60, 345), (cx + 60, 345), color=orange, width=2)
centre("AI-powered RF spectrum intelligence", 392, 15, white, "helv")
centre("Offline  ·  In the browser  ·  Human in the loop", 418, 11.5, dim, "helv")
centre("LIVE CONSOLE", 520, 10, orange)
centre("sedic-ai-nexa.vercel.app", 546, 16, white, "cour")
centre("Scan the codes inside this cover for the console,", 600, 10.5, dim, "helv")
centre("the demo video and this brochure.", 615, 10.5, dim, "helv")
centre("Group NEXA  ·  Universiti Teknologi PETRONAS", 756, 10.5, dim, "helv")
centre("SEDIC 2026 Grand Finale  ·  7 October 2026", 772, 10.5, dim, "helv")
pages = [(src, i) for i in range(7)] + [(bc, 0)]
order = [(8, 1), (2, 7), (6, 3), (4, 5)]                 # sheet 1 front, sheet 1 back, sheet 2 front, sheet 2 back
out = fitz.open(); W, H = A4H, A4W
for l, r in order:
    sh = out.new_page(width=W, height=H)
    for k, n in enumerate((l, r)):
        doc, idx = pages[n - 1]
        sh.show_pdf_page(fitz.Rect(k * W/2, 0, (k + 1) * W/2, H), doc, idx)
out.save("docs/NEXA_eBrochure_booklet.pdf", deflate=True, garbage=4)
print("sides", out.page_count, os.path.getsize("docs/NEXA_eBrochure_booklet.pdf") // 1024, "KB")
