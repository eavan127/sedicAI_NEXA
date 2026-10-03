"""Build docs/NEXA_eBrochure_booklet.pdf from docs/NEXA_eBrochure.pdf: back cover + booklet imposition."""
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
centre("SEE IT RUN", 372, 11, orange)
centre("Scan to open the live console or keep this brochure", 396, 13, white, "helv")
clip = fitz.Rect(163, 610, 433, 790); s = 1.25          # the two QR cards on the team page
w, h = clip.width * s, clip.height * s
box = fitz.Rect(cx - w/2, 425, cx + w/2, 425 + h)
pg.draw_rect(box + (-10, -10, 10, 10), color=None, fill=white, radius=0.06)
pg.show_pdf_page(box, src, 6, clip=clip)
centre("sedic-ai-nexa.vercel.app", 700, 12, white, "cour")
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
