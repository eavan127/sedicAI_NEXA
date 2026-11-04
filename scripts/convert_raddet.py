"""
Convert RadDet (Huang et al., ICASSP 2025; github.com/abcxyzi/RadDet) into the
detector's scene format (scripts/make_scenes.py), so the detector can be
trained or scored on radar scenes another group drew.

    python scripts/convert_raddet.py
    python scripts/convert_raddet.py --src data/external/raddet/RadDet40k128HW009Tv2 --limit 500

Download RadDet40k128HW009Tv2 (RadDet-9T, 128 px: up to 9 radars a frame,
10% noise-only, about 0.8 GB) from kaggle.com/datasets/abcxyzi/raddet-icassp-2025
and unpack it into data/external/raddet/ (gitignored). Licence CC BY-NC-SA 4.0:
research and exhibition use only.

RadDet is NOT our receiver. Each frame is 2 ms of a 500 MHz band (one pixel
row ~3.9 MHz, wider than our whole 3.2 MHz band); ours is 10.24 ms of 3.2 MHz.
It ships finished max-hold spectrogram PNGs, not IQ, so the frames cannot be
redrawn at our band -- only re-shaped to our image size. Treat it as an
experiment: train with and without it and keep it only if the scores improve.

What the conversion does, per frame:
  * Colours -> values: the PNGs are matplotlib's "jet" colour map (checked:
    every pixel is a jet colour), so each pixel is mapped back to 0..1.
  * Values -> our brightness scale: our images are dB above the noise floor
    (src/detect/scenes.spectrogram). RadDet's scale is not published, so each
    frame is set so its noise has the same median and spread as our images'
    noise: (value - median) / noise spread * OUR_NOISE_SPREAD, clipped to 0..1.
  * Shape: frequency is flipped (RadDet has high frequencies at the top, we
    have row 0 = lowest), and time is stretched 128 -> 256 columns (each
    column repeated twice). The 128 frequency rows stay as they are.
  * Boxes: YOLO (class, x centre, y centre, width, height; 0..1) -> our
    [x0, y0, x1, y1] pixels, with the same flip and stretch.
  * snr_db and class name from RadDet's metadata (same order as the labels).
  * visibility: the energy-detection deflection of the box's own pixels,
    sum(pixel z-score) / sqrt(n pixels) -- the pixel version of our rule;
    below scenes.MIN_VISIBILITY the box is marked ignore (about 5% of boxes).

RadDet's boxes mark each radar's nominal band, not what the image shows. A
strong pulse splatters into a full-height stripe in the max-hold image, mostly
outside its box. Measured on 300 converted frames, the share of the image's
bright energy inside the boxes: 80-84% at -12 and -4 dB, 52% at +4 dB, 42% at
+12 dB, 30% at +20 dB (our own scenes hold 90% or more). Frames at +4 dB and
above therefore teach a different box rule from ours; snr_db is stored per
box, so training can keep only frames with snr_db <= -4.

Saved per split in --out (default data/detect_raddet/, gitignored), keys as in
make_scenes.py, but with RADDET_MAX = 9 box slots a scene instead of 6: 30% of
RadDet-9T frames hold 7-9 radars, and dropping boxes would leave radars in the
image unlabelled. The detector scripts read the first n_boxes slots, so they
work with either. group is "raddet", seed is the RadDet frame number.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.detect import scenes  # noqa: E402

RADDET_MAX = 9                     # RadDet-9T: up to 9 radars a frame
CLASS_NAMES = ("Rect", "Barker", "Frank", "P1", "P2", "P3", "P4", "Px", "ZadoffChu", "LFM", "FMCW")
# Spread of the noise in our images, 0..1 scale: measured on 500 frames of
# data/detect/test.npz as median(pixels above the floor) / 0.6745.
OUR_NOISE_SPREAD = 0.145
_JET = None


def jet_lut():
    global _JET
    if _JET is None:
        import matplotlib
        _JET = (matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)
    return _JET


def decode_jet(rgb):
    """(H, W, 3) uint8 jet-coloured image -> (H, W) float values 0..1."""
    flat = rgb.reshape(-1, 3)
    colours, inverse = np.unique(flat, axis=0, return_inverse=True)
    d = ((colours[:, None, :].astype(np.float32) - jet_lut()[None]) ** 2).sum(-1)
    return (d.argmin(1)[inverse.reshape(-1)] / 255).reshape(rgb.shape[:2])


def to_our_scale(v):
    """RadDet values -> our image scale: noise floor at 0, noise spread as ours."""
    med = np.median(v)
    spread = 1.4826 * np.median(np.abs(v - med)) + 1e-6
    return np.clip((v - med) / spread * OUR_NOISE_SPREAD, 0, 1).astype(np.float32), med, spread


def yolo_to_pixels(x, y, w, h, src_h=128):
    """One YOLO box (0..1, y down = high frequency) -> our [x0, y0, x1, y1]
    pixels on the IMG_H x IMG_W image (row 0 = lowest frequency)."""
    x0, x1 = (x - w / 2) * scenes.IMG_W, (x + w / 2) * scenes.IMG_W
    y0, y1 = (1 - (y + h / 2)) * scenes.IMG_H, (1 - (y - h / 2)) * scenes.IMG_H
    return np.clip([x0, y0, x1, y1], 0, [scenes.IMG_W, scenes.IMG_H, scenes.IMG_W, scenes.IMG_H]).astype(np.float32)


def convert_frame(src, split, stem):
    """One RadDet frame -> (image (IMG_H, IMG_W) 0..1, boxes (k, 4), meta)."""
    from PIL import Image

    rgb = np.asarray(Image.open(src / "images" / split / f"{stem}.png"))[..., :3]
    v = decode_jet(rgb)
    img, med, spread = to_our_scale(v)
    img = np.repeat(img[::-1], scenes.IMG_W // img.shape[1], axis=1)      # flip, stretch time
    z = np.repeat(((v - med) / spread)[::-1], scenes.IMG_W // v.shape[1], axis=1)

    label = src / "labels" / split / f"{stem}.txt"
    rows = [list(map(float, line.split())) for line in open(label)] if label.exists() else []
    md = [m for m in json.load(open(src / "metadata" / split / f"{stem}.json")) if m.get("RadarStatus", 1) == 1]
    boxes, meta = [], {"group": [], "cls": [], "snr_db": [], "visibility": [], "ignore": []}
    for j, (c, x, y, w, h) in enumerate(rows):
        b = yolo_to_pixels(x, y, w, h)
        x0, y0, x1, y1 = int(b[0]), int(b[1]), int(np.ceil(b[2])), int(np.ceil(b[3]))
        patch = z[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        vis = float(patch.sum() / np.sqrt(patch.size))
        snr = md[j]["SNR"] if j < len(md) and md[j]["ClassIdx"] == int(c) else np.nan
        boxes.append(b)
        for k, val in zip(meta, ("raddet", CLASS_NAMES[int(c)], snr, vis, vis < scenes.MIN_VISIBILITY)):
            meta[k].append(val)
    return img, np.array(boxes, dtype=np.float32).reshape(-1, 4), meta


def one(job):
    src, split, stem = job
    img, boxes, meta = convert_frame(Path(src), split, stem)
    k = len(boxes)
    row = {
        "images": np.round(img * 255).astype(np.uint8),
        "boxes": np.zeros((RADDET_MAX, 4), np.float32), "n_boxes": k,
        "ignore": np.zeros(RADDET_MAX, bool), "snr_db": np.zeros(RADDET_MAX, np.float32),
        "visibility": np.zeros(RADDET_MAX, np.float32),
        "group": np.full(RADDET_MAX, "", "<U8"), "cls": np.full(RADDET_MAX, "", "<U20"),
        "seed": int(stem),
    }
    row["boxes"][:k] = boxes
    for key in ("ignore", "snr_db", "visibility", "group", "cls"):
        row[key][:k] = meta[key]
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(REPO_ROOT / "data" / "external" / "raddet" / "RadDet40k128HW009Tv2"))
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "detect_raddet"))
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=0, help="convert only the first N frames of each split")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not (src / "images").is_dir():
        sys.exit(f"no RadDet images under {src} -- download and unpack it first (see --help)")
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        t0 = time.time()
        stems = sorted(f[:-4] for f in os.listdir(src / "images" / split) if f.endswith(".png"))
        if args.limit:
            stems = stems[:args.limit]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(one, [(str(src), split, s) for s in stems], chunksize=64))
        data = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
        np.savez_compressed(out / f"{split}.npz", **data)
        print(f"{split}: {len(rows)} frames, {data['n_boxes'].sum()} radars ({data['ignore'].sum()} too faint), "
              f"{time.time() - t0:.0f}s -> {out / (split + '.npz')}")


if __name__ == "__main__":
    main()
