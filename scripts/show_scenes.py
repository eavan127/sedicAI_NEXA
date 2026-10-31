"""
Draw detector scenes with their boxes, so you can see what the detector is
trained on (and, with --model, what it finds).

    python scripts/show_scenes.py                         # 8 new random scenes
    python scripts/show_scenes.py --split test --n 6      # scenes from data/detect/test.npz
    python scripts/show_scenes.py --split test --model    # + the trained detector's boxes

Each panel is 10.24 ms (x, 256 columns of 40 us) by 3.2 MHz (y, 128 rows of
25 kHz, centre = 0 Hz). Brightness is dB above the noise floor (0-40 dB).
White boxes: ground truth, labelled class and SNR; dotted = too faint to see
(ignored in training and scoring). Red boxes: the detector, with its score.
Saves a PNG (default results/detector/scenes.png) and opens it.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.detect import scenes  # noqa: E402


def from_split(path, n, start):
    d = np.load(path)
    for i in range(start, min(start + n, len(d["images"]))):
        k = d["n_boxes"][i]
        yield (d["images"][i].astype(np.float32) / 255, d["boxes"][i, :k], d["ignore"][i, :k],
               d["cls"][i, :k], d["snr_db"][i, :k], f"{Path(path).stem} #{i} (seed {d['seed'][i]})")


def fresh(n, seed):
    for i in range(n):
        _, img, boxes, meta = scenes.make_scene(np.random.default_rng(seed + i))
        yield img, boxes, meta["ignore"], meta["cls"], meta["snr_db"], f"new scene, seed {seed + i}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("train", "val", "test"), help="draw from data/detect/<split>.npz")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--start", type=int, default=0, help="first scene index in the split")
    p.add_argument("--seed", type=int, default=7000, help="seed of the first new scene")
    p.add_argument("--model", action="store_true", help="also draw the trained detector's boxes")
    p.add_argument("--score", type=float, default=None, help="detector score threshold (default: from eval.json)")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "detector" / "scenes.png"))
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    items = list(from_split(REPO_ROOT / "data" / "detect" / f"{args.split}.npz", args.n, args.start)
                 if args.split else fresh(args.n, args.seed))

    net, thr = None, args.score
    if args.model:
        import json
        import torch
        from src.detect import model as det
        ckpt_dir = REPO_ROOT / "results" / "detector"
        ckpt = torch.load(ckpt_dir / "emitter_detector.pt", map_location="cpu")
        net = det.EmitterDetector(ckpt["width"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        if thr is None:
            ev = ckpt_dir / "eval.json"
            thr = json.loads(ev.read_text())["model_score_threshold"] if ev.exists() else 0.5

    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.4 * rows), squeeze=False)
    extent = (0, scenes.IMG_W * scenes.N_FFT / scenes.FS * 1e3, -scenes.FS / 2e6, scenes.FS / 2e6)
    sx = extent[1] / scenes.IMG_W                      # pixels -> ms
    sy = (extent[3] - extent[2]) / scenes.IMG_H        # pixels -> MHz
    for ax, (img, boxes, ignore, cls, snr, title) in zip(axes.ravel(), items):
        ax.imshow(img, origin="lower", aspect="auto", cmap="viridis", extent=extent, vmin=0, vmax=1)
        for b, ign, c, s in zip(boxes, ignore, cls, snr):
            ax.add_patch(Rectangle((b[0] * sx, b[1] * sy + extent[2]), (b[2] - b[0]) * sx, (b[3] - b[1]) * sy,
                                   fill=False, ec="white", lw=1.4, ls=":" if ign else "-"))
            ax.text(b[0] * sx, b[3] * sy + extent[2], f" {c} {s:.0f} dB", color="white", fontsize=6,
                    va="bottom", bbox=dict(fc="black", alpha=0.55, pad=0.4, lw=0))
        if net is not None:
            import torch
            from src.detect import model as det
            with torch.no_grad():
                (pb, ps), = det.decode(net(torch.from_numpy(img)[None, None]), score_thr=thr)
            for b, s in zip(pb, ps):
                ax.add_patch(Rectangle((b[0] * sx, b[1] * sy + extent[2]), (b[2] - b[0]) * sx,
                                       (b[3] - b[1]) * sy, fill=False, ec="red", lw=1.2))
                ax.text(b[2] * sx, b[1] * sy + extent[2], f"{s:.2f}", color="red", fontsize=6, ha="right")
        ax.set_title(f"{title}: {len(boxes)} emitter(s)", fontsize=8)
        ax.set_xlabel("time (ms)", fontsize=7)
        ax.set_ylabel("frequency (MHz)", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=90)
    print(f"saved {out}")
    if not args.no_open and hasattr(os, "startfile"):
        os.startfile(out)


if __name__ == "__main__":
    main()
