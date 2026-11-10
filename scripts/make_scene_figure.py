"""
The brochure's cover picture: one synthetic wideband capture run through the
real pipeline (YOLO + energy detector, the active classifier, tracker and
resolver), with NEXA's own answers drawn on the spectrogram.

The older scene_boxes.png (scripts/make_deck_figures.py) drew the generator's
ground-truth boxes and labels, not what NEXA found. This one draws the
pipeline's tracks: green when the answer covers the true class, red when it
does not, grey when NEXA calls it uncertain.

    python scripts/make_scene_figure.py --seeds 0 40          # score each capture, pick one
    python scripts/make_scene_figure.py --pick 12             # draw capture 12

Writes web/img/foi26/scene_nexa.png and web/img/foi26/scene_nexa.json
(truth, answers and the seed, so the picture can be checked and remade), and
the brochure's first three architecture thumbnails from the same capture:
t_capture.png, t_spectrogram.png and t_detector.png (the detector's boxes).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pipeline import accepted, iou_tf  # noqa: E402
from src.config import CFG  # noqa: E402
from src.detect import scenes  # noqa: E402
from src.pipeline.capture import make_capture  # noqa: E402
from src.pipeline.classify import Classifier  # noqa: E402
from src.pipeline.run import HybridDetector, NetDetector, process_capture  # noqa: E402

FS = CFG["signal"]["fs"]
OUT = ROOT / "web" / "img" / "foi26"
GREEN, RED, GREY = "#7CFC00", "#FF5A4E", "#C9CED6"


def truth_box(t):
    """(t0, t1, f_lo, f_hi) of a true emitter, widened to >= 80 kHz like eval_pipeline."""
    fc, half = (t["f_lo"] + t["f_hi"]) / 2, max(t["f_hi"] - t["f_lo"], 80e3) / 2
    return t["t_start"] / FS, t["t_end"] / FS, fc - half, fc + half


def answer_box(a):
    return a["t_start_ms"] / 1e3, a["t_end_ms"] / 1e3, a["f_lo_khz"] * 1e3, a["f_hi_khz"] * 1e3


def covers(a, cls):
    return cls in a["classes"] or bool(accepted(cls) & set(a["classes"]))


def overlap(truth):
    """Largest time x frequency overlap between two true emitters, as a share of the smaller one."""
    best = 0.0
    for i, a in enumerate(truth):
        for b in truth[i + 1:]:
            A, B = truth_box(a), truth_box(b)
            w = max(0.0, min(A[1], B[1]) - max(A[0], B[0])) * max(0.0, min(A[3], B[3]) - max(A[2], B[2]))
            small = min((A[1] - A[0]) * (A[3] - A[2]), (B[1] - B[0]) * (B[3] - B[2]))
            best = max(best, w / small if small > 0 else 0.0)
    return best


def run(seed, det, clf, n_emitters, snr):
    rng = np.random.default_rng(seed)
    iq, truth = make_capture(rng, n_chunks=1, n_emitters=n_emitters, snr_db=snr)
    answers, _ = process_capture(iq, det, clf, rng=rng)
    for a in answers:
        a["truth"], a["iou"], best = None, 0.0, 0.3
        for t in truth:
            ov = iou_tf(truth_box(t), answer_box(a))
            if ov >= best:
                a["truth"], a["iou"], best = t["cls"], round(ov, 2), ov
        a["right"] = bool(a["truth"] and covers(a, a["truth"]))
    # an emitter counts as found when a box on it names it (one box may name two, e.g. radar + DRFM jammer)
    found = [t["cls"] for t in truth
             if any(iou_tf(truth_box(t), answer_box(a)) >= 0.3 and covers(a, t["cls"]) for a in answers)]
    return iq, truth, answers, found


def boxes_px(a):
    x0, x1 = a["t_start_ms"] / 1e3 * FS / scenes.N_FFT, a["t_end_ms"] / 1e3 * FS / scenes.N_FFT
    y0 = (a["f_lo_khz"] * 1e3 + FS / 2) / FS * scenes.IMG_H
    y1 = (a["f_hi_khz"] * 1e3 + FS / 2) / FS * scenes.IMG_H
    return x0, y0, x1, y1


def thumbs(iq, answers):
    img = scenes.spectrogram(iq)

    def save(name, draw):
        fig, ax = plt.subplots(figsize=(3.2, 1.8))
        draw(ax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.savefig(OUT / f"t_{name}.png", transparent=True, dpi=130, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    def spec(ax):
        ax.imshow(img, aspect="auto", cmap="magma", origin="lower", extent=[0, img.shape[1], 0, img.shape[0]])

    def det(ax):
        spec(ax)
        for a in answers:
            x0, y0, x1, y1 = boxes_px(a)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=GREEN, lw=1.4))
    save("capture", lambda ax: (ax.plot(iq[:6000].real, color="#2E7DB9", lw=.4),
                                ax.plot(iq[:6000].imag, color="#C77A00", lw=.4, alpha=.7)))
    save("spectrogram", spec)
    save("detector", det)


def draw(iq, answers, path):
    img = scenes.spectrogram(iq)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.imshow(img, aspect="auto", cmap="magma", origin="lower", extent=[0, img.shape[1], 0, img.shape[0]])
    for a in answers:
        x0, y0, x1, y1 = boxes_px(a)
        col = GREY if a["band"] == "uncertain" else (GREEN if a["right"] else RED)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.6))
        name = "uncertain" if a["band"] == "uncertain" else a["answer"]
        if y1 + 7 < img.shape[0]:      # label above the box, or inside it when the box touches the top
            ax.text(x0 + 1, y1 + 1.2, name, color=col, fontsize=8, fontweight="bold", va="bottom")
        else:
            ax.text(x0 + 1.5, y1 - 1.2, name, color=col, fontsize=8, fontweight="bold", va="top")
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(0, img.shape[0])
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifier", default=str(ROOT / "results" / "classifier" / "physics_aug_best.pt"))
    p.add_argument("--detector", default=str(ROOT / "results" / "detector" / "emitter_detector.pt"))
    p.add_argument("--emitters", type=int, default=5)
    p.add_argument("--snr", type=float, default=15.0, help="every emitter's SNR in dB")
    p.add_argument("--seeds", type=int, nargs=2, default=[0, 40], help="range of captures to score")
    p.add_argument("--pick", type=int, default=None, help="draw this capture")
    p.add_argument("--name", default="scene_nexa", help="output name: <name>.png, <name>.json")
    p.add_argument("--no-thumbs", action="store_true", help="do not rewrite the t_*.png thumbnails")
    p.add_argument("--overlap", type=float, default=0.0, metavar="FRAC",
                   help="with --clean: two true emitters must overlap by at least FRAC of the smaller one")
    p.add_argument("--clean", type=float, default=None, metavar="IOU",
                   help="search mode: print only captures where every emitter is found, every answer is right "
                        "and every box overlaps its emitter by at least IOU")
    a = p.parse_args()
    det, clf = HybridDetector(NetDetector(a.detector)), Classifier([a.classifier])

    if a.pick is None:
        tot = dict(emitters=0, found=0, right=0)
        for s in range(*a.seeds):
            _, truth, answers, found = run(s, det, clf, a.emitters, a.snr)
            right = sum(x["right"] for x in answers)
            if a.clean is not None:
                if (answers and len(found) == len(truth) and right == len(answers)
                        and min(x["iou"] for x in answers) >= a.clean and overlap(truth) >= a.overlap):
                    print(f"CLEAN seed {s} (overlap {overlap(truth):.2f}): " + ", ".join(f"{x['answer']} ({x['band']}, IoU {x['iou']})" for x in answers),
                          flush=True)
                continue
            tot["emitters"] += len(truth); tot["found"] += len(found); tot["right"] += right
            print(f"seed {s:>3}: {len(truth)} emitters, {len(found)} found, {right} right, "
                  f"{len(answers)} tracks | " + ", ".join(f"{x['answer']}{'' if x['right'] else '(x)'}" for x in answers),
                  flush=True)
        print("total", tot)
        return

    iq, truth, answers, found = run(a.pick, det, clf, a.emitters, a.snr)
    OUT.mkdir(parents=True, exist_ok=True)
    draw(iq, answers, OUT / f"{a.name}.png")
    if not a.no_thumbs:
        thumbs(iq, answers)
    keep = ("answer", "classes", "band", "confidence", "truth", "right", "iou", "t_start_ms", "t_end_ms", "f_lo_khz", "f_hi_khz")
    info = dict(seed=a.pick, snr_db=a.snr, classifier=Path(a.classifier).name,
                truth=[t["cls"] for t in truth], answers=[{k: x[k] for k in keep} for x in answers])
    (OUT / f"{a.name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
