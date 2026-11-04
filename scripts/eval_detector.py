"""
Score the trained emitter detector against the classical energy detector on
the held-out test scenes. Fully offline.

    python scripts/eval_detector.py

Both detectors get a fair operating point: the score threshold (detector) and
the dB threshold (energy detector) are each chosen on the VALIDATION scenes
for best F1, then applied unchanged to the test scenes.

Reports, for each detector:
  * AP@0.5 and AP@0.3 (area under precision-recall; IoU 0.5 is the usual bar,
    0.3 forgives loose edges on very thin boxes)
  * at the chosen operating point: recall, precision, false alarms per scene
  * recall split by emitter SNR, by signal group and by visibility
Ignored (too-faint) emitters count as neither found nor missed.
Writes --out/eval.json and --out/eval_examples.png.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.detect import baseline, model as det  # noqa: E402

SNR_BUCKETS = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20)]
VIS_BUCKETS = [(3, 30), (30, 300), (300, 3000), (3000, 1e9)]


def load(path):
    d = np.load(path)
    n = d["n_boxes"]
    per = lambda key: [d[key][i, :n[i]] for i in range(len(n))]
    return {"images": d["images"].astype(np.float32) / 255, "boxes": per("boxes"),
            "ignore": per("ignore"), "snr_db": per("snr_db"), "group": per("group"),
            "visibility": per("visibility")}


def run_model(net, images, score_thr=0.01, batch=64):
    out = []
    with torch.no_grad():
        for s in range(0, len(images), batch):
            x = torch.from_numpy(images[s:s + batch]).unsqueeze(1)
            out += det.decode(net(x), score_thr=score_thr)
    return out


def run_baseline(images, threshold_db):
    return [baseline.detect(img, threshold_db=threshold_db) for img in images]


def score(preds, data, keep_thr=None, iou_thr=0.5):
    """Pool all scenes; returns AP and, if keep_thr is given, the operating point."""
    tps, scs, founds, n_gt, n_fa = [], [], [], 0, 0
    for (pb, ps), gb, gi in zip(preds, data["boxes"], data["ignore"]):
        if keep_thr is not None:
            pb, ps = pb[ps >= keep_thr], ps[ps >= keep_thr]
        tp, sc, found = det.match(pb, ps, gb, gi, iou_thr)
        tps.append(tp)
        scs.append(sc)
        founds.append(found)
        n_gt += int((~np.asarray(gi, bool)).sum())
        n_fa += int((~tp).sum())
    tp_all = np.concatenate(tps)
    res = {"ap": det.average_precision(tp_all, np.concatenate(scs), n_gt), "n_gt": n_gt}
    if keep_thr is not None:
        n_tp = int(tp_all.sum())
        res.update(recall=n_tp / max(n_gt, 1), precision=n_tp / max(len(tp_all), 1),
                   fa_per_scene=n_fa / len(preds), found=founds)
    return res


def best_f1_threshold(preds, data, candidates):
    best = (-1, None)
    for t in candidates:
        r = score(preds, data, keep_thr=t)
        f1 = 2 * r["recall"] * r["precision"] / max(r["recall"] + r["precision"], 1e-9)
        best = max(best, (f1, t))
    return best[1]


def breakdown(found, data):
    """Recall by SNR bucket, group and visibility bucket (non-ignored only)."""
    f = np.concatenate(found)
    keep = ~np.concatenate(data["ignore"]).astype(bool)
    snr, grp, vis = (np.concatenate(data[k])[keep] for k in ("snr_db", "group", "visibility"))
    f = f[keep]
    out = defaultdict(dict)
    for lo, hi in SNR_BUCKETS:
        m = (snr >= lo) & (snr < hi)
        out["snr_db"][f"{lo}..{hi}"] = (float(f[m].mean()) if m.any() else None, int(m.sum()))
    for g in sorted(set(grp)):
        m = grp == g
        out["group"][g] = (float(f[m].mean()), int(m.sum()))
    for lo, hi in VIS_BUCKETS:
        m = (vis >= lo) & (vis < hi)
        out["visibility"][f"{lo:g}..{hi:g}"] = (float(f[m].mean()) if m.any() else None, int(m.sum()))
    return out


def draw(images, data, model_preds, base_preds, path, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(n // 2, 2, figsize=(14, 3.2 * n // 2))
    for ax, i in zip(axes.ravel(), range(n)):
        ax.imshow(images[i], origin="lower", aspect="auto", cmap="viridis",
                  extent=(0, images.shape[2], 0, images.shape[1]))
        for b, ign in zip(data["boxes"][i], data["ignore"][i]):
            ax.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False,
                                   ec="white", lw=1.5, ls=":" if ign else "-"))
        for b in base_preds[i][0]:
            ax.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec="orange", lw=1, ls="--"))
        for b, s in zip(*model_preds[i]):
            ax.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=1.2))
            ax.text(b[0], b[3], f"{s:.2f}", color="red", fontsize=6)
        ax.set_title(f"test scene {i}: white = truth (dotted = too faint), red = detector, "
                     f"orange = energy detector", fontsize=7)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=80)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "detect"))
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "detector"))
    args = p.parse_args()

    ckpt = torch.load(Path(args.out) / "emitter_detector.pt", map_location="cpu")
    net = det.EmitterDetector(ckpt["width"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    val, test = load(Path(args.data) / "val.npz"), load(Path(args.data) / "test.npz")

    val_model = run_model(net, val["images"])
    model_thr = best_f1_threshold(val_model, val, np.arange(0.05, 0.96, 0.05))
    best = (-1, None)
    for t_db in np.arange(4, 16.5, 1.0):
        r = score(run_baseline(val["images"], t_db), val, keep_thr=0.0)
        f1 = 2 * r["recall"] * r["precision"] / max(r["recall"] + r["precision"], 1e-9)
        best = max(best, (f1, float(t_db)))
    base_thr = best[1]

    test_model = run_model(net, test["images"])
    test_base = run_baseline(test["images"], base_thr)
    report = {"checkpoint_epoch": ckpt["epoch"], "n_test_scenes": len(test["images"]),
              "model_score_threshold": float(model_thr), "baseline_threshold_db": base_thr}
    for name, preds, thr in (("detector", test_model, model_thr), ("energy_detector", test_base, 0.0)):
        op = score(preds, test, keep_thr=thr)
        report[name] = {"ap50": score(preds, test)["ap"], "ap30": score(preds, test, iou_thr=0.3)["ap"],
                        "recall": op["recall"], "precision": op["precision"],
                        "fa_per_scene": op["fa_per_scene"], "breakdown": breakdown(op["found"], test)}
    Path(args.out, "eval.json").write_text(json.dumps(report, indent=2))

    kept_model = [(b[s >= model_thr], s[s >= model_thr]) for b, s in test_model]
    draw(test["images"], test, kept_model, test_base, Path(args.out) / "eval_examples.png")

    print(f"test scenes: {len(test['images'])}, emitters counted: {score(test_model, test)['n_gt']}")
    print(f"operating points chosen on val: detector score >= {model_thr:.2f}, "
          f"energy detector {base_thr:.0f} dB")
    print(f"{'':<16}{'AP@0.5':>8}{'AP@0.3':>8}{'recall':>8}{'precis.':>8}{'FA/scene':>9}")
    for name in ("detector", "energy_detector"):
        r = report[name]
        print(f"{name:<16}{r['ap50']:>8.3f}{r['ap30']:>8.3f}{r['recall']:>8.3f}"
              f"{r['precision']:>8.3f}{r['fa_per_scene']:>9.2f}")
    for key in ("snr_db", "group", "visibility"):
        print(f"\nrecall by {key}:  detector | energy detector  (n)")
        for k, (rec, n) in report["detector"]["breakdown"][key].items():
            brec = report["energy_detector"]["breakdown"][key][k][0]
            fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
            print(f"  {k:<14} {fmt(rec)} | {fmt(brec)}  ({n})")
    print(f"\nwrote {Path(args.out) / 'eval.json'} and eval_examples.png")


if __name__ == "__main__":
    main()
