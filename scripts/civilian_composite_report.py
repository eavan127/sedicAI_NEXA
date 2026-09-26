"""
Why are the civilian classes below the line? Split their recall by whether
anything ELSE shares the window.

The finding this exists to track (measured 2026-09-25, best_model.pt on the
held-out test split, SNR >= +2 dB):

    class    standalone   composite
    BPSK          1.000       0.557
    QPSK          0.997       0.061     <-- the problem
    16QAM         0.941       0.559
    64QAM         0.942       0.415

QPSK is recognised essentially perfectly on its own and almost never when a
radar, hopper or jammer shares the window. The cause is NOT that the evidence
is lost -- on composite QPSK windows the mean 16QAM score RISES from 0.006 to
0.218 and 64QAM from 0.004 to 0.199, both above QPSK's own 0.180. The model
still sees a linearly-modulated civilian signal; it assigns the wrong
constellation order.

The mechanism: QPSK is four tight clusters. A co-present emitter smears them
into a diffuse cloud, and a diffuse cloud is what a denser constellation looks
like. 16QAM survives because it is already a cloud; BPSK survives because two
points is the most distinctive shape of all.

Run this against any experiment checkpoint to see whether a change moved the
composite column. Flags must match the checkpoint or load_state_dict fails
loudly rather than scoring something meaningless -- pass none of them and it
reads the flags from the checkpoint's own _config.json sidecar when there is
one.

Usage:
    python scripts/civilian_composite_report.py
    python scripts/civilian_composite_report.py --checkpoint results/experiment_cum.pt
    python scripts/civilian_composite_report.py --checkpoint X.pt --cumulant-features
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CFG, CLASSES, CLASS_TO_IDX  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIVILIAN = ["BPSK", "QPSK", "16QAM", "64QAM"]
COMPETITORS = ["LFM_RADAR", "FHSS", "JAMMING"]


def flags_for(ckpt_path, args):
    """Explicit flags win; otherwise read the run's own sidecar; otherwise all off."""
    if any([args.freq_summary, args.keep_rows, args.cumulant_features]):
        return dict(stft_freq_summary=args.freq_summary,
                    stft_keep_rows=args.keep_rows,
                    cumulant_features=args.cumulant_features)
    sidecar = ckpt_path.parent / f"{ckpt_path.stem}_config.json"
    if sidecar.exists():
        f = json.loads(sidecar.read_text())
        print(f"(flags read from {sidecar.name})")
        return dict(stft_freq_summary=f.get("stft_freq_summary", False),
                    stft_keep_rows=f.get("stft_keep_rows", False),
                    cumulant_features=f.get("cumulant_features", False))
    return dict(stft_freq_summary=False, stft_keep_rows=False, cumulant_features=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "results" / "best_model.pt")
    ap.add_argument("--min-snr", type=float, default=2.0,
                    help="only score windows at or above this SNR (default +2 dB, where "
                         "standalone civilian recall is already ~1.0, so the composite gap "
                         "is not confounded with plain noise)")
    ap.add_argument("--freq-summary", action="store_true")
    ap.add_argument("--keep-rows", action="store_true")
    ap.add_argument("--cumulant-features", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"no checkpoint at {args.checkpoint}")
        return 1

    flags = flags_for(args.checkpoint, args)
    model = AMC_CNN(num_classes=len(CLASSES), input_len=CFG["signal"]["window_len"], **flags).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    _, _, te = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])
    Xte, yte, ste = X[te], y[te].astype(int), snr_labels[te]

    scores = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            xb = torch.tensor(Xte[i:i + 512], dtype=torch.float32).to(DEVICE)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    scores = np.concatenate(scores)

    thr = CFG.get("multilabel_thresholds_per_class", {})
    fallback = CFG.get("multilabel_threshold", 0.5)
    pred = np.stack([(scores[:, CLASS_TO_IDX[c]] >= thr.get(c, fallback)).astype(int)
                     for c in CLASSES], axis=1)

    hi = ste >= args.min_snr
    print(f"\n{args.checkpoint.name}   flags: "
          + "  ".join(f"{k}={v}" for k, v in flags.items()))
    print(f"test split {len(te):,} windows, scoring SNR >= {args.min_snr:g} dB\n")

    print(f"{'class':<8} {'standalone':>11} {'composite':>11} {'comp frac':>10} {'blended':>9}")
    print("-" * 54)
    out = {}
    for c in CIVILIAN:
        i = CLASS_TO_IDX[c]
        pos = (yte[:, i] == 1) & hi
        al, co = pos & (yte.sum(1) == 1), pos & (yte.sum(1) > 1)
        if not pos.sum():
            continue
        frac = co.sum() / pos.sum()
        a, b = pred[al, i].mean(), pred[co, i].mean()
        out[c] = {"standalone": float(a), "composite": float(b),
                  "composite_fraction": float(frac), "blended": float(a * (1 - frac) + b * frac)}
        print(f"{c:<8} {a:>11.3f} {b:>11.3f} {frac:>10.1%} {out[c]['blended']:>9.3f}")

    print(f"\ncomposite recall by which class shares the window:")
    print(f"{'class':<8} " + " ".join(f"{'+' + o:>14}" for o in COMPETITORS))
    print("-" * 54)
    for c in CIVILIAN:
        i = CLASS_TO_IDX[c]
        cells = []
        for o in COMPETITORS:
            m = (yte[:, i] == 1) & hi & (yte[:, CLASS_TO_IDX[o]] == 1)
            cells.append(f"{pred[m, i].mean():.3f}" if m.sum() else "-")
        print(f"{c:<8} " + " ".join(f"{x:>14}" for x in cells))

    q = CLASS_TO_IDX["QPSK"]
    co = (yte[:, q] == 1) & hi & (yte.sum(1) > 1)
    print(f"\non composite QPSK windows (n={co.sum()}), mean score per civilian class:")
    print("  (if the fix works, QPSK rises above 16QAM/64QAM here)")
    for c in CIVILIAN:
        print(f"   {c:<8} {scores[co, CLASS_TO_IDX[c]].mean():.3f}")

    if "QPSK" in out:
        need = 0.80
        frac = out["QPSK"]["composite_fraction"]
        target = (need - out["QPSK"]["standalone"] * (1 - frac)) / frac if frac else float("nan")
        print(f"\nQPSK blended is {out['QPSK']['blended']:.3f}. To reach {need:.0%} with standalone "
              f"at {out['QPSK']['standalone']:.3f},\ncomposite recall must reach {target:.3f} "
              f"(it is {out['QPSK']['composite']:.3f} now).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
