"""Splits per-class recall by SNR bin AND by what else shares the window.

Why this exists: evaluate.py reports recall-vs-SNR (one line per class) and
recall-by-context (alone / with_emitter / with_jammer) as two separate
breakdowns. Neither can tell you whether a decline at high SNR is caused by
jamming, because the SNR curve averages jammed and unjammed windows together
and the context table averages over every SNR bin.

The question it answers: FHSS recall falls from 90.6% at +2 dB to 81.2% at
+10 dB. Is that masking by a jammer getting worse, or is it the training
sampler seeing high-SNR windows only 3.9% of the time?

Reading the output, for one class:
  - every context line tilts down at high SNR  -> not jamming. Look at the
    SNR sampling weights in src/train.py (compute_snr_weights).
  - only "with_jammer" tilts down              -> masking worsens with SNR.
    The stft_freq_summary flag targets exactly this.
  - "with_jammer" is low but flat, others tilt -> two separate problems.

Also prints how jammed windows are distributed across the SNR bins. The
reasoning above assumes they are spread evenly, which holds only if JSR is
drawn independently of SNR. If that column is lopsided, the two axes are
entangled and neither reading applies.

Buckets match src/evaluate.py's recall_in_context exactly, so the totals
here reconcile with evals/scorecard.json.

Usage:
    python scripts/snr_by_context.py --ensemble --n-models 5
    python scripts/snr_by_context.py --ensemble --classes FHSS LFM_RADAR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (CFG, CLASS_TO_IDX, CLASSES,  # noqa: E402
                         resolve_multilabel_thresholds)
from src.evaluate import _predict_probs  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(ensemble, n_models, window_len):
    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    if ensemble:
        paths = [ckpt_dir / f"ensemble_{i}.pt" for i in range(n_models)]
        desc = f"{n_models}-model ensemble average"
    else:
        paths = [ckpt_dir / "best_model.pt"]
        desc = "best_model.pt"
    models = []
    for p in paths:
        m = AMC_CNN(num_classes=len(CLASSES), input_len=window_len).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        models.append(m)
    return models, desc


def main(ensemble, n_models, want_classes):
    X, y, snr = load_data()
    d = CFG["dataset"]
    _, _, test = stratified_split(y, snr, d["val_frac"], d["test_frac"], d["seed"])
    X, y, snr = X[test], y[test], snr[test]

    models, desc = load_models(ensemble, n_models, X.shape[-1])
    print(f"Checkpoint: {desc}")
    probs = _predict_probs(models, X)
    thresholds = resolve_multilabel_thresholds()

    jam = CLASS_TO_IDX["JAMMING"]
    has_jam = y[:, jam] == 1
    n_pos = y.sum(axis=1)
    bins = sorted({float(s) for s in snr})

    # Is JSR entangled with SNR? The whole reading assumes it is not.
    print("\n--- share of windows carrying a jammer, by SNR bin ---")
    print("  " + "".join(f"{b:>9.0f}" for b in bins) + "   dB")
    print("  " + "".join(f"{has_jam[snr == b].mean() * 100:>8.1f}%" for b in bins))
    print("  Even across bins means SNR and JSR are independent, as intended.")

    for cls in want_classes:
        j = CLASS_TO_IDX[cls]
        present = y[:, j] == 1
        detected = probs[:, j] >= thresholds[j]
        contexts = {
            "alone":        present & (n_pos == 1),
            "with_emitter": present & (n_pos > 1) & ~has_jam,
            "with_jammer":  present & has_jam & (j != jam),
        }
        print(f"\n--- {cls}: recall by SNR and context "
              f"(threshold {thresholds[j]:.3f}) ---")
        print(f"  {'context':<14}" + "".join(f"{b:>9.0f}" for b in bins) + "   dB")
        for name, mask in contexts.items():
            cells = []
            for b in bins:
                sel = mask & (snr == b)
                cells.append(f"{detected[sel].mean() * 100:>8.1f}%"
                             if sel.sum() else f"{'--':>9}")
            print(f"  {name:<14}" + "".join(cells) + f"   n={int(mask.sum())}")
        # slope of each line, high minus low, so the tilt is a single number
        print(f"  {'tilt +10 vs +2':<14}", end="")
        for name, mask in contexts.items():
            hi = mask & (snr == bins[-1])
            lo = mask & (snr == bins[-3])
            if hi.sum() and lo.sum():
                delta = (detected[hi].mean() - detected[lo].mean()) * 100
                print(f"  {name} {delta:+.1f}pts", end="")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--classes", nargs="+", default=["FHSS", "LFM_RADAR", "JAMMING"])
    a = p.parse_args()
    main(a.ensemble, a.n_models, a.classes)
