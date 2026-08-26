"""
How much do the numbers move when NOTHING changes?

Trains the same config several times with different random seeds and reports
the spread. Until you know that spread, you cannot tell a real improvement from
a different weight initialisation.

Motivating case: across three runs today, jamming recall read 0.803, 0.844 and
0.794 — a 5-point swing — while two unrelated config changes were made. Without
a noise floor, there is no way to attribute either.

Usage:
    python scripts/measure_variance.py --runs 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASS_TO_IDX, resolve_multilabel_thresholds  # noqa: E402
from src.evaluate import predict_probs  # noqa: E402
from src.train import load_data, stratified_split, train_model  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_run(X, y, snr_labels, seed, tr, va, te):
    """Train once (on the given split) and return per-class recall on `te`."""
    model, _ = train_model(X, y, snr_labels, tr, va, seed, verbose=False)

    threshold = resolve_multilabel_thresholds()
    present = predict_probs([model], X[te]) > threshold

    y_te = y[te]
    out = {}
    for c in CFG["judged_classes"]:
        idx = CLASS_TO_IDX[c]
        mask = y_te[:, idx] == 1
        if mask.any():
            out[c] = float(present[mask, idx].mean())
    return out


def main(runs):
    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    # Computed once, not per run -- d["seed"] never changes between runs, so
    # every run trained on a different SEED but must still be scored against
    # the SAME held-out split; recomputing it `runs` times was pure repeated
    # work for byte-identical output.
    tr, va, te = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])

    print(f"Training {runs} times on IDENTICAL data and config.")
    print("Only the random seed differs.\n")

    results = []
    for i in range(runs):
        r = one_run(X, y, snr_labels, seed=1000 + i, tr=tr, va=va, te=te)
        results.append(r)
        print(f"  run {i+1}: " + "  ".join(f"{c}={v:.4f}" for c, v in r.items()))

    print(f"\n{'class':<14}{'mean':>9}{'min':>9}{'max':>9}{'spread':>9}")
    print("-" * 50)
    worst = 0.0
    for c in results[0]:
        vals = [r[c] for r in results]
        spread = max(vals) - min(vals)
        worst = max(worst, spread)
        print(f"{c:<14}{np.mean(vals):>9.4f}{min(vals):>9.4f}{max(vals):>9.4f}{spread:>9.4f}")

    print()
    print(f"NOISE FLOOR: {worst:.3f} ({worst*100:.1f} points)")
    print()
    print(f"A config change must move a class by MORE than {worst*100:.1f} points")
    print("before you can call it an improvement. Anything smaller is noise.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    main(p.parse_args().runs)
