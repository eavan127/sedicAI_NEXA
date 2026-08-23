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

from src.config import CFG, CLASSES, CLASS_TO_IDX  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import (compute_class_weights, compute_snr_weights, load_data,  # noqa: E402
                        set_seed, stratified_split)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # noqa: E402
import torch.nn as nn  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_run(X, y, snr_labels, seed):
    """Train once and return per-class recall on the held-out test split."""
    set_seed(seed)
    d, t = CFG["dataset"], CFG["training"]
    tr, va, te = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])

    noise_floor_idx = CLASS_TO_IDX.get("NOISE_FLOOR")
    neutral_classes = [noise_floor_idx] if noise_floor_idx is not None else []
    dampen = {noise_floor_idx: 0.5} if noise_floor_idx is not None else None

    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.float32)  # multi-hot -> BCEWithLogitsLoss wants float targets
    train_sampler = WeightedRandomSampler(
        compute_snr_weights(snr_labels[tr], y[tr], neutral_classes),
        num_samples=len(tr), replacement=True)
    train_loader = DataLoader(TensorDataset(X_t[tr], y_t[tr]),
                              batch_size=t["batch_size"], sampler=train_sampler)
    val_loader = DataLoader(TensorDataset(X_t[va], y_t[va]), batch_size=t["batch_size"])

    model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    # Multi-label: each class is an independent yes/no -- see src/train.py.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_class_weights(y, len(CLASSES), dampen=dampen).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=t["scheduler_patience"])

    best_loss, best_state = float("inf"), None
    for _ in range(t["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vloss += criterion(model(xb), yb).item() * xb.size(0)
        vloss /= len(val_loader.dataset)
        sched.step(vloss)
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    threshold = CFG.get("multilabel_threshold", 0.5)
    # Batched -- X[te] can be thousands of test windows; one unbatched
    # forward pass OOMs a real GPU once the dataset is full-sized (see the
    # same fix in src/evaluate.py and scripts/train_ensemble.py).
    eval_batch_size = 256
    probs_chunks = []
    with torch.no_grad():
        for i in range(0, len(te), eval_batch_size):
            batch = torch.tensor(X[te][i:i + eval_batch_size]).to(DEVICE)
            probs_chunks.append(torch.sigmoid(model(batch)).cpu().numpy())
    present = np.concatenate(probs_chunks, axis=0) > threshold

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
    print(f"Training {runs} times on IDENTICAL data and config.")
    print("Only the random seed differs.\n")

    results = []
    for i in range(runs):
        r = one_run(X, y, snr_labels, seed=1000 + i)
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
