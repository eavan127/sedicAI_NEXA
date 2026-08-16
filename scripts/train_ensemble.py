"""
Train N models with different seeds and average their predictions.

Why this helps here specifically: seed variance measured 2.2 points on radar,
2.5 on FHSS and 8.9 on jamming. A single run lands at a random point inside
that range — sometimes 0.889, sometimes 0.911. Averaging softmax outputs
cancels the initialisation noise, so the result sits near the top of the range
rather than wherever the dice fell.

This is a MODEL-side change. It needs no generator edits and no coordination,
which is why it is the last lever available when the data is settled.

Usage:
    python scripts/train_ensemble.py --models 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.data.preprocess import phase_rotate_batch  # noqa: E402
from src.train import (compute_class_weights, load_data, set_seed,  # noqa: E402
                        stratified_split)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Official rules (SEDIC 2026 RF track, 11 Aug public release) require >80%
# recall on Military/CEMA + Jamming, not 90% — match evaluate.py's benchmark.
BENCHMARK = 0.80


def train_one(X, y, tr, va, seed):
    """Train a single member and return it at its best-validation state."""
    set_seed(seed)
    t = CFG["training"]

    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_t[tr], y_t[tr]),
                              batch_size=t["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_t[va], y_t[va]), batch_size=t["batch_size"])

    model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(y, len(CLASSES)).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=t["scheduler_patience"])

    best_loss, best_state = float("inf"), None
    for _ in range(t["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
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
    return model


def _recalls(preds, y_true):
    return {c: float((preds[y_true == CLASS_TO_IDX[c]] == CLASS_TO_IDX[c]).mean())
            for c in CFG["judged_classes"] if (y_true == CLASS_TO_IDX[c]).any()}


def _predict(model, X_np, tta=0):
    """Softmax probabilities, optionally averaged over TTA phase rotations.

    Phase is arbitrary at the receiver, so a rotated copy is the same signal
    with the same label. Averaging over rotations cancels per-view noise.
    """
    views = [X_np] + [phase_rotate_batch(X_np, t)
                      for t in np.linspace(0, 2 * np.pi, tta, endpoint=False)[1:]]
    out = None
    with torch.no_grad():
        for v in views:
            p = torch.softmax(model(torch.tensor(v).to(DEVICE)), dim=1).cpu().numpy()
            out = p if out is None else out + p
    return out / len(views)


def main(n_models, tta=0):
    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    tr, va, te = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])
    X_test = X[te]
    y_test = y[te]

    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {n_models} members on identical data, different seeds.")
    if tta:
        print(f"Averaging {tta} phase rotations per prediction (TTA).")
    print()

    summed_probs = None
    members = []

    for i in range(n_models):
        model = train_one(X, y, tr, va, seed=2000 + i)
        torch.save(model.state_dict(), ckpt_dir / f"ensemble_{i}.pt")

        probs = _predict(model, X_test, tta)
        summed_probs = probs if summed_probs is None else summed_probs + probs

        r = _recalls(probs.argmax(1), y_test)
        members.append(r)
        print(f"  member {i+1}: " + "  ".join(f"{c}={v:.4f}" for c, v in r.items()))

    ens = _recalls((summed_probs / n_models).argmax(1), y_test)

    print(f"\n{'class':<14}{'single mean':>13}{'single best':>13}{'ENSEMBLE':>11}{'':>3}")
    print("-" * 56)
    scorecard = {"n_models": n_models, "tta": tta, "members": members,
                  "ensemble": ens, "passed": True}

    for c in ens:
        vals = [m[c] for m in members]
        mark = "PASS" if ens[c] >= BENCHMARK else "FAIL"
        scorecard["passed"] &= ens[c] >= BENCHMARK
        print(f"{c:<14}{np.mean(vals):>13.4f}{max(vals):>13.4f}"
              f"{ens[c]:>11.4f}  {mark}")

    print(f"\n  OVERALL: {'PASS' if scorecard['passed'] else 'FAIL'}")

    evals_dir = REPO_ROOT / CFG["paths"]["evals"]
    evals_dir.mkdir(parents=True, exist_ok=True)
    with open(evals_dir / "ensemble_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2)

    print(f"\n  {n_models} checkpoints -> {ckpt_dir}/ensemble_*.pt")
    print(f"  scorecard -> {evals_dir}/ensemble_scorecard.json")
    print("\nIf ENSEMBLE beats the single mean, averaging is cancelling")
    print("initialisation noise and is worth keeping for the submission.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=int, default=5)
    p.add_argument("--tta", type=int, default=0,
                   help="average N phase rotations per prediction (0 = off, 4 is a good start)")
    a = p.parse_args()
    main(a.models, a.tta)
