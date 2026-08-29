"""
Train N models with different seeds and average their predictions.

Why this helps here specifically: seed variance measured 2.2 points on radar,
2.5 on FHSS and 8.9 on jamming. A single run lands at a random point inside
that range — sometimes 0.889, sometimes 0.911. Averaging sigmoid outputs
cancels the initialisation noise, so the result sits near the top of the range
rather than wherever the dice fell.

This is a MODEL-side change. It needs no generator edits and no coordination,
which is why it is the last lever available when the data is settled.

Usage:
    python scripts/train_ensemble.py --models 5
    python scripts/train_ensemble.py --models 5 --eval-only   # re-score existing
                                                                 # checkpoints, e.g.
                                                                 # after a threshold change
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT,  # noqa: E402
                         resolve_class_weight_multipliers, resolve_multilabel_thresholds)
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.data.preprocess import phase_rotate_batch  # noqa: E402
from src.train import (compute_class_weights, compute_snr_weights, load_data,  # noqa: E402
                        set_seed, stratified_split)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Official rules (SEDIC 2026 RF track, 11 Aug public release) require >80%
# recall on Military/CEMA + Jamming, not 90% — match evaluate.py's benchmark.
BENCHMARK = 0.80


def train_one(X, y, snr_labels, tr, va, seed, history_path=None, ckpt_path=None):
    """Train a single member and return it at its best-validation state.

    `history_path` writes one JSON per epoch -- train loss, validation loss and
    the current learning rate -- rewritten after every epoch rather than at the
    end. Losses used to go to stdout and nowhere else, so for the five
    checkpoints in results/ the train/validation gap cannot be inspected at all
    without a 2.5-hour retrain. Rewriting each epoch also means a run killed
    part-way still leaves everything it had reached.

    `ckpt_path` saves the best-so-far weights every time validation improves.
    A 2.5-hour run that dies at epoch 25 with nothing on disk is a total loss;
    this makes it a partial one. Learned the hard way -- the first attempt at
    the stft_freq_summary experiment was killed around epoch 8 and left no
    checkpoint and, because stdout was being piped through tee, no log either.

    Both default to None, so existing callers are unaffected.
    """
    set_seed(seed)
    t = CFG["training"]

    # NOISE_FLOOR needs different treatment from every other class in the
    # sampler -- see compute_snr_weights in src/train.py. The loss's per-class
    # multipliers live in config -- see resolve_class_weight_multipliers.
    noise_floor_idx = CLASS_TO_IDX.get("NOISE_FLOOR")
    neutral_classes = [noise_floor_idx] if noise_floor_idx is not None else []
    dampen = resolve_class_weight_multipliers()

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

    best_loss, best_state, best_epoch = float("inf"), None, None
    history = []
    for epoch in range(t["epochs"]):
        model.train()
        tloss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tloss += loss.item() * xb.size(0)
        tloss /= len(train_loader.dataset)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vloss += criterion(model(xb), yb).item() * xb.size(0)
        vloss /= len(val_loader.dataset)
        sched.step(vloss)
        improved = vloss < best_loss
        if improved:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            if ckpt_path is not None:
                torch.save(best_state, ckpt_path)

        lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{t['epochs']}  train_loss={tloss:.4f}  "
              f"val_loss={vloss:.4f}  lr={lr:g}"
              f"{'  <- best' if improved else ''}", flush=True)
        if history_path is not None:
            history.append({"epoch": epoch + 1, "train_loss": float(tloss),
                             "val_loss": float(vloss), "lr": float(lr),
                             "improved": bool(improved)})
            # Rewritten every epoch, not appended at the end: a killed run
            # still leaves a complete record of everything it reached.
            Path(history_path).write_text(json.dumps(
                {"seed": seed, "epochs_planned": t["epochs"],
                 "best_epoch": best_epoch, "best_val_loss": float(best_loss),
                 "stft_freq_summary": bool(
                     CFG.get("model", {}).get("stft_freq_summary", False)),
                 "epochs": history}, indent=2))

    model.load_state_dict(best_state)
    model.eval()
    return model


def _recalls(present, y_true):
    """present/y_true: (N, len(CLASSES)) 0/1 arrays -- recall of each judged
    class's bit, among examples where that class is truly present (standalone
    or as part of a composite)."""
    out = {}
    for c in CFG["judged_classes"]:
        idx = CLASS_TO_IDX[c]
        mask = y_true[:, idx] == 1
        if mask.any():
            out[c] = float(present[mask, idx].mean())
    return out


def _precisions(present, y_true):
    """Of everything predicted present for a judged class, what fraction
    actually is -- the false-alarm-rate side of the recall/precision
    tradeoff. Not computed anywhere else for the ensemble (only evaluate.py's
    single-model path reports it), so this scorecard has been recall-only
    until now -- easy to miss that a recall win came with a precision cost."""
    out = {}
    for c in CFG["judged_classes"]:
        idx = CLASS_TO_IDX[c]
        pred_pos = present[:, idx] == 1
        if pred_pos.any():
            out[c] = float(y_true[pred_pos, idx].mean())
    return out


EVAL_BATCH_SIZE = 256


def _predict(model, X_np, tta=0):
    """Sigmoid probabilities (each class independent), optionally averaged
    over TTA phase rotations.

    Phase is arbitrary at the receiver, so a rotated copy is the same signal
    with the same label(s). Averaging over rotations cancels per-view noise.

    Batched, not one forward pass over the whole split -- X_np can be
    thousands of test windows, and a single unbatched call OOMs a real GPU
    once the dataset is full-sized (see the same fix in src/evaluate.py).
    """
    views = [X_np] + [phase_rotate_batch(X_np, t)
                      for t in np.linspace(0, 2 * np.pi, tta, endpoint=False)[1:]]
    out = None
    with torch.no_grad():
        for v in views:
            chunks = []
            for i in range(0, len(v), EVAL_BATCH_SIZE):
                batch = torch.tensor(v[i:i + EVAL_BATCH_SIZE]).to(DEVICE)
                chunks.append(torch.sigmoid(model(batch)).cpu().numpy())
            p = np.concatenate(chunks, axis=0)
            out = p if out is None else out + p
    return out / len(views)


def main(n_models, tta=0, eval_only=False):
    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    threshold = resolve_multilabel_thresholds()
    tr, va, te = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])
    X_test = X[te]
    y_test = y[te]

    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if eval_only:
        ckpt_paths = [ckpt_dir / f"ensemble_{i}.pt" for i in range(n_models)]
        missing = [p for p in ckpt_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--eval-only needs existing checkpoints, missing: {missing} -- "
                f"run without --eval-only first to train them")
        print(f"Re-scoring {n_models} EXISTING checkpoints against the current")
        print("thresholds in config -- no retraining. Use this after changing")
        print("multilabel_thresholds_per_class (e.g. from calibrate_thresholds.py")
        print("--ensemble) to check the effect without paying for a fresh 5-model")
        print("training run.")
    else:
        print(f"Training {n_models} members on identical data, different seeds.")
    if tta:
        print(f"Averaging {tta} phase rotations per prediction (TTA).")
    print()

    summed_probs = None
    members = []

    for i in range(n_models):
        if eval_only:
            model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_dir / f"ensemble_{i}.pt", map_location=DEVICE))
            model.eval()
        else:
            model = train_one(X, y, snr_labels, tr, va, seed=2000 + i)
            torch.save(model.state_dict(), ckpt_dir / f"ensemble_{i}.pt")

        probs = _predict(model, X_test, tta)
        summed_probs = probs if summed_probs is None else summed_probs + probs

        r = _recalls((probs > threshold).astype(int), y_test)
        members.append(r)
        print(f"  member {i+1}: " + "  ".join(f"{c}={v:.4f}" for c, v in r.items()))

    ens_probs = summed_probs / n_models
    ens_present = (ens_probs > threshold).astype(int)
    ens = _recalls(ens_present, y_test)
    ens_precision = _precisions(ens_present, y_test)

    print(f"\n{'class':<14}{'single mean':>13}{'single best':>13}{'ENSEMBLE':>11}"
          f"{'precision':>12}{'':>3}")
    print("-" * 68)
    scorecard = {"n_models": n_models, "tta": tta, "members": members,
                  "ensemble": ens, "ensemble_precision": ens_precision, "passed": True}

    for c in ens:
        vals = [m[c] for m in members]
        mark = "PASS" if ens[c] >= BENCHMARK else "FAIL"
        scorecard["passed"] &= ens[c] >= BENCHMARK
        prec_str = f"{ens_precision[c]:.4f}" if c in ens_precision else "n/a"
        print(f"{c:<14}{np.mean(vals):>13.4f}{max(vals):>13.4f}"
              f"{ens[c]:>11.4f}{prec_str:>12}  {mark}")

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
    p.add_argument("--eval-only", action="store_true",
                   help="re-score existing ensemble_*.pt checkpoints against current "
                        "config thresholds, no retraining")
    a = p.parse_args()
    main(a.models, a.tta, a.eval_only)
