"""
Held-out parameter generalisation test.

Our three judged classes are synthetic. A high score on our own test split only
proves the model learned OUR data — it says nothing about the organisers'
Qualifier stream, which nobody on this team has seen. This script answers the
question a technical panel is silently asking:

    Did the model learn the CONCEPT (chirp, frequency hopping, denial), or did
    it memorise the particular parameter values we happened to generate?

Method: train only on the lower half of each judged class's defining parameter
range (configs/heldout_lo.yaml), then score against the disjoint upper half
(configs/heldout_hi.yaml), which the model has never seen. If recall holds, the
model generalises. If it collapses, we have found the fatal flaw ourselves
rather than having the organisers find it.

Prerequisites — build both halves first:

    SEDIC_CONFIG=configs/heldout_lo.yaml python -m src.data.build_dataset
    SEDIC_CONFIG=configs/heldout_hi.yaml python -m src.data.build_dataset

Usage:

    SEDIC_CONFIG=configs/heldout_lo.yaml python scripts/heldout_generalisation.py
    SEDIC_CONFIG=configs/heldout_lo.yaml python scripts/heldout_generalisation.py --runs 3

READ THE RESULT AGAINST THE NOISE FLOOR. Repeated identical runs of this
pipeline have moved a single class by up to 10.8 points on nothing but the
random seed (see scripts/measure_variance.py). A drop smaller than that is not
evidence of anything. --runs 3 measures the spread inline so the comparison is
made against this test's own noise rather than a remembered figure.
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
from src.train import (compute_class_weights, load_data, set_seed,  # noqa: E402
                        stratified_split)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Documented spread across five identical runs; see scripts/measure_variance.py
# and the dataset-size note in configs/default.yaml.
DEFAULT_NOISE_FLOOR = 0.108


def load_heldout(data_dir):
    """Load the disjoint upper-half dataset the model never trains on."""
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    missing = [f for f in ("X.npy", "y.npy", "snr_labels.npy")
               if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Held-out dataset incomplete at {data_dir} (missing {', '.join(missing)}).\n"
            "Build it first:\n"
            "    SEDIC_CONFIG=configs/heldout_hi.yaml python -m src.data.build_dataset"
        )
    return np.load(data_dir / "X.npy"), np.load(data_dir / "y.npy")


def train_one(X, y, train_idx, val_idx, seed):
    """Train on the in-range split and return the best-validation-loss model.

    Mirrors src.train.train() but keeps the model in memory and never writes a
    checkpoint, so this test cannot clobber the real training run's artefacts.
    """
    set_seed(seed)
    t = CFG["training"]

    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_t[train_idx], y_t[train_idx]),
                              batch_size=t["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_t[val_idx], y_t[val_idx]),
                            batch_size=t["batch_size"])

    model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(y, len(CLASSES)).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=t["scheduler_patience"])

    best_loss, best_state = float("inf"), None
    for epoch in range(t["epochs"]):
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
        print(f"    epoch {epoch+1}/{t['epochs']}  val_loss={vloss:.4f}", end="\r")

    print(" " * 60, end="\r")
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, X, batch=1024):
    """Batched — the held-out set is too large for one forward pass on CPU."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch]).to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def per_class_recall(model, X, y, batch=1024):
    """Recall for every class."""
    preds = predict(model, X, batch)
    out = {}
    for cls in CLASSES:
        idx = CLASS_TO_IDX[cls]
        mask = y == idx
        out[cls] = float((preds[mask] == idx).mean()) if mask.any() else None
    return out


def misclassification_breakdown(model, X, y, classes, batch=1024):
    """Where does each class's probability mass GO when it is wrong?

    A recall number says a class failed; this says what it was mistaken for,
    which is the difference between "the generator range is too narrow" and
    "two classes are genuinely not separable at these parameters".
    """
    preds = predict(model, X, batch)
    out = {}
    for cls in classes:
        idx = CLASS_TO_IDX[cls]
        mask = y == idx
        if not mask.any():
            continue
        counts = np.bincount(preds[mask], minlength=len(CLASSES))
        out[cls] = {CLASSES[i]: float(counts[i] / mask.sum())
                    for i in np.argsort(counts)[::-1] if counts[i] > 0}
    return out


def main(runs, heldout_dir, noise_floor):
    X_lo, y_lo, snr_lo = load_data()
    X_hi, y_hi = load_heldout(heldout_dir)

    d = CFG["dataset"]
    train_idx, val_idx, test_idx = stratified_split(
        y_lo, snr_lo, d["val_frac"], d["test_frac"], d["seed"])

    print("Held-out parameter generalisation test")
    print(f"  in-range train : {len(train_idx):,} examples")
    print(f"  in-range test  : {len(test_idx):,} examples  (same parameter range)")
    print(f"  held-out set   : {len(X_hi):,} examples  (disjoint parameter range, never trained on)")
    print(f"  runs           : {runs}\n")

    in_runs, out_runs = [], []
    breakdown = None
    for i in range(runs):
        print(f"  run {i+1}/{runs} — training on the in-range half...")
        model = train_one(X_lo, y_lo, train_idx, val_idx, seed=2000 + i)
        in_runs.append(per_class_recall(model, X_lo[test_idx], y_lo[test_idx]))
        out_runs.append(per_class_recall(model, X_hi, y_hi))
        if i == 0:  # diagnostic from the first run is enough to see the mechanism
            breakdown = misclassification_breakdown(
                model, X_hi, y_hi, CFG["judged_classes"])
        print(f"  run {i+1}/{runs} — done")

    def agg(runs_list, cls):
        vals = [r[cls] for r in runs_list if r[cls] is not None]
        return (float(np.mean(vals)), max(vals) - min(vals)) if vals else (None, None)

    judged = CFG["judged_classes"]
    civilian = [c for c in CLASSES if c not in judged]

    results = {"runs": runs, "judged_classes": {}, "civilian_classes": {}}

    # Aggregate everything BEFORE printing any verdict: with --runs > 1 the
    # noise floor is derived from the full set of runs, so a verdict issued
    # mid-loop would be judged against a partially-accumulated floor.
    rows, worst_drop, measured_spread = [], 0.0, 0.0
    for cls in judged:
        i_mean, i_spread = agg(in_runs, cls)
        o_mean, o_spread = agg(out_runs, cls)
        if i_mean is None or o_mean is None:
            continue
        delta = o_mean - i_mean
        worst_drop = min(worst_drop, delta)
        measured_spread = max(measured_spread, i_spread or 0.0, o_spread or 0.0)
        rows.append((cls, i_mean, o_mean, delta, i_spread, o_spread))

    floor = measured_spread if runs > 1 else noise_floor
    floor_src = (f"measured across {runs} runs" if runs > 1
                 else "documented default — rerun with --runs 3 to measure it here")

    print("\n--- JUDGED CLASSES (the generalisation result) ---")
    print(f"{'class':<12}{'in-range':>10}{'held-out':>11}{'delta':>10}{'verdict':>12}")
    print("-" * 55)
    for cls, i_mean, o_mean, delta, i_spread, o_spread in rows:
        results["judged_classes"][cls] = {
            "in_range_recall": i_mean, "held_out_recall": o_mean, "delta": delta,
            "in_range_spread": i_spread, "held_out_spread": o_spread,
        }
        verdict = "HOLDS" if delta >= -floor else "DROPS"
        print(f"{cls:<12}{i_mean:>10.4f}{o_mean:>11.4f}{delta:>+10.4f}{verdict:>12}")

    print(f"\n  noise floor    : {floor*100:.1f} points ({floor_src})")
    print(f"  worst drop     : {worst_drop*100:.1f} points")
    if worst_drop >= -floor:
        print("\n  VERDICT: GENERALISES — every judged class holds within the noise floor.")
        print("  The model learned the signal concept, not our specific parameter values.")
    else:
        print("\n  VERDICT: DOES NOT GENERALISE — at least one judged class drops by")
        print("  more than seed noise explains. The generator's parameter range is")
        print("  too narrow, or the model is keying on a parameter-specific artefact.")

    if breakdown:
        print("\n--- WHERE THE HELD-OUT PREDICTIONS GO ---")
        print("  Each judged class, evaluated on the unseen parameter range.")
        for cls, dist in breakdown.items():
            parts = "  ".join(f"{k}={v:.3f}" for k, v in list(dist.items())[:4])
            print(f"  {cls:<12} {parts}")
        results["heldout_prediction_breakdown"] = breakdown

    print("\n--- CIVILIAN CLASSES (NOT a generalisation measurement) ---")
    print("  RadioML is real captured data whose parameters we cannot split, so")
    print("  these columns compare different random ROWS, not different parameters.")
    print("  Reported for completeness only — do not cite them as generalisation.")
    print(f"\n{'class':<12}{'in-range':>10}{'held-out':>11}{'delta':>10}")
    print("-" * 43)
    for cls in civilian:
        i_mean, i_spread = agg(in_runs, cls)
        o_mean, o_spread = agg(out_runs, cls)
        if i_mean is None or o_mean is None:
            continue
        results["civilian_classes"][cls] = {
            "in_range_recall": i_mean, "held_out_recall": o_mean,
            "delta": o_mean - i_mean, "note": "different rows, not different parameters",
        }
        print(f"{cls:<12}{i_mean:>10.4f}{o_mean:>11.4f}{o_mean - i_mean:>+10.4f}")

    results["noise_floor"] = floor
    results["noise_floor_source"] = floor_src
    results["worst_judged_drop"] = worst_drop
    results["generalises"] = bool(worst_drop >= -floor)
    results["split_parameters"] = {
        "LFM_RADAR": "radar.bandwidth_hz",
        "FHSS": "fhss.hop_rate_hz",
        "JAMMING": "jamming.sweep_bandwidth_hz + jamming.barrage_bandwidth_hz "
                   "(tone jamming unsplittable — result is a lower bound)",
    }

    out_dir = REPO_ROOT / CFG["paths"]["evals"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "generalisation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {out_dir / 'generalisation.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1,
                   help="training runs; >1 measures this test's own noise floor")
    p.add_argument("--heldout-dir", default="data/processed_heldout_hi")
    p.add_argument("--noise-floor", type=float, default=DEFAULT_NOISE_FLOOR,
                   help="used only when --runs 1")
    a = p.parse_args()
    main(a.runs, a.heldout_dir, a.noise_floor)
