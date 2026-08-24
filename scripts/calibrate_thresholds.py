"""
Pick per-class sigmoid thresholds on the VALIDATION split, not test.

Why this exists: the thresholds currently in configs/default.yaml
(multilabel_thresholds_per_class) were chosen by repeatedly rerunning
evaluate.py -- which only ever scores the test split -- and watching test
recall move as the threshold swept. That tunes a hyperparameter directly
against the number reported to judges, which is leakage regardless of
whether the threshold is global or per-class. The fix isn't "use one
threshold for fairness" -- it's "pick the threshold(s) on data the final
number doesn't come from".

For each class, this sweeps candidate thresholds against val_idx and picks
the highest threshold (best precision) that still clears
CFG["benchmark_recall"] on val. If none clears it, falls back to the
threshold that maximises val recall and flags it. Judged and non-judged
classes are treated the same way -- non-judged classes aren't gated, but a
calibrated threshold is still better than an arbitrary 0.5 default for them.

Run this after any retrain (the model's probability calibration shifts every
time), then run evaluate.py once for the number that goes in the brief.

--ensemble calibrates against the AVERAGE of ensemble_0.pt..ensemble_{n-1}.pt
(train_ensemble.py's checkpoints) instead of the single best_model.pt.
Necessary, not optional, if you're actually submitting the ensemble:
averaging several models' sigmoid outputs compresses extreme probabilities
toward the mean (e.g. several individually-97%-confident models can average
to ~85%), so a threshold calibrated against one model can be too strict for
the ensemble even though it was correctly calibrated for what it was
calibrated against. Measured case: JAMMING's single-model threshold (0.90)
put the 5-member ensemble's recall at 0.8049 -- below every individual
member's own recall except one, and barely above the 80% floor -- purely
from this compression, not from the ensemble actually detecting JAMMING
worse.

Usage:
    python scripts/calibrate_thresholds.py
    python scripts/calibrate_thresholds.py --ensemble --n-models 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BENCHMARK_RECALL = CFG["benchmark_recall"]
CANDIDATES = np.round(np.arange(0.05, 0.96, 0.01), 2)
EVAL_BATCH_SIZE = 256


def _predict_probs(model, X):
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), EVAL_BATCH_SIZE):
            batch = torch.tensor(X[i:i + EVAL_BATCH_SIZE]).to(DEVICE)
            chunks.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _best_threshold(probs_col, true_col):
    """Highest threshold (= best precision) that keeps recall >=
    BENCHMARK_RECALL on this class, among CANDIDATES. Falls back to the
    threshold with the best recall if none clears the floor."""
    support = true_col.sum()
    if support == 0:
        return 0.5, None, False  # nothing to calibrate against -- leave default

    best_passing, best_recall_fallback = None, (None, -1)
    for t in CANDIDATES:
        pred = probs_col > t
        tp = int((pred & (true_col == 1)).sum())
        recall = tp / support
        if recall >= BENCHMARK_RECALL:
            best_passing = t  # CANDIDATES is ascending, so the last one seen is highest
        if recall > best_recall_fallback[1]:
            best_recall_fallback = (t, recall)

    if best_passing is not None:
        pred = probs_col > best_passing
        fp = int((pred & (true_col == 0)).sum())
        tp = int((pred & (true_col == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else None
        return float(best_passing), precision, True

    t, recall = best_recall_fallback
    return float(t), None, False


def _ensemble_probs(ckpt_paths, X_val):
    """Average sigmoid probabilities over several checkpoints -- same
    averaging train_ensemble.py's _predict does, so calibration matches what
    the ensemble actually outputs at inference time."""
    summed = None
    for ckpt in ckpt_paths:
        model = AMC_CNN(num_classes=len(CLASSES), input_len=X_val.shape[-1]).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        p = _predict_probs(model, X_val)
        summed = p if summed is None else summed + p
    return summed / len(ckpt_paths)


def main(ensemble, n_models):
    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    train_idx, val_idx, _ = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])
    X_val, y_val = X[val_idx], y[val_idx].astype(int)

    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    if ensemble:
        ckpt_paths = [ckpt_dir / f"ensemble_{i}.pt" for i in range(n_models)]
        missing = [p for p in ckpt_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing ensemble checkpoints: {missing} -- run "
                f"train_ensemble.py --models {n_models} first")
        probs = _ensemble_probs(ckpt_paths, X_val)
        ckpt_desc = f"{n_models}-model ensemble average ({ckpt_dir}/ensemble_*.pt)"
    else:
        ckpt = ckpt_dir / "best_model.pt"
        model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        probs = _predict_probs(model, X_val)
        ckpt_desc = str(ckpt)

    print(f"Calibrating against: {ckpt_desc}")
    print(f"Calibrating on val split ({len(val_idx)} windows), floor={BENCHMARK_RECALL:.0%}\n")
    print(f"{'class':<12}{'threshold':>10}{'val_recall_ok':>15}{'precision':>11}")
    print("-" * 48)

    results = {}
    for cls in CLASSES:
        idx = CLASS_TO_IDX[cls]
        t, precision, passed = _best_threshold(probs[:, idx], y_val[:, idx])
        results[cls] = t
        prec_str = f"{precision:.3f}" if precision is not None else "n/a"
        flag = "PASS" if passed else "FAIL (best-recall fallback)"
        print(f"{cls:<12}{t:>10.2f}  {flag:<28}{prec_str:>11}")

    print("\nPaste into configs/default.yaml (only judged classes strictly need to")
    print("be listed -- non-judged rows are included here for completeness):")
    print("\nmultilabel_thresholds_per_class:")
    for cls in CFG["judged_classes"]:
        print(f"  {cls}: {results[cls]}")

    print("\nRemember: these numbers are only trustworthy if val_idx wasn't used to")
    print("pick anything else about this checkpoint (early stopping already uses")
    print("val_loss, which is fine -- that's a different quantity from the")
    print("threshold). Report the final recall from evaluate.py's TEST run, once,")
    print("after locking these thresholds in -- not from this script.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", action="store_true",
                    help="calibrate against the ensemble_*.pt average instead of best_model.pt")
    p.add_argument("--n-models", type=int, default=5)
    a = p.parse_args()
    main(a.ensemble, a.n_models)
