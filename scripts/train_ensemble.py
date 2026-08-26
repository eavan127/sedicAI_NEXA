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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT, resolve_multilabel_thresholds  # noqa: E402
from src.evaluate import predict_probs  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import load_data, stratified_split, train_model  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Official rules (SEDIC 2026 RF track, 11 Aug public release) require >80%
# recall on Military/CEMA + Jamming, not 90% -- read from config (not
# hardcoded) so this stays in lockstep with evaluate.py and the organizer's
# actual current number if it's revised again.
BENCHMARK = CFG["benchmark_recall"]


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
            model, _ = train_model(X, y, snr_labels, tr, va, seed=2000 + i, verbose=False)
            torch.save(model.state_dict(), ckpt_dir / f"ensemble_{i}.pt")

        probs = predict_probs([model], X_test, tta=tta)
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
