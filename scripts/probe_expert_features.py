"""Measure the two weaknesses the expert-feature branches target.

Both weaknesses are real, measured, and unfixed in the model itself:

  1. 16QAM vs 64QAM is a coin flip. Choosing the larger of the two
     probabilities is right 51.4% of the time on the held-out split, 49.7%
     after averaging 64 windows, and 47.0% on true 16QAM -- worse than chance,
     because the error is a systematic bias rather than noise. The model
     detects "dense QAM" reliably and then assigns one of the two labels
     close to arbitrarily.

  2. Radar and FHSS confuse each other. 50.7% of LFM_RADAR's false positives
     are genuinely FHSS, and 35.5% of FHSS's are genuinely radar.

The fix under test is expert features -- quantities the samples contain but
the CNN evidently does not learn, computed explicitly and concatenated into
the fused vector:

  model.cumulant_features  normalised |C40|, |C42|, |C63| on the matched-
                           filtered window. Separates 16QAM from 64QAM with
                           AUC 0.609 per window (0.633 with full symbol
                           recovery, 0.576 with no filtering at all).

  model.if_features        max/median of the second difference of unwrapped
                           phase. A chirp sweeps at a near-constant rate; a
                           hopper sits still then jumps. AUC 0.887 pooled,
                           and 0.95-0.97 at SNR >= +2 dB.

Neither is proven to help the trained model. This script is how that gets
decided: run it against the current checkpoints to pin a baseline, retrain
with a flag on, run it again, compare.

Usage
-----
    python scripts/probe_expert_features.py --out docs/experiments/expert_baseline.json
    # ... retrain with model.cumulant_features and/or model.if_features on ...
    python scripts/probe_expert_features.py --checkpoint results/new.pt \\
        --cumulant-features --out docs/experiments/expert_after.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, resolve_multilabel_thresholds  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402
from src.ui.app_models import ensemble_paths  # noqa: E402


def _predict(models, X):
    out = np.zeros((len(X), len(CLASSES)))
    with torch.no_grad():
        for i in range(0, len(X), 512):
            b = torch.from_numpy(X[i:i + 512]).float()
            out[i:i + 512] = np.mean([torch.sigmoid(m(b)).numpy() for m in models], axis=0)
    return out


def qam_discrimination(probs, y):
    """How often is the larger of 16QAM/64QAM the correct one?

    0.5 is a coin flip. Reported on standalone windows of each class, where
    the answer is unambiguous. Also reported per class, because the failure is
    asymmetric -- the model is biased toward 64QAM, so a single pooled figure
    hides that one side is worse than chance.
    """
    i16, i64 = CLASSES.index("16QAM"), CLASSES.index("64QAM")
    out = {}
    correct = []
    for cls, own, other in (("16QAM", i16, i64), ("64QAM", i64, i16)):
        sel = (y[:, own] > 0.5) & (y.sum(axis=1) == 1)
        won = probs[sel][:, own] > probs[sel][:, other]
        out[cls] = {"accuracy": float(won.mean()), "n": int(sel.sum())}
        correct.append(won)
    out["combined"] = {"accuracy": float(np.concatenate(correct).mean()),
                       "n": int(sum(len(c) for c in correct))}
    return out


def cross_confusion(pred, y, a, b):
    """Of class `a`'s false positives, what fraction are genuinely `b`?"""
    ia, ib = CLASSES.index(a), CLASSES.index(b)
    fp = (pred[:, ia] == 1) & (y[:, ia] < 0.5)
    if not fp.any():
        return {"fraction": None, "n_false_positives": 0}
    return {"fraction": float((y[fp, ib] > 0.5).mean()),
            "n_false_positives": int(fp.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="checkpoint path(s); default is the 5-model ensemble")
    ap.add_argument("--cumulant-features", action="store_true",
                    help="build models with model.cumulant_features on -- required "
                          "when probing a checkpoint trained with it, since the "
                          "flag changes the fused width")
    ap.add_argument("--if-features", action="store_true",
                    help="same, for model.if_features")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    CFG.setdefault("model", {})
    if args.cumulant_features:
        CFG["model"]["cumulant_features"] = True
    if args.if_features:
        CFG["model"]["if_features"] = True

    paths = [Path(c) for c in args.checkpoint] if args.checkpoint else ensemble_paths()
    models = []
    for p in paths:
        m = AMC_CNN(num_classes=len(CLASSES), input_len=CFG["signal"]["window_len"])
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)

    X, y, snr = load_data()
    d = CFG["dataset"]
    _, _, test = stratified_split(y, snr, d["val_frac"], d["test_frac"], d["seed"])
    Xt, yt = X[test], y[test]

    probs = _predict(models, Xt)
    pred = (probs > np.array(resolve_multilabel_thresholds())).astype(int)

    qam = qam_discrimination(probs, yt)
    radar_fhss = cross_confusion(pred, yt, "LFM_RADAR", "FHSS")
    fhss_radar = cross_confusion(pred, yt, "FHSS", "LFM_RADAR")
    judged = {}
    for cls in CFG["judged_classes"]:
        j = CLASSES.index(cls)
        pos = yt[:, j] > 0.5
        judged[cls] = float(pred[pos, j].mean())

    print(f"{len(models)} model(s)  ·  cumulant_features="
          f"{CFG['model'].get('cumulant_features', False)}  ·  if_features="
          f"{CFG['model'].get('if_features', False)}\n")
    print("16QAM vs 64QAM -- is the larger probability the right one?")
    for k in ("16QAM", "64QAM", "combined"):
        v = qam[k]
        print(f"  {k:9s} {v['accuracy']:6.1%}  (n={v['n']})"
              f"{'   <- coin flip' if abs(v['accuracy'] - 0.5) < 0.03 else ''}")
    print("\nRadar / FHSS cross-confusion -- of one's false positives, how many "
          "are the other?")
    print(f"  LFM_RADAR FPs that are really FHSS: "
          f"{radar_fhss['fraction']:.1%} (n={radar_fhss['n_false_positives']})")
    print(f"  FHSS FPs that are really LFM_RADAR: "
          f"{fhss_radar['fraction']:.1%} (n={fhss_radar['n_false_positives']})")
    print("\nJudged-class recall (must not regress -- the gate is 0.80)")
    for cls, r in judged.items():
        print(f"  {cls:11s} {r:.4f}{'' if r >= CFG['benchmark_recall'] else '   <- BELOW GATE'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "cumulant_features": bool(CFG["model"].get("cumulant_features", False)),
            "if_features": bool(CFG["model"].get("if_features", False)),
            "n_models": len(models),
            "qam_discrimination": qam,
            "radar_fp_really_fhss": radar_fhss,
            "fhss_fp_really_radar": fhss_radar,
            "judged_recall": judged,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
