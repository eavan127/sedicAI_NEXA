"""
Score every experiment checkpoint on the SAME test split and print one table.

Why this exists
---------------
src/evaluate.py builds its model from configs/default.yaml and loads
best_model.pt / ensemble_*.pt. It cannot score an experiment checkpoint,
because each of those was trained with switches that are OFF in the config
(turning them on in the YAML would stop the shipped checkpoints loading --
see scripts/run_stft_experiment.py's docstring). probe_jsr.py and
high_snr_probe.py do take --checkpoint, but they answer recall-shaped
questions about jammed FHSS and the high-SNR decline.

Nothing scored per-class PRECISION across the experiment cells, which is the
question "did LFM_RADAR precision actually move" needs.

Honest by construction
----------------------
Two numbers per class, neither of them threshold-tuned:

  P@0.5 / R@0.5   the flat 0.5 fallback, NOT the calibrated per-class
                  thresholds in configs/default.yaml. Those were fitted to a
                  specific old ensemble to buy recall margin (LFM_RADAR sits
                  at 0.22), so reading precision through them measures the
                  calibration, not the model.

  AP              average precision -- the area under the precision/recall
                  curve, computed over the raw sigmoid scores at every
                  threshold. This is the headline. A variant whose
                  probabilities are merely shifted scores the same AP; only a
                  variant that genuinely separates the classes better moves
                  it. P@0.5 can flatter or punish a variant for calibration
                  alone; AP cannot.

Each run writes results/experiment_<tag>_config.json recording the switches it
used, so this reads those rather than guessing which flags a checkpoint needs.
A checkpoint whose flags are wrong fails loudly on load_state_dict(strict=True)
rather than scoring something meaningless.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --results-dir results --out evals/experiment_comparison.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CFG, CLASSES, CLASS_TO_IDX  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JUDGED = CFG.get("judged_classes", ["LFM_RADAR", "FHSS", "JAMMING"])


def discover(results_dir):
    """Every experiment checkpoint that has a sidecar config recording its switches."""
    found = []
    for cfg_path in sorted(results_dir.glob("experiment_*_config.json")):
        tag = cfg_path.name[len("experiment_"):-len("_config.json")]
        if tag.endswith("_smoke"):
            continue        # 1 epoch on a tiny slice -- proves the pipeline, means nothing
        ckpt = results_dir / f"experiment_{tag}.pt"
        if not ckpt.exists():
            print(f"  ! {cfg_path.name} has no matching {ckpt.name} -- skipping")
            continue
        found.append((tag, ckpt, json.loads(cfg_path.read_text())))
    return found


def scores_for(ckpt_path, flags, X_test, batch=512):
    """Raw sigmoid scores (N, n_classes) for one checkpoint."""
    model = AMC_CNN(
        num_classes=len(CLASSES),
        input_len=CFG["signal"]["window_len"],
        stft_freq_summary=flags.get("stft_freq_summary", False),
        stft_keep_rows=flags.get("stft_keep_rows", False),
        stamp_branch=flags.get("stamp_branch", False),
    ).to(DEVICE)
    # strict=True: a flag/checkpoint mismatch must fail here, not silently score noise.
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch):
            chunk = torch.tensor(X_test[i:i + batch], dtype=torch.float32).to(DEVICE)
            out.append(torch.sigmoid(model(chunk)).cpu().numpy())
    return np.concatenate(out)


def metrics_for(scores, y_true, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    per_class = {}
    for name in CLASSES:
        i = CLASS_TO_IDX[name]
        t, p, s = y_true[:, i], pred[:, i], scores[:, i]
        tp = int((t & p).sum())
        fp = int(((1 - t) & p).sum())
        fn = int((t & (1 - p)).sum())
        per_class[name] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "ap": float(average_precision_score(t, s)) if t.any() else float("nan"),
            "support": int(t.sum()),
        }
    return per_class


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "experiment_comparison.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    runs = discover(args.results_dir)
    if not runs:
        print(f"No experiment_*.pt + _config.json pairs in {args.results_dir}.")
        print("Run scripts/run_stft_experiment.py first.")
        return 1

    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    # Same split call as training used, so the test third is genuinely held out.
    _, _, te = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])
    X_test, y_test = X[te], y[te].astype(int)
    print(f"test split: {len(te):,} windows  |  device: {DEVICE}  |  threshold {args.threshold}\n")

    results = {}
    for tag, ckpt, flags in runs:
        print(f"scoring {ckpt.name} ...", flush=True)
        try:
            scores = scores_for(ckpt, flags, X_test)
        except RuntimeError as exc:
            print(f"  ! {tag}: {exc}\n  (flag/checkpoint mismatch -- not scored)")
            continue
        results[tag] = {"flags": flags, "per_class": metrics_for(scores, y_test, args.threshold)}

    if not results:
        print("Nothing scored.")
        return 1

    # --- headline: the question this whole exercise is about --------------
    print(f"\n{'='*78}\nLFM_RADAR -- the target metric\n{'='*78}")
    print(f"{'run':<24} | {'AP':>7} | {'P@0.5':>7} | {'R@0.5':>7} | {'params':>9}")
    print("-" * 78)
    for tag, r in sorted(results.items(), key=lambda kv: -kv[1]["per_class"]["LFM_RADAR"]["ap"]):
        m = r["per_class"]["LFM_RADAR"]
        print(f"{tag:<24} | {m['ap']:>7.4f} | {m['precision']:>7.4f} | {m['recall']:>7.4f} | "
              f"{r['flags'].get('parameters', 0):>9,}")

    # --- judged classes, so a radar gain that costs FHSS/JAMMING shows ----
    print(f"\n{'='*78}\nJudged classes -- AP (a gain that costs another class shows here)\n{'='*78}")
    print(f"{'run':<24} | " + " | ".join(f"{c:>12}" for c in JUDGED) + " |     mean")
    print("-" * 78)
    for tag, r in results.items():
        aps = [r["per_class"][c]["ap"] for c in JUDGED]
        print(f"{tag:<24} | " + " | ".join(f"{a:>12.4f}" for a in aps) + f" | {np.mean(aps):>8.4f}")

    print(f"\n{'='*78}\nAll classes -- AP\n{'='*78}")
    print(f"{'run':<24} | " + " | ".join(f"{c[:9]:>9}" for c in CLASSES))
    print("-" * 78)
    for tag, r in results.items():
        print(f"{tag:<24} | " + " | ".join(f"{r['per_class'][c]['ap']:>9.4f}" for c in CLASSES))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {args.out}")
    print("\nRead AP first. P@0.5 moves with calibration; AP does not.")
    print("A variant only wins if it beats the baseline by more than the baseline's own")
    print("run-to-run spread -- which is why the baseline cell is not optional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
