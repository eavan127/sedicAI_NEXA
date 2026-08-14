"""
Evaluation: per-class recall, confusion matrix, accuracy-vs-SNR curve, and a
scorecard that states plainly whether the organiser's benchmark
(`configs/default.yaml: benchmark_recall`) is met.

Usage:
    python -m src.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT
from src.models.amc_cnn import AMC_CNN
from src.train import load_data, stratified_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BENCHMARK_RECALL = CFG["benchmark_recall"]

# Coarse tiers. The 7-class number is what the gate is scored on, but the tier
# call is what matters operationally: mistaking a distant phone for an attack
# (civilian -> hostile) is a false alarm, whereas confusing 16QAM with 64QAM at
# low SNR still yields the correct decision, "this is ordinary traffic".
TIERS = {"Civilian": ["BPSK", "QPSK", "16QAM", "64QAM"],
         "Military": ["LFM_RADAR", "FHSS"],
         "Hostile": ["JAMMING"]}


def _tier_of(class_name):
    for tier, members in TIERS.items():
        if class_name in members:
            return tier
    raise KeyError(f"{class_name} is in no tier — update TIERS")


def coarse_tier_metrics(y_true, y_pred):
    """Accuracy and per-tier recall over Civilian / Military / Hostile."""
    tier_names = list(TIERS)
    tier_idx = {t: i for i, t in enumerate(tier_names)}
    lut = np.array([tier_idx[_tier_of(c)] for c in CLASSES])

    t_true, t_pred = lut[y_true], lut[y_pred]
    out = {"accuracy": float((t_true == t_pred).mean()), "per_tier_recall": {}}
    for t, i in tier_idx.items():
        m = t_true == i
        out["per_tier_recall"][t] = float((t_pred[m] == i).mean()) if m.any() else None
    return out


def comms_vs_jamming(y_true, y_pred):
    """The metric the rules single out for 'significantly higher technical scores':

        "Models that can successfully distinguish between standard communication
         signals and hostile CEMA interference (e.g., RF Jamming)"

    Reported as its own headline number so the panel does not have to dig it out
    of a 7x7 confusion matrix.
    """
    civ = np.array([CLASS_TO_IDX[c] for c in TIERS["Civilian"]])
    jam = CLASS_TO_IDX["JAMMING"]

    mask = np.isin(y_true, civ) | (y_true == jam)
    if not mask.any():
        return None

    true_is_jam = y_true[mask] == jam
    pred_is_jam = y_pred[mask] == jam

    tp = int((true_is_jam & pred_is_jam).sum())
    fn = int((true_is_jam & ~pred_is_jam).sum())
    fp = int((~true_is_jam & pred_is_jam).sum())

    return {
        "accuracy": float((true_is_jam == pred_is_jam).mean()),
        "jamming_recall": tp / (tp + fn) if tp + fn else None,
        "false_alarm_rate": fp / int((~true_is_jam).sum()) if (~true_is_jam).any() else None,
        "n_evaluated": int(mask.sum()),
    }


def evaluate():
    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    _, _, test_idx = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])

    X_test, y_test, snr_test = X[test_idx], y[test_idx], snr_labels[test_idx]

    ckpt = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"
    model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()

    evals_dir = REPO_ROOT / CFG["paths"]["evals"]
    evals_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_test, preds, labels=range(len(CLASSES)),
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    print(classification_report(y_test, preds, labels=range(len(CLASSES)),
                                 target_names=CLASSES, zero_division=0))

    # Scorecard — the number the judges actually check
    scorecard = {"benchmark_recall": BENCHMARK_RECALL, "judged_classes": {}, "passed": True}
    for cls in CFG["judged_classes"]:
        recall = report[cls]["recall"]
        passed = recall >= BENCHMARK_RECALL
        scorecard["judged_classes"][cls] = {"recall": recall, "passed": bool(passed)}
        scorecard["passed"] &= bool(passed)

    coarse = coarse_tier_metrics(y_test, preds)
    cvj = comms_vs_jamming(y_test, preds)

    with open(evals_dir / "scorecard.json", "w") as f:
        json.dump({"per_class": report, "benchmark": scorecard,
                    "coarse_tier": coarse, "comms_vs_jamming": cvj}, f, indent=2)

    print(f"\n--- Benchmark (>{BENCHMARK_RECALL:.0%} recall on judged classes) ---")
    for cls, r in scorecard["judged_classes"].items():
        print(f"  {cls:<12} recall={r['recall']:.4f}  {'PASS' if r['passed'] else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if scorecard['passed'] else 'FAIL'}")

    # Headline numbers for the brief — see docs/WINNING_STRATEGY.md
    if cvj:
        print("\n--- Comms vs Hostile CEMA (the 'Competitive Advantage' criterion) ---")
        print(f"  discrimination accuracy : {cvj['accuracy']:.4f}")
        if cvj["jamming_recall"] is not None:
            print(f"  jamming recall          : {cvj['jamming_recall']:.4f}")
        if cvj["false_alarm_rate"] is not None:
            print(f"  false alarm rate        : {cvj['false_alarm_rate']:.4f}"
                  "   (civilian wrongly flagged as jamming)")

    print("\n--- Coarse tier (Civilian / Military / Hostile) ---")
    print(f"  tier accuracy: {coarse['accuracy']:.4f}")
    for tier, rec in coarse["per_tier_recall"].items():
        print(f"    {tier:<10} recall={rec:.4f}" if rec is not None
              else f"    {tier:<10} recall=n/a")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=range(len(CLASSES)))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(evals_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Accuracy vs SNR, overall and for each judged class
    unique_snrs = sorted(np.unique(snr_test))
    plt.figure()
    plt.plot(unique_snrs, [(preds[snr_test == s] == y_test[snr_test == s]).mean()
                            for s in unique_snrs], marker="o", label="overall", linewidth=2)
    for cls in CFG["judged_classes"]:
        idx = CLASS_TO_IDX[cls]
        accs = []
        for s in unique_snrs:
            m = (snr_test == s) & (y_test == idx)
            accs.append((preds[m] == y_test[m]).mean() if m.any() else np.nan)
        plt.plot(unique_snrs, accs, marker=".", linestyle="--", label=cls)
    plt.axhline(BENCHMARK_RECALL, color="red", linestyle=":",
                label=f"{BENCHMARK_RECALL:.0%} benchmark")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. SNR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(evals_dir / "accuracy_vs_snr.png", dpi=150)
    plt.close()

    print(f"\nArtifacts written to {evals_dir}")


if __name__ == "__main__":
    evaluate()
