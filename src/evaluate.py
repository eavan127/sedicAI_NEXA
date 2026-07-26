"""
Evaluation: per-class recall, confusion matrix, accuracy-vs-SNR curve, and a
scorecard that states plainly whether the >90% benchmark is met.

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
BENCHMARK_RECALL = 0.90


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

    with open(evals_dir / "scorecard.json", "w") as f:
        json.dump({"per_class": report, "benchmark": scorecard}, f, indent=2)

    print("\n--- Benchmark (>90% recall on judged classes) ---")
    for cls, r in scorecard["judged_classes"].items():
        print(f"  {cls:<12} recall={r['recall']:.4f}  {'PASS' if r['passed'] else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if scorecard['passed'] else 'FAIL'}")

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
    plt.axhline(BENCHMARK_RECALL, color="red", linestyle=":", label="90% benchmark")
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
