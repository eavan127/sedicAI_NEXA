"""
Evaluation: per-class recall, confusion matrix, and the accuracy-vs-SNR
curve required for the technical brief (docs section 9).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from model import AMC_CNN
from build_dataset import CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_path="results/best_model.pt"):
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    snr_labels = np.load("data/snr_labels.npy", allow_pickle=True)

    model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X).to(DEVICE)).argmax(1).cpu().numpy()

    print(classification_report(y, preds, target_names=CLASSES))

    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    print("Saved results/confusion_matrix.png")

    # Accuracy vs SNR (only meaningful for examples that have an SNR label)
    valid = np.array([s is not None for s in snr_labels])
    if valid.any():
        snr_vals = np.array([s for s in snr_labels if s is not None])
        unique_snrs = sorted(set(snr_vals))
        accs = []
        for snr in unique_snrs:
            mask = valid & (snr_labels == snr)
            acc = (preds[mask] == y[mask]).mean()
            accs.append(acc)
        plt.figure()
        plt.plot(unique_snrs, accs, marker="o")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. SNR")
        plt.grid(True)
        plt.savefig("results/accuracy_vs_snr.png")
        print("Saved results/accuracy_vs_snr.png")


if __name__ == "__main__":
    evaluate()
