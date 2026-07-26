"""
Runs the trained model on the organizer's "Qualifier IQ Data Stream" and
produces the required classification log (docs section 14 checklist).

Usage:
    python infer.py --input path/to/qualifier_iq_stream.bin --output results/classification_log.csv
"""
import argparse
import csv

import numpy as np
import torch

from model import AMC_CNN
from build_dataset import CLASSES, WINDOW_LEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_iq_file(path):
    """
    TODO: adjust to match the organizer's actual IQ file format
    (e.g. interleaved float32 I/Q, or a specific binary header).
    Placeholder assumes raw interleaved float32 I,Q,I,Q,...
    """
    raw = np.fromfile(path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]
    return iq


def preprocess_window(iq_complex, window_len=WINDOW_LEN):
    iq = iq_complex[:window_len]
    if len(iq) < window_len:
        iq = np.pad(iq, (0, window_len - len(iq)))
    arr = np.stack([iq.real, iq.imag])
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr.astype(np.float32)


def run_inference(input_path, output_path, model_path="results/best_model.pt", window_len=WINDOW_LEN):
    iq = load_iq_file(input_path)

    model = AMC_CNN(num_classes=len(CLASSES), input_len=window_len).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    rows = []
    for start in range(0, len(iq) - window_len, window_len):
        window = iq[start:start + window_len]
        arr = torch.tensor(preprocess_window(window, window_len)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(arr)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        rows.append({
            "sample_start": start,
            "predicted_class": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_start", "predicted_class", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Classification log written to {output_path} ({len(rows)} windows classified)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Qualifier IQ Data Stream file")
    parser.add_argument("--output", default="results/classification_log.csv")
    parser.add_argument("--model", default="results/best_model.pt")
    args = parser.parse_args()
    run_inference(args.input, args.output, args.model)
