"""
Runs the trained model on the organizer's "Qualifier IQ Data Stream" and
writes the classification log required by the submission package.

Usage:
    python -m src.infer --input data/raw/qualifier_iq_stream.bin
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.config import CFG, CLASSES, REPO_ROOT
from src.data.preprocess import preprocess_window
from src.models.amc_cnn import AMC_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_iq_file(path, dtype=np.float32):
    """Read interleaved I,Q,I,Q,... from a binary file.

    NOTE (Person C): confirm the organizer's actual format before submitting —
    it may be complex64, int16, or carry a header. Getting this wrong silently
    produces a garbage log that still "runs".
    """
    raw = np.fromfile(path, dtype=dtype)
    if raw.size % 2:
        raw = raw[:-1]
    return raw[0::2] + 1j * raw[1::2]


def run_inference(input_path, output_path, model_path=None, stride=None):
    cfg_sig = CFG["signal"]
    window_len = cfg_sig["window_len"]
    stride = stride or window_len
    model_path = model_path or REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"

    iq = load_iq_file(input_path)
    if len(iq) < window_len:
        raise ValueError(f"Input has {len(iq)} samples, need at least {window_len}")

    model = AMC_CNN(num_classes=len(CLASSES), input_len=window_len).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    starts = range(0, len(iq) - window_len + 1, stride)
    batch = torch.tensor(
        np.stack([preprocess_window(iq[s:s + window_len], window_len) for s in starts])
    ).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["window_index", "sample_start", "predicted_class",
                            "confidence", "is_threat"]
        )
        writer.writeheader()
        for i, (start, p) in enumerate(zip(starts, probs)):
            cls = CLASSES[p.argmax()]
            writer.writerow({
                "window_index": i,
                "sample_start": start,
                "predicted_class": cls,
                "confidence": round(float(p.max()), 4),
                "is_threat": cls in CFG["judged_classes"],
            })

    n_threat = sum(CLASSES[p.argmax()] in CFG["judged_classes"] for p in probs)
    print(f"Wrote {len(probs)} classified windows to {output_path}")
    print(f"  {n_threat} flagged as Military/CEMA or Jamming")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Qualifier IQ Data Stream file")
    parser.add_argument("--output", default="evals/classification_log.csv")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    run_inference(args.input, args.output, args.model)
