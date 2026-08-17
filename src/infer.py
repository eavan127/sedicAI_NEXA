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


def _load_models(model_path=None, ensemble=False):
    """One checkpoint, or every ensemble member if --ensemble is given.

    Averaging members cancels initialisation noise. Seed variance measured 2.2
    points on radar and 8.9 on jamming, so a single checkpoint lands at a random
    point in that range — not what you want for the submitted run.
    """
    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    if ensemble:
        paths = sorted(ckpt_dir.glob("ensemble_*.pt"))
        if not paths:
            raise FileNotFoundError(
                f"No ensemble members in {ckpt_dir}. "
                "Run: python scripts/train_ensemble.py --models 5")
    else:
        paths = [Path(model_path) if model_path else ckpt_dir / "best_model.pt"]

    models = []
    for p in paths:
        m = AMC_CNN(num_classes=len(CLASSES), input_len=CFG["signal"]["window_len"]).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        models.append(m)
    return models, paths


# Below this softmax confidence, treat the call as unreliable rather than
# trusting a forced guess — mirrors the organizers' own example output
# (briefing slides: "TYPE-C UNKNOWN | AMBIGUOUS | 45.1% | INVESTIGATE").
# This never changes predicted_class or is_threat — the graded classification
# log still forces one of the 7 labels every time. It only adds a display/
# reporting column, since abstaining on the graded output would just count as
# a miss against recall.
LOW_CONFIDENCE = 0.5


def _status(predicted_class, confidence):
    """Operational status label, not a scoring input — see LOW_CONFIDENCE above."""
    if confidence < LOW_CONFIDENCE:
        return "INVESTIGATE"
    return "TRACKED" if predicted_class in CFG["judged_classes"] else "MONITOR"


def run_inference(input_path, output_path, model_path=None, stride=None, ensemble=False):
    cfg_sig = CFG["signal"]
    window_len = cfg_sig["window_len"]
    stride = stride or window_len

    iq = load_iq_file(input_path)
    if len(iq) < window_len:
        raise ValueError(f"Input has {len(iq)} samples, need at least {window_len}")

    models, paths = _load_models(model_path, ensemble)
    print(f"Using {len(models)} model(s): {', '.join(p.name for p in paths)}")

    starts = range(0, len(iq) - window_len + 1, stride)
    batch = torch.tensor(
        np.stack([preprocess_window(iq[s:s + window_len], window_len) for s in starts])
    ).to(DEVICE)

    with torch.no_grad():
        probs = np.mean(
            [torch.softmax(m(batch), dim=1).cpu().numpy() for m in models], axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["window_index", "sample_start", "predicted_class",
                            "confidence", "is_threat", "status"]
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
                "status": _status(cls, float(p.max())),
            })

    n_threat = sum(CLASSES[p.argmax()] in CFG["judged_classes"] for p in probs)
    print(f"Wrote {len(probs)} classified windows to {output_path}")
    print(f"  {n_threat} flagged as Military/CEMA or Jamming")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Qualifier IQ Data Stream file")
    parser.add_argument("--output", default="evals/classification_log.csv")
    parser.add_argument("--model", default=None)
    parser.add_argument("--stride", type=int, default=None,
                        help="overlap windows; smaller catches bursts on a boundary")
    parser.add_argument("--ensemble", action="store_true",
                        help="average every results/ensemble_*.pt instead of one checkpoint")
    args = parser.parse_args()
    run_inference(args.input, args.output, args.model, args.stride, args.ensemble)
