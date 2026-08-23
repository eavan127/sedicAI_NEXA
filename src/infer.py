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

from src.config import CFG, CLASS_TO_IDX, CLASSES, REPO_ROOT
from src.data.preprocess import preprocess_window
from src.evaluate import TIERS
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


def _status(detected_classes):
    """Operational status label, not a scoring input.

    Empty detected_classes means nothing cleared multilabel_threshold for
    this window -- the multi-label equivalent of the old "confidence below
    LOW_CONFIDENCE" case (mirrors the organizers' own example output,
    briefing slides: "TYPE-C UNKNOWN | AMBIGUOUS | 45.1% | INVESTIGATE").
    This never changes detected_classes or is_threat — the graded
    classification log still reports exactly which classes crossed
    threshold. It only adds a display/reporting column.
    """
    if not detected_classes:
        return "INVESTIGATE"
    return "TRACKED" if any(c in CFG["judged_classes"] for c in detected_classes) else "MONITOR"


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
    X_all = np.stack([preprocess_window(iq[s:s + window_len], window_len) for s in starts])

    threshold = CFG.get("multilabel_threshold", 0.5)
    # Batched, not one forward pass over the whole stream -- a real Qualifier
    # stream can be many thousands of windows, and pushing them all through
    # the model at once OOMs a real GPU (same fix as src/evaluate.py and
    # scripts/train_ensemble.py).
    eval_batch_size = 256
    probs_chunks = []
    with torch.no_grad():
        for i in range(0, len(X_all), eval_batch_size):
            batch = torch.tensor(X_all[i:i + eval_batch_size]).to(DEVICE)
            # Multi-label: each class judged independently (sigmoid), not one
            # winner (softmax) -- a window can flag several classes at once,
            # e.g. a real signal AND jamming overlaid on top of it.
            probs_chunks.append(
                np.mean([torch.sigmoid(m(batch)).cpu().numpy() for m in models], axis=0))
    probs = np.concatenate(probs_chunks, axis=0)
    present = probs > threshold

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tier_fields = [f"{t.lower()}_present" for t in TIERS]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["window_index", "sample_start", "detected_classes",
                            "confidences", "is_threat", "status"] + tier_fields
        )
        writer.writeheader()
        for i, (start, p, flags) in enumerate(zip(starts, probs, present)):
            detected = [CLASSES[j] for j in range(len(CLASSES)) if flags[j]]
            row = {
                "window_index": i,
                "sample_start": start,
                "detected_classes": ";".join(detected) if detected else "NONE",
                "confidences": ";".join(f"{c}={p[CLASS_TO_IDX[c]]:.4f}" for c in detected),
                "is_threat": any(c in CFG["judged_classes"] for c in detected),
                "status": _status(detected),
            }
            for t, members in TIERS.items():
                row[f"{t.lower()}_present"] = any(c in members for c in detected)
            writer.writerow(row)

    n_threat = sum(
        any(CLASSES[j] in CFG["judged_classes"] for j in range(len(CLASSES)) if flags[j])
        for flags in present
    )
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
