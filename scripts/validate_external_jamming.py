"""
Independent validation: does JAMMING generalize to real, external jamming data
this model has never seen?

Source: "Raw IQ dataset for GNSS GPS jamming signal classification"
(Zenodo 4629685, CC BY 4.0). Six categories, .mat files, each holding one
variable `GNSS_plus_Jammer_awgn` of shape (1, N) complex128 -- N is much
longer than our window (measured ~16,368 samples per file vs our 512), so
each file is slid into multiple windows, the same way infer.py handles the
Qualifier IQ Stream.

This is GNSS-specific jamming (built to disrupt GPS receivers), not the same
context as our generator's generic barrage/tone/sweep jamming -- which is
exactly why it's a useful check: it's an honest test of whether the model
learned "this is deliberate interference" as a concept, not just our own
generator's specific parameter ranges. Standalone -- does not touch the
submission pipeline, the dataset, or the checkpoint.

Usage:
    python scripts/validate_external_jamming.py --data-dir Raw_IQ_Dataset
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT,  # noqa: E402
                         resolve_multilabel_thresholds)
from src.data.preprocess import preprocess_window  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JAM_IDX = CLASS_TO_IDX["JAMMING"]

# Every folder except NoJam is a genuine jammer -- true_is_jamming for scoring.
CATEGORIES = ["DME", "NB", "NoJam", "SingleAM", "SingleChirp", "SingleFM"]


def load_mat_iq(path):
    """Extract the complex IQ vector from one .mat file."""
    m = sio.loadmat(path)
    keys = [k for k in m if not k.startswith("__")]
    if not keys:
        raise ValueError(f"{path}: no data variable found")
    return np.asarray(m[keys[0]]).flatten()


def windows_from_capture(iq, window_len, stride):
    starts = range(0, len(iq) - window_len + 1, stride)
    return np.stack([preprocess_window(iq[s:s + window_len], window_len) for s in starts])


def main(data_dir, split, model_path, stride, max_files):
    window_len = CFG["signal"]["window_len"]
    stride = stride or window_len

    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    ensemble = sorted(ckpt_dir.glob("ensemble_*.pt"))
    if model_path:
        paths, desc = [Path(model_path)], Path(model_path).name
    elif ensemble:
        paths, desc = ensemble, f"{len(ensemble)}-model ensemble average"
    else:
        paths, desc = [ckpt_dir / "best_model.pt"], "best_model.pt"

    models = []
    for pth in paths:
        m = AMC_CNN(num_classes=len(CLASSES), input_len=window_len).to(DEVICE)
        m.load_state_dict(torch.load(pth, map_location=DEVICE))
        m.eval()
        models.append(m)

    # Per-class threshold on sigmoid outputs -- NOT argmax. This model is
    # multi-label, and argmax is a single-label decision rule left over from
    # before that change.
    #
    # The reason is consistency, not bias direction: the scorecard, the
    # console and this script must all call JAMMING the same way, or an
    # external number cannot be compared with an internal one. Measured on
    # 2,000 true-JAMMING windows from the held-out split, argmax actually
    # scored HIGHER than the threshold (92.8% vs 89.6%) -- JAMMING's threshold
    # of 0.77 is a high bar, so argmax accepting sub-threshold jamming
    # outweighs it rejecting jamming that a victim narrowly outranks. Either
    # way, only the threshold rule is comparable to evals/scorecard.json.
    jam_threshold = float(resolve_multilabel_thresholds()[JAM_IDX])
    print(f"Loaded {desc}")
    print(f"JAMMING threshold {jam_threshold:.2f} (sigmoid, multi-label)\n")

    print(f"{'category':<14}{'files':>7}{'windows':>9}{'jam-recall':>12}{'top mistake':>16}")
    print("-" * 62)

    overall_tp = overall_fn = overall_fp = overall_tn = 0

    for cat in CATEGORIES:
        folder = Path(data_dir) / split / cat
        files = sorted(folder.glob("*.mat"))[:max_files]
        if not files:
            print(f"{cat:<14}  (no files found at {folder})")
            continue

        preds = []
        for f in files:
            iq = load_mat_iq(f)
            batch = windows_from_capture(iq, window_len, stride)
            with torch.no_grad():
                xb = torch.tensor(batch).to(DEVICE)
                probs = sum(torch.sigmoid(m(xb)) for m in models) / len(models)
            preds.append(probs.cpu().numpy())
        probs = np.concatenate(preds)

        true_is_jam = cat != "NoJam"
        pred_is_jam = probs[:, JAM_IDX] > jam_threshold
        preds = probs.argmax(1)   # only used to name the most common mistake

        if true_is_jam:
            tp, fn = int(pred_is_jam.sum()), int((~pred_is_jam).sum())
            overall_tp += tp
            overall_fn += fn
            recall = tp / (tp + fn)
        else:
            fp, tn = int(pred_is_jam.sum()), int((~pred_is_jam).sum())
            overall_fp += fp
            overall_tn += tn
            recall = tn / (tn + fp)  # for NoJam, "recall" = correctly-quiet rate

        wrong = preds[preds != (JAM_IDX if true_is_jam else preds)]
        idx, cnt = (np.unique(preds[preds != JAM_IDX], return_counts=True)
                    if true_is_jam else np.unique(preds[pred_is_jam], return_counts=True))
        top = CLASSES[idx[cnt.argmax()]] if len(idx) else "-"

        label = "jam-recall" if true_is_jam else "quiet-rate"
        print(f"{cat:<14}{len(files):>7}{len(preds):>9}{recall:>11.1%}{'  ' + top:>16}")

    print()
    if overall_tp + overall_fn:
        print(f"Overall external jamming recall : {overall_tp / (overall_tp + overall_fn):.1%}"
              f"  ({overall_tp}/{overall_tp + overall_fn} windows)")
    if overall_fp + overall_tn:
        print(f"Overall false alarm rate (NoJam): {overall_fp / (overall_fp + overall_tn):.1%}"
              f"  ({overall_fp}/{overall_fp + overall_tn} windows wrongly flagged JAMMING)")
    print("\nCompare against your own held-out test set's jamming recall/false-alarm-rate")
    print("in evals/scorecard.json -- a large gap either way is worth a sentence in the brief.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="path to extracted Raw_IQ_Dataset/")
    p.add_argument("--split", default="Testing", choices=["Testing", "Training"])
    p.add_argument("--model", default=None)
    p.add_argument("--stride", type=int, default=None,
                   help="window stride in samples (default: non-overlapping)")
    p.add_argument("--max-files", type=int, default=50,
                   help="cap files per category -- 16,368-sample .mat files add up fast")
    a = p.parse_args()
    main(a.data_dir, a.split, a.model, a.stride, a.max_files)
