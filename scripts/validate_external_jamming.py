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

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT, resolve_multilabel_thresholds  # noqa: E402
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


def _load_models(ckpt_paths, window_len):
    models = []
    for p in ckpt_paths:
        model = AMC_CNN(num_classes=len(CLASSES), input_len=window_len).to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models


def _predict_probs(models, batch):
    """Average sigmoid probabilities over one or more models -- same
    averaging train_ensemble.py's _predict uses, so this checks what the
    actually-submitted ensemble does, not a single unrepresentative model."""
    summed = None
    with torch.no_grad():
        for model in models:
            p = torch.sigmoid(model(batch))
            summed = p if summed is None else summed + p
    return summed / len(models)


def main(data_dir, split, model_path, stride, max_files, ensemble, n_models):
    window_len = CFG["signal"]["window_len"]
    stride = stride or window_len

    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    if ensemble:
        ckpt_paths = [ckpt_dir / f"ensemble_{i}.pt" for i in range(n_models)]
        missing = [p for p in ckpt_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing ensemble checkpoints: {missing} -- run "
                f"train_ensemble.py --models {n_models} first")
        ckpt_desc = f"{n_models}-model ensemble average"
    else:
        ckpt_paths = [Path(model_path) if model_path else ckpt_dir / "best_model.pt"]
        ckpt_desc = ckpt_paths[0].name
    models = _load_models(ckpt_paths, window_len)
    threshold = resolve_multilabel_thresholds()[JAM_IDX]
    print(f"Loaded {ckpt_desc}, JAMMING threshold={threshold:.3f}\n")

    print(f"{'category':<14}{'files':>7}{'windows':>9}{'jam-recall':>12}{'mean prob':>11}")
    print("-" * 57)

    overall_tp = overall_fn = overall_fp = overall_tn = 0

    for cat in CATEGORIES:
        folder = Path(data_dir) / split / cat
        files = sorted(folder.glob("*.mat"))[:max_files]
        if not files:
            print(f"{cat:<14}  (no files found at {folder})")
            continue

        jam_probs = []
        for f in files:
            iq = load_mat_iq(f)
            batch = windows_from_capture(iq, window_len, stride)
            # sigmoid, not argmax -- JAMMING is an independent yes/no call, not
            # a competing single-label prediction, matching every other
            # evaluation in this pipeline post multi-label pivot.
            probs = _predict_probs(models, torch.tensor(batch).to(DEVICE))[:, JAM_IDX]
            jam_probs.append(probs.cpu().numpy())
        jam_probs = np.concatenate(jam_probs)
        pred_is_jam = jam_probs > threshold

        true_is_jam = cat != "NoJam"

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

        print(f"{cat:<14}{len(files):>7}{len(jam_probs):>9}{recall:>11.1%}{jam_probs.mean():>11.3f}")

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
    p.add_argument("--ensemble", action="store_true",
                   help="average over ensemble_*.pt instead of best_model.pt/--model")
    p.add_argument("--n-models", type=int, default=5)
    a = p.parse_args()
    main(a.data_dir, a.split, a.model, a.stride, a.max_files, a.ensemble, a.n_models)
