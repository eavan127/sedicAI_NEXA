"""
JAMMING detection broken down by sub-type: barrage (band-limited noise), tone
(single/multi carrier), sweep (fast repeating chirp) -- src/generators/jamming.py.

A teammate ran something like this once before, pre multi-label pivot
(configs/default.yaml's jamming section still carries the note: barrage 79.5%,
tone 56.5%, sweep 96.5%, with 85/200 tone examples predicted as FHSS). That
result is stale -- softmax/argmax logic from before the sigmoid multi-label
switch, and not run against the current model or ensemble. This is the
correct, current version: generates each sub-type explicitly (not via
random_jamming_example's random pick, which would blend them), runs through
the model with sigmoid + the calibrated JAMMING threshold, and for JAMMING's
misses, reports which other class (if any) got flagged instead -- the
multi-label-correct replacement for "top mistake" (checked against that
class's own threshold, not argmax, since more than one bit can be true or
none at all).

Usage:
    python scripts/jamming_subtype_breakdown.py --n 300
    python scripts/jamming_subtype_breakdown.py --n 300 --ensemble --n-models 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT, resolve_multilabel_thresholds  # noqa: E402
from src.data.preprocess import add_awgn, preprocess_window  # noqa: E402
from src.generators.jamming import (generate_barrage_jamming, generate_sweep_jamming,  # noqa: E402
                                     generate_tone_jamming)
from src.models.amc_cnn import AMC_CNN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LEN = CFG["signal"]["window_len"]
JAM_IDX = CLASS_TO_IDX["JAMMING"]
SUBTYPES = ["barrage", "tone", "sweep"]


def _load_models(ckpt_paths):
    models = []
    for p in ckpt_paths:
        model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN).to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models


def _predict_probs(models, X):
    summed = None
    with torch.no_grad():
        for model in models:
            p = torch.sigmoid(model(torch.tensor(X).to(DEVICE))).cpu().numpy()
            summed = p if summed is None else summed + p
    return summed / len(models)


def _generate_subtype(kind, fs, total_duration, rng):
    n_samples = int(fs * total_duration)
    cfg = CFG["jamming"]
    if kind == "barrage":
        return generate_barrage_jamming(n_samples, rng=rng, fs=fs)
    if kind == "tone":
        n_tones = rng.integers(1, cfg["max_tones"] + 1)
        freqs = rng.uniform(-fs / 4, fs / 4, n_tones)
        return generate_tone_jamming(fs, n_samples, freqs, rng=rng)
    bandwidth = rng.uniform(*cfg["sweep_bandwidth_hz"])
    return generate_sweep_jamming(fs, total_duration, bandwidth)


def main(n, ensemble, n_models):
    rng = np.random.default_rng(CFG["dataset"]["seed"] + 311)
    fs = CFG["signal"]["fs"]
    total_duration = CFG["signal"]["total_duration"]

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
        ckpt_paths = [ckpt_dir / "best_model.pt"]
        ckpt_desc = "best_model.pt"
    models = _load_models(ckpt_paths)
    thresholds = resolve_multilabel_thresholds()
    jam_threshold = thresholds[JAM_IDX]

    print(f"Checkpoint: {ckpt_desc}")
    print(f"JAMMING threshold: {jam_threshold:.3f}\n")
    print(f"{'sub-type':<10}{'n':>6}{'recall':>10}{'mean_prob':>12}{'top confusion (of misses)':>28}")
    print("-" * 66)

    for kind in SUBTYPES:
        windows = []
        for _ in range(n):
            sig = _generate_subtype(kind, fs, total_duration, rng)
            snr_db = rng.choice(CFG["snr_bins_db"])
            noisy = add_awgn(sig, float(snr_db), rng=rng)
            windows.append(preprocess_window(noisy, WINDOW_LEN))
        X = np.stack(windows).astype(np.float32)

        probs = _predict_probs(models, X)
        jam_prob = probs[:, JAM_IDX]
        detected = jam_prob > jam_threshold
        recall = detected.mean()

        missed = ~detected
        top = "-"
        if missed.any():
            other_idx = [i for i in range(len(CLASSES)) if i != JAM_IDX]
            other_present = probs[np.ix_(missed, other_idx)] > thresholds[other_idx]
            counts = other_present.sum(axis=0)
            if counts.max() > 0:
                best = other_idx[counts.argmax()]
                top = f"{CLASSES[best]} ({int(counts.max())}/{int(missed.sum())} misses)"

        print(f"{kind:<10}{n:>6}{recall:>10.1%}{jam_prob.mean():>12.3f}{top:>28}")

    print("\nCompare against Section 6.1's overall JAMMING recall -- a sub-type")
    print("noticeably below that number is the one to call out explicitly in the")
    print("brief, same as the historical single-label finding (tone jamming")
    print("confused with FHSS) this replaces with a current, correct check.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--n-models", type=int, default=5)
    a = p.parse_args()
    main(a.n, a.ensemble, a.n_models)
