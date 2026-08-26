"""
Add the NOISE_FLOOR class to an already-built dataset, without rebuilding
from RadioML/RadChar.

NOISE_FLOOR needs neither raw source -- it's pure generated noise -- so
there's no reason to redo the expensive civilian/radar rebuild just to add
it. This loads an existing (7-class) X/y/snr_labels.npy, generates the same
per-SNR-bin volume of NOISE_FLOOR the real build_dataset.py would (matching
configs/default.yaml: examples_per_class_per_snr), and appends them.

`y` is multi-hot (N, len(CLASSES)), same as build_dataset.py's multi_hot()
produces -- NOT a 1D array of integer class indices. A dataset built before
the multi-label pivot won't load correctly here; rebuild from scratch with
`python -m src.data.build_dataset` instead.

Only valid if the existing dataset's classes are exactly CLASSES[:-1] (every
class except NOISE_FLOOR) -- checked before writing anything.

Usage:
    python scripts/append_noise_floor.py --data-dir data/processed
    python scripts/append_noise_floor.py --data-dir data/processed --out data/processed_v2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT  # noqa: E402
from src.data.preprocess import preprocess_window  # noqa: E402
from src.generators.noise import random_noise_example  # noqa: E402

NOISE_IDX = CLASS_TO_IDX["NOISE_FLOOR"]


def main(data_dir, out_dir, seed):
    data_dir = Path(data_dir)
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    snr_labels = np.load(data_dir / "snr_labels.npy")

    if y.ndim != 2 or y.shape[1] != len(CLASSES):
        raise ValueError(
            f"y.npy has shape {y.shape}, expected (N, {len(CLASSES)}) multi-hot -- "
            "this looks like a pre-multi-label dataset. Rebuild from scratch with "
            "`python -m src.data.build_dataset` instead of appending to it."
        )

    present = set(np.flatnonzero(y.sum(axis=0)))
    expected = set(range(len(CLASSES))) - {NOISE_IDX}
    if present != expected:
        raise ValueError(
            f"Existing dataset's classes ({sorted(present)}) don't match "
            f"'every class except NOISE_FLOOR' ({sorted(expected)}) -- "
            "this script only appends, it doesn't reconcile a mismatched set."
        )

    n_per = CFG["dataset"]["examples_per_class_per_snr"]
    window_len = CFG["signal"]["window_len"]
    rng = np.random.default_rng(seed if seed is not None else CFG["dataset"]["seed"])

    new_X, new_y, new_snr = [], [], []
    for snr_db in CFG["snr_bins_db"]:
        for _ in range(n_per):
            new_X.append(preprocess_window(random_noise_example(rng=rng), window_len))
            row = np.zeros(len(CLASSES), dtype=y.dtype)
            row[NOISE_IDX] = 1
            new_y.append(row)
            new_snr.append(snr_db)

    new_X = np.stack(new_X).astype(X.dtype)
    new_y = np.stack(new_y).astype(y.dtype)
    new_snr = np.array(new_snr, dtype=snr_labels.dtype)

    X_out = np.concatenate([X, new_X])
    y_out = np.concatenate([y, new_y])
    snr_out = np.concatenate([snr_labels, new_snr])

    out_dir = Path(out_dir) if out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X_out)
    np.save(out_dir / "y.npy", y_out)
    np.save(out_dir / "snr_labels.npy", snr_out)

    print(f"Added {len(new_y)} NOISE_FLOOR examples ({n_per} per SNR bin x "
          f"{len(CFG['snr_bins_db'])} bins) to {len(y)} existing examples.")
    print(f"Wrote {len(y_out)} total examples to {out_dir}/")
    print("Class presence counts:", {CLASSES[c]: int(y_out[:, c].sum()) for c in range(len(CLASSES))})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="existing processed/ folder (7-class)")
    p.add_argument("--out", default=None,
                   help="output folder (default: overwrite --data-dir in place)")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    main(a.data_dir, a.out, a.seed)
