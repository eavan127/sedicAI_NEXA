"""
Pull windows for chosen classes out of the external datasets (RadioML 2018,
HisarMod 2019.1), already in the shard format's shape: complex64, first
WINDOW_LEN samples of each 1024-sample example -- the same cut
build_dataset.py has always used for RadioML.

Both readers pick rows at random but DISJOINTLY across train / val / test, so
the same external example can never sit in two splits. The row index in the
source file is returned with each window; it goes in the shard's `seed`
column so any external window can be traced back to its source row.

Neither dataset states a physical sample rate (RadioML is 8 samples/symbol,
see src/dsp.py), so these windows are "the same modulation at some symbol
rate", which is why they are capped at a minority share of each class.
"""
from pathlib import Path

import numpy as np

from src.config import CFG, REPO_ROOT

WINDOW_LEN = CFG["signal"]["window_len"]
RAW = REPO_ROOT / CFG["paths"]["raw_data"]

# --- RadioML 2018.01A -------------------------------------------------------
# Class order and block layout: see the long note above RADIOML_CLASS_ORDER in
# src/data/build_dataset.py (classes.txt shipped with the archive is wrong).
from src.data.build_dataset import (RADIOML_CLASS_ORDER,  # noqa: E402
                                    RADIOML_EXAMPLES_PER_SNR_BLOCK,
                                    RADIOML_MIN_SNR_DB, RADIOML_N_SNR_BINS,
                                    RADIOML_PATH)

# RadioML's own names for the dataset classes it covers. Most are identical;
# keep this table the single place a mismatch would be fixed.
RADIOML_NAMES = {
    "BPSK": "BPSK", "QPSK": "QPSK", "OQPSK": "OQPSK", "8PSK": "8PSK",
    "16PSK": "16PSK", "32PSK": "32PSK", "GMSK": "GMSK",
    "16QAM": "16QAM", "32QAM": "32QAM", "64QAM": "64QAM",
    "128QAM": "128QAM", "256QAM": "256QAM",
    "OOK": "OOK", "4ASK": "4ASK", "8ASK": "8ASK",
    "16APSK": "16APSK", "32APSK": "32APSK", "64APSK": "64APSK", "128APSK": "128APSK",
    "AM_SSB_WC": "AM-SSB-WC", "AM_SSB_SC": "AM-SSB-SC",
    "AM_DSB_WC": "AM-DSB-WC", "AM_DSB_SC": "AM-DSB-SC",
}

# --- HisarMod 2019.1 --------------------------------------------------------
# Integer labels from the dataset's readme.txt. Only the ones a teammate maps
# to a dataset class are listed; extend with care (e.g. HisarMod's FM has an
# unstated deviation, so whether it is WBFM or NBFM is not known).
HISARMOD_LABELS = {
    "2FSK": 2, "4FSK": 12, "8FSK": 22,
    "BPSK": 0, "QPSK": 10, "8PSK": 20, "16PSK": 30, "32PSK": 40,
    "16QAM": 21, "32QAM": 31, "64QAM": 41, "128QAM": 51, "256QAM": 61,
    "AM_DSB_WC": 4, "AM_DSB_SC": 14, "PM": 54,
}
HISARMOD_ROOT = RAW / "hisarmod" / "HisarMod2019.1"


def _disjoint_rows(n_available, counts, rng):
    """Split a random permutation of range(n_available) into consecutive,
    non-overlapping pieces of the requested sizes (fewer if it runs out)."""
    perm = rng.permutation(n_available)
    out, start = {}, 0
    for split, k in counts.items():
        out[split] = np.sort(perm[start:start + k])
        start += k
    return out


def radioml_windows(classes, snr_bins, counts, seed, path=None,
                    examples_per_block=RADIOML_EXAMPLES_PER_SNR_BLOCK,
                    n_snr_bins=RADIOML_N_SNR_BINS):
    """{(class, snr_db, split): (iq complex64 (k, WINDOW_LEN), source_rows int64 (k,))}

    counts = {"train": 240, "val": 30, "test": 30} windows per (class, SNR).
    Classes RadioML does not have, and SNRs outside its -20..+30 dB, are
    simply absent from the result -- the caller fills them from a generator.
    """
    import h5py

    path = Path(path) if path else RADIOML_PATH
    rng = np.random.default_rng(seed)
    out = {}
    with h5py.File(path, "r") as f:
        X = f["X"]
        for cls in classes:
            name = RADIOML_NAMES.get(cls)
            if name is None:
                continue
            class_idx = RADIOML_CLASS_ORDER.index(name)
            for snr in snr_bins:
                snr_idx = (snr - RADIOML_MIN_SNR_DB) // 2
                if snr % 2 or not 0 <= snr_idx < n_snr_bins:
                    continue
                block_start = (class_idx * n_snr_bins + snr_idx) * examples_per_block
                for split, rows in _disjoint_rows(examples_per_block, counts, rng).items():
                    if not len(rows):
                        continue
                    block = X[block_start + rows[0]:block_start + rows[-1] + 1]
                    picked = block[rows - rows[0], :WINDOW_LEN]
                    iq = (picked[..., 0] + 1j * picked[..., 1]).astype(np.complex64)
                    out[(cls, snr, split)] = (iq, (block_start + rows).astype(np.int64))
    return out


def _parse_complex_row(line):
    # MATLAB writes "7.12-7.74i"; Python's complex() wants "j".
    return np.array([complex(tok) for tok in line.strip().replace("i", "j").split(",")[:WINDOW_LEN]],
                    dtype=np.complex64)


def _hisarmod_pick(folder, prefix, wanted_labels, snr_bins, counts, rng):
    """Choose rows from one HisarMod folder, then read only those rows of the
    (multi-GB) data CSV in a single streaming pass."""
    labels = np.loadtxt(folder / f"{prefix}_labels.csv", dtype=np.int64)
    snrs = np.loadtxt(folder / f"{prefix}_snr.csv", dtype=np.int64)
    chosen = {}  # row -> (label, snr, split)
    for label in wanted_labels:
        for snr in snr_bins:
            rows = np.flatnonzero((labels == label) & (snrs == snr))
            for split, idx in _disjoint_rows(len(rows), counts, rng).items():
                for r in rows[idx]:
                    chosen[int(r)] = (int(label), int(snr), split)
    windows = {}
    if chosen:
        last = max(chosen)
        with open(folder / f"{prefix}_data.csv") as f:
            for row, line in enumerate(f):
                if row in chosen:
                    windows[row] = _parse_complex_row(line)
                if row >= last:
                    break
    return chosen, windows


def hisarmod_windows(classes, snr_bins, counts, seed, root=None):
    """Same return shape as radioml_windows. HisarMod's Train folder feeds the
    train split and its Test folder feeds val and test, so no HisarMod example
    crosses from training into evaluation. HisarMod's SNRs run -20..+18 dB in
    2 dB steps; anything else is absent from the result."""
    root = Path(root) if root else HISARMOD_ROOT
    rng = np.random.default_rng(seed)
    label_to_cls = {HISARMOD_LABELS[c]: c for c in classes if c in HISARMOD_LABELS}
    out = {}
    plan = (("Train", "train", {"train": counts.get("train", 0)}),
            ("Test", "test", {k: v for k, v in counts.items() if k != "train"}))
    for folder, prefix, split_counts in plan:
        if not any(split_counts.values()):
            continue
        chosen, windows = _hisarmod_pick(root / folder, prefix, label_to_cls,
                                         snr_bins, split_counts, rng)
        groups = {}
        for row, (label, snr, split) in chosen.items():
            groups.setdefault((label_to_cls[label], snr, split), []).append(row)
        for key, rows in groups.items():
            rows = sorted(rows)
            out[key] = (np.stack([windows[r] for r in rows]), np.asarray(rows, dtype=np.int64))
    return out
