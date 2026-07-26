"""
Assembles the full training dataset: RadioML civilian classes + synthetic
radar/FHSS/jamming, swept across an SNR range, windowed and labeled.

Run this after gen_radar.py / gen_fhss.py / gen_jamming.py have each been
self-QA'd (see docs/SEDIC2026_Track1_Documentation.md section 5.6).
"""
import numpy as np

from gen_radar import random_radar_example
from gen_fhss import random_fhss_example
from gen_jamming import random_jamming_example
from preprocess import add_awgn, preprocess_window

# Class index mapping — extend/adjust once RadioML classes are finalized
CLASSES = ["BPSK", "QPSK", "16QAM", "64QAM", "LFM_RADAR", "FHSS", "JAMMING"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

SNR_BINS_DB = [-10, -5, 0, 5, 10, 15]
FS = 1e6
WINDOW_LEN = 1024
EXAMPLES_PER_CLASS_PER_SNR = 400  # tune based on time budget, see docs section 5


def load_radioml_civilian():
    """
    TODO: replace with actual RadioML2018.01a loading + filtering to
    BPSK/QPSK/16QAM/64QAM classes. Placeholder returns nothing so the
    rest of the pipeline still runs end-to-end for a dry-run test.
    Expected return: list of (iq_complex_array, class_name) tuples.
    """
    return []


def build_synthetic_examples():
    """Generate labeled synthetic examples for the military/jamming classes,
    swept across every SNR bin."""
    examples = []
    generators = {
        "LFM_RADAR": random_radar_example,
        "FHSS": random_fhss_example,
        "JAMMING": random_jamming_example,
    }
    for class_name, gen_fn in generators.items():
        for snr_db in SNR_BINS_DB:
            for _ in range(EXAMPLES_PER_CLASS_PER_SNR):
                sig = gen_fn(fs=FS)
                sig = add_awgn(sig, snr_db)
                examples.append((sig, class_name, snr_db))
    return examples


def build_full_dataset():
    """Combine RadioML civilian data + synthetic data into windowed (X, y, snr) arrays."""
    X, y, snr_labels = [], [], []

    for iq, class_name in load_radioml_civilian():
        X.append(preprocess_window(iq, WINDOW_LEN))
        y.append(CLASS_TO_IDX[class_name])
        snr_labels.append(None)  # fill in from RadioML metadata

    for iq, class_name, snr_db in build_synthetic_examples():
        X.append(preprocess_window(iq, WINDOW_LEN))
        y.append(CLASS_TO_IDX[class_name])
        snr_labels.append(snr_db)

    return np.stack(X), np.array(y), np.array(snr_labels)


if __name__ == "__main__":
    X, y, snr_labels = build_full_dataset()
    print(f"Dataset built: {X.shape[0]} examples, shape per example {X.shape[1:]}")
    print(f"Class counts: {[(c, (y == i).sum()) for c, i in CLASS_TO_IDX.items()]}")
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    np.save("data/snr_labels.npy", snr_labels)
    print("Saved to data/X.npy, data/y.npy, data/snr_labels.npy (gitignored — not committed)")
