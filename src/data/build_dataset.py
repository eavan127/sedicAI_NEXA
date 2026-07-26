"""
Assembles the full training dataset: RadioML civilian classes + synthetic
radar/FHSS/jamming, swept across the configured SNR range.

Run only after each generator has passed `pytest tests/` and its spectrogram
has been eyeballed against reference literature (docs section 5.6).

Usage:
    python -m src.data.build_dataset
"""
import numpy as np

from src.config import CFG, CLASS_TO_IDX, REPO_ROOT
from src.data.preprocess import add_awgn, preprocess_window
from src.generators.fhss import random_fhss_example
from src.generators.jamming import random_jamming_example
from src.generators.radar import random_radar_example

SYNTHETIC_GENERATORS = {
    "LFM_RADAR": random_radar_example,
    "FHSS": random_fhss_example,
    "JAMMING": random_jamming_example,
}


def load_radioml_civilian():
    """
    TODO (Person A): load RadioML2018.01a and filter to BPSK/QPSK/16QAM/64QAM.

    The exact parsing depends on which RadioML release you download, so this
    is left explicit rather than guessed. Expected return:
        list of (iq_complex_array, class_name, snr_db) tuples

    Returning empty keeps the rest of the pipeline runnable as a dry run.
    """
    return []


def build_synthetic_examples(rng=None):
    """Generate labeled synthetic examples across every configured SNR bin."""
    rng = rng or np.random.default_rng(CFG["dataset"]["seed"])
    n_per = CFG["dataset"]["examples_per_class_per_snr"]

    examples = []
    for class_name, gen_fn in SYNTHETIC_GENERATORS.items():
        for snr_db in CFG["snr_bins_db"]:
            for _ in range(n_per):
                sig = add_awgn(gen_fn(rng=rng), snr_db, rng=rng)
                examples.append((sig, class_name, snr_db))
    return examples


def build_full_dataset():
    """Combine civilian + synthetic data into windowed (X, y, snr) arrays."""
    X, y, snr_labels = [], [], []

    for iq, class_name, snr_db in load_radioml_civilian():
        X.append(preprocess_window(iq))
        y.append(CLASS_TO_IDX[class_name])
        snr_labels.append(snr_db)

    for iq, class_name, snr_db in build_synthetic_examples():
        X.append(preprocess_window(iq))
        y.append(CLASS_TO_IDX[class_name])
        snr_labels.append(snr_db)

    if not X:
        raise RuntimeError("No examples generated — check generators and RadioML loader.")

    return np.stack(X), np.array(y), np.array(snr_labels, dtype=float)


def main():
    X, y, snr_labels = build_full_dataset()
    out_dir = REPO_ROOT / CFG["paths"]["processed_data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "snr_labels.npy", snr_labels)

    print(f"Built {X.shape[0]} examples, shape per example {X.shape[1:]}")
    for name, idx in CLASS_TO_IDX.items():
        print(f"  {name:<12} {(y == idx).sum():>6}")
    print(f"Saved to {out_dir} (gitignored)")


if __name__ == "__main__":
    main()
