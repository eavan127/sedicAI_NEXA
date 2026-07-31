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


def load_real_radar():
    """Real LFM waveforms from RadChar, or [] if the file is not present.

    IMPORTANT: these already contain noise at their labelled SNR, so they must
    NOT be passed through add_awgn. Doing so would leave each sample noisier
    than its own label claims, making every RadChar SNR label wrong.

    Only P2 downloads RadChar, so a missing file is not an error — the rest of
    the team still needs build_dataset to run.
    """
    from src.data.radchar import load_radchar_lfm

    n_per = CFG["dataset"]["examples_per_class_per_snr"]
    n_real = int(n_per * CFG["dataset"]["radchar_fraction"])
    if n_real == 0:
        return []

    try:
        return load_radchar_lfm(per_snr=n_real, snr_bins=CFG["snr_bins_db"])
    except FileNotFoundError:
        print("  ! RadChar not found — LFM_RADAR will be fully synthetic.")
        print("    See docs/pipeline/01-data-sources.md to download it.")
        return []


def build_synthetic_examples(n_real_radar=0, rng=None):
    """Generate labeled synthetic examples across every configured SNR bin.

    n_real_radar is how many real RadChar examples were loaded per SNR bin; the
    synthetic radar count is reduced by that much so LFM_RADAR ends up the same
    size as the other classes rather than double.
    """
    rng = rng or np.random.default_rng(CFG["dataset"]["seed"])
    n_per = CFG["dataset"]["examples_per_class_per_snr"]

    examples = []
    for class_name, gen_fn in SYNTHETIC_GENERATORS.items():
        n = n_per - n_real_radar if class_name == "LFM_RADAR" else n_per
        for snr_db in CFG["snr_bins_db"]:
            for _ in range(max(n, 0)):
                # Synthetic signals are generated clean, so they need noise
                # added. Real RadChar waveforms already carry theirs.
                sig = add_awgn(gen_fn(rng=rng), snr_db, rng=rng)
                examples.append((sig, class_name, snr_db))
    return examples


def build_full_dataset():
    """Combine three sources into windowed (X, y, snr) arrays:

        civilian     RadioML          (P1's loader)
        LFM_RADAR    RadChar + ours   (real in its regime, synthetic across a
                                       wider parameter range)
        FHSS/JAM     ours

    Mixing real and synthetic radar is deliberate: RadChar's parameters are a
    dataset-design choice (2-6 pulses packed into a 512-sample frame, 44-94%
    duty), whereas real radar runs at 0.1-10% duty. We do not know which the
    organisers' stream resembles, so we cover both.
    """
    X, y, snr_labels = [], [], []

    def add(iq, class_name, snr_db):
        X.append(preprocess_window(iq))
        y.append(CLASS_TO_IDX[class_name])
        snr_labels.append(snr_db)

    real_radar = load_real_radar()
    n_real_per_bin = (len(real_radar) // max(len(CFG["snr_bins_db"]), 1)) if real_radar else 0

    for iq, class_name, snr_db in load_radioml_civilian():
        add(iq, class_name, snr_db)

    for iq, class_name, snr_db in real_radar:
        add(iq, class_name, snr_db)          # no add_awgn — already noisy

    for iq, class_name, snr_db in build_synthetic_examples(n_real_per_bin):
        add(iq, class_name, snr_db)

    if not X:
        raise RuntimeError("No examples generated — check generators and RadioML loader.")

    print(f"  sources: {len(real_radar)} real RadChar, "
          f"{len(X) - len(real_radar)} synthetic/RadioML")

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
