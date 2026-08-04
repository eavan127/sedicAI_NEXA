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

RADIOML_PATH = REPO_ROOT / CFG["paths"]["raw_data"] / "GOLD_XYZ_OSC.0001_1024.hdf5"

# RadioML2018.01A stores /X, /Y, /Z with /Y a one-hot over 24 classes, laid out
# class-major then SNR-major: 4096 examples per (class, SNR), 26 SNRs per class
# (-20..+30 dB, 2 dB steps), so class c starts at row c*26*4096.
#
# The class-index <-> name mapping is NOT given inside the hdf5 file itself,
# and the classes.txt shipped alongside the archive is WRONG for this file --
# independently confirmed both by our own signal analysis (idx 17/18 sit at
# ~10x the amplitude scale of every other class -- clearly analog, not QAM as
# classes.txt claims; idx 21 is a near-perfectly constant-envelope signal,
# i.e. FM, not AM-DSB-WC) and by a published third-party analysis
# (cyclostationary.blog, "DeepSig's 2018 Dataset", Sept 2020) that reaches the
# same conclusion via cyclostationary signal processing.
#
# The order below instead comes from the ORIGINAL PAPER'S own class listing
# (O'Shea, Roy, Clancy, "Over the Air Deep Learning Based Radio Signal
# Classification", arXiv:1712.04578, Section III "Difficult Classes", same
# order used in Fig. 12/13's legend) rather than the separately-distributed
# classes.txt. It matches every structural check we ran: index 0 = OOK,
# indices 17-21 = the five analog classes (matching the amplitude-scale
# anomaly we measured), index 21 = FM (matching its near-constant envelope),
# indices 22-23 = GMSK, OQPSK as the final two constant-modulus classes.
#
# Residual risk: the exact BPSK/QPSK split within the digital block (indices
# 3-4) wasn't independently nailed down bit-for-bit (would need real carrier
# recovery, not just magnitude/differential-phase heuristics). If these
# labels are wrong, it will show up as a nonsensical confusion matrix for the
# civilian classes once P2 trains -- watch for that on Day 2.
RADIOML_CLASS_ORDER = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]
RADIOML_TARGET_CLASSES = {"BPSK": 3, "QPSK": 4, "16QAM": 12, "64QAM": 14}
RADIOML_EXAMPLES_PER_SNR_BLOCK = 4096
RADIOML_N_SNR_BINS = 26
RADIOML_MIN_SNR_DB = -20


def load_radioml_civilian(path=None, seed=None):
    """Load BPSK/QPSK/16QAM/64QAM from RadioML2018.01A.

    Returns list of (iq_complex_array, class_name, snr_db) tuples, subsampled
    to CFG['dataset']['examples_per_class_per_snr'] per (class, SNR) bin.

    Missing file returns [] (with a warning) so the rest of the pipeline
    stays runnable as a dry run -- same contract as load_real_radar().
    """
    import h5py

    path = path or RADIOML_PATH
    if not path.exists():
        print(f"  ! RadioML not found at {path} -- civilian classes will be empty.")
        print("    See docs/pipeline/01-data-sources.md to download it.")
        return []

    rng = np.random.default_rng(seed if seed is not None else CFG["dataset"]["seed"])
    n_per = CFG["dataset"]["examples_per_class_per_snr"]
    out = []

    with h5py.File(path, "r") as f:
        X = f["X"]
        for class_name, class_idx in RADIOML_TARGET_CLASSES.items():
            for snr_db in CFG["snr_bins_db"]:
                snr_idx = (snr_db - RADIOML_MIN_SNR_DB) // 2
                if not (0 <= snr_idx < RADIOML_N_SNR_BINS) or snr_db % 2 != 0:
                    raise ValueError(
                        f"snr_bins_db must be even and within "
                        f"[{RADIOML_MIN_SNR_DB}, {RADIOML_MIN_SNR_DB + 2*(RADIOML_N_SNR_BINS-1)}], "
                        f"got {snr_db}"
                    )
                block_start = (
                    class_idx * RADIOML_N_SNR_BINS * RADIOML_EXAMPLES_PER_SNR_BLOCK
                    + snr_idx * RADIOML_EXAMPLES_PER_SNR_BLOCK
                )
                block = X[block_start:block_start + RADIOML_EXAMPLES_PER_SNR_BLOCK]

                n = min(n_per, len(block))
                rows = rng.choice(len(block), n, replace=False)
                rows.sort()  # contiguous-ish access is friendlier to h5py than arbitrary order
                iq = block[rows, :, 0] + 1j * block[rows, :, 1]

                out.extend((iq[i], class_name, float(snr_db)) for i in range(n))

    return out


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
    """Yield labeled synthetic examples across every configured SNR bin.

    n_real_radar is how many real RadChar examples were loaded per SNR bin; the
    synthetic radar count is reduced by that much so LFM_RADAR ends up the same
    size as the other classes rather than double.

    A GENERATOR, deliberately. Each raw signal is total_duration * fs samples
    (~6,400 complex values, ~100 KB) while the windowed result is only 4 KB.
    Materialising every raw signal first needed ~1.7 GB at 1000 examples per bin
    and ran the machine out of memory; yielding lets the caller window each one
    and discard it immediately.
    """
    rng = rng or np.random.default_rng(CFG["dataset"]["seed"])
    n_per = CFG["dataset"]["examples_per_class_per_snr"]

    for class_name, gen_fn in SYNTHETIC_GENERATORS.items():
        n = n_per - n_real_radar if class_name == "LFM_RADAR" else n_per
        for snr_db in CFG["snr_bins_db"]:
            for _ in range(max(n, 0)):
                # Synthetic signals are generated clean, so they need noise
                # added. Real RadChar waveforms already carry theirs.
                yield add_awgn(gen_fn(rng=rng), snr_db, rng=rng), class_name, snr_db


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
