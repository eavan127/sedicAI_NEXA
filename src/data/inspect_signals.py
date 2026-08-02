"""
Look at the actual signals — one row per class, waveform beside spectrogram.

Two modes:

    python -m src.data.inspect_signals            # from the built dataset
    python -m src.data.inspect_signals --live     # freshly generated, incl.
                                                  # jamming broken out by sub-type

The --live mode is the one for diagnosing class confusion: it labels barrage,
tone and sweep jamming separately, so you can see which of them resembles FHSS.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from src.config import CFG, CLASSES, REPO_ROOT
from src.data.preprocess import add_awgn, preprocess_window

FS = CFG["signal"]["fs"]
WINDOW = CFG["signal"]["window_len"]


def _panel(ax_wave, ax_spec, iq, title):
    t = np.arange(len(iq)) / FS * 1e6

    ax_wave.plot(t, np.abs(iq), lw=0.8, color="tab:red")
    ax_wave.set_ylabel(title, fontsize=9, rotation=0, ha="right", va="center")
    ax_wave.set_xlim(0, t[-1])
    ax_wave.tick_params(labelsize=7)

    f_, t_, Z = stft(iq, fs=FS, nperseg=32, return_onesided=False)
    ax_spec.pcolormesh(t_ * 1e6, np.fft.fftshift(f_) / 1e6,
                        np.fft.fftshift(np.abs(Z), axes=0), shading="gouraud")
    ax_spec.tick_params(labelsize=7)


def from_dataset(snr_target=10):
    """Plot one example per populated class, straight from data/processed."""
    d = REPO_ROOT / CFG["paths"]["processed_data"]
    X = np.load(d / "X.npy")
    y = np.load(d / "y.npy")
    snr = np.load(d / "snr_labels.npy")

    present = [i for i in range(len(CLASSES)) if (y == i).any()]
    fig, axes = plt.subplots(len(present), 2, figsize=(11, 2.0 * len(present)))

    for row, cls_idx in enumerate(present):
        pool = np.flatnonzero((y == cls_idx) & (snr == snr_target))
        if pool.size == 0:
            pool = np.flatnonzero(y == cls_idx)
        arr = X[pool[0]]
        iq = arr[0] + 1j * arr[1]          # rebuild complex from the (2, N) pair
        _panel(axes[row, 0], axes[row, 1], iq, CLASSES[cls_idx])

    axes[-1, 0].set_xlabel("time (us)   —   amplitude", fontsize=8)
    axes[-1, 1].set_xlabel("time (us)   —   spectrogram (freq MHz)", fontsize=8)
    fig.suptitle(f"Dataset examples at SNR {snr_target} dB", fontsize=11)
    plt.tight_layout()

    out = REPO_ROOT / "results" / "signals_from_dataset.png"
    plt.savefig(out, dpi=140)
    print(f"Saved {out}")


def live(snr_db=10, seed=0):
    """Freshly generated, with jamming split into its three sub-types.

    This is the diagnostic view: tone and sweep jamming sit next to FHSS so you
    can judge by eye whether they are genuinely distinguishable.
    """
    from src.generators.fhss import random_fhss_example
    from src.generators.jamming import (generate_barrage_jamming,
                                         generate_sweep_jamming,
                                         generate_tone_jamming)
    from src.generators.radar import random_radar_example

    rng = np.random.default_rng(seed)
    n = int(FS * CFG["signal"]["total_duration"])

    rows = [
        ("LFM_RADAR", random_radar_example(rng=rng)),
        ("FHSS", random_fhss_example(rng=rng)),
        ("JAM barrage", generate_barrage_jamming(n, rng=rng)),
        ("JAM tone", generate_tone_jamming(FS, n, rng.uniform(-FS / 4, FS / 4, 2))),
        ("JAM sweep", generate_sweep_jamming(
            FS, CFG["signal"]["total_duration"],
            rng.uniform(*CFG["jamming"]["sweep_bandwidth_hz"]))),
    ]

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.0 * len(rows)))
    for row, (name, sig) in enumerate(rows):
        iq = add_awgn(sig[:WINDOW], snr_db, rng=rng)
        _panel(axes[row, 0], axes[row, 1], iq, name)

    axes[-1, 0].set_xlabel("time (us)   —   amplitude", fontsize=8)
    axes[-1, 1].set_xlabel("time (us)   —   spectrogram (freq MHz)", fontsize=8)
    fig.suptitle(f"One window ({WINDOW} samples = {WINDOW/FS*1e6:.0f} us) "
                  f"at SNR {snr_db} dB", fontsize=11)
    plt.tight_layout()

    out = REPO_ROOT / "results" / "signals_live_by_subtype.png"
    plt.savefig(out, dpi=140)
    print(f"Saved {out}")
    print("\nCompare FHSS against 'JAM tone' and 'JAM sweep'.")
    print("If they look alike to you, they look alike to the model — that is")
    print("the 59 jamming->FHSS errors in the confusion matrix.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true",
                   help="generate fresh signals, jamming split by sub-type")
    p.add_argument("--snr", type=float, default=10)
    args = p.parse_args()

    live(args.snr) if args.live else from_dataset(args.snr)
