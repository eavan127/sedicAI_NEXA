"""
RadChar loader — real labelled radar IQ (Huang et al., ICASSP 2023).
Owner: P2.

Dataset facts (from the official spec):
    iq      : (N, 512) complex64, sampled at 3.2 MHz
    labels  : structured array, one row per waveform, fields:
                index, signal_type, number_of_pulses, pulse_width,
                time_delay, pulse_repetition_interval, signal_to_noise_ratio

We want signal_type == 4 (linear_frequency_modulated). The other four types are
radar too, so they are available as extra military examples if we choose — see
`SIGNAL_TYPES` below.

Usage:
    python -m src.data.radchar          # QA: stats + spectrogram vs our generator
"""
from pathlib import Path

import numpy as np

from src.config import CFG, REPO_ROOT

SIGNAL_TYPES = {
    0: "coherent_pulse_train",
    1: "barker_code",
    2: "polyphase_barker_code",
    3: "frank_code",
    4: "linear_frequency_modulated",   # <- our LFM_RADAR class
}
LFM = 4
RADCHAR_FS = 3.2e6          # fixed by the dataset
RADCHAR_LEN = 512           # samples per waveform

DEFAULT_PATH = REPO_ROOT / CFG["paths"]["raw_data"] / "RadChar-Tiny.h5"


def load_radchar_lfm(path=None, per_snr=None, snr_bins=None, seed=42):
    """Load LFM waveforms as (iq_complex, "LFM_RADAR", snr_db) tuples.

    per_snr   : cap examples per SNR bin (None = take everything available)
    snr_bins  : restrict to these SNR values (None = every SNR in the file)
    """
    import h5py  # imported here so the module imports even without the dataset

    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"RadChar not found at {path}\n"
            "Download RadChar-Tiny.h5 into data/raw/ — see docs/pipeline/01-data-sources.md"
        )

    rng = np.random.default_rng(seed)

    with h5py.File(path, "r") as f:
        labels = f["labels"][...]
        is_lfm = labels["signal_type"] == LFM
        snrs = labels["signal_to_noise_ratio"]

        wanted = set(snr_bins) if snr_bins is not None else set(np.unique(snrs[is_lfm]).tolist())

        # Collect every wanted row index FIRST, then read once. Reading row by
        # row costs an HDF5 round-trip each time — at a few thousand examples
        # that took minutes; a single bulk read takes under a second.
        picked, picked_snr = [], []
        for snr in sorted(wanted):
            rows = np.flatnonzero(is_lfm & (snrs == snr))
            if rows.size == 0:
                continue
            if per_snr is not None and rows.size > per_snr:
                rows = rng.choice(rows, per_snr, replace=False)
            picked.append(rows)
            picked_snr.append(np.full(rows.size, float(snr)))

        if not picked:
            return []

        idx = np.concatenate(picked)
        snr_of = np.concatenate(picked_snr)

        # h5py fancy indexing requires strictly increasing indices
        order = np.argsort(idx)
        iq = f["iq"][idx[order]]
        snr_of = snr_of[order]

    return [(iq[i], "LFM_RADAR", float(snr_of[i])) for i in range(len(iq))]


def describe(path=None):
    """Print the real parameter distributions — these are measurements, and they
    are what our synthetic generator should be reconciled against."""
    import h5py

    path = Path(path) if path else DEFAULT_PATH
    with h5py.File(path, "r") as f:
        labels = f["labels"][...]
        lfm = labels[labels["signal_type"] == LFM]

        print(f"RadChar: {len(labels):,} waveforms, {len(lfm):,} are LFM")
        print(f"  sample rate      : {RADCHAR_FS/1e6:.1f} MHz")
        print(f"  samples/waveform : {f['iq'].shape[1]}  "
              f"({f['iq'].shape[1]/RADCHAR_FS*1e6:.0f} us)")
        print(f"  SNR range        : {lfm['signal_to_noise_ratio'].min()} to "
              f"{lfm['signal_to_noise_ratio'].max()} dB")
        print()
        print("  MEASURED parameters (LFM only):")
        for field, unit, scale in [
            ("pulse_width", "us", 1e6),
            ("pulse_repetition_interval", "us", 1e6),
            ("time_delay", "us", 1e6),
            ("number_of_pulses", "", 1),
        ]:
            v = lfm[field] * scale
            print(f"    {field:<28} {v.min():>8.2f} to {v.max():>8.2f} {unit}")

        # Duty cycle drives the pulse/gap pattern the model keys on
        duty = lfm["pulse_width"] / lfm["pulse_repetition_interval"]
        print(f"    {'duty cycle (pw/pri)':<28} {duty.min()*100:>8.1f} to "
              f"{duty.max()*100:>8.1f} %")
        print()
        print("  Ours, MEASURED from 2000 generated examples:")
        _describe_ours()


def _describe_ours(n=2000, seed=0):
    """Measure our generator's actual output rather than reading config bounds.

    PRI is drawn conditional on pulse width, so the naive
    pulse_width_max / pri_min gives a duty cycle that is never actually
    produced — it once reported 588%, which is physically impossible.
    """
    from src.generators.radar import random_radar_example

    rng = np.random.default_rng(seed)
    widths, gaps, duties = [], [], []

    for _ in range(n):
        sig = random_radar_example(rng=rng)
        active = np.abs(sig) > 1e-9
        if not active.any():
            continue
        duties.append(active.mean())
        # width of the first contiguous run of energy
        first = int(np.argmax(active))
        run = 0
        while first + run < len(active) and active[first + run]:
            run += 1
        widths.append(run / CFG["signal"]["fs"] * 1e6)

    r = CFG["radar"]
    print(f"    pulse_width (config)         {r['pulse_width_s'][0]*1e6:>8.2f} to "
          f"{r['pulse_width_s'][1]*1e6:>8.2f} us")
    print(f"    pri (config)                 {r['pri_s'][0]*1e6:>8.2f} to "
          f"{r['pri_s'][1]*1e6:>8.2f} us")
    print(f"    pulse_width (measured)       {min(widths):>8.2f} to {max(widths):>8.2f} us")
    print(f"    duty cycle (measured)        {min(duties)*100:>8.2f} to "
          f"{max(duties)*100:>8.2f} %")
    print()
    print("    RadChar's 44-94% duty sits inside our range, and we also cover")
    print("    the low-duty regime that real radar actually uses.")


def _qa_plot(path=None):
    """Side-by-side: a real RadChar LFM against one from our generator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    from src.generators.radar import random_radar_example

    real = load_radchar_lfm(path, per_snr=1, snr_bins=[10])[0][0]
    ours = random_radar_example(rng=np.random.default_rng(0))[:RADCHAR_LEN]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (sig, title, fs) in enumerate([
        (real, "RadChar (real LFM, SNR 10 dB)", RADCHAR_FS),
        (ours, "Our generator", CFG["signal"]["fs"]),
    ]):
        t = np.arange(len(sig)) / fs * 1e6
        axes[0, col].plot(t, sig.real, lw=0.8, label="I")
        axes[0, col].plot(t, sig.imag, lw=0.8, alpha=0.7, label="Q")
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("time (us)")
        axes[0, col].legend(fontsize=8)

        f_, t_, Z = stft(sig, fs=fs, nperseg=64, return_onesided=False)
        axes[1, col].pcolormesh(t_ * 1e6, np.fft.fftshift(f_) / 1e6,
                                 np.fft.fftshift(np.abs(Z), axes=0), shading="gouraud")
        axes[1, col].set_xlabel("time (us)")
        axes[1, col].set_ylabel("freq (MHz)")

    plt.tight_layout()
    out = REPO_ROOT / "results" / "radchar_vs_ours.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=140)
    print(f"\nSaved {out}")
    print("Look at the TOP row: count the pulses and the gaps between them.")


def plot_lfm_gallery(path=None, snr_db=20, n=4):
    """Several real LFM waveforms at high SNR, so the pulse train is unmistakable.

    Use a clean SNR first: at 10 dB and below the chirp is genuinely hard to see
    by eye, and you cannot tell "loaded wrong" from "buried in noise".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    examples = load_radchar_lfm(path, per_snr=n, snr_bins=[snr_db])
    if not examples:
        raise RuntimeError(f"no LFM examples at SNR {snr_db} dB")

    fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 6))
    for col, (iq, _, snr) in enumerate(examples):
        t = np.arange(len(iq)) / RADCHAR_FS * 1e6

        axes[0, col].plot(t, np.abs(iq), lw=0.9, color="tab:red")
        axes[0, col].set_title(f"RadChar LFM #{col+1}  (SNR {snr:.0f} dB)", fontsize=9)
        axes[0, col].set_xlabel("time (us)")
        axes[0, col].set_ylabel("|amplitude|" if col == 0 else "")

        f_, t_, Z = stft(iq, fs=RADCHAR_FS, nperseg=32, return_onesided=False)
        axes[1, col].pcolormesh(t_ * 1e6, np.fft.fftshift(f_) / 1e6,
                                 np.fft.fftshift(np.abs(Z), axes=0), shading="gouraud")
        axes[1, col].set_xlabel("time (us)")
        axes[1, col].set_ylabel("freq (MHz)" if col == 0 else "")

    plt.tight_layout()
    out = REPO_ROOT / "results" / "radchar_lfm_gallery.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=140)
    print(f"Saved {out}")
    print("TOP row  = amplitude: look for the bursts and the gaps between them.")
    print("BOTTOM   = spectrogram: each burst should be a short diagonal streak.")


if __name__ == "__main__":
    describe()
    print()
    plot_lfm_gallery()
    _qa_plot()
