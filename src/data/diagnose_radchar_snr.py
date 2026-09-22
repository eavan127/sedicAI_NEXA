"""
Does RadChar's labelled SNR mean the same thing as ours?

The proposal to mix RadChar radar into overlays/mixtures (docs discussion,
2026-09-22) depends on this: if RadChar's "-10 dB" is actually cleaner than a
window our own add_awgn labels "-10 dB", training on both together teaches
the model two different meanings of the same number -- a new shortcut, not a
signal feature. This must be checked before radchar_fraction is raised or
overlay_jamming/mix_components are extended to draw real waveforms.

Method: RadChar's labels give exact ground-truth pulse timing (pulse_width,
time_delay, pulse_repetition_interval, number_of_pulses), so the on/off
pulse mask can be reconstructed EXACTLY -- no amplitude threshold needed,
sidestepping the exact failure mode diagnose_duty_cycle.py hit (noise
swamping a fixed dB-below-peak rule). Off-pulse samples are guaranteed pure
noise; measuring their power directly, and comparing to on-pulse power,
gives an independent SNR estimate to check against RadChar's own label.

Usage:
    python -m src.data.diagnose_radchar_snr
"""
import numpy as np

from src.config import CFG, REPO_ROOT
from src.data.radchar import DEFAULT_PATH, LFM, RADCHAR_FS


def pulse_mask(fs, n_samples, pulse_width, time_delay, pri, n_pulses):
    mask = np.zeros(n_samples, dtype=bool)
    t = time_delay
    count = 0
    while t < n_samples / fs and count < n_pulses:
        start = int(t * fs)
        end = min(int((t + pulse_width) * fs), n_samples)
        if start < n_samples:
            mask[start:end] = True
        t += pri
        count += 1
    return mask


def measured_snr_db(iq, mask):
    """Independent SNR estimate from ground-truth on/off segments, active-
    reference convention (matches src/data/preprocess.py::add_awgn)."""
    off = iq[~mask]
    on = iq[mask]
    if off.size == 0 or on.size == 0:
        return np.nan
    noise_power = np.mean(np.abs(off) ** 2)
    on_power = np.mean(np.abs(on) ** 2)
    if noise_power <= 0 or on_power <= noise_power:
        return np.nan
    sig_power = on_power - noise_power
    return float(10 * np.log10(sig_power / noise_power))


def check(n_per_snr=200):
    import h5py

    if not DEFAULT_PATH.exists():
        print(f"RadChar not found at {DEFAULT_PATH}")
        return

    our_snrs = CFG["snr_bins_db"]
    print(f"RadChar label vs INDEPENDENTLY MEASURED SNR (ground-truth pulse mask, n={n_per_snr}/bin)")
    print(f"{'RadChar label':>14} | {'measured (median)':>18} | {'measured (mean)':>16} | {'gap':>7}")
    print("-" * 65)

    with h5py.File(DEFAULT_PATH, "r") as f:
        labels = f["labels"][...]
        is_lfm = labels["signal_type"] == LFM

        for snr_label in our_snrs:
            rows = np.flatnonzero(is_lfm & (labels["signal_to_noise_ratio"] == snr_label))
            if rows.size == 0:
                print(f"{snr_label:>14} | no rows at this label in RadChar")
                continue
            rows = rows[:n_per_snr]

            measured = []
            for r in sorted(rows.tolist()):
                iq = f["iq"][r]
                lab = labels[r]
                mask = pulse_mask(RADCHAR_FS, len(iq), lab["pulse_width"],
                                   lab["time_delay"], lab["pulse_repetition_interval"],
                                   int(lab["number_of_pulses"]))
                m = measured_snr_db(iq, mask)
                if not np.isnan(m):
                    measured.append(m)

            if not measured:
                print(f"{snr_label:>14} | all degenerate (no measurable on/off split)")
                continue
            measured = np.array(measured)
            med, mean = np.median(measured), np.mean(measured)
            print(f"{snr_label:>14} | {med:>18.2f} | {mean:>16.2f} | {med - snr_label:>+7.2f}")

    print("\ngap = measured - labelled. ~0 means RadChar's SNR label means what ours does.")
    print("A large POSITIVE gap means RadChar's label is more pessimistic than the true SNR")
    print("(i.e. RadChar '-10dB' windows are actually cleaner than our '-10dB' windows).")


if __name__ == "__main__":
    check()
