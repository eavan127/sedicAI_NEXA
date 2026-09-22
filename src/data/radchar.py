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

_RAW = REPO_ROOT / CFG["paths"]["raw_data"]
# Prefer the larger variant when present -- RadChar-Small has ~10x Tiny's real
# LFM examples per SNR bin, so a machine with both should use Small. Falls
# back to Tiny so a machine with only Tiny still works unchanged.
DEFAULT_PATH = _RAW / "RadChar-Small.h5" if (_RAW / "RadChar-Small.h5").exists() else _RAW / "RadChar-Tiny.h5"


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


# --- measured (not labelled) SNR -------------------------------------------
#
# RadChar's own `signal_to_noise_ratio` label does not mean what our
# add_awgn-labelled SNR means. Checked 2026-09-22 (src/data/diagnose_radchar_snr.py):
# comparing the label against an independent measurement (using RadChar's own
# ground-truth pulse timing to isolate pure-noise "off" samples from
# pulse+noise "on" samples, active-reference convention matching add_awgn),
# the gap is small at the label's -10dB end (~+1dB) but grows to -15dB at the
# label's own "+20dB" tier -- i.e. RadChar's nominal "clean" examples still
# carry real noise. Bucketing by the label would train the model on two
# different meanings of the same SNR number. load_radchar_lfm_by_measured_snr
# below buckets by the independently measured value instead.

RADCHAR_MEASURED_CACHE = REPO_ROOT / "data" / "processed" / "radchar_measured_snr.npz"


def pulse_mask(fs, n_samples, pulse_width, time_delay, pri, n_pulses):
    """Exact on/off sample mask from RadChar's own ground-truth pulse timing
    -- no amplitude threshold, so it doesn't inherit the failure mode a
    fixed dB-below-peak rule hits once noise is added (see
    src/data/diagnose_duty_cycle.py)."""
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
    """Independent SNR estimate from the ground-truth on/off split, active-
    reference convention (matches src/data/preprocess.py::add_awgn)."""
    off, on = iq[~mask], iq[mask]
    if off.size == 0 or on.size == 0:
        return np.nan
    noise_power = np.mean(np.abs(off) ** 2)
    on_power = np.mean(np.abs(on) ** 2)
    if noise_power <= 0 or on_power <= noise_power:
        return np.nan
    return float(10 * np.log10((on_power - noise_power) / noise_power))


def measure_all_lfm(path=None, cap_per_label=None, seed=0, use_cache=True):
    """Measure every (or a capped-per-label sample of every) LFM waveform's
    true SNR. Returns (rows, measured_snr, labelled_snr), aligned arrays.

    This is a one-time, several-minute pass (row-by-row pulse masking after
    one bulk HDF5 read), so it's cached to RADCHAR_MEASURED_CACHE. Delete
    that file to force a re-measure (e.g. after switching RadChar-Tiny <->
    RadChar-Small, or widening cap_per_label).
    """
    import h5py

    path = Path(path) if path else DEFAULT_PATH
    if use_cache and RADCHAR_MEASURED_CACHE.exists():
        cached = np.load(RADCHAR_MEASURED_CACHE)
        return cached["rows"], cached["measured"], cached["label"]

    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        labels = f["labels"][...]
        is_lfm = labels["signal_type"] == LFM
        lfm_rows = np.flatnonzero(is_lfm)

        by_label = {}
        for lab in np.unique(labels["signal_to_noise_ratio"][is_lfm]):
            rows = lfm_rows[labels["signal_to_noise_ratio"][lfm_rows] == lab]
            if cap_per_label is not None and rows.size > cap_per_label:
                rows = rng.choice(rows, cap_per_label, replace=False)
            by_label[lab] = np.sort(rows)

        all_rows = np.sort(np.concatenate(list(by_label.values())))
        iq_all = f["iq"][all_rows]          # one bulk read
        lab_all = labels[all_rows]

        rows_out, measured_out, label_out = [], [], []
        for i in range(len(all_rows)):
            m = measured_snr_db(
                iq_all[i],
                pulse_mask(RADCHAR_FS, iq_all.shape[1], lab_all[i]["pulse_width"],
                           lab_all[i]["time_delay"], lab_all[i]["pulse_repetition_interval"],
                           int(lab_all[i]["number_of_pulses"])),
            )
            if not np.isnan(m):
                rows_out.append(all_rows[i])
                measured_out.append(m)
                label_out.append(lab_all[i]["signal_to_noise_ratio"])

    rows_out, measured_out, label_out = map(np.array, (rows_out, measured_out, label_out))
    if use_cache:
        RADCHAR_MEASURED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(RADCHAR_MEASURED_CACHE, rows=rows_out, measured=measured_out, label=label_out)
    return rows_out, measured_out, label_out


def load_radchar_lfm_by_measured_snr(path=None, per_snr=None, snr_bins=None,
                                      tolerance_db=3.0, seed=42, exclude_rows=None,
                                      return_rows=False, cap_per_label=300):
    """Drop-in replacement for load_radchar_lfm that buckets each waveform by
    its OWN measured SNR instead of trusting RadChar's label (see the module
    note above this section for why).

    A waveform is only assigned to a snr_bins target if its measured SNR is
    within tolerance_db of it -- a waveform measuring -13dB is not silently
    relabelled into a -10dB bin.

    exclude_rows: row indices to never draw from, so a second caller (e.g.
    composite/mixture examples) can get a pool disjoint from a first caller's
    (e.g. standalone examples) -- reusing the same waveform in both leaks
    train/test the same way the RadioML loader's docstring warns about.

    return_rows: if True, returns (examples, rows_used) instead of just
    examples, so the caller can pass rows_used as the next call's
    exclude_rows.
    """
    import h5py

    path = Path(path) if path else DEFAULT_PATH
    snr_bins = np.array(snr_bins if snr_bins is not None else CFG["snr_bins_db"], dtype=float)

    rows, measured, _ = measure_all_lfm(path, cap_per_label=cap_per_label)
    if exclude_rows is not None and len(exclude_rows):
        keep_mask = ~np.isin(rows, exclude_rows)
        rows, measured = rows[keep_mask], measured[keep_mask]

    dist = np.abs(measured[:, None] - snr_bins[None, :])
    nearest_idx = dist.argmin(axis=1)
    nearest_gap = dist.min(axis=1)
    eligible = nearest_gap <= tolerance_db

    rng = np.random.default_rng(seed)
    picked_rows, picked_snr = [], []
    for i, target in enumerate(snr_bins):
        candidates = rows[eligible & (nearest_idx == i)]
        if per_snr is not None and candidates.size > per_snr:
            candidates = rng.choice(candidates, per_snr, replace=False)
        picked_rows.append(candidates)
        picked_snr.append(np.full(candidates.size, float(target)))

    picked_rows = np.concatenate(picked_rows) if picked_rows else np.array([], dtype=int)
    picked_snr = np.concatenate(picked_snr) if picked_snr else np.array([])
    if picked_rows.size == 0:
        return ([], []) if return_rows else []

    order = np.argsort(picked_rows)
    with h5py.File(path, "r") as f:
        iq = f["iq"][picked_rows[order]]
    snr_of = picked_snr[order]
    examples = [(iq[i], "LFM_RADAR", float(snr_of[i])) for i in range(len(iq))]
    return (examples, picked_rows[order]) if return_rows else examples


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
