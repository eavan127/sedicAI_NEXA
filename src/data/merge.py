"""
Pieces of the week-1 merge (scripts/merge_shards.py) that decide what ends up
in a training window: normalisation, the NOISE_FLOOR SNR fix, and the
mixture builder. Kept here, away from the file handling, so they can be
tested on a few synthetic windows.

Mixtures are built from shard windows, not regenerated: the shards are the
frozen data handoff, and regenerating would need every teammate's generator
at their exact commit. Components are taken from windows at or above
`pool_min_snr_db` and treated as clean, the same convention as
dataset.radioml_clean_min_snr_db, so the mixture gets one add_awgn pass and
its SNR label describes the whole window. Their own residual noise (at least
16 dB down) means a mixture labelled +20 dB really sits a few dB lower.
"""
import hashlib

import numpy as np

from src.config import CFG, CLASS_GROUPS, DATASET_CLASSES
from src.data.composite import unit_power
from src.data.preprocess import add_awgn
from src.generators.jamming import apply_jamming

JAMMERS = frozenset(CLASS_GROUPS["JAMMING"])
# NOISE_FLOOR is never a mixture component: an empty channel next to an
# emitter is just the emitter.
EMITTERS = tuple(c for c in DATASET_CLASSES if c not in JAMMERS and c != "NOISE_FLOOR")
MIX_KINDS = ("signal+jammer", "signal+signal", "three")


def normalize_batch(iq):
    """complex (B, L) -> float32 (B, 2, L), each window zero-mean, unit-std
    over both channels. Same result as preprocess_window, one window at a time."""
    arr = np.stack([iq.real, iq.imag], axis=1).astype(np.float64)
    mean = arr.mean(axis=(1, 2), keepdims=True)
    std = arr.std(axis=(1, 2), keepdims=True)
    return ((arr - mean) / (std + 1e-8)).astype(np.float32)


def fill_noise_floor_snr(snr_db, y, rng):
    """NOISE_FLOOR shards store SNR as NaN (there is no signal to measure
    against). compute_snr_weights turns one NaN into all-NaN weights and the
    sampler crashes, so give those rows an SNR bin drawn uniformly. It is
    bookkeeping only: train.py exempts NOISE_FLOOR from SNR weighting.
    Any other NaN is a real bug in a shard and raises."""
    snr_db = snr_db.astype(np.float32, copy=True)
    nan = np.isnan(snr_db)
    if not nan.any():
        return snr_db
    is_noise = y[:, DATASET_CLASSES.index("NOISE_FLOOR")] == 1
    if (nan & ~is_noise).any():
        raise ValueError(f"{int((nan & ~is_noise).sum())} non-NOISE_FLOOR windows have NaN SNR")
    snr_db[nan] = rng.choice(CFG["dataset_snr_bins_db"], size=int(nan.sum()))
    return snr_db


def pick_mixture_classes(rng, kind, emitters=EMITTERS, jammers=JAMMERS):
    """Distinct classes for one mixture. "three" is two emitters plus either a
    jammer or a third emitter, half the time each. With no jammers available
    (a partial merge), jammer slots become emitters."""
    emitters, jammers = list(emitters), sorted(jammers)
    if kind == "signal+jammer" and jammers:
        return [str(rng.choice(emitters)), str(rng.choice(jammers))]
    if kind in ("signal+jammer", "signal+signal"):
        return [str(c) for c in rng.choice(emitters, size=2, replace=False)]
    if kind == "three":
        if jammers and rng.random() < 0.5:
            return [str(c) for c in rng.choice(emitters, size=2, replace=False)] + [str(rng.choice(jammers))]
        return [str(c) for c in rng.choice(emitters, size=3, replace=False)]
    raise ValueError(f"unknown mixture kind {kind!r}")


def mix_windows(components, snr_db, rng):
    """Sum [(class_name, iq), ...] into one window at `snr_db`.

    Same rules as composite.mix_components, for the 60 week-1 classes: the
    first emitter is the power reference, other emitters come in at an SIR
    from dataset.mixture_sir_db, and each jammer is applied last at a JSR
    from jamming.jsr_db against everything else.
    """
    sir_lo, sir_hi = CFG["dataset"]["mixture_sir_db"]
    signals = [iq for name, iq in components if name not in JAMMERS]
    jams = [iq for name, iq in components if name in JAMMERS]
    if not signals:
        raise ValueError("a mixture needs at least one emitter")
    n = min(len(iq) for _, iq in components)
    mixed = unit_power(signals[0][:n])
    for iq in signals[1:]:
        mixed = mixed + unit_power(iq[:n]) * 10 ** (rng.uniform(sir_lo, sir_hi) / 20)
    for iq in jams:
        mixed = apply_jamming(mixed, unit_power(iq[:n]), rng.uniform(*CFG["jamming"]["jsr_db"]))
    return add_awgn(mixed, snr_db, rng).astype(np.complex64)


def combined_version(manifests, salt=""):
    """Short fingerprint of exactly which shards went in, plus `salt` naming
    any change the merge itself makes to their windows. Two combined sets
    with the same value were built from byte-identical shards the same way."""
    keys = sorted(f"{m['shard']}:{m['sha256']}" for m in manifests)
    if salt:
        keys.append(f"merge:{salt}")
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]


# --- external windows ---------------------------------------------------------
# Shards keep external windows (RadioML 2018, HisarMod 2019) exactly as the
# datasets ship them; the merge brings them into the generated windows'
# conventions. Measured on the test shards (2026-11-04), external windows
# differed from generated ones in two ways a model can latch onto:
#   * RadioML's SNR label is not our full-band SNR. Noise measured beyond
#     +-0.9 MHz (where its 8 samples/symbol signals have no energy) puts a
#     window labelled +10 dB at about +20 dB, +18 dB at about +30 dB, and
#     -10 dB at about -19 dB. At high SNR the band edges are therefore
#     almost empty (0.1% of the power beyond +-0.8 MHz, against 2-19% for
#     generated windows): "signal in the middle, nothing at the edges", the
#     look of JAM_PARTIAL_BAND, which is what the seed-1 model called much of
#     APSK. HisarMod's labels match (within ~1.5 dB).
#   * Both sit at 0 Hz, while impairments.py places every generated window
#     at a random frequency.
# harmonize_external re-noises each window to its label under our convention
# (white noise over the full band, power = signal power / 10^(SNR/10)) and
# moves it to a random frequency, like impairments' placement. A window
# already noisier than its label cannot be cleaned, so its label is moved
# down to the nearest bin at or below its measured SNR instead, when it is
# more than RELABEL_TOLERANCE_DB under.
EXTERNAL_SOURCES = ("radioml2018", "hisarmod2019")
EXTERNAL_FIX = "external-v1"
_EDGE_HZ = 0.9e6
# One window's noise estimate wobbles by about a dB; only relabel a window
# clearly noisier than its label (HisarMod sits within ~1.5 dB, RadioML's low
# bins 3-9 dB under).
RELABEL_TOLERANCE_DB = 1.5


def _snap_down(snr, bins):
    below = [b for b in bins if b <= snr + 1e-6]
    return float(max(below)) if below else float(min(bins))


def harmonize_external(iq, snr_db, rng, fs=None, bins=None):
    """iq complex (B, L), snr_db (B,) -> (new iq complex64 (B, L), new snr_db float32 (B,))."""
    from src.data.impairments import PROFILE, frequency_shift

    fs = fs or CFG["signal"]["fs"]
    bins = bins or CFG["dataset_snr_bins_db"]
    out = np.empty(iq.shape, dtype=np.complex64)
    snr_out = np.asarray(snr_db, dtype=np.float32).copy()
    L = iq.shape[1]
    f = np.fft.fftfreq(L, 1 / fs)
    edge = np.abs(f) > _EDGE_HZ
    room_max = PROFILE["placement"]["band_fraction"] * fs / 2
    for i, x in enumerate(np.asarray(iq, dtype=complex)):
        X = np.abs(np.fft.fft(x)) ** 2 / L                 # sums to L * mean power
        n_bin = float(X[edge].mean())                       # existing noise per bin
        p_noise = n_bin                                     # per-sample noise power
        p_sig = max(float(X.sum() / L) - p_noise, 1e-12)
        target = p_sig / 10 ** (snr_out[i] / 10)
        measured = 10 * np.log10(p_sig / max(p_noise, 1e-12))
        if target > p_noise:
            add = target - p_noise
            x = x + np.sqrt(add / 2) * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
        elif measured < snr_out[i] - RELABEL_TOLERANCE_DB:
            snr_out[i] = _snap_down(measured, bins)
        # Occupied width of the signal itself (noise floor removed), for the
        # placement room: 99% of the above-floor power.
        sig = np.fft.fftshift(np.maximum(X - n_bin, 0))
        c = np.cumsum(sig) / max(sig.sum(), 1e-12)
        lo, hi = np.searchsorted(c, 0.005), np.searchsorted(c, 0.995)
        bw = max((hi - lo + 1) * fs / L, 0.1e6)
        room = max(0.0, room_max - bw / 2)
        out[i] = frequency_shift(x, rng.uniform(-room, room)).astype(np.complex64)
    return out, snr_out
