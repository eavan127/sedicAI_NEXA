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


def combined_version(manifests):
    """Short fingerprint of exactly which shards went in. Two combined sets
    with the same value were built from byte-identical shards."""
    keys = sorted(f"{m['shard']}:{m['sha256']}" for m in manifests)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]
