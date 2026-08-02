"""
Jamming generators: barrage (wideband noise), tone (CW), and sweep.
Owner: Person C.

Verified by tests/test_jamming.py, which checks the achieved Jammer-to-Signal
Ratio matches what was requested.
"""
import numpy as np

from src.config import CFG
from src.generators.radar import generate_lfm_chirp_iq


def generate_barrage_jamming(n_samples, rng=None):
    """Wideband noise jammer."""
    rng = rng or np.random.default_rng()
    return rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)


def generate_tone_jamming(fs, n_samples, freqs, rng=None):
    """Single or multi-tone continuous-wave jammer."""
    t = np.arange(n_samples) / fs
    return sum(np.exp(2j * np.pi * f * t) for f in freqs)


def generate_sweep_jamming(fs, duration, bandwidth):
    """Fast repeating sweep jammer — same chirp math as radar, but tuned to
    jamming-typical sweep rates rather than pulsed radar timing."""
    return generate_lfm_chirp_iq(fs, duration, bandwidth)


def apply_jamming(signal, jammer, jsr_db):
    """Overlay a jammer onto a legitimate signal at a controlled Jammer-to-Signal Ratio."""
    sig_power = np.mean(np.abs(signal) ** 2)
    jam_power = np.mean(np.abs(jammer) ** 2)
    scale = np.sqrt((sig_power * 10 ** (jsr_db / 10)) / jam_power)
    return signal + scale * jammer[:len(signal)]


def random_jamming_example(fs=None, total_duration=None, rng=None):
    """One randomized jamming example; the jamming kind is chosen at random."""
    rng = rng or np.random.default_rng()
    fs = fs or CFG["signal"]["fs"]
    total_duration = total_duration or CFG["signal"]["total_duration"]
    n_samples = int(fs * total_duration)

    kind = rng.choice(["barrage", "tone", "sweep"])
    if kind == "barrage":
        return generate_barrage_jamming(n_samples, rng=rng)
    if kind == "tone":
        n_tones = rng.integers(1, CFG["jamming"]["max_tones"] + 1)
        freqs = rng.uniform(-fs / 4, fs / 4, n_tones)
        return generate_tone_jamming(fs, n_samples, freqs, rng=rng)
    bandwidth = rng.uniform(*CFG["jamming"]["sweep_bandwidth_hz"])
    return generate_sweep_jamming(fs, total_duration, bandwidth)
