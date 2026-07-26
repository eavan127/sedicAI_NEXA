"""
FHSS (Frequency Hopping Spread Spectrum) generator.
Owner: Person B.

Verified by tests/test_fhss.py, which checks each hop segment actually lands
on one of the declared channel frequencies.
"""
import numpy as np

from src.config import CFG


def generate_fhss(fs, total_duration, hop_duration, hop_freqs, rng=None):
    """Generate a frequency-hopping signal from a random hop sequence."""
    rng = rng or np.random.default_rng()
    samples_per_hop = max(int(hop_duration * fs), 1)
    n_hops = max(int(total_duration / hop_duration), 1)
    t_hop = np.arange(samples_per_hop) / fs

    signal = np.zeros(n_hops * samples_per_hop, dtype=complex)
    for i in range(n_hops):
        f = rng.choice(hop_freqs)
        signal[i * samples_per_hop:(i + 1) * samples_per_hop] = np.exp(2j * np.pi * f * t_hop)
    return signal


def random_fhss_example(fs=None, total_duration=None, rng=None):
    """One randomized FHSS example, parameters drawn from the config ranges."""
    rng = rng or np.random.default_rng()
    fs = fs or CFG["signal"]["fs"]
    total_duration = total_duration or CFG["signal"]["total_duration"]

    hop_rate = rng.uniform(*CFG["fhss"]["hop_rate_hz"])
    hop_duration = 1 / hop_rate
    n_channels = rng.integers(*CFG["fhss"]["n_channels"])
    spacing = rng.uniform(*CFG["fhss"]["channel_spacing_hz"])
    hop_freqs = (np.arange(n_channels) - n_channels / 2) * spacing

    return generate_fhss(fs, total_duration, hop_duration, hop_freqs, rng=rng)
