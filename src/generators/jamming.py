"""
Jamming generators: barrage (wideband noise), tone (CW), and sweep.
Owner: Person C.

Verified by tests/test_jamming.py, which checks the achieved Jammer-to-Signal
Ratio matches what was requested.
"""
import numpy as np

from src.config import CFG
from src.generators.radar import generate_lfm_chirp_iq


def generate_barrage_jamming(n_samples, rng=None, fs=None, bandwidth=None, center=None):
    """Band-limited noise jammer.

    Previously this returned pure white noise across the entire band. The
    problem: radar at low duty cycle, buried in AWGN, is ALSO mostly white
    noise — so the two converged. Probing the trained model showed barrage at
    75.0% with 35 of 200 examples predicted as LFM_RADAR, and radar returning
    the favour with 20 of 200 predicted as JAMMING.

    Real barrage jammers flood a targeted band rather than the whole spectrum —
    you jam the frequencies the adversary uses. Band-limited noise has a defined
    spectral shape; white noise has none, which is what made it indistinguishable
    from a weak signal in noise.
    """
    rng = rng or np.random.default_rng()
    fs = fs or CFG["signal"]["fs"]

    white = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)

    cfg = CFG["jamming"]
    if bandwidth is None:
        bandwidth = rng.uniform(*cfg["barrage_bandwidth_hz"])
    nyquist = fs / 2
    if center is None:
        margin = max(nyquist - bandwidth / 2, 0.0)
        center = rng.uniform(-margin, margin)

    # Zero every frequency bin outside the target band.
    freqs = np.fft.fftfreq(n_samples, d=1 / fs)
    mask = np.abs(freqs - center) <= bandwidth / 2
    if not mask.any():                      # degenerate band, fall back to white
        return white

    filtered = np.fft.ifft(np.fft.fft(white) * mask)
    # Renormalise: filtering removed energy, and downstream SNR scaling assumes
    # unit-ish power.
    rms = np.sqrt(np.mean(np.abs(filtered) ** 2))
    return filtered / rms if rms > 0 else white


def generate_tone_jamming(fs, n_samples, freqs, rng=None):
    """Single or multi-tone continuous-wave jammer, with randomized
    per-tone phase and amplitude so the model learns tone structure,
    not a sterile artefact."""
    rng = rng or np.random.default_rng()
    t = np.arange(n_samples) / fs
    sig = np.zeros(n_samples, dtype=complex)
    for f in freqs:
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        sig += amp * np.exp(2j * np.pi * f * t + 1j * phase)
    return sig


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
