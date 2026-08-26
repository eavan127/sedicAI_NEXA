import numpy as np
import pytest

from src.measure import (estimate_snr_db, noise_floor_power, occupancy,
                          power_spectrum_db)


def _noise(n, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)) / np.sqrt(2)


def test_noise_floor_of_pure_noise_is_its_power():
    iq = _noise(32768, sigma=1.0)
    assert noise_floor_power(iq) == pytest.approx(1.0, rel=0.25)


def test_noise_floor_ignores_a_loud_burst():
    """The quiet-percentile estimate must not be dragged up by a strong
    emitter occupying part of the capture."""
    iq = _noise(32768, sigma=1.0)
    iq[:8192] *= 30.0
    assert noise_floor_power(iq) == pytest.approx(1.0, rel=0.35)


def test_estimate_snr_recovers_a_known_ratio():
    """A window with 10x the noise power above the floor should read ~10 dB."""
    noise_power = 1.0
    rng = np.random.default_rng(1)
    n = 512
    signal = (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) / np.sqrt(2)
    signal *= np.sqrt(10.0 / np.mean(np.abs(signal) ** 2))
    window = signal + _noise(n, sigma=1.0, seed=2)
    assert estimate_snr_db(window, noise_power) == pytest.approx(10.0, abs=2.0)


def test_estimate_snr_of_noise_only_is_very_low():
    assert estimate_snr_db(_noise(512, seed=3), 1.0) < 3.0


def test_estimate_snr_never_returns_nan_or_inf():
    """A window at or below the floor must clamp, not divide by zero."""
    assert np.isfinite(estimate_snr_db(np.zeros(512, dtype=complex), 1.0))


def test_occupancy_is_low_for_pure_noise():
    assert occupancy(_noise(32768)) < 0.2


def test_occupancy_is_higher_with_a_strong_tone():
    iq = _noise(32768)
    t = np.arange(32768)
    iq = iq + 20.0 * np.exp(2j * np.pi * 0.1 * t)
    assert occupancy(iq) > 0.0


def test_occupancy_is_a_fraction():
    assert 0.0 <= occupancy(_noise(8192)) <= 1.0


def test_power_spectrum_peaks_at_the_tone_frequency():
    fs = 3_200_000
    t = np.arange(16384) / fs
    iq = np.exp(2j * np.pi * 800_000 * t) + 0.01 * _noise(16384)
    freqs, spectrum = power_spectrum_db(iq, fs)
    assert freqs[np.argmax(spectrum)] == pytest.approx(800_000, abs=20_000)


def test_power_spectrum_covers_full_complex_band():
    freqs, _ = power_spectrum_db(_noise(8192), 3_200_000)
    assert freqs.min() < -1_500_000
    assert freqs.max() > 1_500_000
