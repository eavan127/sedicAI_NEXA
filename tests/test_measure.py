import numpy as np
import pytest

from src.measure import (C42_BOUNDARY, C42_POOLED_ACCURACY, C42_THEORY,
                          MIN_WINDOWS_FOR_C42_DECISION,
                          ConstellationOrderEstimate, _normalized_c42,
                          constellation_order, estimate_snr_db,
                          noise_floor_power, occupancy, power_spectrum_db)
from src.ui.plots import SAMPLES_PER_SYMBOL, rrc_taps


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


# --- constellation_order --------------------------------------------------

def _qam_constellation(order):
    """The ideal, unit-average-power square M-QAM constellation -- built
    directly (no pulse shaping, no windowing, no recover_symbols), so this
    is exactly the abstract set of points the theory constants in
    C42_THEORY are meant to describe."""
    m = int(np.sqrt(order))
    levels = np.arange(-(m - 1), m, 2).astype(float)
    pts = np.array([complex(i, q) for i in levels for q in levels])
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _qam_window(order, n_symbols=64, sps=SAMPLES_PER_SYMBOL, snr_db=5.0,
                 seed=0, offset=0.003):
    """One RRC-shaped, carrier-offset, noisy IQ window of synthetic M-QAM --
    built to go through the real recover_symbols/constellation_order path,
    not the bare-constellation shortcut _qam_constellation is for.

    snr_db=5.0 sits comfortably inside the SNR >= +2 dB regime this
    estimator targets (see C42_POOLED_ACCURACY) -- clean enough that
    pooling several dozen windows separates 16QAM from 64QAM with a
    comfortable margin either side of C42_BOUNDARY (measured directly while
    building this test: pooled means land around 0.62-0.64 for 16QAM and
    0.57-0.585 for 64QAM against a boundary of 0.597, a bigger gap on both
    sides than the ~0.003 SNR-to-SNR jitter), while still being noisy
    enough that a single window is nowhere near definitive -- exactly the
    operating point the MIN_WINDOWS_FOR_C42_DECISION refusal exists for.
    """
    rng = np.random.default_rng(seed)
    const = _qam_constellation(order)
    symbols = const[rng.integers(0, len(const), n_symbols + 8)]
    train = np.zeros(len(symbols) * sps, dtype=complex)
    train[::sps] = symbols
    taps = rrc_taps(sps)
    shaped = np.convolve(train, taps)[len(taps) // 2:][:n_symbols * sps]
    shaped = shaped / np.sqrt(np.mean(np.abs(shaped) ** 2))
    shaped = shaped * np.exp(2j * np.pi * offset * np.arange(len(shaped)))
    noise = (rng.normal(0, 1, len(shaped)) + 1j * rng.normal(0, 1, len(shaped))
              ) * np.sqrt(10 ** (-snr_db / 10) / 2)
    return shaped + noise


def test_constellation_order_boundary_is_the_theoretical_midpoint():
    """Pinned explicitly so a future refactor cannot "improve" the boundary
    by fitting it to this project's own measured means -- see C42_BOUNDARY's
    comment in src/measure.py for why that would destroy the one
    calibration check this estimator has."""
    assert C42_THEORY == {"16QAM": 0.619, "64QAM": 0.5745}
    assert C42_BOUNDARY == pytest.approx((0.619 + 0.5745) / 2)


def test_c42_theory_matches_noiseless_synthetic_constellations():
    """Pins _normalized_c42 against THEORY (the ideal constellation's own
    fourth-order cumulant), not against this estimator's own average --
    build both constellations directly, no recovery pipeline involved.

    _normalized_c42 on the bare ideal square constellations measures
    ~0.680 (16QAM) and ~0.6190 (64QAM) by the exact closed-form kurtosis
    calculation; the constants this project has adopted as C42_THEORY
    (0.619 / 0.5745) sit a consistent ~0.045-0.06 below those bare values,
    reflecting the small shape loss recover_symbols' own matched-filter
    recovery chain introduces even before any channel noise is added
    (finite window, RRC receive filtering of a transmit pulse that only
    matches it exactly at infinite span). A tolerance of 0.07 comfortably
    covers that gap -- generous enough not to be fragile to which exact
    figure recover_symbols' pipeline settles on, but tight enough that it
    could not also accept the OTHER class's theory constant by mistake
    (16QAM's bare value is 0.106 from 64QAM's theory constant, well outside
    this tolerance).
    """
    c16 = _normalized_c42(_qam_constellation(16))
    c64 = _normalized_c42(_qam_constellation(64))
    assert c16 == pytest.approx(C42_THEORY["16QAM"], abs=0.07)
    assert c64 == pytest.approx(C42_THEORY["64QAM"], abs=0.07)


def test_constellation_order_returns_16qam_for_synthetic_16qam():
    windows = [_qam_window(16, seed=i) for i in range(80)]
    est = constellation_order(windows)
    assert est.decision == "16QAM"
    assert est.n_windows == 80
    assert est.mean_c42 > C42_BOUNDARY


def test_constellation_order_returns_64qam_for_synthetic_64qam():
    windows = [_qam_window(64, seed=1000 + i) for i in range(80)]
    est = constellation_order(windows)
    assert est.decision == "64QAM"
    assert est.n_windows == 80
    assert est.mean_c42 < C42_BOUNDARY


def test_constellation_order_refuses_below_the_minimum_window_count():
    """Fewer than MIN_WINDOWS_FOR_C42_DECISION pooled windows: decision must
    be None, but mean_c42/n_windows/accuracy are still reported -- a caller
    can see what was measured even though the estimator won't call it."""
    windows = [_qam_window(16, seed=i) for i in range(3)]
    est = constellation_order(windows)
    assert est.decision is None
    assert est.n_windows == 3
    assert np.isfinite(est.mean_c42)
    assert est.accuracy == C42_POOLED_ACCURACY[2]  # largest measured <= 3


def test_constellation_order_decides_at_exactly_the_minimum():
    windows = [_qam_window(16, seed=i) for i in range(MIN_WINDOWS_FOR_C42_DECISION)]
    est = constellation_order(windows)
    assert est.n_windows == MIN_WINDOWS_FOR_C42_DECISION
    assert est.decision is not None


def test_constellation_order_reports_zero_windows_honestly():
    est = constellation_order([])
    assert est.decision is None
    assert est.n_windows == 0
    assert est.accuracy is None
    assert np.isnan(est.mean_c42)


def test_constellation_order_skips_degenerate_windows_rather_than_scoring_them_zero():
    """A zero-power window comes back from recover_symbols unchanged (see
    its docstring) -- constellation_order must exclude it from the pool
    entirely, not silently count it as a |C42| of 0.0, which would bias a
    pooled average toward "noise-like" for reasons that have nothing to do
    with the actual modulation."""
    good = [_qam_window(16, seed=i) for i in range(10)]
    windows = good + [np.zeros(512, dtype=complex)] * 5
    est = constellation_order(windows)
    assert est.n_windows == 10           # the 5 dead windows contributed nothing
    only_good = constellation_order(good)
    assert est.mean_c42 == pytest.approx(only_good.mean_c42)


def test_constellation_order_returns_a_dataclass_with_the_documented_fields():
    est = constellation_order([_qam_window(16, seed=i) for i in range(10)])
    assert isinstance(est, ConstellationOrderEstimate)
    assert hasattr(est, "decision")
    assert hasattr(est, "mean_c42")
    assert hasattr(est, "n_windows")
    assert hasattr(est, "margin")
    assert hasattr(est, "accuracy")
