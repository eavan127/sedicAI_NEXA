"""Checks that each JAM class (src/generators/jam.py) is built the way its
docstring says: shape, power, bandwidth, tones, repetition period, on/off."""
import numpy as np
import pytest

from src.data import impairments
from src.generators import jam

FS, N = jam.FS, jam.WINDOW_LEN


def clean(cls, seed=0):
    return jam.generate_window(cls, None, np.random.default_rng(seed))


def occupied_bw(x, frac=0.99):
    """Two-sided width holding `frac` of the power (Hz)."""
    p = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    c = np.cumsum(p) / p.sum()
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS))
    lo, hi = np.searchsorted(c, (1 - frac) / 2), np.searchsorted(c, 1 - (1 - frac) / 2)
    return f[min(hi, len(f) - 1)] - f[lo]


@pytest.mark.parametrize("cls", jam.CLASSES)
def test_contract(cls):
    for seed in range(20):
        x, params = clean(cls, seed)
        assert x.shape == (N,) and x.dtype == np.complex64
        assert np.isfinite(x).all()
        assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=1e-3)
        assert set(params) == set(jam.PARAM_KEYS)


@pytest.mark.parametrize("cls", jam.CLASSES)
def test_runs_through_impairments(cls):
    for snr in (-10, 0, 20):
        x, _ = impairments.generate_impaired(jam, cls, snr, np.random.default_rng(snr + 50))
        assert x.shape == (N,) and np.isfinite(x).all()


def test_same_seed_same_window():
    for cls in jam.CLASSES:
        assert np.array_equal(clean(cls, 7)[0], clean(cls, 7)[0])


def test_noise_bandwidths_do_not_overlap():
    spot = max(occupied_bw(clean("JAM_SPOT", s)[0]) for s in range(20))
    part = [occupied_bw(clean("JAM_PARTIAL_BAND", s)[0]) for s in range(20)]
    barr = min(occupied_bw(clean("JAM_BARRAGE", s)[0]) for s in range(20))
    assert spot < 100e3
    assert 150e3 < min(part) and max(part) < 700e3
    assert barr > 800e3


def test_multitone_count():
    for s in range(20):
        x, p = clean("JAM_MULTITONE", s)
        assert jam.MULTITONE_N[0] <= p["n_tones"] <= jam.MULTITONE_N[1]


def test_chirp_faster_than_fmcw():
    for s in range(20):
        _, p = clean("JAM_CHIRP", s)
        assert p["period_samples"] < 64          # RADAR_FMCW starts at 64
        _, p = clean("JAM_SWEPT", s)
        assert p["period_samples"] > N           # slower than one window


def test_pulsed_switches_on_and_off():
    for s in range(20):
        x, p = clean("JAM_PULSED", s)
        on = np.abs(x) > 0
        assert on.any() and (~on).any()
        assert np.count_nonzero(np.diff(on.astype(int)) == 1) >= 2   # two bursts or more


def test_reactive_switches_on_once():
    for s in range(20):
        x, p = clean("JAM_REACTIVE", s)
        on = np.abs(x) > 0
        assert not on[: p["onset_sample"]].any()
        assert on[p["onset_sample"]:].all()


def test_drfm_has_louder_delayed_copy():
    for s in range(20):
        x, p = clean("JAM_DRFM", s)
        assert jam.DRFM_DELAY_SAMPLES[0] <= p["delay_samples"] <= jam.DRFM_DELAY_SAMPLES[1]
        assert np.count_nonzero(np.abs(x)) < N   # silence around the pulse


def test_noise_floor_ignores_snr():
    a, _ = jam.generate_window("NOISE_FLOOR", 20, np.random.default_rng(1))
    b, _ = jam.generate_window("NOISE_FLOOR", None, np.random.default_rng(1))
    assert np.array_equal(a, b)
