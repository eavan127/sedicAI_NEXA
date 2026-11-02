"""
Tests for the week-1 analog and ASK generators.

As with test_digital.py, these check INTERNAL CONSISTENCY -- that each class is
built the way src/generators/analog.py says it is -- not realism.
"""
import numpy as np
import pytest

from src.config import DATASET_CLASSES
from src.data import impairments
from src.generators import analog

N = analog.WINDOW_LEN


def clean(cls, seed=0):
    return analog.generate_window(cls, None, np.random.default_rng(seed))


def is_one_real_axis(sig):
    """True if the I/Q cloud is a line through the origin, i.e. a real signal
    with a phase rotation on it."""
    s = np.linalg.svd(np.column_stack([sig.real, sig.imag]), compute_uv=False)
    return s[1] < 1e-3 * s[0]


def sideband_powers(sig):
    spec = np.abs(np.fft.fft(sig * np.hanning(N))) ** 2
    f = np.fft.fftfreq(N)
    return spec[f > 1 / N].sum(), spec[f < -1 / N].sum()


def test_every_generator_class_is_a_dataset_class():
    assert set(analog.CLASSES) <= set(DATASET_CLASSES)


@pytest.mark.parametrize("cls", analog.CLASSES)
def test_shape_dtype_and_unit_power(cls):
    sig, _ = clean(cls)
    assert sig.shape == (N,) and sig.dtype == np.complex64
    assert np.isfinite(sig).all()
    assert np.mean(np.abs(sig) ** 2) == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("cls", analog.CLASSES)
def test_same_seed_same_window(cls):
    a, _ = analog.generate_window(cls, 4, np.random.default_rng(7))
    b, _ = analog.generate_window(cls, 4, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cls", ["AM_DSB_WC", "AM_SSB_SC", "WBFM", "8ASK"])
@pytest.mark.parametrize("snr_db", [-10, 0, 20])
def test_noise_is_added_at_the_requested_snr(cls, snr_db):
    ratios = []
    for seed in range(20):
        sig, _ = clean(cls, seed)
        noisy, _ = analog.generate_window(cls, snr_db, np.random.default_rng(seed))
        noise = noisy.astype(complex) - sig
        ratios.append(np.mean(np.abs(sig) ** 2) / np.mean(np.abs(noise) ** 2))
    assert 10 * np.log10(np.mean(ratios)) == pytest.approx(snr_db, abs=0.5)


def test_generate_matches_the_contract():
    iq, meta = analog.generate("4ASK", 3, 6, np.random.default_rng(0))
    assert iq.shape == (3, N) and iq.dtype == np.complex64
    assert (meta["sps"] >= 4).all() and np.isnan(meta["fm_hz"]).all()


def test_unknown_class_is_rejected():
    with pytest.raises(ValueError):
        clean("QPSK")


@pytest.mark.parametrize("cls", analog.ANGLE_CLASSES)
def test_angle_classes_have_constant_envelope(cls):
    sig, _ = clean(cls)
    assert np.ptp(np.abs(sig)) < 1e-3


@pytest.mark.parametrize("cls", ["AM_DSB_WC", "AM_DSB_SC", "OOK", "4ASK", "8ASK"])
def test_dsb_and_ask_stay_on_one_axis(cls):
    assert is_one_real_axis(clean(cls)[0])


@pytest.mark.parametrize("cls", ["AM_SSB_WC", "AM_SSB_SC"])
def test_ssb_leaves_the_axis(cls):
    assert not is_one_real_axis(clean(cls)[0])


@pytest.mark.parametrize("cls", ["AM_SSB_WC", "AM_SSB_SC"])
def test_ssb_keeps_only_one_sideband(cls):
    seen = set()
    for seed in range(20):
        upper, lower = sideband_powers(clean(cls, seed)[0])
        assert max(upper, lower) > 5 * min(upper, lower)
        seen.add(upper > lower)
    assert seen == {True, False}  # both sidebands turn up


def test_dsb_keeps_both_sidebands():
    upper, lower = sideband_powers(clean("AM_DSB_SC")[0])
    assert max(upper, lower) < 1.5 * min(upper, lower)


def test_carrier_stays_in_wc_and_goes_in_sc():
    def carrier_share(cls):
        shares = [np.abs(np.mean(clean(cls, s)[0])) ** 2 for s in range(30)]
        return np.mean(shares)
    assert carrier_share("AM_DSB_WC") > 3 * carrier_share("AM_DSB_SC")
    assert carrier_share("AM_SSB_WC") > 3 * carrier_share("AM_SSB_SC")


def test_depth_sets_the_wc_envelope():
    """1 + a*m with |m| <= 1 stays between 1-a and 1+a, so depth 1.0 can touch
    zero but never overmodulates."""
    for seed in range(30):
        sig, p = clean("AM_DSB_WC", seed)
        env, a = np.abs(sig), p["index"]
        assert env.min() / env.max() >= (1 - a) / (1 + a) - 1e-4


def test_sc_envelope_reaches_zero_far_more_often_than_wc():
    def typical_min_env(cls):
        return np.median([np.min(np.abs(clean(cls, s)[0])) for s in range(30)])
    assert typical_min_env("AM_DSB_WC") > 5 * typical_min_env("AM_DSB_SC")


def test_wbfm_swings_further_than_nbfm():
    def swing(cls):
        return np.median([np.ptp(np.unwrap(np.angle(clean(cls, s)[0]))) for s in range(30)])
    assert swing("WBFM") > 3 * swing("NBFM")


@pytest.mark.parametrize("cls", ["WBFM", "NBFM"])
def test_fm_stays_inside_the_stated_deviation(cls):
    for seed in range(10):
        sig, p = clean(cls, seed)
        freq = np.diff(np.unwrap(np.angle(sig))) * analog.FS / (2 * np.pi)
        assert np.max(np.abs(freq)) <= 1.01 * p["index"] * p["fm_hz"]


def test_pm_phase_stays_inside_its_peak():
    """The window's whole phase swing fits in twice the stated peak."""
    for seed in range(10):
        sig, p = clean("PM", seed)
        assert np.ptp(np.unwrap(np.angle(sig))) <= 2 * p["index"] + 1e-3


def test_ask_levels_are_unipolar_so_a_carrier_remains():
    """Unipolar levels have a non-zero mean; bipolar PAM would average out."""
    for cls in analog.ASK_CLASSES:
        means = [np.abs(np.mean(clean(cls, s)[0])) for s in range(20)]
        assert np.mean(means) > 0.3


def test_message_band_is_where_it_says():
    fm = 20e3
    m, _ = analog._message(fm, np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / analog.FS)
    assert spec[f > 1.01 * fm].sum() < 1e-20 * spec.sum()
    assert np.mean(m) == pytest.approx(0, abs=1e-12)
    assert np.max(np.abs(m)) == pytest.approx(1)


@pytest.mark.parametrize("cls", analog.CLASSES)
def test_runs_through_the_impairment_module(cls):
    iq, info = impairments.generate_impaired(analog, cls, 6, np.random.default_rng(0))
    assert iq.shape == (N,) and np.isfinite(iq).all()
    assert np.isfinite(info["bandwidth_hz"])
