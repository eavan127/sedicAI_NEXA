"""
Tests for src/data/impairments.py: one test per impairment, each on a
purpose-built signal where the right answer is known (a tone, an impulse),
plus the apply() / generate_impaired() wiring.

These prove each impairment does what its docstring says. Whether the ranges
in PROFILE match a particular real receiver is a separate question.
"""
import numpy as np
import pytest

from src.data import impairments as imp
from src.generators import digital, radar, spread

N = imp.WINDOW_LEN
FS = imp.FS
t = np.arange(N) / FS


def tone(f_hz, n=N):
    return np.exp(2j * np.pi * f_hz * np.arange(n) / FS)


def phase_slope_hz(x):
    """Frequency of a single tone, from a straight-line fit to its phase."""
    ph = np.unwrap(np.angle(x))
    return np.polyfit(np.arange(len(x)), ph, 1)[0] / (2 * np.pi) * FS


def bin_power(x, f_hz):
    spec = np.abs(np.fft.fft(x)) ** 2
    return spec[int(round(f_hz / FS * len(x))) % len(x)]


# --- channel ------------------------------------------------------------------

def test_multipath_is_a_tapped_delay_line():
    impulse = np.zeros(N, dtype=complex)
    impulse[0] = 1
    taps, delays = np.array([0.8, 0.3j, -0.2]), np.array([0, 3, 7])
    out = imp.multipath(impulse, taps, delays)
    assert np.allclose(out[delays], taps)
    assert np.count_nonzero(np.abs(out) > 1e-12) == 3


@pytest.mark.parametrize("kind", ["rayleigh", "rician"])
def test_fading_keeps_unit_average_energy(kind):
    rng = np.random.default_rng(0)
    energy = [np.sum(np.abs(imp.draw_fading(rng, kind, 6.0, 3, 10)[0]) ** 2) for _ in range(4000)]
    assert np.mean(energy) == pytest.approx(1.0, rel=0.05)


def test_rician_fades_less_than_rayleigh():
    rng = np.random.default_rng(1)
    spread_of = {k: np.std([np.sum(np.abs(imp.draw_fading(rng, k, 10.0, 1, 10)[0]) ** 2)
                            for _ in range(3000)]) for k in ("rayleigh", "rician")}
    assert spread_of["rician"] < spread_of["rayleigh"]


def test_frequency_shift_moves_a_tone():
    assert phase_slope_hz(imp.frequency_shift(tone(0), 250e3)) == pytest.approx(250e3, abs=1)


def test_doppler_drift_ramps_the_frequency():
    rate = 2e9                                   # Hz/s, huge on purpose so 160 us shows it
    f = np.diff(np.unwrap(np.angle(imp.frequency_shift(tone(0), 0.0, rate)))) / (2 * np.pi) * FS
    slope = np.polyfit(t[:-1], f, 1)[0]
    assert slope == pytest.approx(rate, rel=1e-3)


def test_clock_error_scales_frequency_by_ppm():
    x = tone(500e3, 20000)                       # long, so a ppm-scale change is measurable
    shifted = imp.clock_error(x, 1000.0)
    assert phase_slope_hz(shifted[:-10]) == pytest.approx(500e3 * (1 + 1e-3), rel=1e-5)
    assert np.allclose(imp.clock_error(x, 0.0), x)


# --- noise --------------------------------------------------------------------

@pytest.mark.parametrize("ar", [0.0, 0.6, -0.6])
def test_noise_has_the_requested_power(ar):
    rng = np.random.default_rng(2)
    w = imp.noise(200_000, 0.25, rng, ar)
    assert np.mean(np.abs(w) ** 2) == pytest.approx(0.25, rel=0.03)


def test_coloured_noise_tilts_the_spectrum():
    rng = np.random.default_rng(3)
    w = imp.noise(1 << 16, 1.0, rng, ar=0.7)
    spec = np.abs(np.fft.fft(w)) ** 2
    f = np.abs(np.fft.fftfreq(len(w)))
    assert spec[f < 0.05].mean() > 3 * spec[f > 0.45].mean()


def test_impulses_are_sparse_and_loud():
    rng = np.random.default_rng(4)
    x = imp.impulses(200_000, 1.0, rng, rate=5e-3, level_db=20)
    hits = np.abs(x) > 0
    assert hits.mean() == pytest.approx(5e-3, rel=0.15)
    assert np.mean(np.abs(x[hits]) ** 2) == pytest.approx(100, rel=0.15)


def test_interference_sits_where_and_as_loud_as_asked():
    rng = np.random.default_rng(5)
    x = imp.interference(4096, 1.0, rng, level_db=-20, bandwidth_hz=100e3, centre_hz=600e3)
    assert 10 * np.log10(np.mean(np.abs(x) ** 2)) == pytest.approx(-20, abs=0.1)
    spec = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS))
    assert np.abs(f[np.argmax(spec)] - 600e3) < 100e3


# --- receiver -----------------------------------------------------------------

def test_phase_noise_is_a_random_walk_of_the_right_size():
    rng = np.random.default_rng(6)
    lw = 1e3
    out = imp.phase_noise(np.ones(200_000, dtype=complex), lw, rng)
    steps = np.diff(np.unwrap(np.angle(out)))
    assert np.std(steps) == pytest.approx(np.sqrt(2 * np.pi * lw / FS), rel=0.02)
    assert np.allclose(np.abs(out), 1)


@pytest.mark.parametrize("gain_db,phase_deg", [(1.0, 0.0), (0.0, 5.0), (0.7, 3.0)])
def test_iq_imbalance_creates_the_predicted_image(gain_db, phase_deg):
    f0 = 200 * FS / N                            # an exact FFT bin
    out = imp.iq_imbalance(tone(f0), gain_db, phase_deg)
    measured = 10 * np.log10(bin_power(out, -f0) / bin_power(out, f0))
    assert measured == pytest.approx(imp.image_rejection_db(gain_db, phase_deg), abs=0.05)


def test_iq_imbalance_off_changes_nothing():
    x = tone(123e3)
    assert np.allclose(imp.iq_imbalance(x, 0.0, 0.0), x)


def test_frontend_filter_passes_the_middle_and_cuts_the_edge():
    mid, edge = tone(100 * FS / N), tone(250 * FS / N)    # 625 kHz and 1.56 MHz
    assert np.mean(np.abs(imp.frontend_filter(mid, 0.8, 63)[50:-50]) ** 2) == pytest.approx(1, abs=0.05)
    assert np.mean(np.abs(imp.frontend_filter(edge, 0.8, 63)[50:-50]) ** 2) < 0.1


def test_agc_jump_steps_the_gain():
    out = imp.agc_jump(np.ones(N, dtype=complex), at=200, step_db=6.0)
    assert np.allclose(out[:200], 1)
    assert np.allclose(out[210:], 10 ** (6 / 20))


@pytest.mark.parametrize("bits", [8, 12])
def test_adc_quantises_to_the_requested_bits(bits):
    rng = np.random.default_rng(7)
    x = imp.noise(N, 1.0, rng)
    out = imp.adc(x, bits, backoff_db=12)
    assert len(np.unique(out.real)) <= 2 ** bits + 1
    step = 2 * np.sqrt(np.mean(np.abs(x) ** 2)) * 10 ** (12 / 20) / 2 ** bits
    assert np.abs(out - x).max() <= step / np.sqrt(2) * 1.01


def test_adc_clips_when_overdriven():
    x = tone(100e3) * 3
    full_scale = np.sqrt(np.mean(np.abs(x) ** 2)) * 10 ** (-3 / 20)
    out = imp.adc(x, 12, backoff_db=-3)
    assert np.abs(out.real).max() <= full_scale * (1 + 1e-9)
    assert np.isclose(np.abs(out.real), full_scale, rtol=1e-3).mean() > 0.1


def test_dc_offset_is_added_by_apply():
    prof = {k: dict(v, p=0.0) for k, v in imp.PROFILE.items()}
    prof["dc_offset"] = dict(imp.PROFILE["dc_offset"], p=1.0, level=(0.2, 0.2))
    x = tone(200 * FS / N)                       # whole cycles: zero mean
    out, info = imp.apply(x, None, np.random.default_rng(8), profile=prof)
    assert abs(np.mean(out)) == pytest.approx(0.2, rel=1e-3)
    assert info["dc_offset"] == pytest.approx(0.2)


# --- apply() / generate_impaired() ---------------------------------------------

def off_profile(**on):
    prof = {k: dict(v, p=0.0) for k, v in imp.PROFILE.items()}
    for k in on:
        prof[k] = dict(imp.PROFILE[k], p=1.0)
    return prof


def test_snr_is_set_against_the_clean_signal_while_on():
    clean, _ = radar.generate_window("RADAR_PULSE", None, np.random.default_rng(9))
    ratios = []
    for seed in range(200):
        out, _ = imp.apply(clean, 0.0, np.random.default_rng(seed), profile=off_profile())
        ratios.append(np.mean(np.abs(out - clean) ** 2))
    assert np.mean(ratios) == pytest.approx(imp.active_power(clean), rel=0.03)


def test_placement_keeps_the_signal_inside_the_band():
    for seed in range(200):
        out, info = imp.apply(tone(0), None, np.random.default_rng(seed),
                              bandwidth_hz=1.0e6, profile=off_profile(placement=True))
        assert abs(info["offset_hz"]) + 0.5e6 <= 0.95 * imp.NYQUIST + 1e-6
        assert phase_slope_hz(out) == pytest.approx(info["offset_hz"], abs=1)


def test_nothing_on_means_nothing_changes():
    x = tone(77e3)
    out, _ = imp.apply(x, None, np.random.default_rng(10), profile=off_profile())
    assert np.allclose(out, x.astype(np.complex64))


def test_skip_leaves_an_impairment_out():
    x = tone(0)
    out, info = imp.apply(x, None, np.random.default_rng(11), bandwidth_hz=1e5,
                          profile=off_profile(placement=True), skip=("placement",))
    assert info["offset_hz"] == 0.0 and np.allclose(out, x.astype(np.complex64))


def test_apply_is_reproducible_from_the_seed():
    x, _ = spread.generate_window("DSSS", None, np.random.default_rng(12))
    a, ia = imp.apply(x, 3.0, np.random.default_rng(99))
    b, ib = imp.apply(x, 3.0, np.random.default_rng(99))
    assert (a == b).all() and ia.keys() == ib.keys()


def test_occupied_bandwidth_of_a_known_signal():
    x, p = radar.generate_window("LFM_RADAR", None, np.random.default_rng(13))
    assert imp.occupied_bandwidth(x) == pytest.approx(p["bandwidth_hz"], rel=0.35)


@pytest.mark.parametrize("gen,cls", [(radar, c) for c in radar.CLASSES]
                         + [(spread, c) for c in spread.CLASSES]
                         + [(digital, "QPSK"), (digital, "GMSK")])
def test_generate_impaired_works_for_every_team_generator(gen, cls):
    for seed in range(5):
        iq, info = imp.generate_impaired(gen, cls, 4.0, np.random.default_rng(seed))
        assert iq.shape == (N,) and iq.dtype == np.complex64 and np.isfinite(iq).all()
        assert info["snr_db"] == 4.0
        assert abs(info["offset_hz"]) + info["bandwidth_hz"] / 2 <= 0.95 * imp.NYQUIST + 1e-6
        assert any(k.startswith("gen_") for k in info)
