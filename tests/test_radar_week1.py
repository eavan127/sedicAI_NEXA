"""
Tests for the week-1 radar generators in src/generators/radar.py.

As with test_digital.py, these check INTERNAL CONSISTENCY -- that each class is
built the way radar.py says it is (code tables, chirp slope, hop order, whole
pulses in every window, duty cycle, power) -- not realism.
"""
import numpy as np
import pytest

from src.config import CFG, CLASS_GROUPS, DATASET_CLASSES
from src.generators import radar

N = radar.WINDOW_LEN
FS = radar.FS


def inst_freq(sig):
    """Instantaneous frequency in cycles per sample."""
    return np.diff(np.unwrap(np.angle(sig))) / (2 * np.pi)


def runs(mask):
    """(start, length) of every run of True in a boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return [(s, e - s) for s, e in zip(edges[::2], edges[1::2])]


def aperiodic_sidelobes(code):
    s = np.asarray(code, dtype=complex)
    n = len(s)
    return np.array([abs(np.vdot(s[:n - k], s[k:])) for k in range(1, n)])


def whole_pulse(sig, params):
    """The samples of one pulse that sits wholly inside the window."""
    pw = int(round(params["pulse_width_us"] * 1e-6 * FS))
    for start, length in runs(np.abs(sig) > 0):
        if length == pw and start > 0 and start + length < N:
            return sig[start:start + length]
    for start, length in runs(np.abs(sig) > 0):     # a pulse touching an edge
        if length == pw:
            return sig[start:start + length]
    raise AssertionError("no whole pulse in the window")


# --- contract ----------------------------------------------------------------

def test_classes_match_the_radar_group():
    assert set(radar.CLASSES) <= set(DATASET_CLASSES)
    assert list(radar.CLASSES) == CLASS_GROUPS["RADAR"]


@pytest.mark.parametrize("cls", radar.CLASSES)
@pytest.mark.parametrize("snr_db", [None, -10, 20])
def test_shape_dtype_finite(cls, snr_db):
    sig, params = radar.generate_window(cls, snr_db, np.random.default_rng(1))
    assert sig.shape == (N,) and sig.dtype == np.complex64
    assert np.isfinite(sig).all()
    assert set(params) == set(radar.PARAM_KEYS)


@pytest.mark.parametrize("cls", radar.CLASSES)
def test_same_seed_same_window(cls):
    a, pa = radar.generate_window(cls, 0, np.random.default_rng(42))
    b, pb = radar.generate_window(cls, 0, np.random.default_rng(42))
    assert (a == b).all()
    assert pa.keys() == pb.keys()


@pytest.mark.parametrize("cls", radar.CLASSES)
def test_unit_power_while_on(cls):
    for seed in range(20):
        sig, _ = radar.generate_window(cls, None, np.random.default_rng(seed))
        assert radar._active_power(sig) == pytest.approx(1.0, rel=1e-3)


def test_generate_returns_n_windows_and_meta():
    iq, meta = radar.generate("RADAR_FRANK", 5, 6, np.random.default_rng(0))
    assert iq.shape == (5, N) and iq.dtype == np.complex64
    assert set(meta) == set(radar.PARAM_KEYS)
    assert all(len(v) == 5 for v in meta.values())


def test_unknown_class_is_refused():
    with pytest.raises(ValueError):
        radar.generate_window("QPSK", None, np.random.default_rng(0))


@pytest.mark.parametrize("cls", radar.CLASSES)
def test_bandwidth_inside_the_limit(cls):
    for seed in range(50):
        _, params = radar.generate_window(cls, None, np.random.default_rng(seed))
        assert params["bandwidth_hz"] <= radar.MAX_BANDWIDTH_HZ * (1 + 1e-9)


# --- pulsed classes ------------------------------------------------------------

@pytest.mark.parametrize("cls", radar.PULSED_CLASSES)
def test_every_window_holds_a_whole_pulse_and_a_gap(cls):
    for seed in range(60):
        sig, params = radar.generate_window(cls, None, np.random.default_rng(seed))
        assert params["n_full_pulses"] >= 1
        whole_pulse(sig, params)
        assert (np.abs(sig) == 0).any(), "a pulsed window must show silence"
        assert params["duty"] <= CFG["radar"]["max_duty_cycle"] + 1e-9


@pytest.mark.parametrize("cls", radar.PULSED_CLASSES)
def test_pulse_position_varies(cls):
    starts = set()
    for seed in range(30):
        sig, _ = radar.generate_window(cls, None, np.random.default_rng(seed))
        starts.add(runs(np.abs(sig) > 0)[0][0])
    assert len(starts) > 10, "pulses always start in the same place"


def test_barker_codes_have_the_barker_property():
    for code in radar.BARKER_CODES.values():
        assert aperiodic_sidelobes(code).max() <= 1 + 1e-12


def test_polyphase_barker_codes_have_the_barker_property_and_are_not_binary():
    for n, phases in radar.POLY_BARKER_PHASES.items():
        assert len(phases) == n
        code = np.exp(1j * np.array(phases))
        assert aperiodic_sidelobes(code).max() <= 1 + 1e-4
        off_binary = np.minimum(np.abs(np.angle(code)), np.pi - np.abs(np.angle(code)))
        assert off_binary.max() > 0.1, f"length {n} is just a binary code"


def _phases_relative_to_first(sig):
    on = sig[np.abs(sig) > 0]
    return np.angle(on / on[0])


def test_barker_window_uses_only_two_phases():
    for seed in range(20):
        sig, _ = radar.generate_window("RADAR_BARKER", None, np.random.default_rng(seed))
        rel = _phases_relative_to_first(sig)
        assert np.minimum(np.abs(rel), np.pi - np.abs(rel)).max() < 1e-4


def test_frank_phases_step_by_a_constant_per_group():
    for m in radar.FRANK_M:
        ph = radar.frank_phases(m).reshape(m, m)
        for i in range(m):
            assert np.allclose(np.diff(ph[i]), 2 * np.pi * i / m)


def test_frank_window_phases_are_multiples_of_2pi_over_m():
    for seed in range(20):
        sig, params = radar.generate_window("RADAR_FRANK", None, np.random.default_rng(seed))
        m = int(round(np.sqrt(params["code_length"])))
        steps = _phases_relative_to_first(sig) / (2 * np.pi / m)
        assert np.abs(steps - np.round(steps)).max() < 1e-4


@pytest.mark.parametrize("prime", sorted(radar.COSTAS_PRIMES))
def test_costas_sequences_are_costas(prime):
    for shift in range(prime - 1):
        seq = radar.costas_sequence(prime, shift)
        assert sorted(seq) == list(range(1, prime))
        for lag in range(1, len(seq)):
            diffs = seq[lag:] - seq[:-lag]
            assert len(set(diffs.tolist())) == len(diffs), "displacements repeat"


def test_costas_pulse_hops_through_each_frequency_once():
    for seed in range(20):
        sig, params = radar.generate_window("RADAR_COSTAS", None, np.random.default_rng(seed))
        pulse = whole_pulse(sig, params)
        hop, n = int(params["chip_samples"]), int(params["code_length"])
        f = inst_freq(pulse)
        per_hop = [np.median(f[k * hop + 1:(k + 1) * hop - 1]) for k in range(n)]
        levels = np.round(np.array(per_hop) * hop + (n + 1) / 2).astype(int)
        assert sorted(levels.tolist()) == list(range(1, n + 1))


def test_lfm_sweeps_linearly_across_its_bandwidth():
    for seed in range(20):
        sig, params = radar.generate_window("LFM_RADAR", None, np.random.default_rng(seed))
        f = inst_freq(whole_pulse(sig, params))[4:-4]
        t = np.arange(len(f))
        slope, intercept = np.polyfit(t, f, 1)
        assert np.abs(f - (slope * t + intercept)).max() < 1e-3, "not a linear sweep"
        expected = params["bandwidth_hz"] / FS / (params["pulse_width_us"] * 1e-6 * FS)
        assert abs(slope) == pytest.approx(expected, rel=0.02)


def test_plain_pulse_has_no_modulation():
    for seed in range(20):
        sig, params = radar.generate_window("RADAR_PULSE", None, np.random.default_rng(seed))
        rel = _phases_relative_to_first(sig)
        assert np.abs(rel).max() < 1e-4


# --- continuous classes --------------------------------------------------------

@pytest.mark.parametrize("cls", radar.CONTINUOUS_CLASSES)
def test_continuous_classes_never_go_silent(cls):
    for seed in range(20):
        sig, _ = radar.generate_window(cls, None, np.random.default_rng(seed))
        assert np.abs(sig).min() > 0.99


def test_fmcw_covers_its_bandwidth():
    for seed in range(30):
        sig, params = radar.generate_window("RADAR_FMCW", None, np.random.default_rng(seed))
        f = inst_freq(sig) * FS
        assert f.max() - f.min() == pytest.approx(params["bandwidth_hz"], rel=0.05)
        assert np.abs(f).max() <= params["bandwidth_hz"] / 2 * 1.01


def test_cw_is_one_steady_tone_off_dc():
    for seed in range(20):
        sig, params = radar.generate_window("RADAR_CW", None, np.random.default_rng(seed))
        f = inst_freq(sig) * FS
        assert np.ptp(f) < 50.0     # complex64 rounding wobbles a few Hz; a sweep would be kHz
        assert abs(f.mean()) == pytest.approx(params["bandwidth_hz"], rel=1e-3)
        assert abs(f.mean()) >= radar.CW_OFFSET_HZ[0]
