"""
Tests for the week-1 spread-spectrum generators in src/generators/spread.py.

As with test_digital.py and test_radar_week1.py, these check INTERNAL
CONSISTENCY -- hops land on the channel comb, m-sequences have their defining
properties, LoRa chirps sweep at BW^2 / 2^SF -- not realism.
"""
import numpy as np
import pytest
from scipy.signal import welch

from src.config import CLASS_GROUPS, DATASET_CLASSES
from src.generators import radar, spread

N = spread.WINDOW_LEN
FS = spread.FS


def inst_freq_hz(sig):
    return np.diff(np.unwrap(np.angle(sig))) / (2 * np.pi) * FS


# --- contract ----------------------------------------------------------------

def test_classes_match_the_spread_group():
    assert set(spread.CLASSES) <= set(DATASET_CLASSES)
    assert list(spread.CLASSES) == CLASS_GROUPS["SPREAD"]


@pytest.mark.parametrize("cls", spread.CLASSES)
@pytest.mark.parametrize("snr_db", [None, -10, 20])
def test_shape_dtype_finite(cls, snr_db):
    sig, params = spread.generate_window(cls, snr_db, np.random.default_rng(1))
    assert sig.shape == (N,) and sig.dtype == np.complex64
    assert np.isfinite(sig).all()
    assert set(params) == set(spread.PARAM_KEYS)


@pytest.mark.parametrize("cls", spread.CLASSES)
def test_unit_power_and_same_seed_same_window(cls):
    for seed in range(10):
        a, _ = spread.generate_window(cls, None, np.random.default_rng(seed))
        b, _ = spread.generate_window(cls, None, np.random.default_rng(seed))
        assert np.mean(np.abs(a) ** 2) == pytest.approx(1.0, rel=1e-3)
        assert (a == b).all()


@pytest.mark.parametrize("cls", spread.CLASSES)
def test_bandwidth_inside_the_limit(cls):
    for seed in range(50):
        _, params = spread.generate_window(cls, None, np.random.default_rng(seed))
        assert params["bandwidth_hz"] <= spread.MAX_BANDWIDTH_HZ * (1 + 1e-9)


def test_generate_returns_n_windows_and_meta():
    iq, meta = spread.generate("DSSS", 4, 6, np.random.default_rng(0))
    assert iq.shape == (4, N) and iq.dtype == np.complex64
    assert set(meta) == set(spread.PARAM_KEYS) and all(len(v) == 4 for v in meta.values())


def test_unknown_class_is_refused():
    with pytest.raises(ValueError):
        spread.generate_window("RADAR_CW", None, np.random.default_rng(0))


# --- FHSS ----------------------------------------------------------------------

def test_fhss_hops_between_comb_channels_several_times_per_window():
    for seed in range(40):
        sig, p = spread.generate_window("FHSS", None, np.random.default_rng(seed))
        spacing, n_ch = p["channel_spacing_hz"], int(p["n_channels"])
        f = inst_freq_hz(sig)
        idx = np.round(f / spacing + (n_ch - 1) / 2)
        near = np.abs(f - (idx - (n_ch - 1) / 2) * spacing) <= 0.3 * spacing + 1.0
        on_comb = near & (idx >= 0) & (idx <= n_ch - 1)
        # Hop boundaries re-tune the phase, which spikes the instantaneous
        # frequency for one sample; everything else sits on a channel.
        assert on_comb.mean() > 0.95
        held = [int(v) for v, ok in zip(idx, on_comb) if ok]
        changes = sum(1 for a, b in zip(held, held[1:]) if a != b)
        min_hops = int(FS / p["hop_rate_hz"])
        assert changes >= N // min_hops - 1, "too few hops in the window"


def test_fhss_never_stops():
    for seed in range(10):
        sig, _ = spread.generate_window("FHSS", None, np.random.default_rng(seed))
        assert np.abs(sig).min() > 0.99


# --- DSSS ----------------------------------------------------------------------

@pytest.mark.parametrize("degree", sorted(spread.M_SEQUENCE_TAPS))
def test_m_sequences_are_maximal_length(degree):
    seq = spread.m_sequence(degree)
    n = 2 ** degree - 1
    assert len(seq) == n
    assert seq.sum() == 2 ** (degree - 1), "an m-sequence has one more 1 than 0"
    chips = 1 - 2 * seq.astype(int)
    corr = [int(np.dot(chips, np.roll(chips, k))) for k in range(n)]
    assert corr[0] == n and all(c == -1 for c in corr[1:]), "periodic autocorrelation"


def test_m_sequence_refuses_an_all_zero_start():
    with pytest.raises(ValueError):
        spread.m_sequence(5, [0, 0, 0, 0, 0])


def test_dsss_spectrum_matches_its_chip_rate():
    for seed in range(20):
        sig, p = spread.generate_window("DSSS", None, np.random.default_rng(seed))
        f, psd = welch(sig, fs=FS, nperseg=128, return_onesided=False)
        order = np.argsort(f)
        f, psd = f[order], psd[order]
        cum = np.cumsum(psd) / psd.sum()
        width = f[np.searchsorted(cum, 0.995)] - f[np.searchsorted(cum, 0.005)]
        assert 0.6 * p["chip_rate_hz"] < width < 1.4 * p["bandwidth_hz"] + FS / 128


# --- LoRa ----------------------------------------------------------------------

def test_lora_settings_stay_short_and_apart_from_fmcw():
    for bw, sf in spread.LORA_SETTINGS:
        assert 2 ** sf / bw <= 512e-6 + 1e-12
        assert bw < radar.FMCW_BANDWIDTH_HZ[0]


@pytest.mark.parametrize("bw,sf", spread.LORA_SETTINGS)
def test_lora_frame_structure(bw, sf):
    rng = np.random.default_rng(0)
    freq = spread.lora_frame(bw, sf, rng, n_preamble=8, n_data=5)
    ts = 2 ** sf / bw
    assert len(freq) == pytest.approx((8 + 2 + 2.25 + 5) * ts * FS, abs=2)
    assert np.abs(freq).max() <= bw / 2 + 1e-6
    step = np.diff(freq)
    up_rate = bw / (ts * FS)                       # Hz per sample during an up-chirp
    preamble = step[: int(8 * ts * FS) - 2]
    smooth = preamble[np.abs(preamble) < bw / 2]   # skip the wrap at each symbol end
    assert np.allclose(smooth, up_rate, rtol=1e-6)
    n_down = np.sum(np.isclose(step, -up_rate, rtol=1e-6))
    assert n_down == pytest.approx(2.25 * ts * FS, abs=4)


def test_lora_events_find_wraps_and_turns():
    bw = 500e3
    up = np.linspace(-bw / 2, bw / 2, 100, endpoint=False)
    freq = np.concatenate([up, up, up[::-1]])
    ev = spread.lora_events(freq).tolist()
    assert 100 in ev                              # wrap
    assert any(195 <= e <= 205 for e in ev)       # up -> down


def _sweep(rate, n=spread.WINDOW_LEN, wrap_at=None):
    f = (np.arange(n) - n / 2) * rate
    if wrap_at is not None:
        f[wrap_at:] -= 400e3
    return f


def test_looks_like_lora_sweep():
    slopes = spread.lora_slopes()
    assert len(slopes) and slopes.max() < 1300          # only the long-symbol settings
    s = float(slopes[0])
    assert spread.looks_like_lora_sweep(_sweep(s))
    assert spread.looks_like_lora_sweep(-_sweep(s * 1.05))         # down, within 10%
    assert not spread.looks_like_lora_sweep(_sweep(s * 1.4)) or np.any(
        np.abs(s * 1.4 / slopes - 1) <= spread.LORA_SLOPE_TOL)     # off every LoRa rate
    assert not spread.looks_like_lora_sweep(_sweep(s, wrap_at=256))  # visible wrap
    assert spread.looks_like_lora_sweep(_sweep(s, wrap_at=10))       # wrap at the very edge


def test_lora_flag_matches_what_the_window_shows():
    """lora_like_sweep is 1 exactly when the window is a straight LoRa-rate
    sweep -- about a third of windows, as a receiver would cut them."""
    flags = []
    for seed in range(300):
        sig, p = spread.generate_window("LORA_CSS", None, np.random.default_rng(seed))
        assert p["lora_like_sweep"] in (0.0, 1.0)
        assert spread.looks_like_lora_sweep(inst_freq_hz(sig)) == bool(p["lora_like_sweep"]), seed
        flags.append(p["lora_like_sweep"])
    assert 0.15 < np.mean(flags) < 0.5
    _, p = spread.generate_window("FHSS", None, np.random.default_rng(0))
    assert np.isnan(p["lora_like_sweep"])


def test_some_jam_swept_windows_look_like_lora():
    from src.generators import jam
    flags = [jam.generate_window("JAM_SWEPT", None, np.random.default_rng(s))[1]["lora_like_sweep"]
             for s in range(300)]
    assert set(flags) <= {0.0, 1.0}
    assert 0.05 < np.mean(flags) < 0.5


def test_lora_window_stays_in_its_band():
    for seed in range(30):
        sig, p = spread.generate_window("LORA_CSS", None, np.random.default_rng(seed))
        f = inst_freq_hz(sig)
        assert np.abs(f).max() <= p["bandwidth_hz"] / 2 * 1.02
        assert np.abs(sig).min() > 0.99
