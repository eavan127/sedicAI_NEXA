"""
Config sanity tests.

Motivated by a real bug: YAML 1.1 parses `50.0e3` as the *string* "50.0e3",
not a float. Nothing complains until numpy raises deep inside generation — so
these assert the config is numerically usable before any of it reaches the GPU.
"""
import numbers

import pytest

from src.config import CFG, CLASSES, CLASS_TO_IDX

NUMERIC_RANGES = [
    ("radar", "pulse_width_s"), ("radar", "bandwidth_hz"), ("radar", "pri_s"),
    ("fhss", "hop_rate_hz"), ("fhss", "n_channels"), ("fhss", "channel_spacing_hz"),
    ("jamming", "jsr_db"), ("jamming", "sweep_bandwidth_hz"),
]


@pytest.mark.parametrize("section,key", NUMERIC_RANGES)
def test_ranges_are_numeric_and_ordered(section, key):
    lo, hi = CFG[section][key]
    assert isinstance(lo, numbers.Number), f"{section}.{key}[0] parsed as {type(lo)}"
    assert isinstance(hi, numbers.Number), f"{section}.{key}[1] parsed as {type(hi)}"
    assert lo < hi, f"{section}.{key} range is inverted"


@pytest.mark.parametrize("key", ["fs", "window_len", "total_duration"])
def test_signal_params_are_positive_numbers(key):
    val = CFG["signal"][key]
    assert isinstance(val, numbers.Number) and val > 0


def test_judged_classes_exist_in_class_list():
    """The benchmark is computed by name — a typo here means the scorecard
    silently reports on nothing."""
    for cls in CFG["judged_classes"]:
        assert cls in CLASS_TO_IDX, f"judged class {cls!r} is not in classes"


def test_class_list_has_no_duplicates():
    assert len(CLASSES) == len(set(CLASSES))


def test_splits_leave_room_for_training():
    d = CFG["dataset"]
    assert 0 < d["val_frac"] + d["test_frac"] < 1


def test_snr_bins_are_even_and_within_radioml_range():
    """RadioML 2018.01A samples SNR from -20 to +30 dB in 2 dB steps, so every
    available level is an even number.

    An odd bin returns zero civilian examples, leaving that bin populated only
    by radar/FHSS/jamming — which lets the model learn "odd SNR => threat class"
    instead of learning the signals. It would score well on our data and mean
    nothing on the organisers' stream.
    """
    for snr in CFG["snr_bins_db"]:
        assert snr % 2 == 0, f"SNR bin {snr} is odd — RadioML has no such level"
        assert -20 <= snr <= 30, f"SNR bin {snr} is outside RadioML's range"


def test_radar_pulse_fits_inside_generated_window():
    """A pulse longer than the example duration would be silently truncated."""
    assert max(CFG["radar"]["pulse_width_s"]) < CFG["signal"]["total_duration"]


def test_chirp_frequencies_respect_nyquist():
    """A chirp of bandwidth B sweeps -B/2..+B/2, so B/2 must stay under Nyquist.

    This caught a real bug: fs was 1 MHz while radar bandwidth reached 1 MHz,
    putting the sweep endpoints exactly on the Nyquist limit.
    """
    nyquist = CFG["signal"]["fs"] / 2
    assert max(CFG["radar"]["bandwidth_hz"]) / 2 < nyquist
    assert max(CFG["jamming"]["sweep_bandwidth_hz"]) / 2 < nyquist


def test_fhss_hops_are_visible_inside_one_window():
    """The model only ever sees one window. If the dwell time exceeds the window,
    every FHSS example is a single constant tone — indistinguishable from tone
    jamming, and teaching nothing about hopping.

    This caught a real bug: hop rates of 100-1000 Hz gave dwell times of 1-10 ms
    against a 512 us window, so no training example contained a single hop.
    """
    window_s = CFG["signal"]["window_len"] / CFG["signal"]["fs"]
    slowest_hop_rate = min(CFG["fhss"]["hop_rate_hz"])
    hops_in_window = window_s * slowest_hop_rate
    assert hops_in_window >= 3, (
        f"slowest hop rate {slowest_hop_rate} Hz gives only {hops_in_window:.2f} "
        f"hops per {window_s*1e6:.0f} us window — the class would be a constant tone"
    )


def test_radar_pulse_leaves_a_gap_inside_the_window():
    """A pulse filling the whole window hides the listening gap, which is the
    feature separating radar from a continuously-sweeping jammer."""
    window_s = CFG["signal"]["window_len"] / CFG["signal"]["fs"]
    assert max(CFG["radar"]["pulse_width_s"]) < window_s, (
        "longest radar pulse fills the entire window — no visible gap"
    )


def test_fhss_channel_comb_respects_nyquist():
    """Hop channels are laid out as (arange(n) - n/2) * spacing, so the comb
    spans about +/-(n * spacing / 2). If that exceeds Nyquist the outer hops
    alias and land on the wrong frequency entirely — training data that is
    labelled FHSS but is not the FHSS we think it is.

    This caught a real bug: 64 channels at 50 kHz spacing spanned +/-1.6 MHz
    against a 500 kHz Nyquist.
    """
    nyquist = CFG["signal"]["fs"] / 2
    max_n = max(CFG["fhss"]["n_channels"])
    max_spacing = max(CFG["fhss"]["channel_spacing_hz"])
    assert (max_n / 2) * max_spacing < nyquist
