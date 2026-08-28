"""Tests for the civilian constellation panel.

The DSP tests build their own QPSK rather than drawing from the dataset: a
known injected carrier offset and a known symbol timing are the only way to
assert that recovery found the RIGHT answer rather than merely a plausible
one.
"""
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.config import CFG, CLASSES, resolve_multilabel_thresholds
from src.scenarios import CIVILIAN
from src.timeline import TimelineResult
from src.ui.pages.rf_replay import _render
from src.ui.palette import INSTRUMENT, tier_color
from src.ui.plots import (CONSTELLATION_ORDER, RRC_SPAN_SYMBOLS,
                          SAMPLES_PER_SYMBOL, carrier_offset, cluster_score,
                          constellation_figure, recover_symbols, rrc_taps)
from src.ui.session import CaptureSession


def _qpsk(n_symbols=64, sps=SAMPLES_PER_SYMBOL, offset=0.0039, seed=0):
    """QPSK at `sps` samples/symbol with a triangular pulse shape.

    Pulse-shaped, not rectangular, on purpose: with a rectangular pulse every
    sample equals its symbol, so every timing phase is correct and the phase
    search cannot be tested at all. The triangular pulse peaks exactly on the
    symbol instant and mixes adjacent symbols everywhere else -- which is the
    condition the phase search exists to solve.

    The output is sliced so that sample 0 IS a symbol peak, i.e. the correct
    timing phase is 0.
    """
    rng = np.random.default_rng(seed)
    symbols = rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
                          n_symbols + 2) / np.sqrt(2)
    train = np.zeros(len(symbols) * sps, dtype=complex)
    train[::sps] = symbols
    pulse = np.concatenate([np.linspace(0, 1, sps, endpoint=False),
                             np.linspace(1, 0, sps)])
    shaped = np.convolve(train, pulse)[sps:sps + n_symbols * sps]
    return shaped * np.exp(2j * np.pi * offset * np.arange(len(shaped)))


# _kmeans and cluster_score (the k-means-based cluster-separation
# statistic formerly called cluster_score here) now live in
# src.ui.plots -- shipped code, not test-only -- so the panel can
# display the same number this suite verifies. Imported above.



def test_cluster_score_is_high_for_four_separated_blobs():
    """Four tight Gaussian blobs sitting exactly at the QPSK points -- the
    shape a clean, well-separated constellation actually has. Measured
    0.7826 under the centroid-relative null (see cluster_score's
    docstring for why the null is centred on the data's own centroid, not
    the origin); the floor below leaves >0.13 of margin."""
    rng = np.random.default_rng(3)
    centers = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    points = np.concatenate([
        c + 0.05 * (rng.normal(size=50) + 1j * rng.normal(size=50))
        for c in centers])
    assert cluster_score(points) > 0.65


def test_cluster_score_is_near_zero_for_a_single_blob():
    """This is the regression that motivated replacing cluster_score: one
    tight blob sitting off the origin, the exact shape of the jammer window
    that scored 0.90 on the old phase-concentration metric (see that
    function's docstring). Measured here with the OLD one-liner
    (`abs(mean(exp(1j*4*angle(points))))`) restored: 0.997 -- it thinks a
    single blob is essentially a perfect 4-fold constellation, because a
    single, consistent phase is indistinguishable from four points at a
    consistent 90-degree spacing once you throw away which of the four
    non-existent lobes each point is nearest to. Against the SAME points,
    the new metric measures 0.0, using the centroid-relative null (an
    earlier version of this null, phase-shuffled about the ORIGIN rather
    than the data's own centroid, turned this off-origin blob's
    near-constant radius-from-origin into a full ring and scored it 0.795
    -- see cluster_score's docstring)."""
    rng = np.random.default_rng(4)
    points = (0.2 - 2.7j) + 0.05 * (rng.normal(size=200)
                                     + 1j * rng.normal(size=200))
    assert cluster_score(points) < 0.15


def test_cluster_score_is_low_for_a_uniform_ring():
    """A capture with a residual, un-de-rotated carrier looks like a ring of
    phases at a roughly constant radius -- no preferred phase, but also no
    real multi-modal structure a 4-cluster hypothesis should credit.
    Measured 0.060 under the centroid-relative null -- a ring's centroid
    already sits near the origin, so centring changes nothing for this
    case, and shuffling its phase about that centroid reproduces another
    ring indistinguishable from the data (see cluster_score's docstring;
    an earlier Gaussian-fit null scored this same ring 0.361, because a
    Gaussian fitted to a ring's own mean/covariance is a filled disc, not
    a ring, and partitions less tightly than the ring itself does)."""
    rng = np.random.default_rng(5)
    points = np.exp(1j * rng.uniform(0, 2 * np.pi, 200))
    assert cluster_score(points) < 0.15


def test_cluster_score_is_scale_and_rotation_invariant():
    """The constellation's absolute amplitude and absolute phase are both
    arbitrary -- a receiver's AGC sets the first and a residual carrier
    offset sets the second -- so neither may move the score. Measured
    identical (0.7825587667709502 both times, to full float precision,
    since normalising by RMS magnitude removes scale exactly and both the
    k-means partition and the centroid-relative null are geometry-only)
    after scaling by 10x and rotating by an arbitrary 1.23 rad."""
    rng = np.random.default_rng(3)
    centers = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    points = np.concatenate([
        c + 0.05 * (rng.normal(size=50) + 1j * rng.normal(size=50))
        for c in centers])
    base = cluster_score(points)
    transformed = points * 10 * np.exp(1j * 1.23)
    assert cluster_score(transformed) == pytest.approx(base, abs=1e-9)


def test_carrier_offset_finds_the_injected_rotation():
    z = _qpsk(offset=0.0039)
    assert carrier_offset(z) == pytest.approx(0.0039, abs=0.001)


def test_carrier_offset_finds_a_negative_rotation():
    """The direction of a residual carrier error is arbitrary -- a real 16QAM
    capture measured -0.0098 cycles/sample -- and a negative offset lands the
    4th-power peak in the upper half of the FFT, which is a distinct code
    path (the wrap back to a negative bin) from the positive-offset case."""
    z = _qpsk(offset=-0.0039)
    assert carrier_offset(z) == pytest.approx(-0.0039, abs=0.001)


def test_recovered_points_cluster_where_the_raw_samples_do_not():
    """The whole reason the panel shows two axes: raw I/Q of an oversampled,
    rotating capture is a ring, and the same samples de-rotated and decimated
    are four clusters.

    Thresholds re-measured again after cluster_score's null changed to be
    centred on the data's own centroid rather than the origin or a
    Gaussian fit (see cluster_score's docstring for why). On this fixture
    the metric now measures 0.580 for the recovered points and 0.124 for
    the raw samples. The floor below (0.45) sits 0.13 under the measured
    recovered score; the raw-samples ceiling (0.3) sits 0.176 above the
    measured 0.124 -- comfortable margin on both sides.
    """
    z = _qpsk()
    points, _, _ = recover_symbols(z)
    assert cluster_score(points) > 0.45
    assert cluster_score(z) < 0.3


def test_recovery_picks_the_symbol_timing_phase():
    """_qpsk places a symbol peak at sample 0, so phase 0 is the right answer
    and the other seven phases sample the pulse mid-transition."""
    _, _, phase = recover_symbols(_qpsk())
    assert phase == 0


def test_recovery_returns_one_point_per_symbol_minus_the_edges_the_filter_cannot_reach():
    """64 symbols in, but recover_symbols now drops any symbol whose matched
    -filter support runs off the edge of the window into mode="same"'s
    implicit zero padding (see the recover_symbols docstring -- unfiltered,
    edge symbols measured 46% and 12% low). With the default 65-tap RRC
    filter (margin 32 samples = 4 symbols at sps=8) that trims 4 symbols off
    each edge of a 512-sample window: 64 - 4 - 4 = 56. This count was moved
    from 64 deliberately when the edge trim was added, not a regression."""
    points, _, _ = recover_symbols(_qpsk(n_symbols=64))
    assert len(points) == 56


def test_zero_power_window_returns_without_raising():
    """An all-zero window has no carrier to estimate and no power to normalise
    by. It must render as an empty scatter, not crash the page."""
    points, offset, phase = recover_symbols(np.zeros(512, dtype=complex))
    assert offset == 0.0
    assert phase == 0
    assert len(points) == 512


def test_window_shorter_than_one_symbol_is_returned_untouched():
    short = np.ones(4, dtype=complex)
    points, offset, phase = recover_symbols(short)
    assert np.allclose(points, short)
    assert offset == 0.0


def _session(probs_by_class, n_windows=6):
    """A CaptureSession with hand-set probabilities.

    The UI fixtures elsewhere in this suite run an UNTRAINED model, whose
    probabilities come from random weights. That is fine for asserting a page
    renders; it is useless for asserting WHICH window a selector picks.
    Setting probs directly makes the selection logic itself the thing under
    test.
    """
    window_len = hop = 512
    iq = np.concatenate([_qpsk(n_symbols=64, seed=i) for i in range(n_windows)])
    probs = np.full((n_windows, len(CLASSES)), 0.01, dtype=np.float32)
    for cls, column in probs_by_class.items():
        probs[:, CLASSES.index(cls)] = column
    result = TimelineResult(
        probs=probs, starts=np.arange(n_windows) * hop,
        attn=np.zeros((n_windows, window_len), dtype=np.float32),
        hop=hop, window_len=window_len, fs=CFG["signal"]["fs"])
    return CaptureSession(
        iq=iq, result=result, source="scenario", noise_power=0.01,
        thresholds=dict(zip(CLASSES, resolve_multilabel_thresholds())))


def test_civilian_windows_returns_empty_for_radar_only():
    """A radar-only capture has no civilian window, and the panel must be
    hidden rather than showing the noise floor as a constellation."""
    s = _session({"LFM_RADAR": [0.90] * 6})
    s.display_smoothed = False
    assert s.civilian_windows() == []


def test_civilian_windows_returns_count_evenly_spaced_including_first_and_last():
    """20 windows all clear QPSK's threshold; count defaults to 4. The
    selection is by POSITION in the qualifying list, not by probability --
    the probabilities here climb steadily so a confidence-ranked pick would
    have chosen the last four, not four spread across the whole span."""
    s = _session({"QPSK": list(np.linspace(0.30, 0.90, 20))}, n_windows=20)
    s.display_smoothed = False
    picks = s.civilian_windows()
    indices = [p[0] for p in picks]
    assert len(picks) == 4
    assert indices == sorted(indices)              # ascending time order
    assert indices[0] == 0                          # first qualifying window
    assert indices[-1] == 19                        # last qualifying window
    assert all(cls == "QPSK" for _, cls, _ in picks)


def test_civilian_windows_returns_all_qualifying_when_fewer_than_count():
    """Only three of six windows clear threshold; count=4 must not pad or
    repeat -- the panel shows exactly what qualified."""
    s = _session({"QPSK": [0.60, 0.10, 0.10, 0.60, 0.10, 0.70]})
    s.display_smoothed = False
    picks = s.civilian_windows()
    assert [p[0] for p in picks] == [0, 3, 5]
    assert [p[2] for p in picks] == pytest.approx([0.60, 0.60, 0.70], abs=1e-6)


def test_civilian_windows_returns_exactly_count_at_the_boundary():
    """5 qualifying windows, count=4 -- one more than count, the tightest
    case for the evenly-spaced positions to collide under rounding. Must
    still come back as 4 distinct, ascending indices, never fewer."""
    s = _session({"QPSK": [0.30, 0.40, 0.50, 0.60, 0.70]}, n_windows=5)
    s.display_smoothed = False
    picks = s.civilian_windows()
    indices = [p[0] for p in picks]
    assert len(indices) == 4
    assert len(set(indices)) == 4
    assert indices == sorted(indices)


def test_civilian_windows_picks_the_strongest_class_not_the_first():
    """CIVILIAN is iterated in class order, so a selector that returned the
    first class over threshold would answer BPSK here and be wrong."""
    s = _session({"BPSK": [0.40] * 6,
                   "16QAM": [0.10, 0.10, 0.99, 0.10, 0.10, 0.10]})
    s.display_smoothed = False
    picks = s.civilian_windows()
    assert picks == [(2, "16QAM", pytest.approx(0.99, abs=1e-6))]


def test_civilian_windows_does_not_select_by_cluster_quality():
    """Ten windows all clear threshold. Odd-indexed windows carry a high
    class probability (0.99) AND samples that recover to a tight cluster;
    even-indexed windows carry a low-but-qualifying probability (0.30) and
    pure noise that will not cluster at all. Neither "most confident" nor
    "cleanest looking" would choose [0, 3, 6, 9] -- only even spacing in time
    order does. This is the property Task 9 exists to enforce: no quality
    judgement anywhere in the selection.
    """
    rng = np.random.default_rng(2)
    window_len = hop = 512
    chunks = []
    for i in range(10):
        if i % 2 == 1:
            chunks.append(_qpsk(n_symbols=64, seed=i))       # clean, high-prob
        else:
            noise = (rng.normal(0, 1, window_len)
                      + 1j * rng.normal(0, 1, window_len))    # noisy, low-prob
            chunks.append(noise)
    iq = np.concatenate(chunks)
    probs = np.full((10, len(CLASSES)), 0.01, dtype=np.float32)
    probs[:, CLASSES.index("QPSK")] = [0.99 if i % 2 == 1 else 0.30
                                         for i in range(10)]
    result = TimelineResult(
        probs=probs, starts=np.arange(10) * hop,
        attn=np.zeros((10, window_len), dtype=np.float32),
        hop=hop, window_len=window_len, fs=CFG["signal"]["fs"])
    s = CaptureSession(
        iq=iq, result=result, source="scenario", noise_power=0.01,
        thresholds=dict(zip(CLASSES, resolve_multilabel_thresholds())))
    s.display_smoothed = False
    picks = s.civilian_windows()
    assert [p[0] for p in picks] == [0, 3, 6, 9]


def test_civilian_windows_follows_the_sessions_display_mode():
    """Every page reads one view. Smoothing damps the spike, which must show
    up in the probabilities civilian_windows returns."""
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    raw_peak = max(p for _, _, p in s.civilian_windows())
    assert raw_peak == pytest.approx(0.95, abs=1e-6)

    s.display_smoothed = True
    smoothed_peak = max(p for _, _, p in s.civilian_windows())
    assert smoothed_peak < raw_peak


def test_civilian_windows_handles_a_capture_with_no_windows():
    s = _session({"QPSK": [0.95] * 6})
    s.result.probs = s.result.probs[:0]
    s.result.starts = s.result.starts[:0]
    assert s.civilian_windows() == []


def test_figure_is_none_when_there_is_no_civilian_window():
    s = _session({"LFM_RADAR": [0.90] * 6})
    s.display_smoothed = False
    assert constellation_figure(s) is None


def test_figure_has_two_times_count_square_axes():
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        assert len(fig.axes) == 8            # 2 rows x 4 columns
        for ax in fig.axes:
            assert ax.get_aspect() == 1.0
    finally:
        plt.close(fig)


def test_top_row_plots_512_points_bottom_row_plots_56():
    """Top row is the exact (2, 512) array the model is fed; bottom row is
    one point per symbol, MINUS the edge symbols recover_symbols now drops
    because their matched-filter support runs off the window edge (56, not
    the naive 512 // 8 = 64 -- see recover_symbols' docstring). If the two
    rows ever plot the same count, decimation silently stopped happening."""
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        top_row, bottom_row = fig.axes[:4], fig.axes[4:]
        for ax in top_row:
            assert ax.collections[0].get_offsets().shape[0] == 512
        for ax in bottom_row:
            assert ax.collections[0].get_offsets().shape[0] == 56
    finally:
        plt.close(fig)


def test_scatter_points_carry_measured_styling_not_a_tier_colour():
    """Provenance rule: every panel is computed from the capture's own
    samples, so none of them may wear the colour that marks model output."""
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        expected = matplotlib.colors.to_rgb(INSTRUMENT["color"])
        for ax in fig.axes:
            colour = ax.collections[0].get_facecolor()[0]
            assert np.allclose(colour[:3], expected)
    finally:
        plt.close(fig)


def test_per_column_class_probability_carries_tier_colour():
    """The one MODEL element on the panel is each column's class probability
    -- everything else, including the window index and time, is MEASURED."""
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        top_row = fig.axes[:4]
        for ax in top_row:
            prob_texts = [t for t in ax.texts if "QPSK" in t.get_text()]
            assert prob_texts
            assert all(t.get_color() == tier_color("Civilian")
                        for t in prob_texts)
    finally:
        plt.close(fig)


def test_captions_name_class_chain_caveat_and_selection_rule():
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        captions = " ".join(t.get_text() for t in fig.texts)
        assert "QPSK" in captions
        assert "matched filter" in captions
        assert "de-rotate" in captions
        assert "64QAM" in captions                  # cluster-count caveat
        assert "spaced evenly" in captions           # selection rule
        assert "seam" in captions                    # splice caveat
        # The cluster-count caveat interpolates the per-window symbol count.
        # It must quote what recover_symbols ACTUALLY returns (56, after the
        # 10d edge trim) and not the naive window_len // sps (64) -- the top
        # -row/bottom-row test above pins that 56 is what every column
        # actually plots.
        assert "56 symbols" in captions
        assert "64 symbols" not in captions
    finally:
        plt.close(fig)


def test_figure_does_not_claim_recovery_on_a_window_with_no_power():
    """A near-silent window that the model still classified as civilian above
    threshold is not hypothetical for this project -- confident classification
    on near-empty signal has bitten this console before. If a column labels
    512 raw samples "64 symbol points" it is not merely wrong, it is lying
    about the one thing this display exists to prove: that the clusters it
    shows came from real recovery. The zeroed window must still be one of the
    four shown -- degenerate windows are not filtered out of the spread, they
    are shown honestly as having nothing to recover."""
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    picks = s.civilian_windows()
    zeroed_index = picks[1][0]
    s.iq = s.iq.copy()
    s.iq[zeroed_index * 512:(zeroed_index + 1) * 512] = 0
    fig = constellation_figure(s)
    try:
        titles = [ax.get_title() for ax in fig.axes]
        no_power_titles = [t for t in titles if "no power" in t]
        symbol_titles = [t for t in titles if "symbol points" in t]
        assert len(no_power_titles) == 1
        assert len(symbol_titles) == len(picks) - 1
    finally:
        plt.close(fig)


def test_render_returns_a_visible_constellation_for_a_civilian_capture():
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    out = _render(s, "Raw", "single")
    try:
        assert len(out) == 5
        update = out[4]
        assert update["visible"] is True
        assert update["value"] is not None
    finally:
        plt.close("all")


def test_render_hides_the_constellation_when_no_civilian_is_present():
    """A radar-only capture must look exactly as it did before this panel
    existed -- no empty grey box below the console."""
    s = _session({"LFM_RADAR": [0.90] * 6})
    out = _render(s, "Raw", "single")
    try:
        assert out[4]["visible"] is False
    finally:
        plt.close("all")


def _rrc_qpsk(n_symbols=64, sps=SAMPLES_PER_SYMBOL, snr_db=3.0, seed=1):
    """RRC-shaped QPSK with AWGN -- the shape a real receiver actually sees.

    The triangular pulse in _qpsk exists to test the timing search; it is the
    wrong shape for testing a matched filter, which is matched to the RRC the
    transmitter used.
    """
    rng = np.random.default_rng(seed)
    symbols = rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
                          n_symbols + 8) / np.sqrt(2)
    train = np.zeros(len(symbols) * sps, dtype=complex)
    train[::sps] = symbols
    taps = rrc_taps(sps)
    shaped = np.convolve(train, taps)[len(taps) // 2:][:n_symbols * sps]
    shaped = shaped / np.sqrt(np.mean(np.abs(shaped) ** 2))
    noise = (rng.normal(0, 1, len(shaped))
              + 1j * rng.normal(0, 1, len(shaped))) * np.sqrt(
                  10 ** (-snr_db / 10) / 2)
    return shaped + noise


def test_rrc_taps_have_unit_energy_and_odd_length():
    taps = rrc_taps(SAMPLES_PER_SYMBOL)
    assert len(taps) % 2 == 1          # symmetric, so the filter adds no delay
    assert float(np.sum(taps ** 2)) == pytest.approx(1.0, abs=1e-9)


def test_rrc_taps_rejects_a_span_and_sps_that_are_both_odd():
    """span * sps odd means an even-length filter with no centre tap -- the
    t=0 branch never fires and the "adds no delay" claim in the docstring
    silently stops being true. This must raise rather than produce that
    filter silently."""
    with pytest.raises(ValueError, match="span.*sps|sps.*span"):
        rrc_taps(sps=7, span=7)


def test_rrc_taps_are_symmetric_and_peak_at_the_centre_tap():
    """A real RRC is an even function of time. Symmetry alone does not pin
    the general branch's sign (a sign flip there is still even in t, see
    test_rrc_self_convolution_is_nyquist_zero_isi below for the test that
    catches that) -- but it is a real, independent property worth pinning."""
    taps = rrc_taps(SAMPLES_PER_SYMBOL)
    assert np.allclose(taps, taps[::-1])
    assert np.argmax(np.abs(taps)) == len(taps) // 2


@pytest.mark.parametrize("beta", [0.20, 0.35])
def test_rrc_self_convolution_is_nyquist_zero_isi(beta):
    """Pins rrc_taps against a property of a REAL root-raised cosine, not
    against this codebase's own pipeline output.

    _rrc_qpsk (below) shapes its fixture with rrc_taps itself, so every test
    built on it only proves the receive filter matches whatever rrc_taps
    happens to produce -- correct or not. The reviewer demonstrated this by
    flipping the sign of one term in the general branch
    (`- 4*beta*ti*cos(...)` instead of `+`): all four pre-existing RRC/matched
    -filter tests kept passing.

    An RRC convolved with itself is a raised cosine, and a raised cosine is
    the textbook Nyquist pulse: exactly zero at every non-zero integer
    multiple of the symbol period (in samples, `sps`). That property comes
    from the filter's definition, not from any code in this file, so it is
    what actually pins rrc_taps.

    Evaluated at span=32 -- NOT the shipped RRC_SPAN_SYMBOLS=8 -- and at
    both roll-offs the code's own comment treats as equally valid (0.20 and
    0.35). The residual being measured is finite-span truncation error, so
    it shrinks as the span grows; pinning it at the shipped span put the
    correct-case max (~0.0026) and a corrupted-case min (~0.0035) only ~15%
    apart on each side of the tolerance -- fragile enough that changing
    RRC_ROLLOFF or RRC_SPAN_SYMBOLS later could fail this test for a reason
    that has nothing to do with a broken filter. At span=32 the correct-case
    residual shrinks (truncation error falls with span) while a genuine sign
    error stays broken, so the gap widens instead of narrowing.

    Measured at span=32, k=1..15 (k=16 sits at the truncated edge and is
    excluded, same reasoning as before just at 4x the span):
        beta=0.20  correct max   0.000379   corrupted (sign-flipped) min   0.6476
        beta=0.35  correct max   0.000193   corrupted (sign-flipped) min   0.004736
    The tightest gap across both roll-offs is beta=0.20's correct max
    (0.000379) against beta=0.35's corrupted min (0.004736) -- still better
    than 12x apart. tolerance=0.001 sits with >2.5x margin below every
    correct-case value and >4.7x margin above every corrupted-case value
    measured above (beta=0.20's corrupted case is nowhere near the boundary
    at 0.6476, so it is beta=0.35's corrupted case, 0.004736, that actually
    sets the tight side of this margin). Verified: flipping the sign and
    running this test at both roll-offs fails both parametrizations (see the
    commit message for the captured failure output); the sign is restored in
    the shipped code.
    """
    sps, span = SAMPLES_PER_SYMBOL, 32
    taps = rrc_taps(sps, beta=beta, span=span)
    raised_cosine = np.convolve(taps, taps)
    center = len(raised_cosine) // 2
    peak = raised_cosine[center]
    assert peak == pytest.approx(1.0, abs=1e-9)

    # k=span//2 would land exactly on the truncated edge of the span, where
    # finite-length truncation error dominates -- not a meaningful check of
    # the Nyquist property. k=1..span//2-1 sit well inside the span.
    for k in range(1, span // 2):
        isi = raised_cosine[center + k * sps] / peak
        assert abs(isi) < 0.001, (
            f"raised cosine at symbol offset {k} (beta={beta}) is {isi!r}, "
            f"not ~0 -- rrc_taps is not producing a real root-raised cosine")


def test_matched_filter_tightens_clusters_a_raw_decimation_leaves_smeared():
    """The measurement that justifies this filter existing: on the same noisy
    samples, filtering before decimating pulls the constellation together."""
    z = _rrc_qpsk(snr_db=3.0)
    filtered, _, _ = recover_symbols(z)
    unfiltered = z / np.sqrt(np.mean(np.abs(z) ** 2))
    best_unfiltered = max(
        cluster_score(unfiltered[phase::SAMPLES_PER_SYMBOL])
        for phase in range(SAMPLES_PER_SYMBOL))
    # Re-measured again after cluster_score's null moved from a Gaussian
    # fit to a centroid-relative phase shuffle (see cluster_score's
    # docstring). On this fixture (snr_db=3.0, seed=1) the metric now
    # measures ~0.368 for the filtered path and ~0.168 for the best
    # unfiltered decimation phase -- a gap of ~0.20. The floor below (0.25)
    # sits ~0.12 under the measured 0.368. The relative-margin assertion
    # requires the filtered path to beat the best unfiltered phase by 0.10,
    # half the measured 0.20 gap, so reproducibility noise in the reference
    # sampling has real room to move without flipping the assertion.
    assert cluster_score(filtered) > 0.25
    assert cluster_score(filtered) > best_unfiltered + 0.10


def test_matched_filter_leaves_the_symbol_count_the_edge_trim_predicts():
    """Renamed from "does not change the symbol count": the edge-symbol trim
    (10d) means the matched filter DOES change the count relative to the
    naive window_len // sps, from 64 to 56 for this fixture. What this test
    still pins is that the count is exactly what the trim's own accounting
    predicts, not some other number."""
    points, _, _ = recover_symbols(_rrc_qpsk(n_symbols=64))
    assert len(points) == 56


def test_caption_names_the_matched_filter_step():
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        captions = " ".join(t.get_text() for t in fig.texts)
        assert "matched filter" in captions
    finally:
        plt.close(fig)


def test_constellation_order_covers_all_civilian_classes():
    """The score asks "are there `order` distinct clusters", so every
    civilian class must be scored at its own true constellation order --
    BPSK 2, QPSK 4, 16QAM 16, 64QAM 64 -- never a single default. Measured:
    asking order=4 of a genuinely 2-cluster BPSK set returns 0.67, a high
    score for the wrong question."""
    assert set(CONSTELLATION_ORDER) == set(CIVILIAN)
    assert CONSTELLATION_ORDER == {"BPSK": 2, "QPSK": 4, "16QAM": 16,
                                    "64QAM": 64}


def test_qpsk_column_bottom_title_contains_a_cluster_score():
    """56 recovered symbols at QPSK's order-4 is 14 per cluster, comfortably
    over the 8-per-cluster floor, so the title must carry a real number."""
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        bottom_titles = [ax.get_title() for ax in fig.axes[4:]]
        assert all(re.search(r"clusters \d\.\d\d", t) for t in bottom_titles)
    finally:
        plt.close(fig)


def test_16qam_column_says_too_few_symbols_and_prints_no_score():
    """56 symbols at 16QAM's order-16 is 3.5 per cluster -- under the
    8-per-cluster floor -- so the title must say why it cannot score rather
    than print a number dressed up as a measurement."""
    s = _session({"16QAM": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        bottom_titles = [ax.get_title() for ax in fig.axes[4:]]
        assert all("too few" in t for t in bottom_titles)
        assert all("16" in t for t in bottom_titles)   # names the order
        # No number that could be read as a cluster score (a 0.xx figure).
        assert not any(re.search(r"\d\.\d\d", t) for t in bottom_titles)
    finally:
        plt.close(fig)


def test_no_power_column_gets_neither_score_nor_too_few_note():
    """A degenerate window had nothing recovered, so its existing "recovery
    skipped" title stands unchanged -- it earns neither a score nor the
    too-few-symbols note, since both presuppose a real recovery happened."""
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    picks = s.civilian_windows()
    zeroed_index = picks[1][0]
    s.iq = s.iq.copy()
    s.iq[zeroed_index * 512:(zeroed_index + 1) * 512] = 0
    fig = constellation_figure(s)
    try:
        titles = [ax.get_title() for ax in fig.axes[4:]]
        no_power_titles = [t for t in titles if "no power" in t]
        assert len(no_power_titles) == 1
        assert not any("clusters" in t or "too few" in t
                        for t in no_power_titles)
    finally:
        plt.close(fig)


def test_caption_explains_the_cluster_score():
    """The panel must say, in plain language, what the number next to the
    symbol count is: a measured statistic computed from the samples --
    never from the classifier -- with 0 meaning no cluster structure."""
    s = _session({"QPSK": [0.60, 0.60, 0.60, 0.95, 0.60, 0.60]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        captions = " ".join(t.get_text() for t in fig.texts)
        assert "cluster" in captions.lower()
        assert "0" in captions and "no cluster structure" in captions
        assert "classifier" in captions.lower()
        # Existing caption lines must still be present.
        assert "matched filter" in captions
        assert "spaced evenly" in captions
        assert "seam" in captions
    finally:
        plt.close(fig)


def test_civilian_windows_does_not_call_cluster_score(monkeypatch):
    """Selection stays quality-blind and quality-uncomputed: civilian_windows
    must not import or call cluster_score, even indirectly. Monkeypatching
    it to raise and confirming selection is unchanged is the strongest
    available proof that no code path calls it."""
    import src.ui.plots as plots_mod
    import src.ui.session as session_mod

    assert not hasattr(session_mod, "cluster_score")

    def boom(*args, **kwargs):
        raise AssertionError("civilian_windows must not call cluster_score")

    monkeypatch.setattr(plots_mod, "cluster_score", boom)
    s = _session({"QPSK": list(np.linspace(0.30, 0.90, 20))}, n_windows=20)
    s.display_smoothed = False
    picks = s.civilian_windows()
    assert len(picks) == 4
    indices = [p[0] for p in picks]
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == 19
