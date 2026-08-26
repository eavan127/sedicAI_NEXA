"""Tests for the civilian constellation panel.

The DSP tests build their own QPSK rather than drawing from the dataset: a
known injected carrier offset and a known symbol timing are the only way to
assert that recovery found the RIGHT answer rather than merely a plausible
one.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.ui.plots import SAMPLES_PER_SYMBOL, carrier_offset, recover_symbols


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


def _concentration(points, order=4):
    """How tightly points cluster on a 4-fold symmetric constellation.

    1.0 means every point sits on the same 90-degree grid; 0.0 means the
    phases are spread uniformly, which is what a rotating or noise-dominated
    capture looks like.
    """
    return float(abs(np.mean(np.exp(1j * order * np.angle(points)))))


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
    are four clusters."""
    z = _qpsk()
    points, _, _ = recover_symbols(z)
    assert _concentration(points) > 0.9
    assert _concentration(z) < 0.5


def test_recovery_picks_the_symbol_timing_phase():
    """_qpsk places a symbol peak at sample 0, so phase 0 is the right answer
    and the other seven phases sample the pulse mid-transition."""
    _, _, phase = recover_symbols(_qpsk())
    assert phase == 0


def test_recovery_returns_one_point_per_symbol():
    points, _, _ = recover_symbols(_qpsk(n_symbols=64))
    assert len(points) == 64


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


from src.config import CFG, CLASSES, resolve_multilabel_thresholds
from src.timeline import TimelineResult
from src.ui.session import CaptureSession


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


def test_selector_picks_the_strongest_civilian_window():
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    index, cls, prob = s.best_civilian_window()
    assert (index, cls) == (3, "QPSK")
    assert prob == pytest.approx(0.95, abs=1e-6)


def test_selector_prefers_the_strongest_class_not_the_first():
    """CIVILIAN is iterated in class order, so a selector that returned the
    first class over threshold would answer BPSK here and be wrong."""
    s = _session({"BPSK": [0.40] * 6,
                   "16QAM": [0.10, 0.10, 0.99, 0.10, 0.10, 0.10]})
    s.display_smoothed = False
    index, cls, prob = s.best_civilian_window()
    assert (index, cls) == (2, "16QAM")
    assert prob == pytest.approx(0.99, abs=1e-6)


def test_selector_returns_none_when_no_civilian_clears_threshold():
    """A radar-only capture has no civilian window, and the panel must be
    hidden rather than showing the noise floor as a constellation."""
    s = _session({"LFM_RADAR": [0.90] * 6})
    s.display_smoothed = False
    assert s.best_civilian_window() is None


def test_selector_follows_the_sessions_display_mode():
    """Every page reads one view. Smoothing damps the spike but must not move
    the pick off the window that carries it."""
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = True
    index, cls, prob = s.best_civilian_window()
    assert (index, cls) == (3, "QPSK")
    assert prob < 0.95           # smoothed, so damped below the raw peak


def test_selector_handles_a_capture_with_no_windows():
    s = _session({"QPSK": [0.95] * 6})
    s.result.probs = s.result.probs[:0]
    s.result.starts = s.result.starts[:0]
    assert s.best_civilian_window() is None
