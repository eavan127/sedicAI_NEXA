import numpy as np
import pytest

from src.timeline import sliding_windows


def _ramp(n):
    """Complex IQ whose real part is a 0..n-1 ramp, so window contents are
    identifiable by their starting value."""
    return np.arange(n, dtype=np.float64) + 1j * np.zeros(n)


def test_exactly_one_window_when_input_is_exactly_window_len():
    windows, starts = sliding_windows(_ramp(512), window_len=512, hop=256)
    assert windows.shape == (1, 2, 512)
    assert starts.tolist() == [0]


def test_short_input_pads_to_one_window():
    windows, starts = sliding_windows(_ramp(100), window_len=512, hop=256)
    assert windows.shape == (1, 2, 512)
    assert starts.tolist() == [0]


def test_window_count_for_each_hop_setting():
    iq = _ramp(2048)
    # n = 1 + (len - window_len) // hop
    assert sliding_windows(iq, 512, 512)[0].shape[0] == 4
    assert sliding_windows(iq, 512, 256)[0].shape[0] == 7
    assert sliding_windows(iq, 512, 128)[0].shape[0] == 13
    assert sliding_windows(iq, 512, 64)[0].shape[0] == 25


def test_starts_advance_by_hop():
    _, starts = sliding_windows(_ramp(2048), window_len=512, hop=256)
    assert starts.tolist() == [0, 256, 512, 768, 1024, 1280, 1536]


def test_each_window_is_normalized_independently():
    """preprocess_window normalizes per window, so every window must come out
    with ~zero mean and ~unit std regardless of the amplitude of that slice."""
    iq = _ramp(2048) * 1000.0
    windows, _ = sliding_windows(iq, 512, 256)
    for w in windows:
        assert abs(w.mean()) < 1e-4
        assert abs(w.std() - 1.0) < 1e-3


def test_output_is_float32():
    windows, _ = sliding_windows(_ramp(1024), 512, 256)
    assert windows.dtype == np.float32


def test_rejects_non_positive_hop():
    with pytest.raises(ValueError):
        sliding_windows(_ramp(1024), 512, 0)
