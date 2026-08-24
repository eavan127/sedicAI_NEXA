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


import torch

from src.config import CLASSES
from src.models.amc_cnn import AMC_CNN
from src.timeline import TimelineResult, classify_capture


@pytest.fixture(scope="module")
def model():
    """Untrained model. Predictions are meaningless, which is fine -- these
    tests check plumbing, shapes and bookkeeping, not accuracy."""
    m = AMC_CNN(num_classes=len(CLASSES), input_len=512)
    m.eval()
    return m


def test_classify_capture_shapes(model):
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096))
    result = classify_capture(iq, model, hop=256)
    assert isinstance(result, TimelineResult)
    assert result.probs.shape == (result.n_windows, len(CLASSES))
    assert result.starts.shape == (result.n_windows,)
    assert result.attn.shape == (result.n_windows, 512)
    assert result.hop == 256
    assert result.window_len == 512


def test_probabilities_are_independent_not_a_simplex(model):
    """Multi-label sigmoid: rows must NOT sum to 1. If they do, someone
    replaced sigmoid with softmax and co-occurrence is gone."""
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096))
    result = classify_capture(iq, model, hop=256)
    assert ((result.probs >= 0) & (result.probs <= 1)).all()
    assert not np.allclose(result.probs.sum(axis=1), 1.0)


def test_attention_rows_sum_to_one(model):
    """Attention is a per-window softmax over time -- each row is its own
    distribution, which is why heights are not comparable across windows."""
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096))
    result = classify_capture(iq, model, hop=256)
    np.testing.assert_allclose(result.attn.sum(axis=1), 1.0, rtol=1e-4)


def test_times_us_maps_windows_to_capture_position(model):
    iq = (np.random.randn(2048) + 1j * np.random.randn(2048))
    result = classify_capture(iq, model, hop=512, fs=3_200_000)
    # window 1 starts at sample 512 -> 512 / 3.2e6 s = 160 us
    assert result.times_us[0] == pytest.approx(0.0)
    assert result.times_us[1] == pytest.approx(160.0)


def test_batching_does_not_change_results(model):
    iq = (np.random.randn(8192) + 1j * np.random.randn(8192))
    a = classify_capture(iq, model, hop=256, batch_size=4)
    b = classify_capture(iq, model, hop=256, batch_size=4096)
    np.testing.assert_allclose(a.probs, b.probs, atol=1e-5)


def test_hook_is_removed_after_call(model):
    """A leaked forward hook would silently accumulate across calls."""
    before = len(model.attn_pool.score._forward_hooks)
    classify_capture(np.random.randn(1024) + 1j * np.random.randn(1024),
                     model, hop=512)
    assert len(model.attn_pool.score._forward_hooks) == before


from src.timeline import smooth


def _result(probs, hop=256, window_len=512, fs=3_200_000):
    probs = np.asarray(probs, dtype=np.float32)
    n = len(probs)
    return TimelineResult(
        probs=probs,
        starts=np.arange(n) * hop,
        attn=np.full((n, window_len), 1.0 / window_len, dtype=np.float32),
        hop=hop, window_len=window_len, fs=fs,
    )


def test_smoothing_suppresses_a_single_window_spike():
    """One noisy window must not survive as a detection."""
    probs = np.zeros((5, 8), dtype=np.float32)
    probs[:, 5] = 0.9              # FHSS steady
    probs[2, 6] = 0.95             # JAMMING one-window blip
    out = smooth(_result(probs), alpha=0.3)
    assert out.probs[2, 6] < 0.5
    assert out.probs[4, 5] > 0.5


def test_smoothing_preserves_co_occurrence():
    """Two classes true together must both survive -- this is what a majority
    vote over argmax would destroy."""
    probs = np.zeros((10, 8), dtype=np.float32)
    probs[:, 5] = 0.9              # FHSS
    probs[:, 6] = 0.85             # JAMMING, sustained
    out = smooth(_result(probs), alpha=0.3)
    assert out.probs[-1, 5] > 0.8
    assert out.probs[-1, 6] > 0.8


def test_smoothing_never_normalizes_rows():
    probs = np.full((6, 8), 0.9, dtype=np.float32)
    out = smooth(_result(probs), alpha=0.5)
    assert out.probs[-1].sum() > 1.0


def test_smoothing_leaves_first_window_untouched():
    probs = np.zeros((3, 8), dtype=np.float32)
    probs[0, 4] = 0.7
    out = smooth(_result(probs), alpha=0.3)
    assert out.probs[0, 4] == pytest.approx(0.7)


def test_smoothing_does_not_mutate_input():
    r = _result(np.full((4, 8), 0.5, dtype=np.float32))
    original = r.probs.copy()
    smooth(r, alpha=0.4)
    np.testing.assert_array_equal(r.probs, original)


def test_smoothing_preserves_other_fields():
    r = _result(np.full((4, 8), 0.5, dtype=np.float32))
    out = smooth(r, alpha=0.4)
    assert out.hop == r.hop and out.window_len == r.window_len
    np.testing.assert_array_equal(out.starts, r.starts)
