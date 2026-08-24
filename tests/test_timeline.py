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


from src.timeline import Detection, detections, tier_of_classes, tier_track

# Class order is CLASSES: BPSK QPSK 16QAM 64QAM LFM_RADAR FHSS JAMMING NOISE_FLOOR
FHSS_I, JAM_I, RADAR_I = 5, 6, 4
BPSK_I, NOISE_I = 0, 7
FLAT = {c: 0.5 for c in CLASSES}


def test_consecutive_same_class_windows_merge_into_one_event():
    probs = np.zeros((5, 8), dtype=np.float32)
    probs[1:4, FHSS_I] = 0.9
    events = detections(_result(probs), FLAT)
    assert len(events) == 1
    assert events[0].classes == ("FHSS",)
    assert events[0].start_window == 1 and events[0].end_window == 3


def test_two_classes_together_are_one_event_not_two():
    """A run where FHSS and JAMMING are both over threshold is ONE event
    labelled FHSS + JAMMING, not two overlapping single-class events."""
    probs = np.zeros((4, 8), dtype=np.float32)
    probs[:, FHSS_I] = 0.9
    probs[:, JAM_I] = 0.85
    events = detections(_result(probs), FLAT)
    assert len(events) == 1
    assert set(events[0].classes) == {"FHSS", "JAMMING"}


def test_change_in_detected_set_starts_a_new_event():
    probs = np.zeros((4, 8), dtype=np.float32)
    probs[:, FHSS_I] = 0.9
    probs[2:, JAM_I] = 0.9        # jammer joins halfway
    events = detections(_result(probs), FLAT)
    assert len(events) == 2
    assert events[0].classes == ("FHSS",)
    assert set(events[1].classes) == {"FHSS", "JAMMING"}


def test_windows_with_nothing_over_threshold_produce_no_event():
    probs = np.zeros((5, 8), dtype=np.float32)
    assert detections(_result(probs), FLAT) == []


def test_peak_confidence_is_per_class_maximum():
    probs = np.zeros((3, 8), dtype=np.float32)
    probs[:, FHSS_I] = [0.7, 0.94, 0.8]
    events = detections(_result(probs), FLAT)
    assert events[0].peak["FHSS"] == pytest.approx(0.94)


def test_event_times_span_first_window_start_to_last_window_end():
    probs = np.zeros((4, 8), dtype=np.float32)
    probs[1:3, FHSS_I] = 0.9
    r = _result(probs, hop=256, window_len=512, fs=3_200_000)
    e = detections(r, FLAT)[0]
    # starts at sample 256 -> 80 us; last window starts 512, ends 1024 -> 320 us
    assert e.start_us == pytest.approx(80.0)
    assert e.end_us == pytest.approx(320.0)
    assert e.duration_us == pytest.approx(240.0)


def test_per_class_thresholds_are_honoured():
    """LFM_RADAR at 0.30 must fire under a 0.26 threshold but not a 0.5 one."""
    probs = np.zeros((2, 8), dtype=np.float32)
    probs[:, RADAR_I] = 0.30
    lenient = dict(FLAT, LFM_RADAR=0.26)
    assert len(detections(_result(probs), lenient)) == 1
    assert detections(_result(probs), FLAT) == []


def test_event_count_is_events_not_windows():
    probs = np.zeros((40, 8), dtype=np.float32)
    probs[:, FHSS_I] = 0.9
    assert len(detections(_result(probs), FLAT)) == 1


def test_tier_track_returns_one_tier_per_window():
    assert len(tier_track(_result(np.zeros((4, 8), dtype=np.float32)), FLAT)) == 4


def test_hostile_outranks_military_and_civilian():
    """A jammer sitting on top of a civilian signal must colour the ribbon
    Hostile -- the worst thing present is what an operator needs to see."""
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, [BPSK_I, FHSS_I, JAM_I]] = 0.9
    assert tier_track(_result(probs), FLAT)[0] == "Hostile"


def test_military_outranks_civilian():
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, [BPSK_I, RADAR_I]] = 0.9
    assert tier_track(_result(probs), FLAT)[0] == "Military"


def test_nothing_over_threshold_is_empty_tier():
    probs = np.zeros((1, 8), dtype=np.float32)
    assert tier_track(_result(probs), FLAT)[0] == "Empty"


def test_noise_floor_is_empty_tier_not_a_threat():
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, NOISE_I] = 0.95
    assert tier_track(_result(probs), FLAT)[0] == "Empty"


def test_tier_of_classes_is_public_and_worst_first():
    assert tier_of_classes(("BPSK", "JAMMING")) == "Hostile"
    assert tier_of_classes(()) == "Empty"


from src.config import resolve_multilabel_thresholds
from src.measure import estimate_snr_db, noise_floor_power
from src.scenarios import build_scenario


def test_full_core_pipeline_runs_on_a_scenario(model):
    """Scenario -> windows -> model -> smoothing -> events, end to end.

    Uses an untrained model, so this asserts on plumbing and bookkeeping, not
    on whether the right class was found."""
    iq, segments = build_scenario(fs=3_200_000, total_duration=0.01, seed=0)

    result = classify_capture(iq, model, hop=256, fs=3_200_000)
    assert result.n_windows == 1 + (32_000 - 512) // 256

    thresholds = dict(zip(CLASSES, resolve_multilabel_thresholds()))
    events = detections(smooth(result, alpha=0.3), thresholds)
    for e in events:
        assert 0.0 <= e.start_us < e.end_us
        assert set(e.classes).issubset(set(CLASSES))

    assert len(tier_track(result, thresholds)) == result.n_windows

    snr = estimate_snr_db(iq[:512], noise_floor_power(iq))
    assert np.isfinite(snr)


def test_capture_is_never_normalized_by_the_pipeline(model):
    """Guard for the spec's normalization rule: classify_capture must not
    modify the caller's capture, and must not normalize it in place."""
    iq, _ = build_scenario(fs=3_200_000, total_duration=0.005, seed=1)
    before = iq.copy()
    classify_capture(iq, model, hop=512, fs=3_200_000)
    np.testing.assert_array_equal(iq, before)


from src.timeline import apply_hold, apply_noise_gate


def test_noise_gate_drops_weak_emitters_on_an_empty_window():
    """The calibrated thresholds are tuned on the dataset, where every window
    is an emitter or a labelled NOISE_FLOOR example. On a real quiet gap
    LFM_RADAR sits ~0.4, over its 0.26 threshold -- phantom radar on empty
    spectrum. The gate uses the dataset's own invariant that NOISE_FLOOR never
    co-occurs with anything."""
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, RADAR_I] = 0.45
    probs[0, NOISE_I] = 0.94
    thresholds = dict(FLAT, LFM_RADAR=0.26)
    r = _result(probs)
    assert set(detections(r, thresholds)[0].classes) == {"LFM_RADAR", "NOISE_FLOOR"}
    gated = detections(r, thresholds, noise_gate=0.5)
    assert gated[0].classes == ("NOISE_FLOOR",)


def test_noise_gate_leaves_a_genuinely_active_window_alone():
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, FHSS_I] = 0.98
    probs[0, NOISE_I] = 0.02
    assert detections(_result(probs), FLAT, noise_gate=0.5)[0].classes == ("FHSS",)


def test_hold_bridges_a_pulsed_emitters_gap():
    """max_duty_cycle 0.15 means most windows inside a radar's active period
    contain no pulse, fragmenting one emitter into dozens of events."""
    probs = np.zeros((7, 8), dtype=np.float32)
    probs[[0, 3, 6], FHSS_I] = 0.9        # present, gap, present, gap, present
    assert len(detections(_result(probs), FLAT)) == 3
    held = detections(_result(probs), FLAT, hold_us=1000.0)
    assert len(held) == 1
    assert held[0].start_window == 0 and held[0].end_window == 6


def test_hold_does_not_bridge_a_gap_longer_than_the_hold():
    probs = np.zeros((30, 8), dtype=np.float32)
    probs[[0, 29], FHSS_I] = 0.9
    assert len(detections(_result(probs), FLAT, hold_us=100.0)) == 2


def test_hold_is_per_class_and_does_not_chain_transitively():
    """Merging whole events whenever their class sets intersect chains
    radar->FHSS->jamming and collapses a capture into one event. Filling each
    class's own track cannot do that."""
    probs = np.zeros((3, 8), dtype=np.float32)
    probs[0, RADAR_I] = 0.9
    probs[1, FHSS_I] = 0.9
    probs[2, JAM_I] = 0.9
    events = detections(_result(probs), FLAT, hold_us=2000.0)
    assert len(events) > 1, "per-class hold must not merge disjoint emitters"


def test_noise_floor_is_never_held():
    """An empty channel is a state, not a pulsed emitter -- holding it would
    make it overlap a held emitter and produce a co-occurrence the dataset
    says cannot exist."""
    probs = np.zeros((5, 8), dtype=np.float32)
    probs[[0, 4], NOISE_I] = 0.95
    probs[1:4, FHSS_I] = 0.9
    events = detections(_result(probs), FLAT, noise_gate=0.5, hold_us=2000.0)
    for e in events:
        assert not ({"NOISE_FLOOR"} < set(e.classes)), \
            f"NOISE_FLOOR co-occurred with an emitter: {e.classes}"


def test_emitter_wins_over_noise_floor_after_hold():
    probs = np.zeros((3, 8), dtype=np.float32)
    probs[[0, 2], FHSS_I] = 0.9
    probs[1, NOISE_I] = 0.95
    events = detections(_result(probs), FLAT, noise_gate=0.5, hold_us=2000.0)
    assert len(events) == 1
    assert events[0].classes == ("FHSS",)


def test_display_rules_default_to_off():
    """detections() stays a plain primitive so the scorecard path is
    unaffected -- only the UI opts in."""
    probs = np.zeros((3, 8), dtype=np.float32)
    probs[[0, 2], FHSS_I] = 0.9
    assert len(detections(_result(probs), FLAT)) == 2


def test_tier_track_accepts_the_same_display_rules():
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, RADAR_I] = 0.45
    probs[0, NOISE_I] = 0.94
    thresholds = dict(FLAT, LFM_RADAR=0.26)
    assert tier_track(_result(probs), thresholds)[0] == "Military"
    assert tier_track(_result(probs), thresholds, noise_gate=0.5)[0] == "Empty"
