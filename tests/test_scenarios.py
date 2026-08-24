import numpy as np
import pytest

from src.config import CLASSES
from src.scenarios import ScenarioSegment, build_scenario, raised_cosine_ramp


def test_ramp_starts_and_ends_at_zero():
    env = raised_cosine_ramp(1000, ramp_len=100)
    assert env[0] == pytest.approx(0.0, abs=1e-9)
    assert env[-1] == pytest.approx(0.0, abs=1e-9)
    assert env[500] == pytest.approx(1.0)


def test_ramp_is_monotonic_through_the_rise():
    env = raised_cosine_ramp(1000, ramp_len=100)
    assert np.all(np.diff(env[:100]) >= -1e-12)


def test_ramp_handles_segment_shorter_than_two_ramps():
    env = raised_cosine_ramp(10, ramp_len=100)
    assert len(env) == 10
    assert np.all(np.isfinite(env))


def test_scenario_length_matches_requested_duration():
    iq, _ = build_scenario(fs=3_200_000, total_duration=0.01, seed=0)
    assert len(iq) == 32_000


def test_scenario_is_complex():
    iq, _ = build_scenario(fs=3_200_000, total_duration=0.005, seed=0)
    assert np.iscomplexobj(iq)


def test_scenario_returns_ground_truth_segments():
    _, segments = build_scenario(fs=3_200_000, total_duration=0.01, seed=0)
    assert len(segments) > 0
    assert all(isinstance(s, ScenarioSegment) for s in segments)
    assert all(s.end_s > s.start_s for s in segments)


def test_scenario_is_not_normalized():
    """The capture must keep real amplitude -- normalizing it would destroy
    the noise floor the SNR estimate and waterfall depend on."""
    iq, _ = build_scenario(fs=3_200_000, total_duration=0.01, seed=0)
    assert abs(np.std(np.abs(iq)) - 1.0) > 1e-6


def test_quiet_regions_are_quieter_than_active_regions():
    fs = 3_200_000
    iq, segments = build_scenario(fs=fs, total_duration=0.02, snr_db=10, seed=0)
    first = segments[0]
    active = np.mean(np.abs(iq[int(first.start_s * fs):int(first.end_s * fs)]) ** 2)
    gap = np.mean(np.abs(iq[:int(first.start_s * fs)]) ** 2)
    assert active > gap


def test_scenario_is_reproducible_for_a_seed():
    a, _ = build_scenario(fs=3_200_000, total_duration=0.005, seed=7)
    b, _ = build_scenario(fs=3_200_000, total_duration=0.005, seed=7)
    np.testing.assert_array_equal(a, b)


def test_segments_reference_real_class_names():
    _, segments = build_scenario(fs=3_200_000, total_duration=0.01, seed=0)
    for s in segments:
        assert s.class_name in CLASSES
