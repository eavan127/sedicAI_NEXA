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


def test_per_emitter_snr_is_stable_as_emitters_are_added():
    """Adding a second emitter must not quietly raise the noise floor.

    Noise was previously scaled from the pooled power of every active sample,
    so overlapping emitters pushed the mean up and the noise with it -- making
    a two-emitter scenario at a given nominal SNR genuinely harder than a
    one-emitter scenario at the same nominal SNR. That invalidated any
    comparison across scenarios at fixed SNR.
    """
    fs = 3_200_000
    one, _ = build_scenario(fs=fs, total_duration=0.02, snr_db=0, seed=3,
                             script=[("FHSS", 0.2, 0.8)])
    two, _ = build_scenario(fs=fs, total_duration=0.02, snr_db=0, seed=3,
                             script=[("FHSS", 0.2, 0.8), ("JAMMING", 0.4, 0.9)])

    # Quiet head of the capture is noise only, in both cases.
    quiet = slice(0, int(0.15 * 0.02 * fs))
    n1 = float(np.mean(np.abs(one[quiet]) ** 2))
    n2 = float(np.mean(np.abs(two[quiet]) ** 2))
    assert n2 == pytest.approx(n1, rel=0.5), (
        f"noise floor moved when a second emitter was added: {n1:.4g} -> "
        f"{n2:.4g}; per-emitter SNR is not comparable across scenarios"
    )


def test_named_cases_cover_single_through_contested():
    from src.scenarios import CASES
    synthetic = {"Radar only", "FHSS only", "Jamming only",
                  "Radar + FHSS", "FHSS + Jamming", "All three"}
    assert synthetic <= set(CASES), "the generator-only cases must all exist"
    for name, script in CASES.items():
        assert script, f"{name} has an empty script"
        for cls, a, b in script:
            assert cls in CLASSES
            assert 0.0 <= a < b <= 1.0, f"{name}: bad span {a}-{b}"


def test_every_generator_case_builds_and_returns_matching_truth():
    """Cases built purely from generators. Civilian cases need a library of
    real captures and are covered separately."""
    from src.scenarios import CASES, GENERATORS
    for name, script in CASES.items():
        if any(c not in GENERATORS for c, _, _ in script):
            continue
        iq, segments = build_scenario(fs=3_200_000, total_duration=0.01,
                                       snr_db=0, seed=5, script=script)
        assert len(iq) == 32_000, name
        assert {s.class_name for s in segments} == {c for c, _, _ in script}, name


def test_civilian_case_requires_a_library_and_says_so():
    """Civilian classes have no generator. Failing loudly beats silently
    producing a scene with the emitter missing."""
    from src.scenarios import CASES, CIVILIAN
    civ_case = next(s for n, s in CASES.items()
                     if any(c in CIVILIAN for c, _, _ in s))
    with pytest.raises(ValueError, match="no generator"):
        build_scenario(fs=3_200_000, total_duration=0.01, seed=5,
                        script=civ_case)


def test_civilian_case_builds_when_a_library_is_supplied():
    from src.scenarios import CASES, CIVILIAN
    rng = np.random.default_rng(0)
    name, script = next((n, s) for n, s in CASES.items()
                         if any(c in CIVILIAN for c, _, _ in s))
    lib = {c: rng.normal(0, 1, (20, 2, 512)).astype(np.float32)
           for c, _, _ in script if c in CIVILIAN}
    iq, segments = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=0,
                                   seed=5, script=script, library=lib)
    assert len(iq) == 32_000
    assert {s.class_name for s in segments} == {c for c, _, _ in script}


def test_pulsed_emitter_reports_radiating_spans_not_just_schedule():
    """A pulsed radar is scheduled across its whole segment but transmits in
    a small fraction of it. Scoring a detector against the schedule counts
    every silent gap as a miss, which measures duty cycle rather than the
    model -- and does so identically at every SNR."""
    from src.scenarios import CASES
    _, truth = build_scenario(total_duration=0.03, snr_db=60, seed=33,
                               script=CASES["Radar only"])
    seg = truth[0]
    assert seg.radiating_spans, "pulsed emitter reported no radiating spans"
    assert seg.duty < 0.5, f"radar duty {seg.duty:.2f} — expected well under half"
    for a, b in seg.radiating_spans:
        assert seg.start_s <= a < b <= seg.end_s + 1e-9


def test_continuous_emitter_is_radiating_almost_throughout():
    from src.scenarios import CASES
    _, truth = build_scenario(total_duration=0.03, snr_db=60, seed=33,
                               script=CASES["FHSS only"])
    assert truth[0].duty > 0.8, "continuous emitter should radiate throughout"


def test_duty_defaults_to_one_when_spans_are_absent():
    seg = ScenarioSegment("FHSS", 0.0, 0.01)
    assert seg.duty == 1.0
