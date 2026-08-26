import numpy as np
import pytest

from src.config import CFG, CLASSES
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


def test_a_scene_with_no_civilian_emitter_ignores_the_library_snr():
    """The double-noise correction must be invisible to every synthetic case."""
    from src.scenarios import CASES
    a, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=0, seed=7,
                           script=CASES["All three"])
    b, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=0, seed=7,
                           script=CASES["All three"], library_snr_db=10)
    assert np.allclose(a, b)


def test_a_civilian_emitter_is_not_noised_twice():
    """The library capture already carries noise at its labelled SNR. A full
    second helping on top makes a scene labelled +10 dB really +6.9 dB."""
    from src.scenarios import CASES, CIVILIAN
    rng = np.random.default_rng(0)
    script = CASES["Civilian only"]
    lib = {c: rng.normal(0, 1, (20, 2, 512)).astype(np.float32)
           for c, _, _ in script if c in CIVILIAN}
    twice, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=10,
                               seed=7, script=script, library=lib)
    once, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=10,
                              seed=7, script=script, library=lib,
                              library_snr_db=10)
    assert np.mean(np.abs(once) ** 2) < np.mean(np.abs(twice) ** 2)


def test_the_noise_floor_stays_uniform_across_a_civilian_capture():
    """Silence outside the emitter would break the NOISE_FLOOR class and the
    occupancy readout, so the empty stretches must carry the same floor as the
    occupied ones."""
    from src.scenarios import CASES, CIVILIAN
    rng = np.random.default_rng(0)
    script = CASES["Civilian only"]          # the emitter spans 25%-75%
    lib = {c: rng.normal(0, 1, (20, 2, 512)).astype(np.float32)
           for c, _, _ in script if c in CIVILIAN}
    iq, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=10,
                            seed=7, script=script, library=lib,
                            library_snr_db=10)
    quiet = np.mean(np.abs(iq[:2000]) ** 2)
    occupied = np.mean(np.abs(iq[16000:18000]) ** 2)
    assert quiet > 0
    assert quiet < occupied


# --- 10e: carried must come from each civilian span's OWN power ------------
#
# Both tests below fix mixture_sir_db to a single value (min == max) so the
# SIR scale rng.uniform draws is deterministic, and use a CONSTANT-amplitude
# library recording (not the Gaussian-noise stand-in the other tests use) so
# unit_power's normalisation lands on an exactly-known power every time.
# That makes the noise floor analytically predictable:
#
#   reference_power is always ~1.0 -- unit_power forces the FIRST non-jamming
#   emitter's OWN measured power to exactly 1.0, regardless of which class it
#   is (this holds for any continuous emitter; a pulsed one like LFM_RADAR
#   has a lower whole-window mean because unit_power normalises ACTIVE power,
#   which is why these fixtures use FHSS/QPSK rather than radar).
#
#   carried_i = span_i's own placed power / 10**(library_snr_db / 10)
#   floor = max(target, carried_1, carried_2, ...)
#
# With a forced SIR of +6 dB (power gain 10**0.6 = 3.981x) and
# snr_db == library_snr_db == 10, target = carried_for_the_UNSCALED_span =
# 0.1, but the SCALED civilian span's own carried is 0.398 -- four times
# larger, so it alone should set `floor`. The bug used reference_power
# (~1.0, or whatever emitter happens to be first) for EVERY span's carried,
# which reproduces 0.1 regardless of the scaled span's real power. That is
# not a subtle numerical difference -- it was verified directly against the
# pre-fix code (git stash the fix and re-run this scenario): the pre-fix
# quiet-region floor measured ~0.098, ~4x low, matching the 0.1 the buggy
# formula predicts; post-fix it measures ~0.39, matching the 0.398 the
# per-span formula predicts.
def _constant_library(cls, n=20):
    """A deterministic, non-random stand-in for a real dataset recording.
    Every sample has the same |amplitude|, so unit_power's normalisation
    lands on an EXACTLY known power (not a statistical estimate) -- needed
    so the expected noise floor below can be computed analytically rather
    than approximately."""
    return {cls: np.ones((n, 2, 512), dtype=np.float32)}


def test_carried_noise_uses_the_civilian_spans_own_power_not_the_first_emitter(monkeypatch):
    """Non-civilian emitter FIRST (sets reference_power), civilian SECOND and
    SIR-scaled well above the reference. The floor must reflect the
    civilian's OWN carried noise, not the unrelated first emitter's -- and
    that floor must be the SAME number everywhere outside an emitter span
    (before, between, and after), not just self-consistent within one
    implementation."""
    monkeypatch.setitem(CFG["dataset"], "mixture_sir_db", (6.0, 6.0))
    fs = 3_200_000
    lib = _constant_library("QPSK")
    script = [("FHSS", 0.05, 0.30), ("QPSK", 0.40, 0.65)]
    iq, segments = build_scenario(fs=fs, total_duration=0.02, snr_db=10,
                                   seed=5, script=script, library=lib,
                                   library_snr_db=10)
    n = len(iq)
    before = np.mean(np.abs(iq[:int(0.03 * n)]) ** 2)
    gap = np.mean(np.abs(iq[int(0.32 * n):int(0.38 * n)]) ** 2)
    after = np.mean(np.abs(iq[int(0.68 * n):]) ** 2)

    expected_floor = 10 ** 0.6 / 10  # QPSK's own carried noise, ~0.398
    wrong_floor = 1.0 / 10           # what reference_power (FHSS, ~1.0) predicts

    for label, measured in [("before", before), ("gap", gap), ("after", after)]:
        assert measured == pytest.approx(expected_floor, rel=0.25), (
            f"{label} region floor {measured:.4f} is not the civilian span's "
            f"own carried noise (~{expected_floor:.4f})"
        )
        assert abs(measured - wrong_floor) > abs(measured - expected_floor), (
            f"{label} region floor {measured:.4f} is closer to the "
            f"reference-emitter-derived (wrong) floor {wrong_floor:.4f} than "
            f"to the civilian span's own (correct) floor {expected_floor:.4f}"
        )
    assert before == pytest.approx(gap, rel=0.15)
    assert gap == pytest.approx(after, rel=0.15)


def test_carried_noise_is_correct_for_each_of_two_civilian_spans(monkeypatch):
    """Two civilian emitters at different positions -- the first sets
    reference_power (so its own carried noise happens to match the old
    single shared scalar), the second is SIR-scaled well above it. The old
    code applied ONE shared top-up scalar (derived from the first span) to
    BOTH spans; this asserts the floor everywhere -- including outside
    either civilian span -- reflects the correct per-span accounting, which
    for `floor = max(target, carried_1, carried_2)` is dominated by the
    second (higher-power) span's own carried noise."""
    monkeypatch.setitem(CFG["dataset"], "mixture_sir_db", (6.0, 6.0))
    fs = 3_200_000
    lib = _constant_library("QPSK")
    script = [("QPSK", 0.05, 0.30), ("QPSK", 0.40, 0.65)]
    iq, segments = build_scenario(fs=fs, total_duration=0.02, snr_db=10,
                                   seed=5, script=script, library=lib,
                                   library_snr_db=10)
    n = len(iq)
    before = np.mean(np.abs(iq[:int(0.03 * n)]) ** 2)
    gap = np.mean(np.abs(iq[int(0.32 * n):int(0.38 * n)]) ** 2)
    after = np.mean(np.abs(iq[int(0.68 * n):]) ** 2)

    expected_floor = 10 ** 0.6 / 10  # second span's own carried noise, ~0.398
    wrong_floor = 1.0 / 10           # what the shared scalar (first span) predicts

    for label, measured in [("before", before), ("gap", gap), ("after", after)]:
        assert measured == pytest.approx(expected_floor, rel=0.25), (
            f"{label} region floor {measured:.4f} does not reflect the "
            f"second civilian span's own carried noise (~{expected_floor:.4f})"
        )
        assert abs(measured - wrong_floor) > abs(measured - expected_floor)
    assert before == pytest.approx(gap, rel=0.15)
    assert gap == pytest.approx(after, rel=0.15)
