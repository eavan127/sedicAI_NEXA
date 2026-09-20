"""
Tests for the pre-registered verdict in scripts/evaluate_experiment.py::judge().

The verdict decides whether a retrain "fixed FHSS without hurting JAMMING". It must be
impossible to pass by trading one for the other, and impossible to pass on a check whose
data was never supplied. These tests pin that behaviour with tiny hand-built results.
"""
import copy

import pytest

from scripts.evaluate_experiment import judge

CLASSES = ["BPSK", "QPSK", "16QAM", "64QAM", "LFM_RADAR", "FHSS", "JAMMING", "NOISE_FLOOR"]


def _ev(fhss_with_jammer=0.55, jam_recall=0.845, jam_precision=0.988, fhss_precision=0.54,
        fhss_recall=0.826, radar_recall=0.847, radar_precision=0.514, far=0.0007, cvj_acc=0.970,
        jam_alone=0.842):
    per_class = {c: {"precision": 0.5, "recall": 0.85, "f1": 0.6, "support": 100} for c in CLASSES}
    per_class["FHSS"].update(recall=fhss_recall, precision=fhss_precision)
    per_class["JAMMING"].update(recall=jam_recall, precision=jam_precision)
    per_class["LFM_RADAR"].update(recall=radar_recall, precision=radar_precision)
    ctx = {c: {b: {"recall": 0.8, "support": 10} for b in ("alone", "with_emitter", "with_jammer")}
           for c in CLASSES}
    ctx["FHSS"]["with_jammer"]["recall"] = fhss_with_jammer
    ctx["JAMMING"]["alone"]["recall"] = jam_alone
    ctx["JAMMING"]["with_jammer"]["recall"] = None
    return {"per_class": per_class, "recall_in_context": ctx,
            "comms_vs_jamming": {"accuracy": cvj_acc, "false_alarm_rate": far, "jamming_recall": jam_recall}}


def _high(both10=0.19, both6=0.28, class_only10=0.05, class_only6=0.03, standalone=99.5):
    bins = [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0]
    return {
        "snr_bins": bins,
        "recall": {"standalone": [77.7] + [standalone] * 5},          # -10 dB is allowed to be low
        "overlay_outcomes": {
            "6.0": {"both": 100 * both6, "jamming_only": 0, "class_only": 100 * class_only6, "neither": 0},
            "10.0": {"both": 100 * both10, "jamming_only": 0, "class_only": 100 * class_only10, "neither": 0},
        },
    }


def _jsr(fhss10=0.025, called=0.018):
    return {"rows": [{"jsr_db": 10, "fhss_recall": fhss10, "jammer_called_fhss": called}]}


def _status(checks):
    ran = [c for c in checks if c["passed"] is not None]
    return all(c["passed"] for c in ran)


def _by_name(checks, fragment):
    hits = [c for c in checks if fragment in c["name"]]
    assert hits, f"no check named like {fragment!r}"
    return hits[0]


def test_an_unchanged_model_passes_every_guardrail_but_fails_the_goal():
    base = _ev()
    checks = judge(copy.deepcopy(base), base)
    assert not _by_name(checks, "FHSS recall with a jammer")["passed"]          # no improvement
    guardrails = [c for c in checks if c["group"] != "FHSS goes up"]
    assert all(c["passed"] for c in guardrails)
    assert not _status(checks)


def test_a_real_win_passes():
    base = _ev()
    new = _ev(fhss_with_jammer=0.68)          # +13 points where FHSS is masked, everything else unchanged
    assert _status(judge(new, base))


def test_a_win_bought_by_hurting_jamming_recall_fails():
    base = _ev()
    new = _ev(fhss_with_jammer=0.70, jam_recall=0.80)      # 4.5 points down: too much
    checks = judge(new, base)
    assert not _by_name(checks, "JAMMING recall")["passed"]
    assert not _status(checks)


def test_jamming_recall_below_the_benchmark_fails_even_if_the_drop_is_small():
    base = _ev(jam_recall=0.815)
    new = _ev(fhss_with_jammer=0.70, jam_recall=0.795)     # only 2 points down, but under 80%
    assert not _by_name(judge(new, base), "JAMMING recall")["passed"]


def test_a_win_bought_by_false_jamming_alarms_fails():
    base = _ev()
    new = _ev(fhss_with_jammer=0.70, far=0.003)
    assert not _by_name(judge(new, base), "false alarm rate")["passed"]


def test_a_win_bought_by_lower_jamming_precision_fails():
    base = _ev()
    new = _ev(fhss_with_jammer=0.70, jam_precision=0.95)
    assert not _by_name(judge(new, base), "JAMMING precision")["passed"]


def test_a_win_bought_by_radar_or_fhss_dropping_under_the_benchmark_fails():
    base = _ev()
    assert not _status(judge(_ev(fhss_with_jammer=0.70, radar_recall=0.79), base))
    assert not _status(judge(_ev(fhss_with_jammer=0.70, fhss_recall=0.79), base))


def test_fhss_precision_collapsing_fails():
    base = _ev()
    new = _ev(fhss_with_jammer=0.70, fhss_precision=0.48)   # 6 points more false FHSS alarms
    assert not _by_name(judge(new, base), "FHSS precision")["passed"]


def test_probe_checks_are_skipped_when_no_probe_data_is_supplied():
    checks = judge(_ev(fhss_with_jammer=0.70), _ev())
    assert not any("BOTH" in c["name"] for c in checks)     # never added, never counted


def test_probe_both_column_must_reach_its_floor():
    base, hb = _ev(), _high()
    good = judge(_ev(fhss_with_jammer=0.70), base, _high(both10=0.50, both6=0.60), hb)
    assert _status(good)
    weak = judge(_ev(fhss_with_jammer=0.70), base, _high(both10=0.30, both6=0.60), hb)
    assert not _by_name(weak, "BOTH, +10 dB")["passed"]
    assert not _status(weak)


def test_fhss_rising_because_the_jammer_is_now_missed_fails():
    """The trap: 'FHSS only' growing means FHSS was reported and the jammer was not."""
    base, hb = _ev(), _high(class_only10=0.05, class_only6=0.03)
    new_high = _high(both10=0.50, both6=0.60, class_only10=0.15, class_only6=0.03)
    checks = judge(_ev(fhss_with_jammer=0.70), base, new_high, hb)
    assert not _by_name(checks, "jammer MISSED (FHSS reported without JAMMING), +10 dB")["passed"]
    assert not _status(checks)


def test_standalone_fhss_must_not_regress_at_useful_snr():
    base, hb = _ev(), _high()
    checks = judge(_ev(fhss_with_jammer=0.70), base, _high(both10=0.5, both6=0.6, standalone=93.0), hb)
    assert not _by_name(checks, "standalone FHSS recall")["passed"]


def test_jsr_probe_bar_and_the_jammer_called_fhss_guardrail():
    base = _ev()
    ok = judge(_ev(fhss_with_jammer=0.70), base, jsr_new=_jsr(fhss10=0.40, called=0.02), jsr_base=_jsr())
    assert _by_name(ok, "JSR +10 dB")["passed"] and _by_name(ok, "wrongly called FHSS")["passed"]
    leak = judge(_ev(fhss_with_jammer=0.70), base, jsr_new=_jsr(fhss10=0.40, called=0.12), jsr_base=_jsr())
    assert not _by_name(leak, "wrongly called FHSS")["passed"]
    below = judge(_ev(fhss_with_jammer=0.70), base, jsr_new=_jsr(fhss10=0.20, called=0.02), jsr_base=_jsr())
    assert not _by_name(below, "JSR +10 dB")["passed"]
