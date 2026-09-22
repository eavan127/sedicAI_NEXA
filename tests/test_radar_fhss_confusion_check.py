"""scripts/radar_fhss_confusion_check.py reads evaluate_experiment.py's JSON
output and reports the ONE thing stft_dwell_feature targets -- unlike
evaluate_experiment.py --verdict-only, whose 10 checks are pre-registered for
a different experiment (C2 / masking) and can PASS a run that does nothing
for radar-called-FHSS confusion."""
import json

import pytest

from scripts.radar_fhss_confusion_check import _confusion, summarize


def _fake_eval(radar_precision, radar_recall, fhss_precision, radar_called_fhss_frac,
              checkpoint="ckpt.pt"):
    return {
        "checkpoints": [checkpoint],
        "per_class": {
            "LFM_RADAR": {"precision": radar_precision, "recall": radar_recall},
            "FHSS": {"precision": fhss_precision, "recall": 0.8},
        },
        "radar_fhss_confusion": {
            "LFM_RADAR_fp_that_are_true_FHSS": {
                "false_positives": 1000,
                "fraction_that_are_true_FHSS": radar_called_fhss_frac,
            },
        },
    }


def test_confusion_extracts_the_right_fraction():
    ev = _fake_eval(0.5, 0.85, 0.55, 0.464)
    assert _confusion(ev) == pytest.approx(0.464)


def test_summarize_matches_baseline_exactly_when_run_equals_baseline():
    baseline = _fake_eval(0.514, 0.847, 0.543, 0.464)
    rows, base = summarize([baseline], baseline)
    assert rows[0]["lfm_radar_precision"] == base["lfm_radar_precision"]
    assert rows[0]["radar_called_fhss"] == base["radar_called_fhss"]


def test_summarize_reports_a_real_improvement():
    baseline = _fake_eval(0.514, 0.847, 0.543, 0.464)
    improved = _fake_eval(0.60, 0.85, 0.55, 0.30)
    rows, base = summarize([improved], baseline)
    assert rows[0]["lfm_radar_precision"] > base["lfm_radar_precision"]
    assert rows[0]["radar_called_fhss"] < base["radar_called_fhss"]


def test_summarize_reports_the_measured_dwell_seed2000_regression():
    # The actual numbers from the seed-2000 dwell run: precision down,
    # confusion up -- this is what the earlier misleading PASS verdict hid.
    baseline = _fake_eval(0.514, 0.847, 0.543, 0.4644607843137255)
    dwell_seed2000 = _fake_eval(0.502, 0.844, 0.548, 0.498)
    rows, base = summarize([dwell_seed2000], baseline)
    assert rows[0]["lfm_radar_precision"] < base["lfm_radar_precision"]
    assert rows[0]["radar_called_fhss"] > base["radar_called_fhss"]


def test_multiple_seeds_are_all_summarized():
    baseline = _fake_eval(0.514, 0.847, 0.543, 0.464)
    seed_a = _fake_eval(0.50, 0.84, 0.55, 0.50)
    seed_b = _fake_eval(0.52, 0.85, 0.54, 0.45)
    rows, base = summarize([seed_a, seed_b], baseline)
    assert len(rows) == 2
