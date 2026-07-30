"""
Tests for the headline metrics that target the rules' scoring criteria.

These numbers go straight into the technical brief, so a silent error here is
worse than a training bug — it would be a false claim to the judges.
"""
import numpy as np
import pytest

from src.config import CLASSES, CLASS_TO_IDX
from src.evaluate import TIERS, coarse_tier_metrics, comms_vs_jamming, _tier_of

CIV = [CLASS_TO_IDX[c] for c in ("BPSK", "QPSK", "16QAM", "64QAM")]
RADAR = CLASS_TO_IDX["LFM_RADAR"]
FHSS = CLASS_TO_IDX["FHSS"]
JAM = CLASS_TO_IDX["JAMMING"]


class TestTiers:
    def test_every_class_belongs_to_exactly_one_tier(self):
        """A class in no tier raises; a class in two would silently skew metrics."""
        members = [c for group in TIERS.values() for c in group]
        assert sorted(members) == sorted(CLASSES)
        assert len(members) == len(set(members))

    def test_unknown_class_raises_rather_than_guessing(self):
        with pytest.raises(KeyError):
            _tier_of("NOT_A_CLASS")


class TestCoarseTier:
    def test_perfect_predictions(self):
        y = np.array(CIV + [RADAR, FHSS, JAM])
        m = coarse_tier_metrics(y, y.copy())
        assert m["accuracy"] == 1.0
        assert all(v == 1.0 for v in m["per_tier_recall"].values())

    def test_intra_tier_confusion_does_not_hurt_tier_accuracy(self):
        """16QAM mistaken for 64QAM is still 'civilian' — the operational call is
        right, and that is the whole point of reporting a coarse tier."""
        y_true = np.array([CLASS_TO_IDX["16QAM"], CLASS_TO_IDX["QPSK"]])
        y_pred = np.array([CLASS_TO_IDX["64QAM"], CLASS_TO_IDX["BPSK"]])
        assert coarse_tier_metrics(y_true, y_pred)["accuracy"] == 1.0

    def test_inter_tier_confusion_is_penalised(self):
        """Civilian mistaken for jamming is a false alarm and must show up."""
        y_true = np.array([CLASS_TO_IDX["QPSK"], CLASS_TO_IDX["QPSK"]])
        y_pred = np.array([JAM, CLASS_TO_IDX["BPSK"]])
        assert coarse_tier_metrics(y_true, y_pred)["accuracy"] == 0.5

    def test_radar_fhss_confusion_stays_within_military(self):
        y_true, y_pred = np.array([RADAR]), np.array([FHSS])
        assert coarse_tier_metrics(y_true, y_pred)["accuracy"] == 1.0


class TestCommsVsJamming:
    def test_ignores_military_classes_entirely(self):
        """The criterion is comms vs jamming. Radar/FHSS must not dilute it."""
        y_true = np.array([CLASS_TO_IDX["QPSK"], JAM, RADAR, FHSS])
        y_pred = np.array([CLASS_TO_IDX["QPSK"], JAM, FHSS, RADAR])
        r = comms_vs_jamming(y_true, y_pred)
        assert r["n_evaluated"] == 2
        assert r["accuracy"] == 1.0

    def test_civilian_confused_with_civilian_still_counts_as_correct(self):
        """This is a binary comms-vs-jamming call, so BPSK predicted as QPSK is
        still 'not jamming' and must not be scored as an error."""
        y_true = np.array([CLASS_TO_IDX["BPSK"]])
        y_pred = np.array([CLASS_TO_IDX["QPSK"]])
        assert comms_vs_jamming(y_true, y_pred)["accuracy"] == 1.0

    def test_missed_jamming_lowers_recall(self):
        y_true = np.array([JAM, JAM])
        y_pred = np.array([JAM, CLASS_TO_IDX["QPSK"]])
        assert comms_vs_jamming(y_true, y_pred)["jamming_recall"] == 0.5

    def test_false_alarm_rate(self):
        """Two civilians, one wrongly flagged as jamming -> 50%."""
        y_true = np.array([CLASS_TO_IDX["QPSK"], CLASS_TO_IDX["BPSK"], JAM])
        y_pred = np.array([JAM, CLASS_TO_IDX["BPSK"], JAM])
        assert comms_vs_jamming(y_true, y_pred)["false_alarm_rate"] == 0.5

    def test_returns_none_when_no_relevant_classes(self):
        y = np.array([RADAR, FHSS])
        assert comms_vs_jamming(y, y.copy()) is None
