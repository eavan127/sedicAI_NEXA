"""
Tests for the headline metrics that target the rules' scoring criteria.

These numbers go straight into the technical brief, so a silent error here is
worse than a training bug — it would be a false claim to the judges.

coarse_tier_metrics/comms_vs_jamming now take multi-hot (N, len(CLASSES))
arrays instead of a scalar class index per example, since a window can
contain more than one class at once (a composite: jammer overlaid on a real
signal). The `mh()` helper below builds those arrays; pass a single int for a
standalone row, or a list of ints for a composite row.
"""
import numpy as np
import pytest

from src.config import CLASSES, CLASS_TO_IDX
from src.evaluate import TIERS, coarse_tier_metrics, comms_vs_jamming, _tier_of

CIV = [CLASS_TO_IDX[c] for c in ("BPSK", "QPSK", "16QAM", "64QAM")]
RADAR = CLASS_TO_IDX["LFM_RADAR"]
FHSS = CLASS_TO_IDX["FHSS"]
JAM = CLASS_TO_IDX["JAMMING"]
NOISE = CLASS_TO_IDX["NOISE_FLOOR"]


def mh(*rows):
    """Build a multi-hot (n_rows, len(CLASSES)) array. Each positional arg is
    one row's active class index (int, standalone) or indices (list, a
    composite row with more than one bit set)."""
    out = []
    for idxs in rows:
        idxs = [idxs] if isinstance(idxs, int) else list(idxs)
        v = np.zeros(len(CLASSES))
        v[idxs] = 1
        out.append(v)
    return np.array(out)


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
        y = mh(*CIV, RADAR, FHSS, JAM, NOISE)
        m = coarse_tier_metrics(y, y.copy())
        assert m["accuracy"] == 1.0
        assert all(v == 1.0 for v in m["per_tier_recall"].values())

    def test_intra_tier_confusion_does_not_hurt_tier_accuracy(self):
        """16QAM mistaken for 64QAM is still 'civilian' — the operational call is
        right, and that is the whole point of reporting a coarse tier."""
        y_true = mh(CLASS_TO_IDX["16QAM"], CLASS_TO_IDX["QPSK"])
        y_pred = mh(CLASS_TO_IDX["64QAM"], CLASS_TO_IDX["BPSK"])
        assert coarse_tier_metrics(y_true, y_pred)["accuracy"] == 1.0

    def test_inter_tier_confusion_is_penalised(self):
        """Civilian mistaken for jamming is a false alarm and must show up as
        a miss on Civilian recall specifically -- checking per-tier recall
        rather than the aggregate accuracy number, since "accuracy" is now a
        Hamming bit-accuracy across all 4 independent tier flags (not a
        single categorical match), and averages in the 3 tiers that were not
        confused here."""
        y_true = mh(CLASS_TO_IDX["QPSK"], CLASS_TO_IDX["QPSK"])
        y_pred = mh(JAM, CLASS_TO_IDX["BPSK"])
        assert coarse_tier_metrics(y_true, y_pred)["per_tier_recall"]["Civilian"] == 0.5

    def test_radar_fhss_confusion_stays_within_military(self):
        y_true, y_pred = mh(RADAR), mh(FHSS)
        assert coarse_tier_metrics(y_true, y_pred)["accuracy"] == 1.0

    def test_composite_window_reads_as_both_tiers_present(self):
        """A JAMMING-overlaid-on-LFM_RADAR window is BOTH Military and
        Hostile present at once -- neither tier should be forced to lose to
        the other the way a single predicted class used to force a choice."""
        y = mh([RADAR, JAM])
        m = coarse_tier_metrics(y, y.copy())
        assert m["per_tier_recall"]["Military"] == 1.0
        assert m["per_tier_recall"]["Hostile"] == 1.0

    def test_missing_one_component_of_a_composite_is_penalised(self):
        """Catching the jammer but missing the civilian signal underneath (or
        vice versa) must show up as a miss on the tier that was missed, not
        get averaged away by the tier that was caught."""
        y_true = mh([CLASS_TO_IDX["QPSK"], JAM])
        y_pred = mh(JAM)  # jammer caught, civilian signal underneath missed
        m = coarse_tier_metrics(y_true, y_pred)
        assert m["per_tier_recall"]["Hostile"] == 1.0
        assert m["per_tier_recall"]["Civilian"] == 0.0


class TestCommsVsJamming:
    def test_ignores_military_classes_entirely(self):
        """The criterion is comms vs jamming. Radar/FHSS must not dilute it."""
        y_true = mh(CLASS_TO_IDX["QPSK"], JAM, RADAR, FHSS)
        y_pred = mh(CLASS_TO_IDX["QPSK"], JAM, FHSS, RADAR)
        r = comms_vs_jamming(y_true, y_pred)
        assert r["n_evaluated"] == 2
        assert r["accuracy"] == 1.0

    def test_civilian_confused_with_civilian_still_counts_as_correct(self):
        """This is a binary comms-vs-jamming call, so BPSK predicted as QPSK is
        still 'not jamming' and must not be scored as an error."""
        y_true = mh(CLASS_TO_IDX["BPSK"])
        y_pred = mh(CLASS_TO_IDX["QPSK"])
        assert comms_vs_jamming(y_true, y_pred)["accuracy"] == 1.0

    def test_missed_jamming_lowers_recall(self):
        y_true = mh(JAM, JAM)
        y_pred = mh(JAM, CLASS_TO_IDX["QPSK"])
        assert comms_vs_jamming(y_true, y_pred)["jamming_recall"] == 0.5

    def test_false_alarm_rate(self):
        """Two civilians, one wrongly flagged as jamming -> 50%."""
        y_true = mh(CLASS_TO_IDX["QPSK"], CLASS_TO_IDX["BPSK"], JAM)
        y_pred = mh(JAM, CLASS_TO_IDX["BPSK"], JAM)
        assert comms_vs_jamming(y_true, y_pred)["false_alarm_rate"] == 0.5

    def test_returns_none_when_no_relevant_classes(self):
        y = mh(RADAR, FHSS)
        assert comms_vs_jamming(y, y.copy()) is None

    def test_civilian_jammed_on_top_counts_as_relevant_and_caught(self):
        """The exact composite scenario this metric exists to headline: a
        civilian signal WITH a jammer overlaid on top. Must be evaluated
        (civilian present -> relevant) and correctly count as jamming caught,
        instead of the old either/or framing which had no way to represent
        both being true on the same window."""
        y_true = mh([CLASS_TO_IDX["QPSK"], JAM])
        y_pred = mh([CLASS_TO_IDX["QPSK"], JAM])
        r = comms_vs_jamming(y_true, y_pred)
        assert r["n_evaluated"] == 1
        assert r["jamming_recall"] == 1.0

    def test_civilian_jammed_on_top_but_jammer_missed_is_a_miss(self):
        y_true = mh([CLASS_TO_IDX["QPSK"], JAM])
        y_pred = mh(CLASS_TO_IDX["QPSK"])  # civilian signal seen, jammer missed
        r = comms_vs_jamming(y_true, y_pred)
        assert r["jamming_recall"] == 0.0
