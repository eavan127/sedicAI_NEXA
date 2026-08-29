"""
Tests for src/evaluate.py's scorecard metrics.

coarse_tier_metrics and comms_vs_jamming already have thorough coverage in
tests/test_metrics.py; this file adds the still-missing coverage for
confusion_between, plus full coverage for the new context-conditioned recall
metric (recall_in_context) that this change introduces.

recall_in_context exists because a single per-class recall number averages
over three very different situations -- a class predicted while it is the
only thing in the window, a class predicted alongside another comms/military
emitter, and a class predicted while a jammer is also present -- and that
average can make a class that is actually fine "alone" look like a uniform
mediocre performer, or hide a company-only collapse entirely. Splitting the
bucket is only useful if the split is trustworthy, so the tests here are
built so that a bug that pools the buckets (or mislabels one) changes the
answer, not just the presentation.
"""
import numpy as np
import pytest

from src.config import CLASSES, CLASS_TO_IDX
from src.evaluate import (coarse_tier_metrics, comms_vs_jamming,
                           confusion_between, dense_qam_recall,
                           recall_in_context)

BPSK = CLASS_TO_IDX["BPSK"]
QPSK = CLASS_TO_IDX["QPSK"]
QAM16 = CLASS_TO_IDX["16QAM"]
QAM64 = CLASS_TO_IDX["64QAM"]
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


class TestRecallInContext:
    def test_all_positives_standalone(self):
        """A class that only ever appears alone in the test split should have
        support in the "alone" bucket and empty (None-recall, zero-support)
        "with_emitter"/"with_jammer" buckets -- there is nothing to average
        in over company that never occurs."""
        y_true = mh(BPSK, BPSK, BPSK)
        y_pred = mh(BPSK, BPSK, BPSK)  # all correctly predicted
        out = recall_in_context(y_true, y_pred)

        bpsk = out["BPSK"]
        assert bpsk["alone"] == {"recall": 1.0, "support": 3}
        assert bpsk["with_emitter"] == {"recall": None, "support": 0}
        assert bpsk["with_jammer"] == {"recall": None, "support": 0}

    def test_mixed_buckets_are_not_pooled(self):
        """BPSK appears in all three contexts with a DIFFERENT recall in
        each: 2/2 alone, 1/2 with another emitter, 0/2 with a jammer. A
        pooling bug would report (2+1+0)/(2+2+2) = 0.5 for everything; the
        real per-bucket answers are 1.0, 0.5 and 0.0."""
        y_true = mh(
            BPSK, BPSK,                       # alone x2
            [BPSK, QAM16], [BPSK, QAM64],      # with another emitter x2
            [BPSK, JAM], [BPSK, JAM],          # with a jammer x2
        )
        y_pred = mh(
            BPSK, BPSK,                        # both alone caught
            [BPSK, QAM16], QAM64,              # one of two "with emitter" caught (BPSK bit missed on 2nd)
            JAM, JAM,                          # both "with jammer" BPSK bits missed
        )
        out = recall_in_context(y_true, y_pred)["BPSK"]

        assert out["alone"] == {"recall": 1.0, "support": 2}
        assert out["with_emitter"] == {"recall": 0.5, "support": 2}
        assert out["with_jammer"] == {"recall": 0.0, "support": 2}

    def test_jamming_has_no_with_jammer_bucket(self):
        """JAMMING can't be 'with a jammer' relative to itself -- that bucket
        must read None, not silently compute a recall against the class's
        own presence bit, and must NOT be counted against it under a
        different bucket either."""
        y_true = mh(JAM, [JAM, BPSK], [JAM, QAM16])
        y_pred = mh(JAM, [JAM, BPSK], JAM)  # 3rd row: JAM caught, QAM16 missed (irrelevant to JAM's own bit)
        out = recall_in_context(y_true, y_pred)["JAMMING"]

        assert out["with_jammer"] == {"recall": None, "support": 0}
        # JAMMING + another emitter, no other jammer to speak of -> with_emitter
        assert out["with_emitter"] == {"recall": 1.0, "support": 2}
        assert out["alone"] == {"recall": 1.0, "support": 1}

    def test_noise_floor_never_cooccurs_so_only_alone_has_support(self):
        y_true = mh(NOISE, NOISE, NOISE, NOISE)
        y_pred = mh(NOISE, NOISE, NOISE, BPSK)  # last one missed
        out = recall_in_context(y_true, y_pred)["NOISE_FLOOR"]

        assert out["alone"] == {"recall": 0.75, "support": 4}
        assert out["with_emitter"] == {"recall": None, "support": 0}
        assert out["with_jammer"] == {"recall": None, "support": 0}

    def test_support_counts_across_all_buckets_for_one_class(self):
        """QPSK: 1 alone, 2 with another emitter, 3 with a jammer -- support
        must reflect exactly that split, independent of recall."""
        y_true = mh(
            QPSK,
            [QPSK, BPSK], [QPSK, QAM16],
            [QPSK, JAM], [QPSK, JAM], [QPSK, JAM, FHSS],
        )
        y_pred = y_true.copy()  # perfect predictions, isolate support-counting
        out = recall_in_context(y_true, y_pred)["QPSK"]

        assert out["alone"]["support"] == 1
        assert out["with_emitter"]["support"] == 2
        assert out["with_jammer"]["support"] == 3
        assert out["alone"]["recall"] == 1.0
        assert out["with_emitter"]["recall"] == 1.0
        assert out["with_jammer"]["recall"] == 1.0

    def test_every_class_reported(self):
        """The judged-class blind spot this change fixes: every class gets a
        breakdown, not just the ones judged in the competition scorecard."""
        y_true = mh(BPSK, QPSK, QAM16, QAM64, RADAR, FHSS, JAM, NOISE)
        y_pred = y_true.copy()
        out = recall_in_context(y_true, y_pred)
        assert set(out.keys()) == set(CLASSES)


class TestConfusionBetween:
    def test_no_false_positives_returns_none(self):
        y_true = mh(RADAR, FHSS)
        y_pred = mh(RADAR, FHSS)  # perfect, no FPs
        assert confusion_between(y_true, y_pred, "LFM_RADAR", "FHSS") is None

    def test_all_false_positives_are_the_other_class(self):
        """Every LFM_RADAR false positive happens on a window where FHSS is
        truly present -- the model is substituting FHSS for LFM_RADAR, not
        just generally unsure."""
        y_true = mh(FHSS, FHSS, BPSK)
        y_pred = mh([FHSS, RADAR], [FHSS, RADAR], BPSK)  # FP on rows 0,1 only
        out = confusion_between(y_true, y_pred, "LFM_RADAR", "FHSS")
        assert out["false_positives"] == 2
        assert out["fraction_that_are_true_FHSS"] == 1.0

    def test_false_positives_spread_elsewhere_give_low_fraction(self):
        y_true = mh(BPSK, QAM16, FHSS)
        y_pred = mh(RADAR, RADAR, RADAR)  # 3 FPs, only 1 co-occurs with true FHSS
        out = confusion_between(y_true, y_pred, "LFM_RADAR", "FHSS")
        assert out["false_positives"] == 3
        assert out["fraction_that_are_true_FHSS"] == pytest.approx(1 / 3)


class TestCoarseTierMetricsSanity:
    """Light hand-checked coverage -- the deep behavioural cases already live
    in tests/test_metrics.py; these just pin the basic contract so a future
    refactor of evaluate.py can't silently drop it."""

    def test_perfect_predictions_are_perfect(self):
        y = mh(BPSK, RADAR, JAM, NOISE)
        out = coarse_tier_metrics(y, y.copy())
        assert out["accuracy"] == 1.0
        assert all(v == 1.0 for v in out["per_tier_recall"].values())

    def test_missed_hostile_tier_is_visible(self):
        y_true = mh(JAM)
        y_pred = mh(BPSK)  # jamming entirely missed
        out = coarse_tier_metrics(y_true, y_pred)
        assert out["per_tier_recall"]["Hostile"] == 0.0


class TestDenseQamRecall:
    def test_none_when_neither_class_present(self):
        y_true = mh(RADAR, FHSS, JAM)
        y_pred = mh(RADAR, FHSS, JAM)
        assert dense_qam_recall(y_true, y_pred) is None

    def test_per_class_split_is_wrong_but_combined_is_right(self):
        """The exact failure mode this metric exists to fix: the model
        always predicts the WRONG one of the two dense-QAM classes (16QAM
        truth predicted as 64QAM and vice versa), so BOTH per-class
        recalls read 0.0 -- yet on every single window, the model
        correctly flagged that dense QAM traffic was present. A metric
        that measures what it claims must read 1.0 here, not 0.0."""
        y_true = mh(QAM16, QAM16, QAM64, QAM64)
        y_pred = mh(QAM64, QAM64, QAM16, QAM16)  # always the other one

        recall_16 = float((y_pred[y_true[:, QAM16] == 1, QAM16] == 1).mean())
        recall_64 = float((y_pred[y_true[:, QAM64] == 1, QAM64] == 1).mean())
        assert recall_16 == 0.0
        assert recall_64 == 0.0

        out = dense_qam_recall(y_true, y_pred)
        assert out["recall"] == 1.0
        assert out["n_evaluated"] == 4

    def test_a_miss_of_both_classes_is_a_combined_miss(self):
        y_true = mh(QAM16, QAM64)
        y_pred = mh(RADAR, RADAR)  # neither dense-QAM bit predicted
        out = dense_qam_recall(y_true, y_pred)
        assert out["recall"] == 0.0
        assert out["n_evaluated"] == 2

    def test_support_counts_only_windows_where_a_dense_qam_class_is_true(self):
        y_true = mh(QAM16, RADAR, FHSS, QAM64)
        y_pred = mh(QAM16, QAM16, QAM16, QAM64)  # 3rd row: FP on non-dense-QAM row
        out = dense_qam_recall(y_true, y_pred)
        assert out["n_evaluated"] == 2  # rows 0 and 3 only
        assert out["recall"] == 1.0

    def test_composite_window_with_both_classes_counts_once(self):
        """A window truly carrying BOTH 16QAM and 64QAM (a composite
        example) is still ONE window for this metric -- catching either
        bit is a hit, not two separate opportunities to score."""
        y_true = mh([QAM16, QAM64])
        y_pred = mh(QAM16)  # only the 16QAM bit predicted
        out = dense_qam_recall(y_true, y_pred)
        assert out["n_evaluated"] == 1
        assert out["recall"] == 1.0


class TestCommsVsJammingSanity:
    def test_perfect_discrimination(self):
        y = mh(BPSK, JAM, QAM16)
        out = comms_vs_jamming(y, y.copy())
        assert out["accuracy"] == 1.0
        assert out["jamming_recall"] == 1.0
        assert out["false_alarm_rate"] == 0.0

    def test_missed_jamming_is_reflected_in_recall_not_accuracy_alone(self):
        y_true = mh(JAM, JAM, BPSK)
        y_pred = mh(BPSK, JAM, BPSK)  # first jamming window missed
        out = comms_vs_jamming(y_true, y_pred)
        assert out["jamming_recall"] == 0.5
