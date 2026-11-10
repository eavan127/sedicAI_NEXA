"""
Join detector boxes into tracks: one track per emitter over the whole capture.

The detector sees one 10.24 ms chunk at a time, so an emitter that lasts
longer arrives as one box per chunk. Each new box is matched to an open track
by frequency overlap (IoU of the two frequency ranges), with
scipy.optimize.linear_sum_assignment picking the best one-to-one pairing; a
box nobody matches starts a new track, and a track not seen for max_gap_s is
closed.

Frequency alone is not enough: in a busy band a different emitter often comes
up in the same band while the first is still on or just after it stops, and
joining it made one track that matched neither (about half the "missed"
emitters of scripts/eval_pipeline.py). So a box only continues a track when
the classifier sees much the same thing in both: the overlap of their class
profiles (each the per-class probabilities normalised to sum to 1) must be at
least min_class_overlap. On 200 captures this found 79% of emitters instead
of 76% and named 42% right instead of 38%.

Along a track the classifier's window scores are averaged in log-odds, so the
noise of single windows cancels (an ensemble over time), and the behaviour
features of every box (src/pipeline/extract.py) are combined.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import CFG, DATASET_CLASSES

FS = CFG["signal"]["fs"]
P_CLIP = 1e-4


@dataclass
class Detection:
    """One box in one chunk, cut out and scored."""
    t0: int                     # samples, from the start of the capture
    t1: int
    f_lo: float                 # Hz
    f_hi: float
    score: float                # detector confidence
    probs: np.ndarray           # (k, 60) classifier probabilities, one row per window
    features: dict
    chunk: int = 0

    @property
    def width_hz(self):
        return self.f_hi - self.f_lo


def _profile(probs):
    """Per-class probabilities normalised to sum to 1."""
    p = np.asarray(probs, dtype=float)
    return p / max(p.sum(), 1e-12)


def class_overlap(p, q):
    """Shared mass of two class profiles: 1 = the same, 0 = nothing in common."""
    return float(np.minimum(_profile(p), _profile(q)).sum())


def freq_iou(a_lo, a_hi, b_lo, b_hi):
    inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = max(a_hi, b_hi) - min(a_lo, b_lo)
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    dets: list = field(default_factory=list)
    closed: bool = False

    def add(self, d):
        self.dets.append(d)

    @property
    def t_first(self):
        return min(d.t0 for d in self.dets)

    @property
    def t_last(self):
        return max(d.t1 for d in self.dets)

    @property
    def f_lo(self):
        return float(np.median([d.f_lo for d in self.dets]))

    @property
    def f_hi(self):
        return float(np.median([d.f_hi for d in self.dets]))

    @property
    def n_windows(self):
        return int(sum(len(d.probs) for d in self.dets))

    def probs(self):
        """Per-class probability of the whole track: the mean log-odds of all
        its windows, back through the sigmoid."""
        p = np.clip(np.concatenate([d.probs for d in self.dets]), P_CLIP, 1 - P_CLIP)
        logit = np.log(p / (1 - p)).mean(axis=0)
        return 1 / (1 + np.exp(-logit))

    def window_top1(self):
        """Each window's own answer (argmax), before any averaging."""
        return np.concatenate([d.probs.argmax(axis=1) for d in self.dets])

    def features(self):
        """Behaviour over the whole track."""
        def med(key):
            v = np.array([d.features.get(key, np.nan) for d in self.dets], dtype=float)
            return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
        return dict(
            duration_s=(self.t_last - self.t_first) / FS,
            n_boxes=len(self.dets),
            # A sweep shows its full span in any chunk that holds a whole
            # period, so the widest occupied band is the honest width.
            width_hz=float(max(d.features.get("occupied_hz", d.width_hz) for d in self.dets)),
            wraps=int(sum(d.features.get("wraps", 0) for d in self.dets)),
            sideband_db=med("sideband_db"),
            line_db=med("line_db"),
            snr_db=med("snr_db"),
            power_cv=med("power_cv"),
            envelope_m4=med("envelope_m4"),
        )


class Tracker:
    """Feed it the detections of each chunk in time order with update(); read
    every track (open and closed) from .tracks."""

    def __init__(self, max_gap_s=5e-3, min_iou=0.3, min_class_overlap=0.5):
        self.max_gap = int(max_gap_s * FS)
        self.min_iou = min_iou
        self.min_class_overlap = min_class_overlap      # None: match on frequency only (old behaviour)
        self.tracks = []

    def _link(self, d, tr):
        """How well box d continues track tr: frequency IoU, or 0 when the
        classifier sees something different in the two."""
        iou = freq_iou(d.f_lo, d.f_hi, tr.f_lo, tr.f_hi)
        if iou > 0 and self.min_class_overlap is not None:
            if class_overlap(Track(0, [d]).probs(), tr.probs()) < self.min_class_overlap:
                return 0.0
        return iou

    def _open(self, t):
        return [tr for tr in self.tracks if not tr.closed and t - tr.t_last <= self.max_gap]

    def update(self, dets):
        if not dets:
            return
        t_now = min(d.t0 for d in dets)
        for tr in self.tracks:
            if not tr.closed and t_now - tr.t_last > self.max_gap:
                tr.closed = True
        cands = self._open(t_now)
        matched = {}
        if cands:
            iou = np.array([[self._link(d, tr) for tr in cands] for d in dets])
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.min_iou:
                    matched[r] = cands[c]
        touched = list(matched.values())
        for i in sorted(range(len(dets)), key=lambda i: (i not in matched, -dets[i].width_hz)):
            d = dets[i]
            tr = matched.get(i)
            if tr is None:
                # The detector sometimes draws two boxes for one emitter in the
                # same chunk: join a box to a track this chunk already extended
                # or started, instead of starting a twin. (A narrow tone over a
                # wide signal overlaps it far less than min_iou, so it stays
                # its own track.)
                same = [t for t in touched if freq_iou(d.f_lo, d.f_hi, t.f_lo, t.f_hi) >= self.min_iou]
                tr = max(same, key=lambda t: freq_iou(d.f_lo, d.f_hi, t.f_lo, t.f_hi)) if same else None
            if tr is None:
                tr = Track(id=len(self.tracks) + 1)
                self.tracks.append(tr)
            if tr not in touched:
                touched.append(tr)
            tr.add(d)

    def finish(self):
        for tr in self.tracks:
            tr.closed = True
        return self.tracks


def top_classes(probs, k=5):
    order = np.argsort(-probs)[:k]
    return [(DATASET_CLASSES[i], float(probs[i])) for i in order]
