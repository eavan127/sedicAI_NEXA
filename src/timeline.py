"""Sliding-window inference over a continuous IQ capture.

Everything in this module is MODEL-derived -- windows fed to the classifier,
the probabilities it returns, its attention weights, and the events grouped
from them. Nothing here measures the signal itself; that lives in
src/measure.py. Keeping the two apart is what makes the UI's provenance rule
(see docs/superpowers/specs/2026-08-24-omni-ui-design.md) structural rather
than a naming convention.
"""
from dataclasses import dataclass, replace

import numpy as np
import torch

from src.config import CFG, CLASS_TO_IDX, CLASSES, TIERS
from src.data.preprocess import preprocess_window

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sliding_windows(iq, window_len=None, hop=None):
    """Cut a continuous complex IQ capture into overlapping model inputs.

    Returns (windows, starts): windows is (n, 2, window_len) float32, already
    normalized per window by preprocess_window -- which is exactly how every
    training example was normalized, so the model sees the same distribution
    it was trained on. `starts` holds each window's first sample index, which
    is what maps a window back to a position in the capture.

    The capture itself is NEVER normalized as a whole -- see the spec's
    normalization rule. Raw amplitude has to survive for the waterfall and the
    noise-floor estimate to mean anything.
    """
    window_len = window_len or CFG["signal"]["window_len"]
    # `hop or window_len` would silently turn an invalid hop=0 into a valid
    # default, so the guard below could never fire. Check for None explicitly.
    hop = window_len if hop is None else hop
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}")

    iq = np.asarray(iq)
    if len(iq) < window_len:
        iq = np.pad(iq, (0, window_len - len(iq)))

    n = 1 + (len(iq) - window_len) // hop
    starts = np.arange(n) * hop
    out = np.empty((n, 2, window_len), dtype=np.float32)
    for i, s in enumerate(starts):
        out[i] = preprocess_window(iq[s:s + window_len], window_len)
    return out, starts


@dataclass
class TimelineResult:
    """One sliding-window pass over a capture.

    probs: (n_windows, num_classes) INDEPENDENT sigmoid probabilities -- rows
           do not sum to 1, and must never be normalized so that they do.
    attn:  (n_windows, window_len) attention weights, each row a softmax over
           time. Rows are separate distributions: a tall spike in one window
           says nothing about magnitude relative to another window.
    """
    probs: np.ndarray
    starts: np.ndarray
    attn: np.ndarray
    hop: int
    window_len: int
    fs: float

    @property
    def n_windows(self):
        return len(self.starts)

    @property
    def times_us(self):
        """Start time of each window, in microseconds from capture start."""
        return self.starts / self.fs * 1e6

    @property
    def window_duration_us(self):
        return self.window_len / self.fs * 1e6


def classify_capture(iq, model, hop=None, window_len=None, fs=None,
                     batch_size=256):
    """Run the model over every sliding window of a capture.

    Batched deliberately: looping one window at a time through a model this
    small is dominated by per-call overhead, and a 0.1 s capture at hop 256 is
    ~1,250 windows.

    Attention is captured with a forward hook on attn_pool.score, the same
    approach scripts/inspect_attention.py uses. The hook sees the RAW scores,
    before the softmax the model applies internally, so this function applies
    that softmax itself.
    """
    window_len = window_len or CFG["signal"]["window_len"]
    hop = window_len if hop is None else hop
    fs = fs or CFG["signal"]["fs"]

    windows, starts = sliding_windows(iq, window_len, hop)

    captured = {}

    def _hook(module, inputs, output):
        captured["scores"] = output.detach()

    handle = model.attn_pool.score.register_forward_hook(_hook)
    probs_batches, attn_batches = [], []
    try:
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                xb = torch.tensor(windows[i:i + batch_size]).to(DEVICE)
                logits = model(xb)
                probs_batches.append(torch.sigmoid(logits).cpu().numpy())
                scores = captured["scores"]           # (batch, 1, time)
                weights = torch.softmax(scores, dim=2)[:, 0, :]
                attn_batches.append(weights.cpu().numpy())
    finally:
        handle.remove()

    return TimelineResult(
        probs=np.concatenate(probs_batches).astype(np.float32),
        starts=starts,
        attn=np.concatenate(attn_batches).astype(np.float32),
        hop=hop, window_len=window_len, fs=fs,
    )


def smooth(result, alpha=0.3):
    """Exponential moving average applied to EACH CLASS independently.

    Per-class, not majority-vote-over-argmax, and the difference is the whole
    point: this model is multi-label, so "jammer overlaid on a victim signal"
    is a legitimate two-class answer. Voting for a single winner would delete
    exactly the case the composite training examples exist to teach.

    DISPLAY ONLY. Benchmark numbers (src/evaluate.py) stay per-window and
    unsmoothed -- smoothing suppresses brief events, which on a benchmark
    judged by recall would quietly cost real detections. See the spec's
    two-pipeline table.

    alpha is the weight on the newest window: higher = more responsive, lower
    = steadier.
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    p = result.probs
    out = np.empty_like(p)
    acc = p[0].astype(np.float64)
    out[0] = acc
    for i in range(1, len(p)):
        acc = alpha * p[i] + (1 - alpha) * acc
        out[i] = acc
    return replace(result, probs=out.astype(np.float32))


@dataclass
class Detection:
    """One grouped detection event, spanning a run of consecutive windows
    that all reported the same set of classes."""
    start_us: float
    end_us: float
    duration_us: float
    classes: tuple
    peak: dict
    start_window: int
    end_window: int

    @property
    def label(self):
        return " + ".join(self.classes)


def _over_threshold(result, thresholds):
    """(n_windows, n_classes) boolean: is each class over its OWN threshold."""
    thr = np.array([thresholds[c] for c in CLASSES], dtype=np.float32)
    return result.probs > thr


def _sets_from_matrix(over):
    return [tuple(c for i, c in enumerate(CLASSES) if row[i]) for row in over]


def _detected_sets(result, thresholds):
    """Per window, the tuple of class names over their own threshold."""
    return _sets_from_matrix(_over_threshold(result, thresholds))


def apply_noise_gate(probs, over, gate=0.5):
    """Where NOISE_FLOOR dominates, the window is empty -- drop everything else.

    Grounded in how the dataset was built: NOISE_FLOOR never co-occurs with
    any other class (see compute_snr_weights' docstring in src/train.py), so a
    window where NOISE_FLOOR is strong AND some emitter is weakly over its
    threshold is out of distribution, and NOISE_FLOOR is the answer.

    This matters because the per-class thresholds are calibrated on the
    DATASET, where every window is either an emitter or a labelled NOISE_FLOOR
    example. A continuous capture has genuinely quiet gaps, and in those gaps
    LFM_RADAR sits around 0.38-0.50 -- comfortably over its 0.26 threshold.
    Without this gate a 3-emitter scenario reported 243 events, nearly all of
    them phantom radar on empty spectrum.

    DISPLAY ONLY, like smoothing. The scorecard stays per-window and ungated.
    """
    if "NOISE_FLOOR" not in CLASS_TO_IDX:
        return over
    noise_idx = CLASS_TO_IDX["NOISE_FLOOR"]
    out = over.copy()
    empty = probs[:, noise_idx] > gate
    out[empty, :] = False
    out[empty, noise_idx] = True
    return out


def apply_hold(over, hold_windows):
    """Bridge short gaps in each class's presence, INDEPENDENTLY per class.

    A pulsed emitter is genuinely absent between pulses: with
    max_duty_cycle 0.15, most windows inside a radar's active period contain
    no pulse at all, so one radar fragments into dozens of events. Real
    detectors solve this with a hangover timer; this is that.

    Per class, not per detected-set. Merging whole events whenever their class
    sets intersect chains transitively -- radar overlaps FHSS overlaps
    jamming -- and collapses an entire capture into a single event. Filling
    gaps in each class's own track and only then regrouping cannot do that.
    """
    if hold_windows <= 0:
        return over
    out = over.copy()
    # NOISE_FLOOR is excluded: it denotes an EMPTY channel, which is a state,
    # not a pulsed emitter. Holding it would bridge quiet gaps the same way it
    # bridges a radar's inter-pulse gaps, and the filled NOISE_FLOOR span would
    # then overlap the filled emitter span -- producing "LFM_RADAR +
    # NOISE_FLOOR" events, which the dataset says cannot exist.
    noise_idx = CLASS_TO_IDX.get("NOISE_FLOOR")
    for j in range(over.shape[1]):
        if j == noise_idx:
            continue
        idx = np.flatnonzero(over[:, j])
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a - 1 <= hold_windows:
                out[a + 1:b, j] = True
    return out


def _resolved_matrix(result, thresholds, noise_gate=None, hold_us=0.0):
    """Threshold, then apply the two deployment-layer rules in order."""
    over = _over_threshold(result, thresholds)
    if noise_gate is not None:
        over = apply_noise_gate(result.probs, over, noise_gate)
    if hold_us > 0:
        hold_windows = int(round(hold_us * result.fs / 1e6 / result.hop))
        over = apply_hold(over, hold_windows)
        # Re-assert mutual exclusion AFTER hold, and ONLY after hold. Holding
        # an emitter across a gap can extend it over windows where NOISE_FLOOR
        # was legitimately detected; "an emitter is present AND the channel is
        # empty" is not a state the dataset contains, so the emitter wins.
        #
        # Scoped to this branch deliberately: applying it unconditionally would
        # change plain detections(result, thresholds) output, and this function
        # has to stay a primitive that the scorecard path can rely on.
        noise_idx = CLASS_TO_IDX.get("NOISE_FLOOR")
        if noise_idx is not None:
            others = np.delete(over, noise_idx, axis=1).any(axis=1)
            over = over.copy()
            over[others, noise_idx] = False
    return over


def detections(result, thresholds, noise_gate=None, hold_us=0.0):
    """Group consecutive windows into events.

    `noise_gate` and `hold_us` are DISPLAY-LAYER rules and default to off, so
    this stays a plain primitive and the scorecard path is unaffected. The UI
    turns them on; src/evaluate.py never does. See apply_noise_gate and
    apply_hold for why each exists.

    Grouping keys on the WHOLE detected set, not one class at a time. A run
    where FHSS and JAMMING both fire is one `FHSS + JAMMING` event -- grouping
    per class would emit two overlapping rows describing one situation.

    Without grouping, a detections table shows one emitter once per window
    (~40 rows for a 6 ms emitter at hop 256) and any "Detections: N" readout
    counts windows rather than signals.
    """
    sets = _sets_from_matrix(
        _resolved_matrix(result, thresholds, noise_gate, hold_us))
    events, i = [], 0
    while i < len(sets):
        current = sets[i]
        if not current:
            i += 1
            continue
        j = i
        while j + 1 < len(sets) and sets[j + 1] == current:
            j += 1

        peak = {c: float(result.probs[i:j + 1, CLASSES.index(c)].max())
                for c in current}
        start_us = float(result.starts[i]) / result.fs * 1e6
        end_us = float(result.starts[j] + result.window_len) / result.fs * 1e6
        events.append(Detection(
            start_us=start_us, end_us=end_us, duration_us=end_us - start_us,
            classes=current, peak=peak, start_window=i, end_window=j,
        ))
        i = j + 1
    return events


# Worst-first. The ribbon shows the most serious thing present in a window,
# so a jammer overlaid on civilian traffic reads Hostile, not Civilian.
TIER_PRIORITY = ("Hostile", "Military", "Civilian", "Empty")

_TIER_OF = {cls: tier for tier, members in TIERS.items() for cls in members}


def tier_of_classes(class_names):
    """Worst tier present among a set of class names.

    Public because the UI needs it for detection-box and alert colouring --
    keeping it private would have every call site reaching into _TIER_OF and
    reimplementing the priority order.
    """
    tiers = {_TIER_OF[c] for c in class_names}
    return next((t for t in TIER_PRIORITY if t in tiers), "Empty")


def tier_track(result, thresholds, noise_gate=None, hold_us=0.0):
    """One tier name per window, for the ribbon.

    NOISE_FLOOR maps to Empty, the same as nothing-detected: both mean "no
    emitter here". They are distinguishable in the probability list; on the
    ribbon they are the same operational state.
    """
    return [tier_of_classes(d) for d in _sets_from_matrix(
        _resolved_matrix(result, thresholds, noise_gate, hold_us))]
