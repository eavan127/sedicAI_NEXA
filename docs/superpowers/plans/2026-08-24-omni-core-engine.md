# OMNI Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure-Python inference core the OMNI console needs — sliding-window classification, per-class EMA smoothing, multi-class event grouping, DSP measurement, and synthesized scenario capture — with no Gradio and no matplotlib, so all of it is unit-testable.

**Architecture:** Three new modules under `src/`. `timeline.py` owns everything derived from the MODEL (windows, probabilities, attention, smoothing, events). `measure.py` owns everything derived from DSP on the samples (noise floor, SNR estimate, occupancy, power spectrum). `scenarios.py` builds long synthetic captures with ground truth. The provenance rule from the spec is enforced structurally: MODEL and MEASURED quantities live in different modules and never mix inside one function.

**Tech Stack:** Python 3.14, NumPy, PyTorch, SciPy, pytest. Existing project modules: `src.config`, `src.data.preprocess`, `src.models.amc_cnn`, `src.generators.*`, `src.data.composite`.

**Spec:** `docs/superpowers/specs/2026-08-24-omni-ui-design.md`

**Follow-on plan:** `docs/superpowers/plans/2026-08-24-omni-console.md` (the six pages) consumes this.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/timeline.py` (create) | MODEL-derived: `sliding_windows`, `TimelineResult`, `classify_capture`, `smooth`, `Detection`, `detections`, `tier_track` |
| `src/measure.py` (create) | MEASURED-derived: `noise_floor_power`, `estimate_snr_db`, `occupancy`, `power_spectrum_db` |
| `src/scenarios.py` (create) | `raised_cosine_ramp`, `ScenarioSegment`, `build_scenario` |
| `tests/test_timeline.py` (create) | Covers `src/timeline.py` |
| `tests/test_measure.py` (create) | Covers `src/measure.py` |
| `tests/test_scenarios.py` (create) | Covers `src/scenarios.py` |

**Note on `src/measure.py`:** the spec's module list names `src/timeline.py` and `src/scenarios.py` but folds DSP into `src/ui/plots.py`. That would make SNR estimation and occupancy untestable without matplotlib, and it would put MEASURED logic in a MODEL-adjacent file. Splitting it out is a deliberate refinement of the spec; update the spec's "Module layout" section when this plan is done.

---

## Task 1: Sliding windows

**Files:**
- Create: `src/timeline.py`
- Test: `tests/test_timeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_timeline.py`:

```python
import numpy as np
import pytest

from src.timeline import sliding_windows


def _ramp(n):
    """Complex IQ whose real part is a 0..n-1 ramp, so window contents are
    identifiable by their starting value."""
    return np.arange(n, dtype=np.float64) + 1j * np.zeros(n)


def test_exactly_one_window_when_input_is_exactly_window_len():
    windows, starts = sliding_windows(_ramp(512), window_len=512, hop=256)
    assert windows.shape == (1, 2, 512)
    assert starts.tolist() == [0]


def test_short_input_pads_to_one_window():
    windows, starts = sliding_windows(_ramp(100), window_len=512, hop=256)
    assert windows.shape == (1, 2, 512)
    assert starts.tolist() == [0]


def test_window_count_for_each_hop_setting():
    iq = _ramp(2048)
    # n = 1 + (len - window_len) // hop
    assert sliding_windows(iq, 512, 512)[0].shape[0] == 4
    assert sliding_windows(iq, 512, 256)[0].shape[0] == 7
    assert sliding_windows(iq, 512, 128)[0].shape[0] == 13
    assert sliding_windows(iq, 512, 64)[0].shape[0] == 25


def test_starts_advance_by_hop():
    _, starts = sliding_windows(_ramp(2048), window_len=512, hop=256)
    assert starts.tolist() == [0, 256, 512, 768, 1024, 1280, 1536]


def test_each_window_is_normalized_independently():
    """preprocess_window normalizes per window, so every window must come out
    with ~zero mean and ~unit std regardless of the amplitude of that slice."""
    iq = _ramp(2048) * 1000.0
    windows, _ = sliding_windows(iq, 512, 256)
    for w in windows:
        assert abs(w.mean()) < 1e-4
        assert abs(w.std() - 1.0) < 1e-3


def test_output_is_float32():
    windows, _ = sliding_windows(_ramp(1024), 512, 256)
    assert windows.dtype == np.float32


def test_rejects_non_positive_hop():
    with pytest.raises(ValueError):
        sliding_windows(_ramp(1024), 512, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_timeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.timeline'`

- [ ] **Step 3: Write minimal implementation**

Create `src/timeline.py`:

```python
"""Sliding-window inference over a continuous IQ capture.

Everything in this module is MODEL-derived -- windows fed to the classifier,
the probabilities it returns, its attention weights, and the events grouped
from them. Nothing here measures the signal itself; that lives in
src/measure.py. Keeping the two apart is what makes the UI's provenance rule
(see docs/superpowers/specs/2026-08-24-omni-ui-design.md) structural rather
than a naming convention.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import preprocess_window


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
    hop = hop or window_len
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_timeline.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/timeline.py tests/test_timeline.py
git commit -m "feat(timeline): sliding_windows over a continuous IQ capture"
```

---

## Task 2: TimelineResult and classify_capture

**Files:**
- Modify: `src/timeline.py`
- Test: `tests/test_timeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_timeline.py`:

```python
import torch

from src.config import CLASSES
from src.models.amc_cnn import AMC_CNN
from src.timeline import TimelineResult, classify_capture


@pytest.fixture(scope="module")
def model():
    """Untrained model. Predictions are meaningless, which is fine -- these
    tests check plumbing, shapes and bookkeeping, not accuracy."""
    m = AMC_CNN(num_classes=len(CLASSES), input_len=512)
    m.eval()
    return m


def test_classify_capture_shapes(model):
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096))
    result = classify_capture(iq, model, hop=256)
    assert isinstance(result, TimelineResult)
    assert result.probs.shape == (result.n_windows, len(CLASSES))
    assert result.starts.shape == (result.n_windows,)
    assert result.attn.shape == (result.n_windows, 512)
    assert result.hop == 256
    assert result.window_len == 512


def test_probabilities_are_independent_not_a_simplex(model):
    """Multi-label sigmoid: rows must NOT sum to 1. If they do, someone
    replaced sigmoid with softmax and co-occurrence is gone."""
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096))
    result = classify_capture(iq, model, hop=256)
    assert ((result.probs >= 0) & (result.probs <= 1)).all()
    assert not np.allclose(result.probs.sum(axis=1), 1.0)


def test_attention_rows_sum_to_one(model):
    """Attention is a per-window softmax over time -- each row is its own
    distribution, which is why heights are not comparable across windows."""
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096))
    result = classify_capture(iq, model, hop=256)
    np.testing.assert_allclose(result.attn.sum(axis=1), 1.0, rtol=1e-4)


def test_times_us_maps_windows_to_capture_position(model):
    iq = (np.random.randn(2048) + 1j * np.random.randn(2048))
    result = classify_capture(iq, model, hop=512, fs=3_200_000)
    # window 1 starts at sample 512 -> 512 / 3.2e6 s = 160 us
    assert result.times_us[0] == pytest.approx(0.0)
    assert result.times_us[1] == pytest.approx(160.0)


def test_batching_does_not_change_results(model):
    iq = (np.random.randn(8192) + 1j * np.random.randn(8192))
    a = classify_capture(iq, model, hop=256, batch_size=4)
    b = classify_capture(iq, model, hop=256, batch_size=4096)
    np.testing.assert_allclose(a.probs, b.probs, atol=1e-5)


def test_hook_is_removed_after_call(model):
    """A leaked forward hook would silently accumulate across calls."""
    before = len(model.attn_pool.score._forward_hooks)
    classify_capture(np.random.randn(1024) + 1j * np.random.randn(1024),
                     model, hop=512)
    assert len(model.attn_pool.score._forward_hooks) == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_timeline.py -k "classify or attention or times_us or batching or hook or independent" -v`
Expected: FAIL — `ImportError: cannot import name 'TimelineResult'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/timeline.py` (imports at top, rest appended):

```python
from dataclasses import dataclass, replace

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    hop = hop or window_len
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_timeline.py -v`
Expected: PASS, 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/timeline.py tests/test_timeline.py
git commit -m "feat(timeline): classify_capture with batched inference and attention capture"
```

---

## Task 3: Per-class EMA smoothing

**Files:**
- Modify: `src/timeline.py`
- Test: `tests/test_timeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_timeline.py`:

```python
from src.timeline import smooth


def _result(probs, hop=256, window_len=512, fs=3_200_000):
    probs = np.asarray(probs, dtype=np.float32)
    n = len(probs)
    return TimelineResult(
        probs=probs,
        starts=np.arange(n) * hop,
        attn=np.full((n, window_len), 1.0 / window_len, dtype=np.float32),
        hop=hop, window_len=window_len, fs=fs,
    )


def test_smoothing_suppresses_a_single_window_spike():
    """One noisy window must not survive as a detection."""
    probs = np.zeros((5, 8), dtype=np.float32)
    probs[:, 5] = 0.9              # FHSS steady
    probs[2, 6] = 0.95             # JAMMING one-window blip
    out = smooth(_result(probs), alpha=0.3)
    assert out.probs[2, 6] < 0.5
    assert out.probs[4, 5] > 0.5


def test_smoothing_preserves_co_occurrence():
    """Two classes true together must both survive -- this is what a majority
    vote over argmax would destroy."""
    probs = np.zeros((10, 8), dtype=np.float32)
    probs[:, 5] = 0.9              # FHSS
    probs[:, 6] = 0.85             # JAMMING, sustained
    out = smooth(_result(probs), alpha=0.3)
    assert out.probs[-1, 5] > 0.8
    assert out.probs[-1, 6] > 0.8


def test_smoothing_never_normalizes_rows():
    probs = np.full((6, 8), 0.9, dtype=np.float32)
    out = smooth(_result(probs), alpha=0.5)
    assert out.probs[-1].sum() > 1.0


def test_smoothing_leaves_first_window_untouched():
    probs = np.zeros((3, 8), dtype=np.float32)
    probs[0, 4] = 0.7
    out = smooth(_result(probs), alpha=0.3)
    assert out.probs[0, 4] == pytest.approx(0.7)


def test_smoothing_does_not_mutate_input():
    r = _result(np.full((4, 8), 0.5, dtype=np.float32))
    original = r.probs.copy()
    smooth(r, alpha=0.4)
    np.testing.assert_array_equal(r.probs, original)


def test_smoothing_preserves_other_fields():
    r = _result(np.full((4, 8), 0.5, dtype=np.float32))
    out = smooth(r, alpha=0.4)
    assert out.hop == r.hop and out.window_len == r.window_len
    np.testing.assert_array_equal(out.starts, r.starts)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_timeline.py -k smooth -v`
Expected: FAIL — `ImportError: cannot import name 'smooth'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/timeline.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_timeline.py -v`
Expected: PASS, 19 passed

- [ ] **Step 5: Commit**

```bash
git add src/timeline.py tests/test_timeline.py
git commit -m "feat(timeline): per-class EMA smoothing, display-only"
```

---

## Task 4: Multi-class event grouping

**Files:**
- Modify: `src/timeline.py`
- Test: `tests/test_timeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_timeline.py`:

```python
from src.timeline import Detection, detections

# Class order is CLASSES: BPSK QPSK 16QAM 64QAM LFM_RADAR FHSS JAMMING NOISE_FLOOR
FHSS_I, JAM_I, RADAR_I = 5, 6, 4
FLAT = {c: 0.5 for c in CLASSES}


def test_consecutive_same_class_windows_merge_into_one_event():
    probs = np.zeros((5, 8), dtype=np.float32)
    probs[1:4, FHSS_I] = 0.9
    events = detections(_result(probs), FLAT)
    assert len(events) == 1
    assert events[0].classes == ("FHSS",)
    assert events[0].start_window == 1 and events[0].end_window == 3


def test_two_classes_together_are_one_event_not_two():
    """A run where FHSS and JAMMING are both over threshold is ONE event
    labelled FHSS + JAMMING, not two overlapping single-class events."""
    probs = np.zeros((4, 8), dtype=np.float32)
    probs[:, FHSS_I] = 0.9
    probs[:, JAM_I] = 0.85
    events = detections(_result(probs), FLAT)
    assert len(events) == 1
    assert set(events[0].classes) == {"FHSS", "JAMMING"}


def test_change_in_detected_set_starts_a_new_event():
    probs = np.zeros((4, 8), dtype=np.float32)
    probs[:, FHSS_I] = 0.9
    probs[2:, JAM_I] = 0.9        # jammer joins halfway
    events = detections(_result(probs), FLAT)
    assert len(events) == 2
    assert events[0].classes == ("FHSS",)
    assert set(events[1].classes) == {"FHSS", "JAMMING"}


def test_windows_with_nothing_over_threshold_produce_no_event():
    probs = np.zeros((5, 8), dtype=np.float32)
    events = detections(_result(probs), FLAT)
    assert events == []


def test_peak_confidence_is_per_class_maximum():
    probs = np.zeros((3, 8), dtype=np.float32)
    probs[:, FHSS_I] = [0.7, 0.94, 0.8]
    events = detections(_result(probs), FLAT)
    assert events[0].peak["FHSS"] == pytest.approx(0.94)


def test_event_times_span_first_window_start_to_last_window_end():
    probs = np.zeros((4, 8), dtype=np.float32)
    probs[1:3, FHSS_I] = 0.9
    r = _result(probs, hop=256, window_len=512, fs=3_200_000)
    e = detections(r, FLAT)[0]
    # starts at sample 256 -> 80 us; last window starts 512, ends 1024 -> 320 us
    assert e.start_us == pytest.approx(80.0)
    assert e.end_us == pytest.approx(320.0)
    assert e.duration_us == pytest.approx(240.0)


def test_per_class_thresholds_are_honoured():
    """LFM_RADAR at 0.30 must fire under a 0.26 threshold but not a 0.5 one."""
    probs = np.zeros((2, 8), dtype=np.float32)
    probs[:, RADAR_I] = 0.30
    lenient = dict(FLAT, LFM_RADAR=0.26)
    assert len(detections(_result(probs), lenient)) == 1
    assert detections(_result(probs), FLAT) == []


def test_event_count_is_events_not_windows():
    probs = np.zeros((40, 8), dtype=np.float32)
    probs[:, FHSS_I] = 0.9
    assert len(detections(_result(probs), FLAT)) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_timeline.py -k "event or threshold or peak or merge" -v`
Expected: FAIL — `ImportError: cannot import name 'Detection'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/timeline.py` (add `from src.config import CFG, CLASSES` to the existing config import):

```python
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


def _detected_sets(result, thresholds):
    """Per window, the tuple of class names over their own threshold."""
    thr = np.array([thresholds[c] for c in CLASSES], dtype=np.float32)
    over = result.probs > thr
    return [tuple(c for i, c in enumerate(CLASSES) if row[i]) for row in over]


def detections(result, thresholds):
    """Group consecutive windows into events.

    Grouping keys on the WHOLE detected set, not one class at a time. A run
    where FHSS and JAMMING both fire is one `FHSS + JAMMING` event -- grouping
    per class would emit two overlapping rows describing one situation.

    Without grouping, a detections table shows one emitter once per window
    (~40 rows for a 6 ms emitter at hop 256) and any "Detections: N" readout
    counts windows rather than signals.
    """
    sets = _detected_sets(result, thresholds)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_timeline.py -v`
Expected: PASS, 27 passed

- [ ] **Step 5: Commit**

```bash
git add src/timeline.py tests/test_timeline.py
git commit -m "feat(timeline): multi-class event grouping"
```

---

## Task 5: Tier track (and move TIERS somewhere importable)

`TIERS` currently lives in `src/evaluate.py`, which imports matplotlib and
sklearn at module scope. Importing it from `src/timeline.py` would drag both
into the core and break this plan's "core needs no matplotlib" property. Move
the constant to `src/config.py` (which imports only `os`, `pathlib` and
`yaml`) and re-export it from `src/evaluate.py` so existing call sites —
including `scripts/inference_ui.py` — keep working unchanged.

**Files:**
- Modify: `src/config.py`
- Modify: `src/evaluate.py:43-49`
- Modify: `src/timeline.py`
- Test: `tests/test_timeline.py`

- [ ] **Step 0: Move TIERS to src/config.py**

Cut the `TIERS` definition (including its comment) out of `src/evaluate.py`
and paste it into `src/config.py`, after `CLASSES` is defined:

```python
# Operational grouping used by the UI ribbon and the Alerts page, and by
# evaluate.py's per-tier reporting. Lives here rather than in evaluate.py so
# that importing it does not pull in matplotlib and sklearn -- src/timeline.py
# needs it and must stay dependency-light.
TIERS = {"Civilian": ["BPSK", "QPSK", "16QAM", "64QAM"],
         "Military": ["LFM_RADAR", "FHSS"],
         "Hostile": ["JAMMING"],
         # Its own tier, not folded into Civilian: an empty channel is not
         # "ordinary traffic", it is the absence of any emitter. Merging the
         # two would hide the false alarm this class exists to prevent.
         "Empty": ["NOISE_FLOOR"]}
```

Then in `src/evaluate.py`, replace the deleted definition with a re-export so
existing imports (`from src.evaluate import TIERS`) still resolve:

```python
from src.config import (CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT, TIERS,
                         resolve_multilabel_thresholds)
```

- [ ] **Step 0b: Verify nothing broke**

Run: `python -c "from src.evaluate import TIERS; from src.config import TIERS as T2; assert TIERS is T2; print('ok')"`
Expected: `ok`

Run: `python -m pytest tests/ -q`
Expected: PASS, no new failures

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_timeline.py`:

```python
from src.timeline import tier_track

BPSK_I, NOISE_I = 0, 7


def test_tier_track_returns_one_tier_per_window():
    probs = np.zeros((4, 8), dtype=np.float32)
    assert len(tier_track(_result(probs), FLAT)) == 4


def test_hostile_outranks_military_and_civilian():
    """A jammer sitting on top of a civilian signal must colour the ribbon
    Hostile -- the worst thing present is what an operator needs to see."""
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, BPSK_I] = 0.9
    probs[0, FHSS_I] = 0.9
    probs[0, JAM_I] = 0.9
    assert tier_track(_result(probs), FLAT)[0] == "Hostile"


def test_military_outranks_civilian():
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, BPSK_I] = 0.9
    probs[0, RADAR_I] = 0.9
    assert tier_track(_result(probs), FLAT)[0] == "Military"


def test_nothing_over_threshold_is_empty_tier():
    probs = np.zeros((1, 8), dtype=np.float32)
    assert tier_track(_result(probs), FLAT)[0] == "Empty"


def test_noise_floor_is_empty_tier_not_a_threat():
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[0, NOISE_I] = 0.95
    assert tier_track(_result(probs), FLAT)[0] == "Empty"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_timeline.py -k tier -v`
Expected: FAIL — `ImportError: cannot import name 'tier_track'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/timeline.py`:

```python
from src.config import TIERS

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


def tier_track(result, thresholds):
    """One tier name per window, for the ribbon.

    NOISE_FLOOR maps to Empty, the same as nothing-detected: both mean "no
    emitter here". They are distinguishable in the probability list; on the
    ribbon they are the same operational state.
    """
    out = []
    for detected in _detected_sets(result, thresholds):
        out.append(tier_of_classes(detected))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_timeline.py -v`
Expected: PASS, 32 passed

- [ ] **Step 5: Commit**

```bash
git add src/timeline.py tests/test_timeline.py
git commit -m "feat(timeline): tier_track for the ribbon"
```

---

## Task 6: DSP measurement module

**Files:**
- Create: `src/measure.py`
- Test: `tests/test_measure.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_measure.py`:

```python
import numpy as np
import pytest

from src.measure import (estimate_snr_db, noise_floor_power, occupancy,
                          power_spectrum_db)


def _noise(n, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)) / np.sqrt(2)


def test_noise_floor_of_pure_noise_is_its_power():
    iq = _noise(32768, sigma=1.0)
    assert noise_floor_power(iq) == pytest.approx(1.0, rel=0.25)


def test_noise_floor_ignores_a_loud_burst():
    """The quiet-percentile estimate must not be dragged up by a strong
    emitter occupying part of the capture."""
    iq = _noise(32768, sigma=1.0)
    iq[:8192] *= 30.0
    assert noise_floor_power(iq) == pytest.approx(1.0, rel=0.35)


def test_estimate_snr_recovers_a_known_ratio():
    """A window with 10x the noise power above the floor should read ~10 dB."""
    noise_power = 1.0
    rng = np.random.default_rng(1)
    n = 512
    signal = (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) / np.sqrt(2)
    signal *= np.sqrt(10.0 / np.mean(np.abs(signal) ** 2))
    window = signal + _noise(n, sigma=1.0, seed=2)
    assert estimate_snr_db(window, noise_power) == pytest.approx(10.0, abs=2.0)


def test_estimate_snr_of_noise_only_is_very_low():
    assert estimate_snr_db(_noise(512, seed=3), 1.0) < 3.0


def test_estimate_snr_never_returns_nan_or_inf():
    """A window at or below the floor must clamp, not divide by zero."""
    value = estimate_snr_db(np.zeros(512, dtype=complex), 1.0)
    assert np.isfinite(value)


def test_occupancy_is_low_for_pure_noise():
    assert occupancy(_noise(32768)) < 0.2


def test_occupancy_is_higher_with_a_strong_tone():
    iq = _noise(32768)
    t = np.arange(32768)
    iq += 20.0 * np.exp(2j * np.pi * 0.1 * t)
    assert occupancy(iq) > 0.0


def test_occupancy_is_a_fraction():
    assert 0.0 <= occupancy(_noise(8192)) <= 1.0


def test_power_spectrum_peaks_at_the_tone_frequency():
    fs = 3_200_000
    t = np.arange(16384) / fs
    iq = np.exp(2j * np.pi * 800_000 * t) + 0.01 * _noise(16384)
    freqs, spectrum = power_spectrum_db(iq, fs)
    assert freqs[np.argmax(spectrum)] == pytest.approx(800_000, abs=20_000)


def test_power_spectrum_covers_full_complex_band():
    freqs, _ = power_spectrum_db(_noise(8192), 3_200_000)
    assert freqs.min() < -1_500_000
    assert freqs.max() > 1_500_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_measure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.measure'`

- [ ] **Step 3: Write minimal implementation**

Create `src/measure.py`:

```python
"""DSP measurement over a raw IQ capture.

Everything here is MEASURED -- computed from the samples themselves, with no
model involved. Under the UI's provenance rule these values are rendered in
neutral instrument styling and are never coloured as detections. Keeping them
in a separate module from src/timeline.py makes that separation structural.

Requires a capture that has NOT been through preprocess_window: that function
normalizes to zero mean and unit variance, which destroys the absolute
amplitude every function here depends on.
"""
import numpy as np
from scipy.signal import stft

from src.config import CFG

_EPS = 1e-20


def _frame_powers(iq, frame_len=512):
    """Mean power per non-overlapping frame."""
    n_frames = max(len(iq) // frame_len, 1)
    frames = np.asarray(iq)[:n_frames * frame_len].reshape(n_frames, frame_len)
    return np.mean(np.abs(frames) ** 2, axis=1)


def noise_floor_power(iq, percentile=10.0, frame_len=512):
    """Noise power estimated from the quietest frames of the capture.

    Uses a low percentile rather than the minimum so a single anomalously
    quiet frame cannot set the floor, and so a capture that is mostly quiet
    still yields a stable estimate. Assumes the capture contains SOME region
    without a strong emitter -- true for scenario captures by construction,
    and typical of real recordings.
    """
    return float(max(np.percentile(_frame_powers(iq, frame_len), percentile),
                     _EPS))


def estimate_snr_db(window_iq, noise_power):
    """Estimated SNR of one window, in dB.

    Subtracts the noise floor before taking the ratio: total window power is
    signal PLUS noise, so 10*log10(total / noise) reads +3 dB for a window
    that actually sits at 0 dB SNR. Subtracting first gives an unbiased
    estimate.

    ALWAYS an estimate. The UI must render the result with a visible `est.`
    prefix -- the classifier does not produce SNR, and this is not a
    calibrated receiver measurement.
    """
    total = float(np.mean(np.abs(np.asarray(window_iq)) ** 2))
    signal = max(total - noise_power, _EPS)
    return float(10.0 * np.log10(signal / max(noise_power, _EPS)))


def occupancy(iq, nperseg=256, margin_db=6.0):
    """Fraction of time-frequency cells sitting above the noise floor.

    This is a MEASURED spectrum-occupancy figure. It deliberately replaces the
    "Channel Load" readout an OmniSIG-style console would show: the obvious
    implementation of that name here (fraction of windows where a class fired)
    would be MODEL output wearing a measurement's name, which the provenance
    rule forbids.
    """
    _, _, Z = stft(np.asarray(iq), nperseg=nperseg, return_onesided=False)
    power = np.abs(Z) ** 2
    floor = max(np.percentile(power, 10.0), _EPS)
    return float(np.mean(power > floor * 10 ** (margin_db / 10.0)))


def power_spectrum_db(iq, fs=None, nperseg=1024):
    """Average power spectrum in dB, over the full complex band.

    Returns (freqs_hz, spectrum_db), both fftshifted so frequency runs
    -fs/2 .. +fs/2 -- the capture is complex baseband, so the negative half is
    real signal, not a mirror.
    """
    fs = fs or CFG["signal"]["fs"]
    f, _, Z = stft(np.asarray(iq), fs=fs, nperseg=nperseg,
                    return_onesided=False)
    mean_power = np.mean(np.abs(Z) ** 2, axis=1)
    order = np.argsort(np.fft.fftshift(f))
    freqs = np.fft.fftshift(f)[order]
    spectrum = 10.0 * np.log10(np.fft.fftshift(mean_power)[order] + _EPS)
    return freqs, spectrum
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_measure.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/measure.py tests/test_measure.py
git commit -m "feat(measure): noise floor, SNR estimate, occupancy, power spectrum"
```

---

## Task 7: Synthesized scenario captures

**Files:**
- Create: `src/scenarios.py`
- Test: `tests/test_scenarios.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scenarios.py`:

```python
import numpy as np
import pytest

from src.scenarios import ScenarioSegment, build_scenario, raised_cosine_ramp


def test_ramp_starts_at_zero_and_ends_at_one():
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
    iq, segments = build_scenario(fs=3_200_000, total_duration=0.02,
                                   snr_db=10, seed=0)
    fs = 3_200_000
    active = segments[0]
    a = np.mean(np.abs(iq[int(active.start_s * fs):int(active.end_s * fs)]) ** 2)
    gap_start = int(segments[-1].end_s * fs)
    if gap_start < len(iq) - 1000:
        q = np.mean(np.abs(iq[gap_start:]) ** 2)
        assert a > q


def test_scenario_is_reproducible_for_a_seed():
    a, _ = build_scenario(fs=3_200_000, total_duration=0.005, seed=7)
    b, _ = build_scenario(fs=3_200_000, total_duration=0.005, seed=7)
    np.testing.assert_array_equal(a, b)


def test_segments_reference_real_class_names():
    from src.config import CLASSES
    _, segments = build_scenario(fs=3_200_000, total_duration=0.01, seed=0)
    for s in segments:
        assert s.class_name in CLASSES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scenarios.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.scenarios'`

- [ ] **Step 3: Write minimal implementation**

Create `src/scenarios.py`:

```python
"""Synthesized multi-emitter captures, long enough to drive the replay deck.

Why generate rather than stitch stored dataset windows: every stored window
went through preprocess_window, which normalizes it to zero mean and unit
variance. Concatenating them yields a capture with no amplitude dynamics (so
no usable noise floor and no SNR), and a phase discontinuity at every splice
-- which at hop 256 would corrupt roughly half of all sliding windows.

The generators already accept total_duration. build_dataset asks for 0.002 s
(6,400 samples) and then keeps only the first 512, discarding 92% of it. Here
we simply ask for more and keep all of it.
"""
from dataclasses import dataclass

import numpy as np

from src.config import CFG
from src.generators.fhss import random_fhss_example
from src.generators.jamming import random_jamming_example
from src.generators.radar import random_radar_example

GENERATORS = {
    "LFM_RADAR": random_radar_example,
    "FHSS": random_fhss_example,
    "JAMMING": random_jamming_example,
}


@dataclass
class ScenarioSegment:
    """Ground truth for one emitter's active period. TRUTH provenance -- the
    UI renders these in outline styling, never as if they were detections,
    and never at all for uploaded captures."""
    class_name: str
    start_s: float
    end_s: float


def raised_cosine_ramp(n_samples, ramp_len=256):
    """Envelope that fades in and out with a raised-cosine shape.

    Switching an emitter on with a hard edge produces broadband splatter
    across the whole band -- which would show on the waterfall as a vertical
    stripe and could plausibly trigger a JAMMING detection that nothing in the
    scenario put there. Real transmitters ramp; so do these.
    """
    env = np.ones(n_samples)
    ramp_len = int(min(ramp_len, n_samples // 2))
    if ramp_len < 1:
        return env
    rise = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len)))
    env[:ramp_len] = rise
    env[-ramp_len:] = rise[::-1]
    return env


# Fractions of total_duration. Reads as: quiet, radar sweeping, jammer joins
# on top of the radar, quiet again.
DEFAULT_SCRIPT = [
    ("LFM_RADAR", 0.10, 0.45),
    ("FHSS",      0.30, 0.70),
    ("JAMMING",   0.55, 0.85),
]


def build_scenario(fs=None, total_duration=0.1, snr_db=-6, seed=0,
                    script=None):
    """Build one continuous capture with known ground truth.

    Returns (iq, segments). The capture is NOT normalized: absolute amplitude
    has to survive so the waterfall, the noise floor and the SNR readout mean
    something.

    Noise is added once, at the end, scaled against the power of the ACTIVE
    regions only. Scaling against the whole capture would let the quiet gaps
    drag the mean down and make the active regions land at a higher SNR than
    requested.
    """
    fs = fs or CFG["signal"]["fs"]
    script = script or DEFAULT_SCRIPT
    rng = np.random.default_rng(seed)

    n_total = int(round(total_duration * fs))
    iq = np.zeros(n_total, dtype=np.complex128)
    segments = []
    active = np.zeros(n_total, dtype=bool)

    for class_name, start_frac, end_frac in script:
        start = int(start_frac * n_total)
        end = min(int(end_frac * n_total), n_total)
        if end - start < 2:
            continue
        length = end - start

        emitter = GENERATORS[class_name](fs=fs, total_duration=length / fs,
                                          rng=rng)
        emitter = np.asarray(emitter)
        if len(emitter) < length:
            emitter = np.pad(emitter, (0, length - len(emitter)))
        emitter = emitter[:length] * raised_cosine_ramp(length)

        iq[start:end] += emitter
        active[start:end] = True
        segments.append(ScenarioSegment(class_name, start / fs, end / fs))

    signal_power = float(np.mean(np.abs(iq[active]) ** 2)) if active.any() else 1.0
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = (rng.normal(0, 1, n_total) + 1j * rng.normal(0, 1, n_total))
    iq += noise * np.sqrt(noise_power / 2.0)

    return iq, segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scenarios.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/scenarios.py tests/test_scenarios.py
git commit -m "feat(scenarios): synthesized multi-emitter captures with ground truth"
```

---

## Task 8: End-to-end core integration test

**Files:**
- Modify: `tests/test_timeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_timeline.py`:

```python
from src.config import resolve_multilabel_thresholds
from src.measure import estimate_snr_db, noise_floor_power
from src.scenarios import build_scenario


def test_full_core_pipeline_runs_on_a_scenario(model):
    """Scenario -> windows -> model -> smoothing -> events, end to end.

    Uses an untrained model, so this asserts on plumbing and bookkeeping, not
    on whether the right class was found."""
    iq, segments = build_scenario(fs=3_200_000, total_duration=0.01, seed=0)

    result = classify_capture(iq, model, hop=256, fs=3_200_000)
    assert result.n_windows == 1 + (32_000 - 512) // 256

    thresholds = dict(zip(CLASSES, resolve_multilabel_thresholds()))
    events = detections(smooth(result, alpha=0.3), thresholds)
    for e in events:
        assert 0.0 <= e.start_us < e.end_us
        assert e.end_us <= result.n_windows * 256 / 3_200_000 * 1e6 + 200
        assert set(e.classes).issubset(set(CLASSES))

    assert len(tier_track(result, thresholds)) == result.n_windows

    floor = noise_floor_power(iq)
    snr = estimate_snr_db(iq[:512], floor)
    assert np.isfinite(snr)


def test_capture_is_never_normalized_by_the_pipeline(model):
    """Guard for the spec's normalization rule: classify_capture must not
    modify the caller's capture, and must not normalize it in place."""
    iq, _ = build_scenario(fs=3_200_000, total_duration=0.005, seed=1)
    before = iq.copy()
    classify_capture(iq, model, hop=512, fs=3_200_000)
    np.testing.assert_array_equal(iq, before)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_timeline.py -k "full_core or never_normalized" -v`
Expected: FAIL — `ImportError` on `src.scenarios` if Task 7 was skipped; otherwise PASS immediately (this is an integration test over already-built parts, so it may pass on first run — that is acceptable here).

- [ ] **Step 3: Fix anything the integration test surfaces**

No new production code is expected. If the test fails, the defect is in Tasks 1–7; fix it there rather than adding compensating logic here.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: PASS — all existing project tests plus 34 timeline, 10 measure, 10 scenarios

- [ ] **Step 5: Commit**

```bash
git add tests/test_timeline.py
git commit -m "test: end-to-end core pipeline over a synthesized scenario"
```

---

## Task 9: Update the spec's module layout

**Files:**
- Modify: `docs/superpowers/specs/2026-08-24-omni-ui-design.md`

- [ ] **Step 1: Update the module layout section**

Replace the `src/timeline.py` line block in the "Module layout" section so it reads:

```
src/timeline.py          MODEL-derived: windows, probabilities, attention,
                         smoothing, event grouping, tier track
src/measure.py           MEASURED-derived: noise floor, SNR estimate,
                         occupancy, power spectrum
src/scenarios.py         synthesized captures + ground truth
src/ui/palette.py        dark console palette, tier colors
src/ui/plots.py          waterfall, spectrum trace, ribbon, attention
src/ui/pages/*.py        one module per page
src/ui/app.py            assembles tabs
scripts/inference_ui.py  entry point -> src.ui.app
```

Add below it:

```markdown
`src/measure.py` was split out of the original plan for `src/ui/plots.py` so
that SNR estimation and occupancy are testable without matplotlib, and so
MEASURED logic never shares a module with MODEL logic — making the provenance
rule structural rather than a naming convention.
```

- [ ] **Step 2: Verify the spec still reads consistently**

Run: `grep -n "measure.py" docs/superpowers/specs/2026-08-24-omni-ui-design.md`
Expected: at least two matches

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-24-omni-ui-design.md
git commit -m "docs: record src/measure.py split in the OMNI spec"
```

---

## Definition of Done

- `python -m pytest tests/ -v` passes
- `src/timeline.py`, `src/measure.py`, `src/scenarios.py` exist and import without Gradio or matplotlib installed
- No function in `src/timeline.py` computes a MEASURED quantity, and no function in `src/measure.py` calls the model
- A scenario capture round-trips: `build_scenario` → `classify_capture` → `smooth` → `detections` produces well-formed events
