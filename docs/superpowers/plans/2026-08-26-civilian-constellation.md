# Civilian Constellation Panel Implementation Plan

> **Status: superseded in part.** Tasks 1-5 below shipped as written. What
> they produced did not yet do its job -- see
> `2026-08-27-civilian-constellation-addendum.md` (matched filter, and the
> double-noising of civilian scenes) and
> `2026-08-27-civilian-constellation-addendum-2.md` (four windows, and the
> findings from the Task 6 and 7 reviews) for what changed and why.
> `best_civilian_window` named here no longer exists; it is now
> `civilian_windows(count=4)`.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an IQ constellation panel to the RF Replay page that appears only for captures containing civilian traffic, showing raw I/Q beside recovered symbol points for the strongest civilian window.

**Architecture:** Three new units. `recover_symbols()` in `src/ui/plots.py` turns one complex window into symbol points (unit-power scale, blind 4th-power carrier de-rotation, best-phase decimation). `CaptureSession.best_civilian_window()` in `src/ui/session.py` picks which window to show. `constellation_figure()` in `src/ui/plots.py` composes the two-axis figure and returns `None` when there is no civilian window. `src/ui/pages/rf_replay.py` adds one `gr.Plot` below the console figure and toggles its visibility. Everything reads `session.iq` — the capture's own samples — so both panels are MEASURED; the only MODEL element is the caption's class label, which takes its tier colour.

**Tech Stack:** Python, NumPy, Matplotlib (Agg backend), Gradio 6.25, pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-civilian-constellation-design.md`

**Working directory:** `C:/Users/eilee/Documents/Projects/sedicAI_NEXA/.worktrees/eileen-omni-ui` (branch `eileen-omni-ui`). Run every command from there.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ui/plots.py` (modify) | `SAMPLES_PER_SYMBOL`, `carrier_offset()`, `recover_symbols()`, `constellation_figure()`. Existing figure functions untouched. |
| `src/ui/session.py` (modify) | `CaptureSession.best_civilian_window()` — which window the panel shows. Selection is a query over `result.probs`, so it belongs with the session, not with the plotting code. |
| `src/ui/pages/rf_replay.py` (modify) | One new `gr.Plot`, one extra return value from `_render`, visibility toggling. |
| `tests/test_ui_constellation.py` (create) | Every test for this feature: recovery DSP, window selection, figure construction, page wiring. |

Tests live in one new file rather than being split across `test_ui_plots`/`test_ui_session`/a page test, because all four levels share the same fabricated-session helper and splitting them would duplicate it.

---

## Task 1: Symbol recovery

**Files:**
- Modify: `src/ui/plots.py` (add after the `plt.rcParams["font.sans-serif"] = MPL_FONT` line, near line 24)
- Test: `tests/test_ui_constellation.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ui_constellation.py`:

```python
"""Tests for the civilian constellation panel.

The DSP tests build their own QPSK rather than drawing from the dataset: a
known injected carrier offset and a known symbol timing are the only way to
assert that recovery found the RIGHT answer rather than merely a plausible
one.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.ui.plots import SAMPLES_PER_SYMBOL, carrier_offset, recover_symbols


def _qpsk(n_symbols=64, sps=SAMPLES_PER_SYMBOL, offset=0.0039, seed=0):
    """QPSK at `sps` samples/symbol with a triangular pulse shape.

    Pulse-shaped, not rectangular, on purpose: with a rectangular pulse every
    sample equals its symbol, so every timing phase is correct and the phase
    search cannot be tested at all. The triangular pulse peaks exactly on the
    symbol instant and mixes adjacent symbols everywhere else -- which is the
    condition the phase search exists to solve.

    The output is sliced so that sample 0 IS a symbol peak, i.e. the correct
    timing phase is 0.
    """
    rng = np.random.default_rng(seed)
    symbols = rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
                          n_symbols + 2) / np.sqrt(2)
    train = np.zeros(len(symbols) * sps, dtype=complex)
    train[::sps] = symbols
    pulse = np.concatenate([np.linspace(0, 1, sps, endpoint=False),
                             np.linspace(1, 0, sps)])
    shaped = np.convolve(train, pulse)[sps:sps + n_symbols * sps]
    return shaped * np.exp(2j * np.pi * offset * np.arange(len(shaped)))


def _concentration(points, order=4):
    """How tightly points cluster on a 4-fold symmetric constellation.

    1.0 means every point sits on the same 90-degree grid; 0.0 means the
    phases are spread uniformly, which is what a rotating or noise-dominated
    capture looks like.
    """
    return float(abs(np.mean(np.exp(1j * order * np.angle(points)))))


def test_carrier_offset_finds_the_injected_rotation():
    z = _qpsk(offset=0.0039)
    assert carrier_offset(z) == pytest.approx(0.0039, abs=0.001)


def test_recovered_points_cluster_where_the_raw_samples_do_not():
    """The whole reason the panel shows two axes: raw I/Q of an oversampled,
    rotating capture is a ring, and the same samples de-rotated and decimated
    are four clusters."""
    z = _qpsk()
    points, _, _ = recover_symbols(z)
    assert _concentration(points) > 0.9
    assert _concentration(z) < 0.5


def test_recovery_picks_the_symbol_timing_phase():
    """_qpsk places a symbol peak at sample 0, so phase 0 is the right answer
    and the other seven phases sample the pulse mid-transition."""
    _, _, phase = recover_symbols(_qpsk())
    assert phase == 0


def test_recovery_returns_one_point_per_symbol():
    points, _, _ = recover_symbols(_qpsk(n_symbols=64))
    assert len(points) == 64


def test_zero_power_window_returns_without_raising():
    """An all-zero window has no carrier to estimate and no power to normalise
    by. It must render as an empty scatter, not crash the page."""
    points, offset, phase = recover_symbols(np.zeros(512, dtype=complex))
    assert offset == 0.0
    assert phase == 0
    assert len(points) == 512


def test_window_shorter_than_one_symbol_is_returned_untouched():
    short = np.ones(4, dtype=complex)
    points, offset, phase = recover_symbols(short)
    assert np.allclose(points, short)
    assert offset == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_constellation.py -v`

Expected: collection error — `ImportError: cannot import name 'SAMPLES_PER_SYMBOL' from 'src.ui.plots'`

- [ ] **Step 3: Implement recovery**

In `src/ui/plots.py`, immediately after the `plt.rcParams["font.sans-serif"] = MPL_FONT` line:

```python
# RadioML 2018.01A is stored at 8 samples per symbol. Named rather than
# inlined because it is the one constant a capture at another rate would
# invalidate: the decimation below would then sample the pulse shape instead
# of the symbol instants, and the constellation would be wrong without looking
# wrong.
SAMPLES_PER_SYMBOL = 8


def carrier_offset(window, order=4):
    """Blind estimate of residual carrier offset, in cycles per sample.

    Raising the signal to the 4th power collapses a QPSK or QAM constellation
    onto a single tone at 4x the offset, which then shows as an FFT peak. The
    4th power is used for BPSK too: it locks there as well, at the cost of a
    90-degree phase ambiguity, which is harmless here because the panel only
    de-rotates and never labels an axis with an absolute phase.

    MEASURED, not MODEL -- this reads the capture's own samples and fits
    nothing to an expected constellation, so it cannot manufacture clusters
    the samples do not contain.
    """
    z = np.asarray(window)
    if len(z) < order * 2:
        return 0.0
    spectrum = np.abs(np.fft.fft(z ** order))
    k = int(np.argmax(spectrum))
    if k >= len(z) / 2:          # negative frequencies live in the upper half
        k -= len(z)
    return k / len(z) / order


def recover_symbols(window, sps=SAMPLES_PER_SYMBOL):
    """Symbol points from one raw IQ window.

    Returns (points, offset_estimate, timing_phase).

    Three operations, none of them model-derived: unit-power scaling,
    de-rotation by the estimated carrier offset, and decimation to one sample
    per symbol at the timing phase whose points have the tightest amplitude
    spread.

    Degenerate windows -- shorter than one symbol, or carrying no power --
    come back unchanged rather than raising. This feeds a display; a capture
    with a silent stretch in it must render, not crash the page.
    """
    z = np.asarray(window).astype(complex)
    power = float(np.mean(np.abs(z) ** 2)) if len(z) else 0.0
    if len(z) < sps or power <= 0:
        return z, 0.0, 0

    z = z / np.sqrt(power)
    offset = carrier_offset(z)
    z = z * np.exp(-2j * np.pi * offset * np.arange(len(z)))

    best_phase, best_score, best_points = 0, -np.inf, z[::sps]
    for phase in range(sps):
        points = z[phase::sps]
        # Power over amplitude spread. At the symbol instant the amplitudes
        # take the constellation's own discrete levels; between symbols they
        # smear across the pulse shape, which widens the spread.
        score = float(np.mean(np.abs(points) ** 2) /
                       (np.var(np.abs(points)) + 1e-9))
        if score > best_score:
            best_phase, best_score, best_points = phase, score, points
    return best_points, offset, best_phase
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui_constellation.py -v`

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_ui_constellation.py src/ui/plots.py
git commit -m "feat(ui): recover symbol points from a raw IQ window"
```

---

## Task 2: Civilian window selection

**Files:**
- Modify: `src/ui/session.py` (add a method to `CaptureSession`, directly after `judged_events`)
- Test: `tests/test_ui_constellation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_constellation.py`:

```python
from src.config import CFG, CLASSES, resolve_multilabel_thresholds
from src.timeline import TimelineResult
from src.ui.session import CaptureSession


def _session(probs_by_class, n_windows=6):
    """A CaptureSession with hand-set probabilities.

    The UI fixtures elsewhere in this suite run an UNTRAINED model, whose
    probabilities come from random weights. That is fine for asserting a page
    renders; it is useless for asserting WHICH window a selector picks.
    Setting probs directly makes the selection logic itself the thing under
    test.
    """
    window_len = hop = 512
    iq = np.concatenate([_qpsk(n_symbols=64, seed=i) for i in range(n_windows)])
    probs = np.full((n_windows, len(CLASSES)), 0.01, dtype=np.float32)
    for cls, column in probs_by_class.items():
        probs[:, CLASSES.index(cls)] = column
    result = TimelineResult(
        probs=probs, starts=np.arange(n_windows) * hop,
        attn=np.zeros((n_windows, window_len), dtype=np.float32),
        hop=hop, window_len=window_len, fs=CFG["signal"]["fs"])
    return CaptureSession(
        iq=iq, result=result, source="scenario", noise_power=0.01,
        thresholds=dict(zip(CLASSES, resolve_multilabel_thresholds())))


def test_selector_picks_the_strongest_civilian_window():
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    index, cls, prob = s.best_civilian_window()
    assert (index, cls) == (3, "QPSK")
    assert prob == pytest.approx(0.95, abs=1e-6)


def test_selector_prefers_the_strongest_class_not_the_first():
    """CIVILIAN is iterated in class order, so a selector that returned the
    first class over threshold would answer BPSK here and be wrong."""
    s = _session({"BPSK": [0.40] * 6,
                   "16QAM": [0.10, 0.10, 0.99, 0.10, 0.10, 0.10]})
    s.display_smoothed = False
    index, cls, prob = s.best_civilian_window()
    assert (index, cls) == (2, "16QAM")
    assert prob == pytest.approx(0.99, abs=1e-6)


def test_selector_returns_none_when_no_civilian_clears_threshold():
    """A radar-only capture has no civilian window, and the panel must be
    hidden rather than showing the noise floor as a constellation."""
    s = _session({"LFM_RADAR": [0.90] * 6})
    s.display_smoothed = False
    assert s.best_civilian_window() is None


def test_selector_follows_the_sessions_display_mode():
    """Every page reads one view. Smoothing damps the spike but must not move
    the pick off the window that carries it."""
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = True
    index, cls, prob = s.best_civilian_window()
    assert (index, cls) == (3, "QPSK")
    assert prob < 0.95           # smoothed, so damped below the raw peak


def test_selector_handles_a_capture_with_no_windows():
    s = _session({"QPSK": [0.95] * 6})
    s.result.probs = s.result.probs[:0]
    s.result.starts = s.result.starts[:0]
    assert s.best_civilian_window() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_constellation.py -k selector -v`

Expected: FAIL with `AttributeError: 'CaptureSession' object has no attribute 'best_civilian_window'`

- [ ] **Step 3: Implement the selector**

In `src/ui/session.py`, add to `CaptureSession` directly after `judged_events`:

```python
    def best_civilian_window(self, smoothed=None):
        """The window carrying the strongest civilian evidence.

        Returns (index, class_name, probability), or None when no window
        clears its class threshold -- which is what a radar-only or empty
        capture looks like, and what tells the page to hide the constellation
        panel entirely rather than plot a noise floor as a constellation.

        Strongest across ALL civilian classes, not the first one over
        threshold: BPSK sits first in CLASSES, so first-match would answer
        BPSK for a capture whose actual emitter is 16QAM.
        """
        smoothed = self.display_smoothed if smoothed is None else smoothed
        probs = self._resolved(smoothed).probs
        if not len(probs):
            return None

        best = None
        for cls in CIVILIAN:
            column = probs[:, CLASSES.index(cls)]
            index = int(np.argmax(column))
            prob = float(column[index])
            if prob < self.thresholds.get(cls, 0.5):
                continue
            if best is None or prob > best[2]:
                best = (index, cls, prob)
        return best
```

`CIVILIAN`, `CLASSES` and `np` are already imported at the top of `src/ui/session.py`; no import changes are needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui_constellation.py -v`

Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_ui_constellation.py src/ui/session.py
git commit -m "feat(ui): pick the strongest civilian window for the constellation"
```

---

## Task 3: The constellation figure

**Files:**
- Modify: `src/ui/plots.py` (add directly after `recover_symbols`)
- Test: `tests/test_ui_constellation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_constellation.py`:

```python
from src.ui.palette import INSTRUMENT, tier_color
from src.ui.plots import constellation_figure


def test_figure_is_none_when_there_is_no_civilian_window():
    s = _session({"LFM_RADAR": [0.90] * 6})
    s.display_smoothed = False
    assert constellation_figure(s) is None


def test_figure_has_two_square_axes():
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        assert len(fig.axes) == 2
        for ax in fig.axes:
            assert ax.get_aspect() == 1.0
    finally:
        plt.close(fig)


def test_raw_axis_plots_every_sample_and_symbol_axis_one_per_symbol():
    """The left panel is the model's actual input; the right is one point per
    symbol. If they ever plot the same count, the decimation silently stopped
    happening."""
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        raw_ax, sym_ax = fig.axes
        assert raw_ax.collections[0].get_offsets().shape[0] == 512
        assert (sym_ax.collections[0].get_offsets().shape[0]
                 == 512 // SAMPLES_PER_SYMBOL)
    finally:
        plt.close(fig)


def test_scatter_points_carry_measured_styling_not_a_tier_colour():
    """Provenance rule: both panels are computed from the capture's own
    samples, so they must not wear the colour that marks model output."""
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        expected = matplotlib.colors.to_rgb(INSTRUMENT["color"])
        for ax in fig.axes:
            colour = ax.collections[0].get_facecolor()[0]
            assert np.allclose(colour[:3], expected)
    finally:
        plt.close(fig)


def test_caption_names_the_class_the_window_and_the_recovery_chain():
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        captions = " ".join(t.get_text() for t in fig.texts)
        assert "QPSK" in captions
        assert "window 3" in captions
        assert "de-rotate" in captions
        assert "64QAM" in captions        # the point-count caveat
        model_text = [t for t in fig.texts if "QPSK" in t.get_text()]
        assert any(t.get_color() == tier_color("Civilian") for t in model_text)
    finally:
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_constellation.py -k figure -v`

Expected: collection error — `ImportError: cannot import name 'constellation_figure' from 'src.ui.plots'`

- [ ] **Step 3: Implement the figure**

In `src/ui/plots.py`, add directly after `recover_symbols`:

```python
def constellation_figure(session, smoothed=None):
    """IQ constellation for the strongest civilian window, or None.

    Why this panel exists: the waterfall cannot tell civilian modulations
    apart. BPSK, QPSK, 16QAM and 64QAM are the same flat wideband smear on it
    at every SNR. Cluster count IS the modulation order -- 2, 4, 16, 64 -- so
    this is the one display that carries the distinction.

    Two axes, both MEASURED. The left is the exact (2, 512) array the model is
    fed. The right is the SAME samples through recover_symbols: unit-power
    scaling, de-rotation, decimation. Neither is model output, so neither
    wears a tier colour. The single MODEL element is the caption's detected
    class, which does.

    Deliberately ONE window. Pooling several would give more points, but the
    4th-power carrier estimate leaves a 90-degree ambiguity per window, so
    pooled BPSK would render four clusters instead of two -- the display would
    assert the wrong modulation order. The caption states the resulting point
    limit instead.

    Returns None when no window carries a civilian class above threshold; the
    page hides the component rather than drawing an empty panel.
    """
    pick = session.best_civilian_window(smoothed)
    if pick is None:
        return None
    index, class_name, prob = pick

    start = int(session.result.starts[index])
    window = session.iq[start:start + session.result.window_len]
    points, offset, phase = recover_symbols(window)
    raw = window / np.sqrt(np.mean(np.abs(window) ** 2) + 1e-20)

    fig, (ax_raw, ax_sym) = plt.subplots(1, 2, figsize=(9.5, 5.0))
    ax_raw.scatter(raw.real, raw.imag, s=4, alpha=0.45, linewidths=0,
                    color=INSTRUMENT["color"])
    ax_raw.set_title(f"raw I/Q — {len(raw)} samples, as the model is fed",
                      fontsize=8, color=TEXT_DIM)
    ax_sym.scatter(points.real, points.imag, s=20, alpha=0.85, linewidths=0,
                    color=INSTRUMENT["color"])
    ax_sym.set_title(f"recovered — {len(points)} symbol points",
                      fontsize=8, color=TEXT_DIM)

    for ax in (ax_raw, ax_sym):
        ax.set_xlabel("I (measured)")
        ax.set_ylabel("Q (measured)")
        # Equal aspect, or a QPSK square renders as a rectangle and the eye
        # reads a constellation that is not there.
        ax.set_aspect("equal")

    t_ms = start / CFG["signal"]["fs"] * 1e3
    snr_text = f"est. {estimate_snr_db(window, session.noise_power):.1f} dB"
    fig.text(0.01, 0.055,
              f"window {index} @ {t_ms:.2f} ms · {snr_text} · unit-power scale "
              f"→ de-rotate {offset:+.4f} cyc/sample → decimate 1-in-"
              f"{SAMPLES_PER_SYMBOL} at phase {phase}",
              color=TEXT_DIM, fontsize=7)
    fig.text(0.01, 0.005, f"{class_name} {prob * 100:.0f}%",
              color=tier_color("Civilian"), fontsize=8, fontweight="bold")
    fig.text(0.16, 0.005,
              f"cluster count is the modulation order — {len(points)} symbols "
              f"separates 2 clusters from 4, not enough to resolve 64QAM",
              color=TEXT_DIM, fontsize=7)

    style_axes(fig, [ax_raw, ax_sym])
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    return fig
```

`CFG`, `estimate_snr_db`, `INSTRUMENT`, `TEXT_DIM`, `style_axes` and `tier_color` are already imported at the top of `src/ui/plots.py`; no import changes are needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui_constellation.py -v`

Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_ui_constellation.py src/ui/plots.py
git commit -m "feat(ui): constellation panel for the strongest civilian window"
```

---

## Task 4: Wire the panel into RF Replay

**Files:**
- Modify: `src/ui/pages/rf_replay.py` (`_render`, the component list, the two no-capture fallbacks)
- Test: `tests/test_ui_constellation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_constellation.py`:

```python
def test_render_returns_a_visible_constellation_for_a_civilian_capture():
    from src.ui.pages.rf_replay import _render
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    out = _render(s, "Raw", "single")
    try:
        assert len(out) == 5
        update = out[4]
        assert update["visible"] is True
        assert update["value"] is not None
    finally:
        plt.close("all")


def test_render_hides_the_constellation_when_no_civilian_is_present():
    """A radar-only capture must look exactly as it did before this panel
    existed -- no empty grey box below the console."""
    from src.ui.pages.rf_replay import _render
    s = _session({"LFM_RADAR": [0.90] * 6})
    out = _render(s, "Raw", "single")
    try:
        assert out[4]["visible"] is False
    finally:
        plt.close("all")
```

If Gradio 6.25 returns an object rather than a dict from `gr.update`, read the fields as attributes (`update.visible`) instead — check with `python -c "import gradio as gr; print(type(gr.update(visible=True)), gr.update(visible=True))"` and write the assertions to match what it prints. Do not change the implementation to suit the test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_constellation.py -k render -v`

Expected: FAIL with `assert 4 == 5` — `_render` still returns four values

- [ ] **Step 3: Modify the page**

In `src/ui/pages/rf_replay.py`, replace the `return` at the end of `_render`:

```python
    # The constellation is a separate component, not another panel inside the
    # console figure: that figure's whole premise is one shared time axis, and
    # a constellation has no time axis at all. Hidden outright when the
    # capture has no civilian window, so military-only cases look exactly as
    # they did before this panel existed.
    constellation = plots.constellation_figure(session, smoothed=smoothed)
    constellation_update = (gr.update(value=constellation, visible=True)
                             if constellation is not None
                             else gr.update(visible=False))
    return (session, head, plots.console_figure(session, smoothed=smoothed),
            rows, constellation_update)
```

Add the component immediately after the `console = gr.Plot(...)` line:

```python
    # Starts hidden: most cases are military-only, and an empty panel below
    # the console would read as a broken plot rather than as "not applicable".
    constellation = gr.Plot(
        label="Civilian constellation — raw I/Q vs recovered symbols",
        visible=False)
```

Extend the outputs list:

```python
    outputs = [state, header, console, events, constellation]
```

Both no-capture fallbacks currently return four values. Update them to five — in `model_sel.change`:

```python
        else (s, "Load a capture first.", None, [], gr.update(visible=False)),
```

and identically in `smoothing.change`:

```python
        else (s, "Load a capture first.", None, [], gr.update(visible=False)),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui_constellation.py -v`

Expected: 18 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_ui_constellation.py src/ui/pages/rf_replay.py
git commit -m "feat(ui): show the constellation panel on civilian captures only"
```

---

## Task 5: Verify against the running app

**Files:** none modified unless a defect is found.

- [ ] **Step 1: Tidy the test file's imports**

Tasks 2 and 3 appended `from src.config import ...`, `from src.timeline import ...`, `from src.ui.session import ...`, `from src.ui.palette import ...` and `from src.ui.plots import constellation_figure` partway down `tests/test_ui_constellation.py`, because each task adds only what it needs. Move all of them into the import block at the top of the file, alphabetised alongside the existing imports, and merge the two `src.ui.plots` lines into one.

Run: `python -m pytest tests/test_ui_constellation.py -q`

Expected: 18 passed — the move is pure housekeeping and must not change a single result.

```bash
git add tests/test_ui_constellation.py
git commit -m "test: hoist the constellation test imports to the top of the file"
```

- [ ] **Step 2: Run the whole suite**

Run: `python -m pytest tests -q`

Expected: no failures. If something fails, confirm whether it also fails at `git stash` on clean HEAD before attributing it to this work.

- [ ] **Step 3: Launch the app**

Use the `preview_start` tool with the project's `.claude/launch.json` entry. Do not run the server through Bash.

- [ ] **Step 4: Check a civilian case**

On RF Replay, set Scenario case to `Civilian only`, SNR to `+10 dB`, click Synthesize scenario. Confirm:
- the constellation panel appears below the console figure
- the right axis shows visible clusters, the left a ring or blob
- the caption names QPSK, a window index, a time offset and the three recovery steps

- [ ] **Step 5: Check the low-SNR and military cases**

Re-synthesize `Civilian only` at `-10 dB`: the panel must still appear, with the clusters dissolved into a cloud — that is the honest result, not a bug.

Then switch to `Radar only` and synthesize: the constellation panel must disappear entirely.

- [ ] **Step 6: Check the display toggle**

With a `Civilian + Jamming` capture loaded, flip Display between Smoothed and Raw. The panel must update rather than freeze, and must not throw. Check `read_console_messages` and `preview_logs` for errors.

- [ ] **Step 7: Screenshot, and commit any fix**

Capture a screenshot of the civilian case for the record. If steps 4-6 surfaced a defect, fix it, add a test that fails without the fix, and commit:

```bash
git add -A
git commit -m "fix(ui): <what the app showed that the tests did not>"
```
