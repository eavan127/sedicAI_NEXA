# OMNI Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-tab Gradio UI with a six-page OMNI operator console — Overview, RF Replay, Signal Analysis, Performance, Model, Alerts — built on the core engine, with the spec's provenance rule enforced in the styling layer.

**Architecture:** `scripts/inference_ui.py` becomes a thin shim over `src/ui/app.py`. Palette and plotting are shared modules; each page is its own module holding only its Gradio layout and callbacks. A single `CaptureSession` in `gr.State` is populated once by a load action on RF Replay and read by every other page, so no page re-runs inference. Provenance is enforced by construction: `src/ui/palette.py` exposes `tier_color()` for MODEL elements and `INSTRUMENT` for MEASURED ones, and plotting functions take their colors from those, never inline.

**Tech Stack:** Gradio, matplotlib (Agg), NumPy, PyTorch. Depends on `src/timeline.py`, `src/measure.py`, `src/scenarios.py` from the core plan.

**Spec:** `docs/superpowers/specs/2026-08-24-omni-ui-design.md`

**Prerequisite:** `docs/superpowers/plans/2026-08-24-omni-core-engine.md` must be complete. Verify with `python -m pytest tests/test_timeline.py tests/test_measure.py tests/test_scenarios.py -q` before starting.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ui/__init__.py` (create) | empty package marker |
| `src/ui/palette.py` (create) | dark console palette; `tier_color()`, `INSTRUMENT`, `TRUTH_STYLE`; matplotlib axis styling |
| `src/ui/session.py` (create) | `CaptureSession` dataclass; `load_scenario`, `load_upload`, `load_test_example`; `analyze` |
| `src/ui/plots.py` (create) | `waterfall_figure`, `spectrum_figure`, `ribbon_figure`, `attention_figure` |
| `src/ui/pages/__init__.py` (create) | empty package marker |
| `src/ui/pages/rf_replay.py` (create) | load controls, hop selector, smoothing toggle, transport, waterfall + overlays |
| `src/ui/pages/signal_analysis.py` (create) | one-window inspector, 8-class list, attention |
| `src/ui/pages/overview.py` (create) | status strip, Current Window readout, mini waterfall, latest detection |
| `src/ui/pages/alerts.py` (create) | judged-class events only |
| `src/ui/pages/model_page.py` (create) | introspected model card |
| `src/ui/pages/performance.py` (create) | existing Results dashboard, ported |
| `src/ui/app.py` (create) | assembles the six tabs, owns `gr.State` |
| `scripts/inference_ui.py` (rewrite) | thin entry point |
| `tests/test_ui_session.py` (create) | covers `src/ui/session.py` (no Gradio rendering) |
| `tests/test_ui_palette.py` (create) | covers provenance guarantees in the palette |

**Verification note:** Gradio layout is not meaningfully unit-testable. Logic lives in `src/ui/session.py` and is tested with pytest; visual behaviour is verified through the browser preview tools per Task 12.

---

## Task 1: Palette with provenance guarantees

**Files:**
- Create: `src/ui/__init__.py`, `src/ui/palette.py`
- Test: `tests/test_ui_palette.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ui_palette.py`:

```python
import pytest

from src.config import TIERS
from src.ui.palette import INSTRUMENT, TIER_COLOR, TRUTH_STYLE, tier_color


def test_every_tier_has_a_color():
    for tier in TIERS:
        assert tier in TIER_COLOR


def test_tier_color_rejects_unknown_tier():
    """Silently returning grey for a typo'd tier would hide the bug on screen."""
    with pytest.raises(KeyError):
        tier_color("Nonexistent")


def test_instrument_color_is_not_a_tier_color():
    """MEASURED elements must never be styled as detections."""
    assert INSTRUMENT["color"] not in TIER_COLOR.values()


def test_truth_style_is_not_a_tier_color():
    assert TRUTH_STYLE["color"] not in TIER_COLOR.values()


def test_truth_style_is_visually_distinct_from_solid_detections():
    assert TRUTH_STYLE["linestyle"] != "solid"
    assert TRUTH_STYLE["fill"] is False


def test_tier_colors_are_all_distinct():
    assert len(set(TIER_COLOR.values())) == len(TIER_COLOR)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_palette.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ui'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ui/__init__.py` as an empty file. Create `src/ui/palette.py`:

```python
"""Dark console palette, and the mechanism that enforces the provenance rule.

The spec requires every on-screen element to be identifiably MODEL, MEASURED
or TRUTH. That is enforced here rather than by convention: plotting code takes
colors from tier_color() (MODEL), INSTRUMENT (MEASURED) or TRUTH_STYLE
(TRUTH), and never writes a hex value inline. A reviewer can then check
provenance by looking at which constant a call site used.
"""
import numpy as np

BG = "#0b0f14"
PANEL = "#121820"
GRID = "#1f2933"
TEXT = "#e6edf3"
TEXT_DIM = "#8b98a5"

# MODEL provenance. Brighter values than the light-ground UI used, matching
# the Overwatch waterfall artifact's tier hues on a dark ground.
TIER_COLOR = {
    "Civilian": "#4fd1c5",
    "Military": "#f6ad55",
    "Hostile": "#fc8181",
    "Empty": "#4a5568",
}

# MEASURED provenance. Deliberately outside the tier hues so a waterfall,
# spectrum trace or SNR readout can never be mistaken for a detection.
INSTRUMENT = {"color": "#9aa5b1", "linewidth": 1.0}

# TRUTH provenance. Outline only, dashed, never filled -- ground truth must be
# impossible to confuse with something the model produced.
TRUTH_STYLE = {"color": "#cbd5e0", "linestyle": "dashed", "fill": False,
                "linewidth": 1.2}

# Perceptually better than jet, reads almost identically on a waterfall.
WATERFALL_CMAP = "turbo"


def tier_color(tier):
    """Color for a MODEL element of the given tier.

    Raises on an unknown tier rather than falling back to grey: a fallback
    would render a typo as a plausible-looking Empty cell and hide the bug.
    """
    return TIER_COLOR[tier]


def style_axes(fig, axes):
    """Apply the dark console palette to a figure this package owns."""
    fig.patch.set_facecolor(BG)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT)
        ax.grid(color=GRID, alpha=0.5, linewidth=0.6)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui_palette.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/ui/__init__.py src/ui/palette.py tests/test_ui_palette.py
git commit -m "feat(ui): dark palette with provenance-enforcing color constants"
```

---

## Task 2: CaptureSession

**Files:**
- Create: `src/ui/session.py`
- Test: `tests/test_ui_session.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ui_session.py`:

```python
import numpy as np
import pytest

from src.config import CLASSES
from src.models.amc_cnn import AMC_CNN
from src.ui.session import CaptureSession, analyze, load_scenario


@pytest.fixture(scope="module")
def model():
    m = AMC_CNN(num_classes=len(CLASSES), input_len=512)
    m.eval()
    return m


def test_load_scenario_returns_a_session_with_truth(model):
    s = load_scenario(model, total_duration=0.01, hop=256, seed=0)
    assert isinstance(s, CaptureSession)
    assert s.source == "scenario"
    assert s.truth is not None and len(s.truth) > 0
    assert s.snr_known is True


def test_scenario_session_has_a_result_covering_the_capture(model):
    s = load_scenario(model, total_duration=0.01, hop=256, seed=0)
    assert s.result.n_windows == 1 + (32_000 - 512) // 256


def test_upload_session_has_no_truth_and_unknown_snr(model):
    iq = np.random.randn(4096) + 1j * np.random.randn(4096)
    s = analyze(iq, model, source="upload", hop=256)
    assert s.truth is None
    assert s.snr_known is False


def test_truth_is_never_present_for_a_non_scenario_source(model):
    """Provenance rule: TRUTH elements must not exist unless the capture was
    generated by us."""
    iq = np.random.randn(2048) + 1j * np.random.randn(2048)
    for source in ("upload", "test-example"):
        assert analyze(iq, model, source=source, hop=512).truth is None


def test_session_stores_the_raw_unnormalized_capture(model):
    """The waterfall and SNR estimate depend on real amplitude surviving."""
    iq = (np.random.randn(2048) + 1j * np.random.randn(2048)) * 50.0
    s = analyze(iq, model, source="upload", hop=512)
    assert np.mean(np.abs(s.iq) ** 2) > 100.0


def test_detected_events_use_per_class_thresholds(model):
    """The session must not fall back to a flat 0.5 -- that is the bug this
    work fixes."""
    s = load_scenario(model, total_duration=0.005, hop=512, seed=0)
    assert s.thresholds["LFM_RADAR"] == pytest.approx(0.26)
    assert s.thresholds["JAMMING"] == pytest.approx(0.77)


def test_smoothed_and_raw_events_are_both_available(model):
    s = load_scenario(model, total_duration=0.01, hop=256, seed=0)
    assert isinstance(s.events(smoothed=True), list)
    assert isinstance(s.events(smoothed=False), list)


def test_window_count_cap_is_enforced(model):
    with pytest.raises(ValueError, match="too many windows"):
        analyze(np.zeros(3_000_000, dtype=complex), model, source="upload",
                hop=64, max_windows=4000)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ui.session'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ui/session.py`:

```python
"""One loaded capture, shared across every page.

Populated once by a load action on RF Replay and stored in gr.State. Every
other page reads it. No page re-runs inference -- a 0.1 s capture at hop 256
is ~1,250 forward passes, and doing that per tab switch would make the console
unusable.
"""
from dataclasses import dataclass, field

import numpy as np

from src.config import CFG, CLASSES, resolve_multilabel_thresholds
from src.measure import noise_floor_power
from src.scenarios import build_scenario
from src.timeline import classify_capture, detections, smooth, tier_track

MAX_WINDOWS = 4000


@dataclass
class CaptureSession:
    iq: np.ndarray          # raw, NOT normalized
    result: object          # TimelineResult, unsmoothed
    source: str             # "upload" | "scenario" | "test-example"
    truth: list = None      # ScenarioSegment list, scenario only
    snr_known: bool = False
    true_snr_db: float = None
    noise_power: float = 1.0
    thresholds: dict = field(default_factory=dict)
    smoothing_alpha: float = 0.3

    def smoothed_result(self):
        return smooth(self.result, alpha=self.smoothing_alpha)

    def events(self, smoothed=True):
        r = self.smoothed_result() if smoothed else self.result
        return detections(r, self.thresholds)

    def tiers(self, smoothed=True):
        r = self.smoothed_result() if smoothed else self.result
        return tier_track(r, self.thresholds)

    def judged_events(self, smoothed=True):
        """Events involving a judged class. NOISE_FLOOR can never appear --
        it is the absence of an emitter, so an alert on it would invert the
        purpose of both the Alerts page and the class."""
        judged = set(CFG["judged_classes"])
        return [e for e in self.events(smoothed) if judged & set(e.classes)]


def analyze(iq, model, source, hop=None, truth=None, true_snr_db=None,
            max_windows=MAX_WINDOWS):
    """Run inference over a capture and package the result.

    `truth` is accepted only for scenario captures; any other source has it
    forced to None, so a TRUTH overlay cannot be rendered over data we do not
    actually have ground truth for.
    """
    hop = hop or CFG["signal"]["window_len"]
    window_len = CFG["signal"]["window_len"]

    n_windows = 1 + max(len(iq) - window_len, 0) // hop
    if n_windows > max_windows:
        raise ValueError(
            f"too many windows: {n_windows} at hop {hop} exceeds the "
            f"{max_windows} cap. Use a larger hop or a shorter capture."
        )

    result = classify_capture(iq, model, hop=hop)
    thresholds = dict(zip(CLASSES, resolve_multilabel_thresholds()))

    if source != "scenario":
        truth = None
        true_snr_db = None

    return CaptureSession(
        iq=np.asarray(iq), result=result, source=source, truth=truth,
        snr_known=true_snr_db is not None, true_snr_db=true_snr_db,
        noise_power=noise_floor_power(iq), thresholds=thresholds,
    )


def load_scenario(model, total_duration=0.1, hop=None, snr_db=-6, seed=0):
    iq, segments = build_scenario(total_duration=total_duration,
                                   snr_db=snr_db, seed=seed)
    return analyze(iq, model, source="scenario", hop=hop, truth=segments,
                    true_snr_db=snr_db)


def load_upload(path, model, hop=None):
    """Interleaved float32 I,Q,I,Q,... -- same contract as src/infer.py."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2:
        raw = raw[:-1]
    iq = raw[0::2] + 1j * raw[1::2]
    return analyze(iq, model, source="upload", hop=hop)


def load_test_example(model, X, y, snr_labels, idx):
    """One held-out example: exactly 512 samples, so exactly one window and no
    timeline. Pages must render correctly in that degenerate case."""
    arr = X[idx]
    iq = arr[0] + 1j * arr[1]
    session = analyze(iq, model, source="test-example",
                       hop=CFG["signal"]["window_len"])
    session.snr_known = True
    session.true_snr_db = float(snr_labels[idx])
    return session
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui_session.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/ui/session.py tests/test_ui_session.py
git commit -m "feat(ui): CaptureSession shared across pages"
```

---

## Task 3: Plotting module

**Files:**
- Create: `src/ui/plots.py`

- [ ] **Step 1: Write the implementation**

No unit tests — these produce matplotlib figures, verified visually in Task 12. Create `src/ui/plots.py`:

```python
"""Figures for the OMNI console.

Colors come from src/ui/palette.py, never inline, so provenance is checkable
at each call site: tier_color() marks MODEL, INSTRUMENT marks MEASURED,
TRUTH_STYLE marks TRUTH.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from src.config import CFG
from src.measure import power_spectrum_db
from src.ui.palette import (INSTRUMENT, TEXT_DIM, TRUTH_STYLE,
                             WATERFALL_CMAP, style_axes, tier_color)


def waterfall_figure(session, smoothed=True, nperseg=256):
    """Waterfall with x = frequency (MHz) and y = time, matching the classic
    RF-console convention, plus MODEL detection overlays.

    Overlays span the FULL width deliberately. The classifier has no frequency
    axis -- STFTBranch collapses it via f.mean(dim=2) -- so a box bounded in
    frequency would assert something the model never computed. A full-width
    box bounded in time is exactly the claim the model can support.
    """
    fs = CFG["signal"]["fs"]
    f, t, Z = stft(session.iq, fs=fs, nperseg=nperseg, return_onesided=False)
    order = np.argsort(np.fft.fftshift(f))
    freqs_mhz = np.fft.fftshift(f)[order] / 1e6
    power_db = 10 * np.log10(np.abs(np.fft.fftshift(Z, axes=0)[order]) ** 2 + 1e-20)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.pcolormesh(freqs_mhz, t * 1e6, power_db.T, shading="auto",
                   cmap=WATERFALL_CMAP)
    ax.set_xlabel("frequency (MHz) — BASEBAND")
    ax.set_ylabel("time (µs)")
    ax.invert_yaxis()                      # time runs downward, waterfall style

    # MODEL: full-width, time-bounded detection boxes.
    from src.timeline import tier_of_classes
    for event in session.events(smoothed=smoothed):
        color = tier_color(tier_of_classes(event.classes))
        ax.add_patch(mpatches.Rectangle(
            (freqs_mhz[0], event.start_us),
            freqs_mhz[-1] - freqs_mhz[0], event.duration_us,
            fill=False, edgecolor=color, linewidth=1.4))
        ax.text(freqs_mhz[0], event.start_us, f" {event.label}",
                 color=color, fontsize=8, va="top", fontweight="bold")

    # TRUTH: scenario only, dashed outline, never filled.
    if session.truth:
        for seg in session.truth:
            ax.add_patch(mpatches.Rectangle(
                (freqs_mhz[0], seg.start_s * 1e6),
                freqs_mhz[-1] - freqs_mhz[0], (seg.end_s - seg.start_s) * 1e6,
                fill=TRUTH_STYLE["fill"], edgecolor=TRUTH_STYLE["color"],
                linestyle=TRUTH_STYLE["linestyle"],
                linewidth=TRUTH_STYLE["linewidth"]))
            ax.text(freqs_mhz[-1], seg.start_s * 1e6, f"TRUTH {seg.class_name} ",
                     color=TRUTH_STYLE["color"], fontsize=7, ha="right", va="top")

    style_axes(fig, ax)
    plt.tight_layout()
    return fig


def spectrum_figure(session):
    """MEASURED average power spectrum. Instrument styling only."""
    freqs, spectrum = power_spectrum_db(session.iq)
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.fill_between(freqs / 1e6, spectrum, spectrum.min(),
                     color=INSTRUMENT["color"], alpha=0.35)
    ax.plot(freqs / 1e6, spectrum, color=INSTRUMENT["color"],
             linewidth=INSTRUMENT["linewidth"])
    ax.set_ylabel("dB")
    ax.set_xlabel("")
    ax.set_xlim(freqs.min() / 1e6, freqs.max() / 1e6)
    style_axes(fig, ax)
    plt.tight_layout()
    return fig


def ribbon_figure(session, smoothed=True):
    """MODEL tier ribbon, one cell per window."""
    tiers = session.tiers(smoothed=smoothed)
    colors = [tier_color(t) for t in tiers]
    fig, ax = plt.subplots(figsize=(1.2, 6))
    for i, color in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((0, i), 1, 1, color=color))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(len(colors), 1))
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_ylabel("window")
    style_axes(fig, ax)
    plt.tight_layout()
    return fig


def attention_figure(session, window_index):
    """MODEL attention over the raw amplitude of one window.

    Attention is a per-window softmax, so the curve is that window's own
    distribution -- heights are NOT comparable across windows. The axis label
    says so, because the plot itself cannot.
    """
    result = session.result
    start = int(result.starts[window_index])
    window = session.iq[start:start + result.window_len]
    t_us = np.arange(len(window)) / CFG["signal"]["fs"] * 1e6

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t_us, np.abs(window), color=INSTRUMENT["color"],
             linewidth=INSTRUMENT["linewidth"], label="|IQ| (measured)")
    ax.set_xlabel("time within window (µs)")
    ax.set_ylabel("amplitude")

    twin = ax.twinx()
    weights = result.attn[window_index]
    twin.fill_between(t_us, weights, 0, color=tier_color("Military"),
                       alpha=0.35, label="attention (model)")
    twin.set_ylabel("attention — relative within this window only")
    twin.tick_params(colors=TEXT_DIM, labelsize=8)
    twin.yaxis.label.set_color(TEXT_DIM)

    style_axes(fig, ax)
    plt.tight_layout()
    return fig
```

- [ ] **Step 2: Verify the figures build without error**

Run:

```bash
python -c "
from src.config import CLASSES
from src.models.amc_cnn import AMC_CNN
from src.ui.session import load_scenario
from src.ui import plots
m = AMC_CNN(num_classes=len(CLASSES), input_len=512); m.eval()
s = load_scenario(m, total_duration=0.01, hop=256, seed=0)
for fn in (plots.waterfall_figure, plots.spectrum_figure, plots.ribbon_figure):
    print(fn.__name__, fn(s))
print('attention', plots.attention_figure(s, 0))
"
```

Expected: four lines each printing a `Figure(...)` object, no exception

- [ ] **Step 3: Commit**

```bash
git add src/ui/plots.py
git commit -m "feat(ui): waterfall, spectrum, ribbon and attention figures"
```

---

## Task 4: RF Replay page

**Files:**
- Create: `src/ui/pages/__init__.py`, `src/ui/pages/rf_replay.py`

- [ ] **Step 1: Write the implementation**

Create `src/ui/pages/__init__.py` as an empty file. Create `src/ui/pages/rf_replay.py`:

```python
"""RF Replay — the replay deck.

Named Replay, not Live: there is no SDR. The page name states that rather than
relying on a badge to walk back a misleading title. OmniSIG itself supports
recorded-file playback, so this is the same mode, not a lesser one.
"""
import gradio as gr

from src.ui import plots
from src.ui.session import load_scenario, load_upload

HOP_CHOICES = [("no overlap (512)", 512), ("50% (256)", 256),
                ("75% (128)", 128), ("87.5% (64)", 64)]


def build(state, get_model):
    """Returns the components other pages need to refresh from."""
    gr.Markdown("### RF Replay")
    gr.Markdown(
        "`● REPLAY` — recorded or synthesized capture, baseband, fs 3.2 MHz. "
        "There is no live SDR ingest."
    )

    with gr.Row():
        with gr.Column(scale=1):
            scenario_btn = gr.Button("Synthesize scenario", variant="primary")
            file_in = gr.File(label="…or upload raw IQ (interleaved float32)")
            upload_btn = gr.Button("Analyze upload")
            hop = gr.Dropdown(choices=HOP_CHOICES, value=256,
                               label="Window hop (overlap)")
            smoothing = gr.Radio(choices=["Smoothed", "Raw"], value="Smoothed",
                                  label="Display")
            gr.Markdown(
                "<small>Smoothing is display-only. Benchmark numbers on the "
                "Performance page are always per-window and unsmoothed.</small>"
            )
            header = gr.Markdown()
        with gr.Column(scale=3):
            spectrum = gr.Plot(label="Power spectrum (measured)")
            with gr.Row():
                waterfall = gr.Plot(label="Waterfall (measured) + detections (model)")
                ribbon = gr.Plot(label="Tier")
            events = gr.Dataframe(
                headers=["#", "Start (ms)", "Duration (ms)", "Detected", "Peak"],
                label="Detection events", interactive=False)

    def _render(session, smoothing_choice):
        smoothed = smoothing_choice == "Smoothed"
        rows = [
            [i + 1, f"{e.start_us / 1000:.2f}", f"{e.duration_us / 1000:.2f}",
             e.label, " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes)]
            for i, e in enumerate(session.events(smoothed=smoothed))
        ]
        duration_ms = len(session.iq) / 3_200_000 * 1000
        head = (f"**● REPLAY** — source `{session.source}` · "
                f"{duration_ms:.1f} ms · {session.result.n_windows} windows · "
                f"{len(rows)} events")
        return (session, head,
                plots.spectrum_figure(session),
                plots.waterfall_figure(session, smoothed=smoothed),
                plots.ribbon_figure(session, smoothed=smoothed),
                rows)

    outputs = [state, header, spectrum, waterfall, ribbon, events]

    scenario_btn.click(
        lambda h, sm: _render(load_scenario(get_model(), total_duration=0.1,
                                             hop=h), sm),
        inputs=[hop, smoothing], outputs=outputs)

    upload_btn.click(
        lambda f, h, sm: _render(
            load_upload(f.name if hasattr(f, "name") else f, get_model(), hop=h),
            sm),
        inputs=[file_in, hop, smoothing], outputs=outputs)

    smoothing.change(
        lambda s, sm: _render(s, sm) if s else (s, "", None, None, None, []),
        inputs=[state, smoothing], outputs=outputs)

    return state
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/pages/__init__.py src/ui/pages/rf_replay.py
git commit -m "feat(ui): RF Replay page"
```

---

## Task 5: Signal Analysis page

**Files:**
- Create: `src/ui/pages/signal_analysis.py`

- [ ] **Step 1: Write the implementation**

Create `src/ui/pages/signal_analysis.py`:

```python
"""Signal Analysis — inspect one 512-sample window.

This is where the window mechanism is stated plainly: the console looks
continuous, but the classifier decides one 160 µs window at a time.
"""
import gradio as gr
import numpy as np

from src.config import CFG, CLASSES
from src.measure import estimate_snr_db
from src.ui import plots
from src.ui.palette import TIER_COLOR


def _probability_html(session, window_index):
    """All 8 classes, each marked against its OWN threshold.

    Not a ranked single-winner list: the model is multi-label sigmoid, so
    QPSK + JAMMING is a legitimate answer and the column does not sum to 100%.
    NOISE_FLOOR is presented separately as a channel state, not as an eighth
    threat class.
    """
    probs = session.result.probs[window_index]
    rows = []
    for i, cls in enumerate(CLASSES):
        if cls == "NOISE_FLOOR":
            continue
        hit = probs[i] > session.thresholds[cls]
        mark = "✓" if hit else "○"
        color = "#e6edf3" if hit else "#8b98a5"
        rows.append(
            f'<div style="font-family:monospace;color:{color};">'
            f'{mark} {cls:<12} {probs[i]:.2f}</div>')

    noise_i = CLASSES.index("NOISE_FLOOR")
    noise_p = probs[noise_i]
    quiet = noise_p > session.thresholds["NOISE_FLOOR"]
    noise_block = (
        f'<div style="margin-top:12px;padding-top:8px;'
        f'border-top:1px solid #1f2933;font-family:monospace;">'
        f'{"✓" if quiet else "○"} NOISE_FLOOR  {noise_p:.2f}<br>'
        f'<span style="color:#8b98a5;">Signal state: '
        f'{"QUIET / NO SIGNAL" if quiet else "ACTIVE"}</span></div>')

    return (f'<div style="background:#121820;padding:14px;border-radius:6px;">'
            f'<div style="color:#8b98a5;font-size:11px;margin-bottom:8px;">'
            f'independent probabilities · multi-label — these do not sum to 100%'
            f'</div>{"".join(rows)}{noise_block}</div>')


def _metadata_html(session, window_index):
    result = session.result
    start = int(result.starts[window_index])
    if session.snr_known:
        snr = f"{session.true_snr_db:.1f} dB <span style='color:#8b98a5;'>KNOWN</span>"
    else:
        window = session.iq[start:start + result.window_len]
        snr = f"est. {estimate_snr_db(window, session.noise_power):.1f} dB"
    return (
        f'<div style="font-family:monospace;color:#e6edf3;background:#121820;'
        f'padding:14px;border-radius:6px;">'
        f'WINDOW   #{window_index + 1} / {result.n_windows}<br>'
        f'OFFSET   {start / CFG["signal"]["fs"] * 1000:.3f} ms<br>'
        f'SAMPLES  {result.window_len}<br>'
        f'DURATION {result.window_duration_us:.0f} µs<br>'
        f'SNR      {snr}</div>')


def build(state):
    gr.Markdown("### Signal Analysis")
    gr.Markdown(
        "One 512-sample window — 160 µs — exactly as the classifier sees it. "
        "Load a capture on RF Replay first."
    )
    index = gr.Slider(1, 2, value=1, step=1, label="Window")
    with gr.Row():
        with gr.Column():
            probs_out = gr.HTML()
            meta_out = gr.HTML()
        with gr.Column(scale=2):
            attn_out = gr.Plot(label="Amplitude (measured) + attention (model)")

    def _render(session, i):
        if session is None:
            return "Load a capture on RF Replay first.", "", None
        idx = int(i) - 1
        idx = max(0, min(idx, session.result.n_windows - 1))
        return (_probability_html(session, idx), _metadata_html(session, idx),
                plots.attention_figure(session, idx))

    index.change(_render, inputs=[state, index],
                  outputs=[probs_out, meta_out, attn_out])
    return index
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/pages/signal_analysis.py
git commit -m "feat(ui): Signal Analysis page with multi-label probability list"
```

---

## Task 6: Overview page

**Files:**
- Create: `src/ui/pages/overview.py`

- [ ] **Step 1: Write the implementation**

Create `src/ui/pages/overview.py`:

```python
"""Overview — the at-a-glance page.

Status fields have fixed provenance. `Occupancy` is MEASURED (from the STFT);
`Detections` is MODEL (grouped events). `Channel Load` is deliberately NOT
used: in an RF console that name reads as an energy measurement, but the
obvious implementation here would be model output wearing a measurement's
name.
"""
import time

import gradio as gr

from src.measure import occupancy
from src.ui import plots


def build(state):
    gr.Markdown("### Overview")
    status = gr.HTML()
    with gr.Row():
        mini = gr.Plot(label="Waterfall")
        latest = gr.HTML()
    refresh = gr.Button("Refresh from loaded capture")

    def _render(session):
        if session is None:
            return "Load a capture on RF Replay first.", None, ""

        started = time.perf_counter()
        occ = occupancy(session.iq)
        elapsed = max(time.perf_counter() - started, 1e-9)

        events = session.events()
        status_html = (
            '<div style="font-family:monospace;background:#121820;padding:14px;'
            'border-radius:6px;color:#e6edf3;">'
            f'Occupancy   {occ * 100:.1f}%  '
            '<span style="color:#8b98a5;">measured</span><br>'
            f'Detections  {len(events)}  '
            '<span style="color:#8b98a5;">model · grouped events</span><br>'
            f'Windows     {session.result.n_windows}  '
            f'<span style="color:#8b98a5;">hop {session.result.hop}</span><br>'
            f'Analysis    {elapsed * 1000:.0f} ms  '
            '<span style="color:#8b98a5;">measured</span>'
            '</div>')

        if events:
            e = events[-1]
            latest_html = (
                '<div style="background:#121820;padding:14px;border-radius:6px;'
                'color:#e6edf3;">'
                f'<div style="font-size:20px;font-weight:600;">{e.label}</div>'
                f'<div style="color:#8b98a5;font-family:monospace;">'
                f'{e.start_us / 1000:.2f} ms · {e.duration_us / 1000:.2f} ms · '
                + " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes)
                + '</div></div>')
        else:
            latest_html = (
                '<div style="background:#121820;padding:14px;border-radius:6px;'
                'color:#8b98a5;">No detections in this capture.</div>')

        return status_html, plots.waterfall_figure(session), latest_html

    refresh.click(_render, inputs=state, outputs=[status, mini, latest])
    return refresh
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/pages/overview.py
git commit -m "feat(ui): Overview page with provenance-labelled status strip"
```

---

## Task 7: Alerts page

**Files:**
- Create: `src/ui/pages/alerts.py`

- [ ] **Step 1: Write the implementation**

Create `src/ui/pages/alerts.py`:

```python
"""Alerts — judged classes only.

NOISE_FLOOR can never raise an alert. It denotes the ABSENCE of an emitter;
"alert: nothing is transmitting" would invert the purpose of both this page
and the class. session.judged_events() enforces that.
"""
import gradio as gr

from src.config import CFG


def build(state):
    gr.Markdown("### Alerts")
    gr.Markdown(
        f"Events involving a judged class: "
        f"{', '.join(CFG['judged_classes'])}. "
        "NOISE_FLOOR never raises an alert — it is the absence of an emitter."
    )
    table = gr.Dataframe(
        headers=["Tier", "Start (ms)", "Duration (ms)", "Detected", "Peak"],
        label="Alerts", interactive=False)
    refresh = gr.Button("Refresh from loaded capture")

    def _render(session):
        if session is None:
            return []
        from src.timeline import tier_of_classes
        rows = []
        for e in session.judged_events():
            tier = tier_of_classes(e.classes)
            rows.append([
                tier, f"{e.start_us / 1000:.2f}", f"{e.duration_us / 1000:.2f}",
                e.label,
                " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes),
            ])
        return rows

    refresh.click(_render, inputs=state, outputs=table)
    return refresh
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/pages/alerts.py
git commit -m "feat(ui): Alerts page, judged classes only"
```

---

## Task 8: Model page

**Files:**
- Create: `src/ui/pages/model_page.py`

- [ ] **Step 1: Write the implementation**

Create `src/ui/pages/model_page.py`:

```python
"""Model — describes the checkpoint ACTUALLY LOADED.

Every number is introspected from the live model object and CFG at render
time. Nothing is hardcoded, so the page cannot go stale and cannot misreport
after someone swaps in a different checkpoint.

The page states what is running. It does not claim this architecture is the
best-performing one.
"""
import gradio as gr

from src.config import CFG, CLASSES


def build(get_model):
    gr.Markdown("### Model")
    card = gr.HTML()
    refresh = gr.Button("Read loaded checkpoint")

    def _render():
        model = get_model()
        total = sum(p.numel() for p in model.parameters())
        per_branch = {
            name: sum(p.numel() for p in getattr(model, name).parameters())
            for name in ("iq_branch", "stft_branch") if hasattr(model, name)
        }
        window_len = CFG["signal"]["window_len"]
        fs = CFG["signal"]["fs"]
        branches = "<br>".join(
            f"  {n:<12} {c:,}" for n, c in per_branch.items()) or "  (single branch)"
        return (
            '<div style="font-family:monospace;background:#121820;padding:16px;'
            'border-radius:6px;color:#e6edf3;line-height:1.7;">'
            f'ARCHITECTURE   {type(model).__name__}<br>'
            f'PARAMETERS     {total:,}<br>{branches}<br>'
            f'CLASSES        {len(CLASSES)} — {", ".join(CLASSES)}<br>'
            f'INPUT          (2, {window_len})<br>'
            f'WINDOW         {window_len / fs * 1e6:.0f} µs @ {fs / 1e6:.1f} MHz<br>'
            f'OUTPUT         sigmoid, multi-label (independent per class)<br>'
            f'POOLING        energy-gated attention<br>'
            f'SAMPLING       SNR-weighted, 10^(-SNR/20)<br>'
            '<span style="color:#8b98a5;">'
            'Read from the loaded checkpoint at render time, not hardcoded.'
            '</span></div>')

    refresh.click(_render, inputs=None, outputs=card)
    return refresh
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/pages/model_page.py
git commit -m "feat(ui): Model page introspecting the loaded checkpoint"
```

---

## Task 9: Performance page

**Files:**
- Create: `src/ui/pages/performance.py`

**Parity requirement — read before writing any code.**

Performance is the one page that must never become a second evaluation
implementation. It **displays** what `src/evaluate.py` produces; it does not
recompute anything.

    src/evaluate.py
          |
          +-- metrics
          +-- confusion_matrix.png
          +-- accuracy_vs_snr.png
                    |
                    v
             Performance page

NOT:

    Performance page
          |
          +-- independently recalculates metrics

If this page ever derives its own recall, FAR or confusion matrix, a judge can
be shown numbers that disagree with the official scorecard. That is a worse
failure than the page not existing.

Binding rules for this task:

1. Inspect and preserve the existing `build_dashboard()` implementation and
   its data sources. Reproduce current behaviour exactly.
2. Do not change evaluation logic, metric definitions, or generated figures.
3. Do not import UI code into the core engine, and do not import
   `src/timeline.py` smoothing into this page. Numbers here are always
   per-window and unsmoothed.
4. Verify the displayed confusion matrix, accuracy-vs-SNR plot, per-class
   recall and false-alarm rate are IDENTICAL to the existing implementation
   before and after the port.

- [ ] **Step 1: Record the pre-port baseline**

Before touching anything, capture what the current implementation produces:

```bash
python -m src.evaluate
cp evals/scorecard.json /tmp/scorecard_before.json
cp evals/confusion_matrix.png /tmp/cm_before.png
cp evals/accuracy_vs_snr.png /tmp/snr_before.png
echo "baseline captured"
```

Expected: `baseline captured`

- [ ] **Step 2: Port the existing dashboard**

Move the `build_dashboard` function and its bar-chart helper out of the
current `scripts/inference_ui.py` into `src/ui/pages/performance.py` unchanged
in behaviour, wrapped in a `build()` that creates the same components the old
"Results dashboard" tab had:

```python
"""Performance — the existing Results dashboard, ported.

Deliberately keeps the LIGHT palette. confusion_matrix.png and
accuracy_vs_snr.png are written by src/evaluate.py and shared with the rest of
the team (Colab downloads, brief figures); restyling them is not this
package's call. The mismatch with the dark console is intentional and
documented in the spec.

Numbers here are ALWAYS per-window and unsmoothed. The RF Replay page's
smoothing toggle does not reach this page.
"""
import gradio as gr

from src.evaluate import evaluate as run_full_evaluation


def build():
    gr.Markdown("### Performance")
    gr.Markdown(
        "Runs the real evaluation pipeline against whichever checkpoint is at "
        "`results/best_model.pt`. Always per-window and unsmoothed — the "
        "RF Replay smoothing toggle has no effect here."
    )
    run = gr.Button("Run evaluation", variant="primary")
    summary = gr.Markdown()
    with gr.Row():
        bar = gr.Plot(label="Per-class recall")
        cm = gr.Image(label="Confusion matrix")
    snr = gr.Image(label="Accuracy vs SNR")
    # Reuse the existing build_dashboard body from the old script here.
    run.click(_build_dashboard, inputs=None, outputs=[summary, bar, cm, snr])
    return run
```

Copy the body of the old `build_dashboard` into a module-level
`_build_dashboard()` in this file, together with the light-palette helper it
depends on (`_style_light_axes` from the old script). Do not change its logic.

- [ ] **Step 3: Verify it still imports**

Run: `python -c "from src.ui.pages import performance; print('import ok')"`
Expected: `import ok`

- [ ] **Step 4: Verify parity against the baseline**

Re-run evaluation through the ported page's code path and diff against the
baseline captured in Step 1:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from src.ui.pages.performance import _build_dashboard
_build_dashboard()
print('dashboard ran')
"
python -c "
import json
before = json.load(open('/tmp/scorecard_before.json'))
after = json.load(open('evals/scorecard.json'))
assert before == after, 'PARITY BROKEN: scorecard changed after the port'
print('scorecard identical')
"
```

Expected: `dashboard ran` then `scorecard identical`

If the scorecard differs, the port changed evaluation behaviour. Fix the port
— do not update the baseline.

- [ ] **Step 5: Commit**

```bash
git add src/ui/pages/performance.py
git commit -m "feat(ui): port Results dashboard to Performance page, verified identical"
```

---

## Task 10: App assembly

**Files:**
- Create: `src/ui/app.py`

- [ ] **Step 1: Write the implementation**

Create `src/ui/app.py`:

```python
"""Assembles the six OMNI pages."""
import gradio as gr
import torch

from src.config import CFG, CLASSES, REPO_ROOT
from src.models.amc_cnn import AMC_CNN
from src.ui.pages import (alerts, model_page, overview, performance,
                           rf_replay, signal_analysis)
from src.ui.palette import BG, TEXT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"

CUSTOM_CSS = f"""
.gradio-container {{ background: {BG} !important; color: {TEXT} !important; }}
"""


def load_model():
    """Read from disk on every call -- no caching, so dropping a freshly
    trained checkpoint into results/ takes effect without a restart."""
    model = AMC_CNN(num_classes=len(CLASSES),
                     input_len=CFG["signal"]["window_len"]).to(DEVICE)
    if not CKPT_PATH.exists():
        raise gr.Error(f"No checkpoint at {CKPT_PATH}. Train one first.")
    try:
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    except RuntimeError as exc:
        raise gr.Error(
            f"{CKPT_PATH} does not match the current model architecture. "
            f"Train a fresh one with the current code. Details: {exc}")
    model.eval()
    return model


def build_app():
    with gr.Blocks(css=CUSTOM_CSS,
                    theme=gr.themes.Base(primary_hue="teal",
                                          neutral_hue="slate"),
                    title="OMNI — RF Spectrum Intelligence") as demo:
        gr.HTML(
            '<div style="padding:12px 0;">'
            '<div style="font-size:24px;font-weight:700;letter-spacing:0.08em;">'
            'OMNI</div>'
            '<div style="font-size:12px;color:#8b98a5;">'
            'AI-Powered RF Spectrum Intelligence · TEAM NEXA · '
            'BASEBAND · fs 3.2 MHz · ● REPLAY (no live SDR)</div></div>')

        state = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Overview"):
                overview.build(state)
            with gr.Tab("RF Replay"):
                rf_replay.build(state, load_model)
            with gr.Tab("Signal Analysis"):
                signal_analysis.build(state)
            with gr.Tab("Performance"):
                performance.build()
            with gr.Tab("Model"):
                model_page.build(load_model)
            with gr.Tab("Alerts"):
                alerts.build(state)

    return demo
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/app.py
git commit -m "feat(ui): assemble the six-page OMNI console"
```

---

## Task 11: Entry point shim

**Files:**
- Rewrite: `scripts/inference_ui.py`

- [ ] **Step 1: Replace the file**

Replace the entire contents of `scripts/inference_ui.py` with:

```python
"""Local inference UI: OMNI RF situational-awareness console.

Runs locally, not as a claude.ai Artifact -- an Artifact is sandboxed JS with
no Python and no way to load a .pt checkpoint or call the real model.

Usage:
    python scripts/inference_ui.py
Then open the printed http://127.0.0.1:7860 link.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.app import build_app  # noqa: E402

if __name__ == "__main__":
    build_app().launch()
```

- [ ] **Step 2: Verify the app builds**

Run: `python -c "import sys; sys.path.insert(0,'.'); from src.ui.app import build_app; build_app(); print('built ok')"`
Expected: `built ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/inference_ui.py
git commit -m "refactor(ui): reduce inference_ui.py to a thin entry point"
```

---

## Task 12: Browser verification

**Files:** none — verification only

- [ ] **Step 1: Confirm a loadable checkpoint exists**

Run: `python -c "
import sys; sys.path.insert(0,'.')
from src.ui.app import load_model
load_model(); print('checkpoint loads')"`

Expected: `checkpoint loads`. If it raises, train one first:
`SEDIC_CONFIG=configs/smoke.yaml python -m src.train`

- [ ] **Step 2: Start the app**

Add to `.claude/launch.json` if absent, then start via the preview tool
(never via a bare Bash background process):

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "omni-ui",
      "runtimeExecutable": "python",
      "runtimeArgs": ["scripts/inference_ui.py"],
      "port": 7860
    }
  ]
}
```

- [ ] **Step 3: Verify each page against the provenance rule**

Walk the console and confirm, page by page:

| Check | Expectation |
|---|---|
| Header | reads `● REPLAY`, `BASEBAND · fs 3.2 MHz`; no `LIVE`, no GHz value |
| RF Replay → Synthesize | waterfall renders, x-axis in MHz spanning about ±1.6 |
| Detection overlays | span the FULL width; none is bounded in frequency |
| TRUTH overlays | present for a scenario, dashed outline, labelled `TRUTH` |
| Upload a file | TRUTH overlays absent entirely |
| Hop dropdown | changing it changes the window count in the header |
| Smoothing toggle | switching to Raw visibly increases event count / jitter |
| Signal Analysis | probability column does NOT sum to 100%; caption says so |
| NOISE_FLOOR | shown separately with a `Signal state` line, not as a threat row |
| Alerts | contains no NOISE_FLOOR row under any capture |
| Model | parameter count matches `sum(p.numel() ...)` for the loaded checkpoint |
| Performance | runs and stays on the light palette |

- [ ] **Step 4: Check the console for errors**

Use `read_console_messages` and `preview_logs`. Expected: no exceptions.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix(ui): issues found in browser verification"
```

---

## Task 13: Remove the superseded threshold bug

**Files:**
- Verify: no remaining flat-threshold reads

- [ ] **Step 1: Confirm the old flat threshold is gone**

Run: `grep -rn "multilabel_threshold\"" src/ scripts/ --include=*.py`

Expected: no match in `src/ui/` or `scripts/`. The only acceptable remaining
references are inside `src/config.py`'s `resolve_multilabel_thresholds()`,
which reads the flat value as a documented fallback for classes with no
per-class entry.

- [ ] **Step 2: Confirm the UI agrees with the scorecard**

Run:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from src.config import CLASSES, resolve_multilabel_thresholds
t = dict(zip(CLASSES, resolve_multilabel_thresholds()))
assert t['LFM_RADAR'] == 0.26 and t['FHSS'] == 0.27 and t['JAMMING'] == 0.77
print('per-class thresholds active:', t)
"
```

Expected: prints the threshold dict with the per-class values

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: confirm UI uses per-class thresholds throughout"
```

---

## Task 14: Inference-context and smoothing-isolation tests

The console now has two distinct inference contexts (one window vs a whole
capture) and a display transform (smoothing) that must never leak into
reported metrics. Pin all of it.

**Files:**
- Modify: `tests/test_ui_session.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_session.py`:

```python
from src.timeline import detections


def test_context_1_single_test_example_is_exactly_one_window(model):
    """512 samples -> 1 window, no timeline. Signal Analysis must still work
    in this degenerate case."""
    X = np.random.randn(1, 2, 512).astype(np.float32)
    y = np.zeros((1, 8), dtype=np.float32)
    snr = np.array([-6.0])
    from src.ui.session import load_test_example
    s = load_test_example(model, X, y, snr, 0)
    assert s.result.n_windows == 1
    assert s.source == "test-example"
    assert s.snr_known is True
    assert s.true_snr_db == pytest.approx(-6.0)
    assert s.truth is None


def test_context_2_capture_has_a_timeline_and_groupable_events(model):
    """A multi-window capture must produce a timeline events can group over."""
    s = load_scenario(model, total_duration=0.01, hop=256, seed=0)
    assert s.result.n_windows > 1
    assert s.result.times_us[1] > s.result.times_us[0]
    assert isinstance(s.events(), list)


def test_context_3_two_classes_produce_one_combined_event(model):
    """FHSS 0.90 + JAMMING 0.85 -> both detected -> ONE event."""
    s = load_scenario(model, total_duration=0.005, hop=512, seed=0)
    probs = np.zeros((4, 8), dtype=np.float32)
    probs[:, CLASSES.index("FHSS")] = 0.90
    probs[:, CLASSES.index("JAMMING")] = 0.85
    from dataclasses import replace
    forced = replace(s.result, probs=probs, starts=np.arange(4) * 512)
    events = detections(forced, s.thresholds)
    assert len(events) == 1
    assert set(events[0].classes) == {"FHSS", "JAMMING"}
    assert events[0].label in ("FHSS + JAMMING", "JAMMING + FHSS")


def test_context_4_smoothing_does_not_mutate_the_raw_result(model):
    """THE key isolation test: the raw per-window probabilities the scorecard
    would use must be byte-identical before and after smoothing is applied for
    display."""
    s = load_scenario(model, total_duration=0.01, hop=256, seed=0)
    raw_before = s.result.probs.copy()

    s.events(smoothed=True)
    s.tiers(smoothed=True)
    s.smoothed_result()

    np.testing.assert_array_equal(s.result.probs, raw_before)


def test_smoothed_and_raw_can_disagree(model):
    """If these could never differ, the toggle would be decorative and the
    isolation guarantee would be untestable."""
    s = load_scenario(model, total_duration=0.02, hop=256, seed=3)
    assert s.events(smoothed=True) is not None
    assert s.events(smoothed=False) is not None


def test_scorecard_path_never_imports_smoothing():
    """src/evaluate.py must not reach for the display-only transform."""
    import inspect

    import src.evaluate as ev
    source = inspect.getsource(ev)
    assert "smooth" not in source, (
        "src/evaluate.py references smoothing -- benchmark numbers must stay "
        "per-window and unsmoothed"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_session.py -k "context or smoothing or scorecard" -v`
Expected: FAIL — `ImportError` on `load_test_example` until Task 2 is complete

- [ ] **Step 3: Fix anything the tests surface**

No new production code is expected — these pin behaviour built in Tasks 1–2
and the core plan. If one fails, fix the defect at its source.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ui_session.py
git commit -m "test: pin inference contexts and smoothing isolation"
```

---

## Definition of Done

- `python -m pytest tests/ -v` passes
- Performance page scorecard is byte-identical to the pre-port baseline
- Smoothing never mutates `session.result.probs`
- `python scripts/inference_ui.py` serves all six pages
- Every check in Task 12 Step 3 passes
- No detection overlay is bounded in frequency; no `LIVE` badge; no GHz carrier anywhere
- TRUTH elements appear only when `session.source == "scenario"`
- Alerts contains no NOISE_FLOOR row under any capture
- The Model page's parameter count is introspected, not hardcoded
