# OMNI — RF Situational-Awareness UI

Design spec, 2026-08-24. Replaces the current two-tab Gradio UI in
`scripts/inference_ui.py` with a six-page operator console inspired by
DeepSig OmniSIG Sensor, adapted to what this project's model can honestly
support.

## Guiding principle

**What the operator sees is not what the AI uses.**

The waterfall, spectrum trace and IQ waveform are human-facing
visualizations of the RF capture. The classifier consumes normalized
512-sample windows. Neither has to pretend to be the other, and nothing on
screen may imply a capability the model lacks.

Three distinct things, kept distinct throughout:

| Layer | Nature |
|---|---|
| RF capture | continuous |
| Processing window | discrete, 512 samples / 160 µs |
| Classification decision | discrete, one per window |

The UI composes discrete decisions into a continuous operational picture.

## The provenance rule

> **The UI should visualize more than the classifier knows, but never claim
> that the classifier knows what it doesn't.**

Both halves are binding.

*Visualize more* is a licence, not a grudging allowance. A waterfall,
spectrum trace, band-occupancy overlay or noise-floor readout all show the
operator far more than 8 probabilities ever could, and they belong on
screen. Withholding them would make a worse tool, not a more honest one.

*Never claim* is enforced by making provenance visible. Every element on
screen belongs to exactly one of three classes, and each class has a
distinct visual treatment so an operator can tell them apart at a glance
without reading a legend:

| Class | Source | Treatment |
|---|---|---|
| **MODEL** | classifier output — probabilities, detections, attention | tier colors (Civilian / Military / Hostile / Empty) |
| **MEASURED** | DSP on the samples — waterfall, spectrum, power, estimated SNR | neutral instrument styling, never tier colors |
| **TRUTH** | ground truth, synthesized captures only | distinct dashed/outline styling, always labelled `TRUTH` |

Rules that follow:

- No MEASURED element may be styled as a detection, and no MODEL element
  may borrow instrument styling.
- TRUTH is never rendered when `source != "scenario"`, and is never
  visually adjacent enough to a MODEL element to be mistaken for it.
- Any derived statistic states which class it came from. A number whose
  provenance is ambiguous from its name must be renamed or annotated.
- Frequency-resolved *measurement* is permitted and encouraged. Frequency-
  resolved *classification* is not, because the model has no frequency
  axis. A band-occupancy overlay computed from the STFT is fine when drawn
  in instrument styling; the same overlay in tier colors is a lie.

## Honesty constraints (non-negotiable)

These exist because violating them would claim capabilities the model does
not have, and a judge reading the model source would find it.

1. **No frequency localization in MODEL elements.** `STFTBranch.forward`
   ends in `f.mean(dim=2)` — frequency is averaged away before fusion. The
   classifier never sees which frequency anything occupied. Detection
   overlays therefore span the **full frequency height** of the waterfall
   and are bounded only in time. No detection box is ever drawn at a
   frequency. This constrains MODEL elements only; MEASURED frequency
   structure is welcome under the provenance rule.
2. **No carrier frequency.** No RF centre frequency exists anywhere in the
   codebase; `fs: 3200000` is a baseband sample rate. The UI displays
   `BASEBAND · fs 3.2 MHz` and a ±1.6 MHz axis. It never displays a GHz
   carrier.
3. **No live stream.** There is no SDR. The status badge reads `REPLAY` or
   `FILE`. Never `LIVE`.
4. **Smoothing is display-only.** Reported performance stays strictly
   per-window and unsmoothed, matching `src/evaluate.py`.
5. **Attention is per-window softmax.** Each window's 512 weights sum to 1
   independently, so heights are not comparable across windows. The UI
   labels this explicitly.
6. **Estimated SNR is labelled estimated.** See SNR policy below.

## Target architecture

Current `main` (177cb78), dual-branch `AMC_CNN`, unchanged:

- 8 classes: BPSK, QPSK, 16QAM, 64QAM, LFM_RADAR, FHSS, JAMMING, NOISE_FLOOR
- Multi-label sigmoid output — probabilities are independent and do **not**
  sum to 1; more than one class may fire on the same window
- 148,938 parameters (IQ branch 87,680 / STFT branch 9,600)
- Input `(2, 512)`; 160 µs at 3.2 MHz

NOISE_FLOOR is load-bearing and must appear everywhere the other classes
do. It exists to stop empty spectrum being reported as JAMMING.

## Capture sources

Two capture sources, both producing a continuous IQ capture. Neither is
continuous RF in the physical sense an SDR would give — a synthesized
scenario is a continuous digital IQ sequence assembled from scripted signal
segments, and an upload is a recording. Stitching stored
dataset windows into a pseudo-stream is **rejected**: `preprocess_window`
normalizes each stored window to mean 0 / std 1, destroying absolute
amplitude (so SNR and noise-floor estimation become impossible), and every
splice introduces a phase discontinuity that would corrupt the ~50% of
sliding windows straddling a seam.

### Synthesized scenario

Generators already accept `total_duration`; `build_dataset` currently
discards 92% of every generation by keeping only `iq[:512]` of 6,400
samples. Generate long instead:

- `total_duration` ~0.1 s at 3.2 MHz = 320,000 samples (~2.5 MB), ~1,249
  windows at hop 256
- Scenario scripted over time segments (quiet → radar → radar+jammer →
  quiet), composed with existing `mix_components` / `overlay_jamming`
- Raised-cosine ramp at each emitter on/off, so segment boundaries do not
  produce click artifacts (physically realistic — real transmitters ramp)
- Ground truth returned alongside, for the truth strip and true SNR

### Upload

Any interleaved float32 IQ file, full length. The current
`iq = iq[:WINDOW_LEN]` truncation in `scripts/inference_ui.py` is removed.

### The normalization rule

**Never call `preprocess_window` on the capture. Only on each window, at
inference time.**

- Raw capture retains true amplitude → honest waterfall, real noise floor,
  working SNR estimate
- Each 512-sample window is normalized immediately before the model sees
  it → exactly matches how every training example was normalized

## Sliding-window inference

- Window 512; hop selectable in the UI: 512 / **256 (default)** / 128 / 64
- Overlapping windows are correlated, not independent
- Windows batched through the model in chunks, never looped one at a time
- Cap total windows (~4,000) with a warning rather than hanging

### Temporal smoothing

Per-class EMA over each class's sigmoid probability independently, **then**
threshold via `resolve_multilabel_thresholds()`. Per-class smoothing (not
majority vote over argmax) is required to preserve co-occurrence — a jammer
overlaid on a victim signal must keep both classes.

On by default, with a raw/smoothed toggle so the distinction can be
demonstrated. Never applied to scorecard numbers.

Two pipelines, deliberately separate:

| | Path |
|---|---|
| **Benchmark** (`src/evaluate.py`) | window → model → raw sigmoid → recall / F1 / FAR |
| **Operational** (OMNI) | capture → sliding windows → model → sigmoid → EMA → threshold → event grouping → display |

Suggested wording for the technical report: *temporal probability
smoothing and event grouping are applied exclusively at the deployment
visualization layer and are not applied during benchmark evaluation.*

### Event grouping

Consecutive windows sharing a class merge into one detection event with
start time, duration and peak confidence. Without this the detections table
shows one emitter many times and `Detections: N` counts windows rather than
signals.

An event may be multi-class. A run of windows where both FHSS and JAMMING
are over threshold is **one** event labelled `FHSS + JAMMING`, not two
overlapping single-class events — grouping keys on the whole detected set,
not on one class at a time. Event rows carry: start time, duration,
detected set, peak confidence per class.

Headline the event count, not the window count — "3 detection events", not
"43 detections". Rendered form:

    DETECTION EVENTS

    01  12.4 ms - 17.2 ms   (4.8 ms)
        FHSS + JAMMING
        Peak: FHSS 94% - JAMMING 87%

    02  38.1 ms - 40.6 ms   (2.5 ms)
        LFM_RADAR
        Peak: 91%

### SNR policy

SNR is never MODEL output — the classifier does not estimate SNR. It is
either TRUTH or MEASURED, and which one is always visible:

| Source | Class | SNR shown |
|---|---|---|
| Test-set example | TRUTH | true value from `snr_labels` |
| Synthesized scenario | TRUTH | true value, we generated it |
| Upload | MEASURED | estimated from noise floor, `est. −8.4 dB` |

Estimate: noise floor from the quietest percentile of the capture, then
`10*log10(window power / noise power)`.

Both cases are marked, not just the estimate — a bare `SNR: -8.4 dB` is
never displayed for any source:

    known:      SNR  -8.4 dB  KNOWN
    estimated:  SNR  est. -8.4 dB

This is what makes "how do you know the SNR of this recording?" answerable:
for uploads it is estimated from the capture's own noise floor; for
generated scenarios and labelled test samples it is known.

Because a detection overlay labels `class / confidence / SNR`, that single
label spans two provenance classes — the first two fields are MODEL, the
third is not. The SNR field is therefore visually separated within the
label (instrument styling, not tier color) so it cannot be read as
something the classifier produced.

## Pages

Shared `CaptureSession` in `gr.State`, populated once by a load action on
RF Replay and read by every other page. No page re-runs inference.

```
CaptureSession
  iq          complex capture, raw amplitude
  result      TimelineResult (probs, starts, attn)
  source      "upload" | "scenario" | "test-example"
  truth       ground-truth segments, or None
  snr_known   bool
```

**Overview** — status strip, miniature waterfall, latest detection card,
alert counts by tier.

Status strip fields, each with its provenance fixed:

| Field | Class | Definition |
|---|---|---|
| `Inferences/s` | MEASURED | wall-clock throughput of the last run |
| `Occupancy` | MEASURED | fraction of the capture above the noise floor, from the STFT |
| `Detections` | MODEL | count of grouped events |

Alongside the strip, a **Current Window** readout — the one place the
window mechanism is stated plainly, so the underlying design is visible
without making the whole console look like a debugger:

    WINDOW   #384 / 1249
    OFFSET   98.3 ms
    SNR      est. -8.4 dB

`Channel Load` is deliberately **not** used. In an RF console that name
reads as an energy-occupancy measurement, but the obvious implementation
here — fraction of windows where a class crossed threshold — is model
output. The two are split instead: `Occupancy` is honestly MEASURED, and
detection counts are honestly MODEL. Naming a model-derived number after a
spectrum measurement is exactly the ambiguity the provenance rule exists to
catch.

**RF Replay** — the replay deck. Load controls (upload / synthesize), hop
selector, smoothing toggle, transport (play / step / scrub). Transport
reads in milliseconds, not mm:ss — a capture is ~100 ms of signal, not
minutes. Replay rate is decoupled from signal rate: 100 ms of IQ is played
back over tens of seconds of wall clock so a human can watch it, and the
header states the scenario and position:

    * REPLAY   Scenario: Radar + Jamming   42.6 ms / 100.0 ms

Power-spectrum
trace on top sharing the x-axis with the waterfall below (x = MHz,
y = time). Full-width time-bounded detection overlays labelled
class / confidence / SNR. Tier ribbon down the side.

**Signal Analysis** — inspect one window: I/Q waveform, all 8 class
probabilities as independent bars (not a single winner), each marked
detected (`✓`) or not (`○`) against its own per-class threshold rather than
ranked into a single answer, attention overlay,
window metadata (index, offset µs, 512 samples, 160 µs). Two ways in:
select a window from the RF Replay capture, or load a random held-out test
example directly (the existing "Try a random real test example" feature).
A test example is exactly one 512-sample window, so it sets
`source = "test-example"` with `result` holding a single window and no
timeline — the page must render correctly in that degenerate case.

The probability list carries a permanent caption — *independent
probabilities - multi-label* - because a reader trained on softmax UIs will
otherwise assume the column sums to 100%. It does not, and must never be
normalized to.

NOISE_FLOOR is presented apart from the other seven rather than as an
eighth threat class. When it is the detected class the page reports a
channel state, not an emitter:

    NOISE FLOOR   v 0.93
    Signal state: QUIET / NO SIGNAL

This makes its job legible: the model is not only asking "which signal is
this?" but also "is anything transmitting at all?".

**Performance** — existing Results dashboard, unchanged. Keeps the light
palette, because `confusion_matrix.png` / `accuracy_vs_snr.png` are written
by `src/evaluate.py` and shared with the team; restyling them is out of
scope.

**Model** — card describing **the checkpoint actually loaded**, not an
idealized architecture. Parameter count, class list and input shape are
introspected from the live model object and `CFG` at render time, never
hardcoded, so the page cannot go stale or misreport a swapped checkpoint.
At time of writing that resolves to 148,938 parameters, 8 classes, input
`(2, 512)`, 160 µs @ 3.2 MHz. Static prose alongside: energy-gated
attention pooling, SNR-weighted sampling `10^(-SNR/20)` (3.16× at −10 dB;
10× spread between −10 dB and +10 dB).

The page describes what is running. It does not assert that this
architecture is the best-performing one — see Out of scope.

**Alerts** — rows where a judged class (LFM_RADAR, FHSS, JAMMING) crosses
its per-class threshold, tiered Military / Hostile. Real model output only.

NOISE_FLOOR never raises an alert, by construction — it is the *absence* of
an emitter. An "alert: nothing is transmitting" would invert the purpose
of the page and of the class.

## Module layout

`scripts/inference_ui.py` remains the entry point (preserving the
documented command) but becomes a thin shim.

```
src/timeline.py          sliding-window inference, pure, testable
src/scenarios.py         synthesized captures + ground truth
src/ui/palette.py        dark console palette, tier colors
src/ui/plots.py          waterfall, spectrum trace, ribbon, attention
src/ui/pages/*.py        one module per page
src/ui/app.py            assembles tabs
scripts/inference_ui.py  entry point -> src.ui.app
```

`src/timeline.py` holds no Gradio and no matplotlib, so it is unit-testable
in isolation:

- `sliding_windows(iq, window_len, hop) -> (n, 2, 512)`
- `classify_capture(iq, model, hop) -> TimelineResult`
- `smooth(result, alpha) -> TimelineResult`
- `detections(result, thresholds) -> list[Detection]` (grouped events)
- `tier_track(result, thresholds)`

## Known bug fixed as part of this work

`scripts/inference_ui.py` uses the flat `CFG.get("multilabel_threshold",
0.5)` while `src/evaluate.py` and `src/train.py` both use
`resolve_multilabel_thresholds()` (LFM_RADAR 0.26, FHSS 0.27, JAMMING
0.77). The UI therefore under-reports LFM_RADAR badly against the project's
own scorecard. All threshold decisions move to
`resolve_multilabel_thresholds()`.

## Testing

`tests/test_timeline.py`:

- window-count math for each hop setting
- exactly-512 input yields exactly 1 window
- input shorter than 512 pads to 1 window
- a scenario with radar in a known time range places radar detections in
  that range
- event grouping merges consecutive same-class windows into one event
- smoothing preserves co-occurrence (a two-class window stays two-class)
- no MODEL element is emitted for a capture with no windows over threshold
  (guards against a UI that always shows *something*)
- TRUTH elements are absent whenever `source != "scenario"`

## Error handling

- Input < 512 samples → pad to one window, note "single window, no timeline"
- Very long file → cap window count with a warning
- Missing/stale checkpoint → existing `gr.Error` path, unchanged

## Notes for the report

Training windows are **not** overlapping slices of a shared stream. Each
training example is an independently generated signal, and
`build_dataset.py` keeps only `iq[:512]` of each. At deployment, sliding
windows land at arbitrary offsets while every training window was anchored
to sample 0 of a fresh generation. Generators randomize `time_delay_s`, so
event position within the window does vary — but the distinction is worth
stating rather than fixing.

Separately, a measured argument for the next architecture experiment, not a
UI concern: FHSS dwell is 6.7 µs (21 samples) to 40 µs (128 samples), all
comfortably inside the 160 µs window — so the information needed to
identify FHSS *is* present in every window. The `IQBranch` effective
receptive field is ~43 samples ≈ 13.4 µs, which fully spans a fast hop but
only about a third of a slow one. These are measurements; the conclusion
drawn from them is not. Suggested phrasing: *the current results suggest
that the primary limitation is the network's ability to integrate temporal
structure across the window, rather than the 160 µs observation window
itself.* "Suggest" is doing real work there — this is an architectural
inference from receptive-field analysis, not an established result. Stated
that way it is a sharper motivation for the next experiment than "add
STFT", and it is what attention pooling already exists to address.

## Out of scope

- Restyling `src/evaluate.py`'s shared figures
- Any live SDR ingest
- Reverting to a single-branch IQ-only model. The claimed 87% vs 84%
  IQ-vs-dual-branch comparison is not recorded anywhere in this repo; it
  should be reproduced and written down before any architecture change is
  considered, and that decision belongs with whoever owns training.
