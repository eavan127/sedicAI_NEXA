# Civilian constellation panel — RF Replay

Design spec, 2026-08-26. Adds an IQ constellation view to the RF Replay page
for captures containing civilian traffic, alongside the existing
spectrum/waterfall console figure.

## Why

The waterfall cannot distinguish civilian modulations. BPSK, QPSK, 16QAM and
64QAM are the same flat wideband smear on it at every SNR, so a case named
"Civilian only" shows an operator nothing that identifies the emitter as
civilian, let alone which scheme it uses.

The constellation is the display that carries that information: cluster count
IS the modulation order — 2 points for BPSK, 4 for QPSK, 16 for 16QAM, 64 for
64QAM. This is a display for a human. The classifier does not count clusters;
it is a CNN over I/Q plus STFT, and the panel must never be read as depicting
its reasoning.

## What the data actually allows

Three measured facts constrain the design. All three were checked against
`data/processed/X.npy` before the design was settled.

**1. Civilian scenario emitters are concatenations of independent captures.**
`_from_library` (src/scenarios.py) builds a long civilian stretch by
crossfading unrelated 512-sample RadioML windows. Each carries its own carrier
phase, so a scatter pooled over the whole civilian span sums incoherent
segments into a smear at any SNR. **The constellation must be per-window.**

**2. RadioML windows carry a residual carrier offset.** Measured by 4th-power
FFT peak: 0.002–0.01 cycles/sample, i.e. 1–5 full rotations across 512
samples. Raw I-vs-Q therefore plots a ring, not clusters.

**3. The captures are ~8 samples/symbol.** Plotting every sample plots the
pulse-shaping transitions between symbols. Symbol points require decimating at
a timing phase.

With unit-power scaling, blind 4th-power carrier removal and best-phase
decimation, QPSK cluster structure resolves at +10 dB (4th-power phase
concentration 0.66) and collapses at 0 dB (0.09) and −10 dB (0.12). The panel
is diagnostic at high SNR and honestly featureless at low SNR — which is
itself the story of why civilian recall falls off.

## Provenance

Both panels are MEASURED. They are computed from `session.iq` — the capture's
own samples — never from model output. The recovery chain applies only
scaling, de-rotation and decimation to those samples; it does not fit to an
expected constellation, so it cannot invent clusters the samples do not
contain. Points therefore take INSTRUMENT styling, the same as the spectrum
trace.

The single MODEL element on the panel is the caption's detected class and
confidence, which takes its tier colour, exactly as elsewhere in the console.

Pooling multiple windows was rejected. Per-window 4th-power de-rotation leaves
a 90° phase ambiguity; pooled BPSK would then show four clusters instead of
two, i.e. the display would assert the wrong modulation order. One window, and
the caption states the resulting point-count limit.

## Placement and visibility

A new `gr.Plot` sits directly below the console figure on RF Replay, not
inside it. The console figure's premise is one shared time axis; a
constellation has no time axis, so folding it in would break the property that
layout exists to defend.

The panel appears only when a civilian class is present in the truth script or
in the detected events, via `gr.update(value=fig, visible=True)`. It is hidden
entirely otherwise, so military-only cases look exactly as they do today.

## Components

### `recover_symbols(window)` — src/ui/plots.py

Pure function. Takes a complex window, returns `(points, offset_estimate,
timing_phase)`.

1. Unit-power normalise.
2. Estimate carrier offset from the 4th-power FFT peak.
3. De-rotate by `exp(-2πjft)`.
4. For each of the 8 timing phases, take every 8th sample; keep the phase
   whose decimated points have the tightest amplitude spread.

No model involvement. A window of fewer than 8 samples, or one with zero
power, returns the samples unchanged with a zero offset estimate rather than
raising — the caller renders it as-is.

### `CaptureSession.best_civilian_window(smoothed=None)` — src/ui/session.py

Returns `(index, class_name, prob)` for the highest civilian-probability
window, or `None` when no window carries a civilian class above threshold.
Reads `result.probs` and defaults `smoothed` to `self.display_smoothed`, so
the panel agrees with the rest of the page.

### `constellation_figure(session, smoothed=None)` — src/ui/plots.py

Returns `None` when the selector returns `None`. Otherwise builds a figure
with two square axes:

- **Left — raw I/Q.** All 512 samples, the exact `(2, 512)` array the model is
  fed.
- **Right — recovered.** The same samples through `recover_symbols`, 64 symbol
  points.

Caption strip carries: window index and time offset (ms), detected class and
confidence (tier colour, MODEL), estimated SNR for the window, the three
recovery steps by name, and the point-count limit — 64 symbols separates 2
clusters from 4 but cannot resolve 64QAM.

### Page wiring — src/ui/pages/rf_replay.py

`_render` gains one output. It returns `gr.update(value=fig, visible=True)`
when `constellation_figure` yields a figure and `gr.update(visible=False)`
otherwise. The component is added to the shared `outputs` list, so every
existing handler (scenario, upload, model change, smoothing change) updates it
without further change.

## Error handling

| Case | Behaviour |
|---|---|
| No civilian class in truth or detections | panel hidden |
| Upload with no detections at all | panel hidden |
| Window shorter than 8 samples | recovery returns samples unchanged |
| Zero-power window | de-rotation skipped, offset estimate 0 |
| No capture loaded | existing "Load a capture first." path, panel hidden |

## Testing

`tests/test_ui_plots.py` (new), against synthetic QPSK at 8 samples/symbol
with a known injected carrier offset:

- Recovered points reach 4th-power phase concentration > 0.9; the raw samples
  do not.
- The estimated offset lands within tolerance of the injected one.
- `recover_symbols` on a zero-power window returns without raising.

`tests/test_ui_session.py` (existing), for the selector:

- `None` for a radar-only capture.
- Correct window index and class for a civilian capture.

Figure-level: `constellation_figure` builds for `Civilian only` and returns
`None` for `Radar only`.

## Out of scope

- A time slider or table-driven window selection. The panel auto-picks the
  strongest civilian window; adding a control to an already-busy control bar
  is a separate decision.
- Any use of the constellation for measurement or scoring. It is a display.
  Civilian detection performance is measured on the held-out test split, as
  `_from_library` already states.
