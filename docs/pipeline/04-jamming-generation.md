# 04 — Jamming Class

**Owner:** Person C · **Day:** 1–2 · **Judged class — must clear >90% recall**

## Is jamming actually mandatory?

The rules list distinguishing jamming under *"Competitive Advantage"*, which
reads optional — but the Evaluation section requires **">90% recall specifically
on the High Priority (Military/CEMA) and Jamming classes."**

**Treat it as mandatory.** The benchmark names it explicitly.

## What jamming is

Deliberate interference intended to deny use of the spectrum. Three forms we
generate:

| Type | What it is | Spectrogram look |
|---|---|---|
| **Barrage** | Wideband noise across the whole band | Uniform haze everywhere |
| **Tone (CW)** | One or a few continuous carriers | Solid horizontal lines |
| **Sweep** | A carrier swept rapidly and repeatedly | Repeating diagonals |

**JSR (Jammer-to-Signal Ratio)** sets how much stronger the jammer is than the
victim signal, in dB.

### The hard part: sweep jamming vs. LFM radar

Both are swept chirps. The distinction is behavioural, not structural — radar
sweeps in short pulses at a regular PRI with silence between; a sweep jammer
sweeps continuously and much faster, with no listening gaps.

Expect these two to be the confusion pair, and **check that cell of the
confusion matrix first** when recall falls short. If they blur together,
separate them by making sweep-jammer sweep rates distinctly faster than radar
chirp rates in the config, and keep the radar pulse train's duty cycle low.

## Implementation

`src/generators/jamming.py`

| Function | Does |
|---|---|
| `generate_barrage_jamming(n_samples, rng)` | Complex Gaussian noise |
| `generate_tone_jamming(fs, n_samples, freqs)` | Sum of complex exponentials |
| `generate_sweep_jamming(fs, duration, bandwidth)` | Reuses the chirp generator at jamming-typical rates |
| `apply_jamming(signal, jammer, jsr_db)` | Overlays a jammer at a controlled JSR |
| `random_jamming_example(rng=...)` | One example, type and parameters randomised |

### Tools

| Tool | Use |
|---|---|
| NumPy | Noise generation, complex exponentials, power scaling |
| SciPy (`stft`) | Spectrogram QA |
| `src/generators/radar.py` | Chirp maths reused for sweep jamming |

No dedicated jamming library exists that fits raw-IQ classification. Published
[RF jamming datasets](https://ieee-dataport.org/documents/rf-jamming-dataset-vehicular-wireless-networks)
are mostly link-layer metrics (RSSI, SINR, packet delivery ratio) rather than raw
IQ, so they do not fit our input format.

### Parameters (`configs/default.yaml`)

| Parameter | Range |
|---|---|
| `jsr_db` | 0–20 dB |
| `max_tones` | 3 |
| `sweep_bandwidth_hz` | 100–500 kHz |

## Verification

`pytest tests/test_generators.py::TestJamming -v`

| Test | Asserts |
|---|---|
| `test_applied_jsr_matches_request` | Measured JSR matches the request within 0.5 dB across 0/5/10/20 dB — if this drifts, every JSR label is wrong |
| `test_tone_jammer_sits_at_requested_frequencies` | Each requested tone dominates its spectral neighbourhood |
| `test_barrage_jammer_is_broadband` | Energy is spread, not concentrated in one bin |

Plus `test_chirp_frequencies_respect_nyquist` covers sweep-jamming bandwidth.

## Manual QA (still required)

1. Plot all three types — each should match its row in the table above
2. Overlay a jammer on a civilian signal at 0 dB and 20 dB JSR; at 20 dB the
   victim should be visibly swamped
3. **Put a sweep-jamming and an LFM-radar spectrogram side by side.** If you
   cannot tell them apart by eye, the model will not either — widen the
   sweep-rate separation before training

## Design decision to make on Day 1

Is jamming a **standalone class**, or **jammed-civilian** examples?

- *Standalone* (current implementation): jammer alone, labelled JAMMING. Simple,
  matches the seven-class scheme, already tested.
- *Overlay*: jammer applied on top of a civilian signal via `apply_jamming()`.
  More realistic — real jamming targets something — but blurs the class boundary
  with the civilian classes.

Current code supports both; `random_jamming_example()` does standalone.
**Recommendation:** start standalone for a clean baseline, then add a portion of
overlay examples if radar/FHSS recall is already clearing 90% and time remains.
Document whichever you ship.

## Definition of done

- [ ] `pytest tests/test_generators.py::TestJamming` passes
- [ ] All three types' spectrograms saved for the brief
- [ ] Sweep-jamming vs. LFM-radar visually separable
- [ ] Standalone-vs-overlay decision made and written down
