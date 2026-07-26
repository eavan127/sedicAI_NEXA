# 02 — Radar (LFM) Class

**Owner:** Person A · **Day:** 1–2 · **Judged class — must clear >90% recall**

## What an LFM pulse is

A **chirp**: a pulse whose frequency sweeps linearly from low to high (or high
to low) over its duration. Phase is quadratic in time, so frequency — the phase
derivative — is linear. Radars use it because sweeping across a wide bandwidth
gives fine range resolution while keeping the transmitter at constant power.

Repeated at a fixed **PRI** (Pulse Repetition Interval), it forms a pulse train:
short bursts of energy separated by silence. That burst-and-gap pattern, plus
the diagonal streak in a spectrogram, is what the model learns.

Key parameters: **pulse width** (how long each pulse lasts), **bandwidth** (how
far the frequency sweeps), **PRI** (gap between pulses), **sweep direction**.

## Primary source: RadChar

Use the [RadChar](https://github.com/abcxyzi/RadChar) dataset's LFM pulses as
the main training data — real published labelled radar IQ, SNR -20…+20 dB. See
[01-data-sources.md](01-data-sources.md).

This matters: radar is a judged class, and using published data instead of our
own synthesis removes the "nobody qualified validated this" risk for it.

## Secondary: our generator

`src/generators/radar.py` — kept for augmentation and for topping up SNR bins
RadChar covers thinly.

| Function | Does |
|---|---|
| `generate_lfm_chirp_iq(fs, duration, bandwidth, f_start)` | One complex-baseband chirp |
| `embed_pulse_train(pulse, pri, fs, total_duration)` | Repeats it at a PRI |
| `random_radar_example(rng=...)` | One example, parameters randomised from config |

### Tools

| Tool | Use |
|---|---|
| NumPy | Phase/complex-exponential maths — this is all it takes |
| SciPy (`scipy.signal.stft`) | Spectrogram for QA |
| Matplotlib | Viewing that spectrogram |
| [gr-plasma](https://github.com/ShaneFlandermeyer/gr-plasma) (optional) | GNU Radio radar module, if we want a GNU Radio cross-check for the brief |

We generate the chirp directly as `exp(j·2π(f₀t + ½kt²))` rather than via
`scipy.signal.chirp` + Hilbert transform — fewer steps, no Hilbert edge
artefacts, and the complex baseband form is what we actually want.

### Parameters (`configs/default.yaml`)

| Parameter | Range | Note |
|---|---|---|
| `pulse_width_s` | 10–100 µs | literature-informed |
| `bandwidth_hz` | 50 kHz – 1 MHz | |
| `pri_s` | 1–10 ms | |
| sweep direction | up / down | randomised per example |

Randomised **per example**. A fixed parameter set teaches the model one specific
emitter instead of the general concept of a chirp — and the organisers' stream
will not use our numbers.

## Verification

`pytest tests/test_generators.py::TestRadar -v`

| Test | Asserts |
|---|---|
| `test_chirp_sweeps_linearly_at_requested_rate` | Instantaneous frequency is linear in time, slope = bandwidth/duration, residual < 1% |
| `test_chirp_spans_the_requested_bandwidth` | Sweep covers the bandwidth asked for |
| `test_chirp_has_constant_envelope` | Amplitude flat — phase modulation only |
| `test_pulse_train_repeats_at_requested_pri` | Bursts with gaps, spaced at the PRI |

Plus `tests/test_config.py::test_chirp_frequencies_respect_nyquist` — a chirp of
bandwidth B spans ±B/2, so B/2 must stay under Nyquist. **This caught a real
bug:** the original 1 MHz sample rate put a 1 MHz-bandwidth sweep exactly on the
Nyquist limit. Sample rate is now 2 MHz.

## Manual QA (still required)

The tests prove the code implements a chirp correctly. They cannot prove the
*parameters* resemble a real radar. So:

1. Plot a spectrogram — expect a clear diagonal streak, repeating at the PRI
2. Compare against published LFM spectrogram figures
3. Compare our synthetic pulses against RadChar's real ones — if they look
   materially different, RadChar is right and we are wrong
4. Write what you compared against in the technical brief, limitations included

## Definition of done

- [ ] RadChar LFM extracted and loading
- [ ] Our generator's spectrogram compared side-by-side with RadChar's
- [ ] `pytest tests/test_generators.py::TestRadar` passes
- [ ] Spectrogram QA plot saved for the brief/poster
