# 03 — FHSS Class

**Owner:** Person B · **Day:** 1–2 · **Judged class — must clear >90% recall**

> **Highest-risk class in the project.** No public raw-IQ FHSS dataset exists
> that we can use, so unlike radar (which has RadChar) this class is *entirely*
> our own synthesis, validated only by our own tests. Read
> [02-radar-generation.md](02-radar-generation.md) for the contrast.

## What FHSS is

**Frequency Hopping Spread Spectrum**: the carrier jumps between predetermined
channels many times per second, following a pseudorandom hop sequence both ends
agree on in advance. Militaries use it because an eavesdropper who does not know
the sequence sees only brief fragments, and a jammer must cover every channel at
once instead of parking on one.

In a spectrogram it looks like scattered rectangular blocks at different
frequencies — visually nothing like radar's smooth diagonal streak, which is why
a CNN separates them readily *if the data is right*.

Key parameters: **hop rate** (hops/second), **number of channels**, **channel
spacing**, **dwell time** (1/hop rate).

## Implementation

`src/generators/fhss.py`

| Function | Does |
|---|---|
| `generate_fhss(fs, total_duration, hop_duration, hop_freqs, rng)` | Builds the signal by concatenating segments, each a complex exponential at a randomly chosen channel |
| `random_fhss_example(rng=...)` | One example, parameters randomised from config |

### Tools

| Tool | Use |
|---|---|
| NumPy | Complex exponentials, hop-sequence selection |
| SciPy (`stft`) | Spectrogram QA |
| Matplotlib | Viewing it |

No specialist library — a hop is a tone, and the signal is tones concatenated.
[TorchSig](https://github.com/TorchDSP/torchsig) does **not** cover FHSS
(communications modulations only), so there is no shortcut here.

### Parameters (`configs/default.yaml`)

| Parameter | Range | Note |
|---|---|---|
| `hop_rate_hz` | 25–150 kHz | **set by window length — see below** |
| `n_channels` | 8–64 | |
| `channel_spacing_hz` | 10–48 kHz | **capped by Nyquist — see below** (widened from 40 kHz after the dwell-time fix was confirmed) |

## The dwell-time bug — read this before touching hop rate

The model only ever sees **one window** (512 samples at 3.2 MHz = 160 µs). If the
dwell time is longer than that window, the window captures a single hop — and the
example is a **constant tone**, indistinguishable from tone jamming.

This was a real bug. Hop rates of 100–1000 Hz give dwell times of 1–10 ms against
a 512 µs window, so **not one training example contained a single hop**. The
generator was correct and its tests passed (hopping does occur across the full
2 ms signal), but windowing discarded all of it. Measured directly:

```
before:  every FHSS window contained 1 distinct frequency
after:   6-8 distinct frequencies per window
```

On identical smoke data, FHSS recall went from **0.17 to 0.92**.

`tests/test_config.py::test_fhss_hops_are_visible_inside_one_window` now blocks
any hop rate too slow for the window, and
`tests/test_format_contract.py` verifies multiple frequencies actually appear.

**Scoping note for the brief:** 25–150 kHz is *fast* frequency hopping. That is
the regime observable at a 160 µs window, which RadChar's format fixes. State
this as a deliberate scoping decision rather than leaving it unexplained.

## The aliasing bug this class already had

Channels are laid out as `(arange(n) - n/2) * spacing`, so the comb spans about
±(n · spacing / 2). The original config — 64 channels at 50 kHz — spanned
**±1.6 MHz against a 500 kHz Nyquist limit**. The outer hops aliased and folded
back to entirely wrong frequencies.

Training on that would have meant a class labelled FHSS that was not FHSS: the
model learns fold-back artefacts, scores well on our data, and fails the
organisers' stream. Precisely the silent failure this project cannot afford.

Fixed by raising `fs` to 3.2 MHz (shared with RadChar's native rate) and
capping `channel_spacing_hz`.
`tests/test_config.py::test_fhss_channel_comb_respects_nyquist` now blocks any
config that reintroduces it: `(n_channels/2) * channel_spacing_hz` must stay
under Nyquist (`fs/2` = 1.6 MHz). **Do not raise `n_channels` or
`channel_spacing_hz` without re-running the tests.**

Once the dwell-time fix confirmed FHSS was learnable at all, `channel_spacing_hz`
was widened further, from a 40 kHz cap to 48 kHz — using 1.536 of the 1.6 MHz
Nyquist budget (96%) instead of 1.28 MHz (80%), for broader channel-spread
coverage while staying strictly inside the enforced ceiling.

## Verification

`pytest tests/test_generators.py::TestFHSS -v`

| Test | Asserts |
|---|---|
| `test_each_hop_lands_on_a_declared_channel` | Every segment's FFT peak sits on a declared channel — the sequence is what we labelled |
| `test_signal_actually_hops` | More than one distinct frequency appears (catches a constant tone mislabelled as FHSS) |
| `test_length_matches_requested_duration` | No silent truncation |

## Results (latest run)

| Metric | Value |
|---|---|
| FHSS recall | **92.2%** — PASSES the 90% benchmark |
| Precision | 73% |
| F1 | 0.82 |

Crosses 90% recall above -2 dB SNR; degrades at low SNR (-10 dB, -6 dB) as
expected — a healthy accuracy-vs-SNR shape, not a flat/suspicious one.

Dominant confusion is with JAMMING, not radar. Cross-checking the confusion
matrix against the per-SNR curve shows this is almost entirely concentrated in
the two lowest SNR bins — consistent with faded signals of any class becoming
hard to distinguish from noise-like classes generally, not a FHSS generation
defect.

**Cross-class interaction worth tracking:** across successive fixes (radar
duty-cycle cap, then this class's parameter widening), FHSS recall rose
(82.5% -> 89.7% -> 92.2%) while JAMMING recall fell in the same three runs
(80.0% -> 73.3% -> 67.5%). Flagged to the jamming owner — may indicate the
model is trading decision-boundary space between these two classes rather than
improving both independently.

## Manual QA (still required)

1. Spectrogram should show discrete blocks scattered across frequency, each one
   dwell-time long — not a continuous band, not a diagonal
2. Count blocks against the configured hop rate — they must agree
3. Confirm no energy near the spectrum edges (a wrap-around smear means aliasing
   is back)
4. Sanity-check hop rate and channel count against published FHSS figures, and
   write the comparison into the brief

## Residual risk — state it plainly in the brief

Our hop rates and channel plans come from general literature ranges, not from any
specific emitter. If the organisers' FHSS uses markedly different timing, our
model may generalise poorly. Nothing in our control fixes that in four days
without a signal-processing expert. Two mitigations:

**1. Widen the randomisation rather than narrow it.** The instinct is to tune
these ranges until the spectrograms look convincing. Do the opposite — a broad
training distribution is more likely to contain the organisers' actual signal
than a tight guess. Real tactical radios span far wider hop rates than our
current 100–1000 Hz. Widen until Nyquist binds; `tests/test_config.py` tells you
where that ceiling is.

Then **prove it generalised**: train on one hop-rate subset, evaluate on a
disjoint one. If accuracy holds, the model learned "frequency hopping" rather
than "our hop rates". See [`07-evaluation.md`](07-evaluation.md).

**2. Say so in the technical brief.** Judges in a technical field respect a team
that names its own limitation over one that hides it.

## Definition of done

- [x] `pytest tests/test_generators.py::TestFHSS` passes
- [x] Spectrogram QA plot saved, block pattern verified against hop rate
- [ ] Parameter ranges cross-checked against a citable source
- [x] Limitation written into the brief's methodology section (drafted; pending
      insertion into the shared brief doc by P4)
