# 05 — Preprocessing & Dataset Assembly

**Owner:** Person A + Person D · **Day:** 2

## Goal

Turn heterogeneous IQ from three sources into one uniform tensor the model can
train on: `X` of shape `(N, 2, window_len)`, `y` of class indices, `snr_labels`
for the accuracy-vs-SNR curve.

## Tools

| Tool | Use | Licence |
|---|---|---|
| NumPy | Windowing, padding, normalisation | BSD-3-Clause |
| SciPy | `resample_poly` (rate matching), `stft` (optional spectrograms) | BSD-3-Clause |
| PyYAML | Config | MIT |
| [scikit-commpy](https://github.com/veeresht/CommPy) (optional) | Rayleigh/Rician fading, if we go beyond AWGN | BSD-3-Clause |

## Steps

### 1. Rate reconciliation — do this first

The three sources disagree:

| Source | Samples/example | Rate |
|---|---|---|
| RadioML 2018.01A | 1024 | normalised |
| RadChar | 512 | 3.2 MHz |
| Our generators | configurable | 2 MHz |

**If different classes arrive at different lengths or rates, the model can learn
the artefact instead of the signal** — "512 samples ⇒ radar" scores brilliantly
on our data and collapses on the organisers' stream. This is the same class of
silent failure as the FHSS aliasing bug.

Pick one target rate and one window length, resample everything to it with
`scipy.signal.resample_poly`, and record the choice in the brief.

### 2. Windowing & normalisation

`preprocess_window()` in `src/data/preprocess.py`: truncate or zero-pad to
`window_len`, stack real and imaginary into `(2, N)`, then zero-mean/unit-std.

Normalisation matters because absolute amplitude is a receiver-gain artefact,
not a class property — without it the model can key on loudness.

### 3. SNR control

`add_awgn(signal, snr_db)` scales noise to hit a target SNR exactly. Applied
across every bin in `snr_bins_db` for every class, so the accuracy-vs-SNR curve
has data at every point.

Verified by `tests/test_generators.py::TestPreprocess::test_awgn_achieves_requested_snr`
(within 0.3 dB across -10…+15 dB). Every SNR label in the dataset depends on it.

### 4. Augmentation (training only)

`augment_iq()`: random phase rotation and time shift. Both are label-preserving
— a receiver's phase offset and capture start time carry no class information —
so they multiply effective dataset size for free.

Do **not** augment validation or test data.

### 5. Assembly

`src/data/build_dataset.py` combines everything and writes to `data/processed/`.

```bash
SEDIC_CONFIG=configs/smoke.yaml python -m src.data.build_dataset
```
```bash
python -m src.data.build_dataset
```

## Optional: spectrogram representation

**This is an alternative path, not an extra step.** Either raw IQ → 1D-CNN
(what we do), or spectrogram → 2D-CNN. You pick one; they need different input
shapes and different architectures.

```
Path A (ours):  IQ → window + normalise → (2, 1024)        → 1D-CNN
Path B:         IQ → window + STFT      → magnitude image  → 2D-CNN
```

### ⚠️ The magnitude spectrogram destroys phase

`to_spectrogram()` returns `np.abs(Zxx)` — magnitude only. The STFT itself is
invertible, but taking the magnitude **discards phase permanently**.

Our four civilian classes are *defined* by phase: BPSK and QPSK are Phase Shift
Keying; 16/64QAM are amplitude *and* phase constellations. Remove phase and they
become very hard to separate.

So Path B would help radar/FHSS (distinctive time-frequency shapes) while
badly hurting the civilian classes. Raw IQ keeps phase — it is encoded in the
relationship between the I and Q channels — which is why published AMC work uses
it and why it is our default.

### When to reconsider

Only if the 1D-CNN misses 90% **and** the confusion matrix shows radar/FHSS
confused with each other. Even then the sane version is a two-branch model (raw
IQ for civilian, spectrogram for military), not switching everything over. That
is a Phase-2 idea, not a four-day one. Do not start here.

## Class balance

`compute_class_weights()` in `src/train.py` applies inverse-frequency weighting,
so the judged classes are not drowned out by whichever source happens to be
largest. Verified by `tests/test_pipeline.py::TestClassWeights`.

RadioML is enormous — do not dump all of it in. Subsample the civilian classes to
roughly match the synthetic class counts.

## Definition of done

- [ ] Target sample rate and window length agreed, resampling implemented
- [ ] All seven classes present with non-zero counts
- [ ] SNR labels populated for every example, including RadioML/RadChar
- [ ] Class counts within roughly an order of magnitude of each other
- [ ] `python -m src.data.build_dataset` prints a sane per-class table
- [ ] `pytest tests/test_generators.py::TestPreprocess` passes
