# SEDIC 2026 — RF/Signal Track ("Project Overwatch")
## Full Technical Documentation — Phase 1 Preliminary Qualifier

---

## 1. Executive Summary

This document is the complete technical plan for attempting the SEDIC 2026 RF/Signal Track: an AI model that detects and classifies radio signals from raw IQ data into civilian modulation types, military/tactical signals (radar, frequency-hopping), and hostile jamming — evaluated at both clean (high-SNR) and noisy (low-SNR) conditions.

**Known risk (stated up front, not buried):** the mandatory benchmark (>90% recall on Military/CEMA and Jamming classes) is measured against the organizer's own "Qualifier IQ Data Stream" — a file your team has not seen. Your training data for the military/jamming classes must be synthesized yourselves (no public dataset covers it), and there is no signal-processing expert available on your team's timeline to validate that synthetic data before submission. This document includes a self-QA methodology to partially mitigate that, but it does not eliminate the risk. Treat this as the single biggest go/no-go factor for this track.

---

## 2. Competition Requirements Recap (Phase 1 only)

| Requirement | Detail |
|---|---|
| Format | Online technical proof-of-concept, no live component in Phase 1 |
| Mandatory classes | Civilian: BPSK, QPSK, 16QAM, 64QAM. Military/CEMA: Radar Pulses (LFM), FHSS bursts |
| Bonus differentiator | Distinguishing standard comms vs. hostile jamming — but note: the Evaluation section explicitly folds Jamming into the >90% mandatory benchmark, so treat it as required, not optional |
| Conditions | Must work across high-SNR (clean) and low-SNR (faded/noisy) |
| Submission package | Model source code, classification log & results (run on provided Qualifier IQ Data Stream), performance benchmark (>90% recall on Military/CEMA + Jamming), technical brief PDF, video demo (≤5 min, YouTube) |
| NOT required in Phase 1 | GUI, live demo station, poster, jury presentation — these are Phase 2 (Top 10 only) |

---

## 3. Glossary

- **IQ data**: In-phase/Quadrature — the raw complex-valued sample format a radio receiver outputs; every signal in this project is represented as a `(2, N)` real array or `N`-length complex array.
- **SNR (Signal-to-Noise Ratio)**: ratio of signal power to noise power, in dB. High SNR = clean signal. Low SNR = noisy/faded.
- **BPSK/QPSK/16QAM/64QAM**: digital modulation schemes, differing in how many bits are encoded per symbol.
- **LFM (Linear Frequency Modulation)**: a "chirp" — a pulse whose frequency sweeps linearly over time. Classic radar signature.
- **FHSS (Frequency Hopping Spread Spectrum)**: a signal that rapidly switches carrier frequency according to a hop sequence, used by military radios to resist jamming/interception.
- **Jamming**: deliberate interference (barrage/wideband noise, single/multi-tone, or sweep) meant to disrupt reception.
- **AMC (Automatic Modulation Classification)**: the academic field this whole task belongs to — well-published, with known reference architectures and known accuracy-vs-SNR behavior.
- **JSR (Jammer-to-Signal Ratio)**: how much stronger the jammer is relative to the legitimate signal.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  DATA LAYER                                              │
│  RadioML2018.01a (civilian) + synthetic generators        │
│  (radar / FHSS / jamming), all producing (2, N) IQ arrays │
└───────────────────────────┬───────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  PREPROCESSING                                            │
│  Fixed-length windowing, normalization,                   │
│  optional STFT spectrogram representation                 │
└───────────────────────────┬───────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  MODEL                                                     │
│  1D-CNN / CLDNN trained from scratch                       │
│  (no pretrained backbone exists for this domain)           │
└───────────────────────────┬───────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  EVALUATION                                                │
│  Per-class recall across SNR bins,                        │
│  confusion matrix, accuracy-vs-SNR curve                   │
└───────────────────────────┬───────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  SUBMISSION                                                │
│  Run on organizer's Qualifier IQ Data Stream               │
│  → classification log + technical brief + video demo       │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Data Strategy

### 5.1 Civilian classes — use existing data
Download **RadioML2018.01a** (DeepSig). Covers BPSK, QPSK, 16QAM, 64QAM (and others you can ignore/exclude) across SNR -20dB to +18dB, ~4096 samples/example. No generation needed — just filter to the classes you need and reformat.

### 5.2 Military — Radar (LFM chirp)

Generate directly as a complex baseband chirp (quadratic phase), not via `scipy.signal.chirp` + Hilbert transform (simpler and avoids Hilbert-transform artifacts):

```python
import numpy as np

def generate_lfm_chirp_iq(fs, duration, bandwidth, f_start=None):
    """Generate a complex-baseband LFM radar pulse (IQ)."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    f_start = f_start if f_start is not None else -bandwidth / 2
    k = bandwidth / duration  # chirp rate (Hz/s)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return np.exp(1j * phase)

def embed_pulse_train(pulse, pri, fs, total_duration):
    """Embed repeating pulses at a Pulse Repetition Interval (PRI) inside a noise floor."""
    total_samples = int(total_duration * fs)
    pri_samples = int(pri * fs)
    out = np.zeros(total_samples, dtype=complex)
    for start in range(0, total_samples - len(pulse), pri_samples):
        out[start:start + len(pulse)] += pulse
    return out
```

**Parameters to vary across your dataset** (for diversity): pulse width (10–100 µs typical), bandwidth (50kHz–1MHz), PRI (1–10ms), sweep direction (up/down chirp). Vary these randomly per generated example — don't use one fixed set of parameters for all examples, or your model will overfit to one specific radar signature instead of learning the general LFM concept.

### 5.3 Military — FHSS

```python
def generate_fhss(fs, total_duration, hop_duration, hop_freqs):
    """Generate a frequency-hopping signal from a random hop sequence."""
    samples_per_hop = int(hop_duration * fs)
    n_hops = int(total_duration / hop_duration)
    t_hop = np.arange(samples_per_hop) / fs
    signal = np.zeros(n_hops * samples_per_hop, dtype=complex)
    for i in range(n_hops):
        f = np.random.choice(hop_freqs)
        signal[i*samples_per_hop:(i+1)*samples_per_hop] = np.exp(2j * np.pi * f * t_hop)
    return signal
```

**Parameters to vary**: hop rate (100–1000 hops/sec is a reasonable literature-informed range), number of hop channels (8–64), channel spacing.

### 5.4 Jamming (barrage / tone / sweep)

```python
def generate_barrage_jamming(n_samples):
    """Wideband noise jammer."""
    return (np.random.randn(n_samples) + 1j*np.random.randn(n_samples))

def generate_tone_jamming(fs, n_samples, freqs):
    """Single or multi-tone continuous-wave jammer."""
    t = np.arange(n_samples) / fs
    return sum(np.exp(2j*np.pi*f*t) for f in freqs)

def generate_sweep_jamming(fs, duration, bandwidth, sweep_rate):
    """Repeating fast sweep jammer (distinct from a single radar chirp by faster/repeating sweep)."""
    return generate_lfm_chirp_iq(fs, duration, bandwidth)  # reuse chirp generator, tune params to jamming-typical sweep rates
```

**Overlay onto a legitimate signal** at a controlled Jammer-to-Signal Ratio:
```python
def apply_jamming(signal, jammer, jsr_db):
    sig_power = np.mean(np.abs(signal)**2)
    jam_power = np.mean(np.abs(jammer)**2)
    scale = np.sqrt((sig_power * 10**(jsr_db/10)) / jam_power)
    return signal + scale * jammer[:len(signal)]
```

### 5.5 Channel realism — AWGN + SNR control

```python
def add_awgn(signal, snr_db):
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise
```

Sweep this across your full SNR range (e.g. -10, -5, 0, 5, 10, 15 dB) for **every** class, so your model is evaluated (and the accuracy-vs-SNR curve is generated) across the same conditions the rules describe.

### 5.6 Self-QA methodology (your substitute for expert validation)

Since no signal-processing expert is confirmed available this week, do this for every synthetic class before using it in training:
1. Plot the spectrogram (`matplotlib.pyplot.specgram` or `scipy.signal.stft` + `pcolormesh`) of several generated examples.
2. Compare visually against reference spectrogram images from published radar/FHSS papers or textbook figures (search "LFM radar spectrogram example", "FHSS spectrogram example" for reference images).
3. Sanity-check parameter ranges against publicly published literature values (pulse width, hop rate, bandwidth) — never invent numbers from nothing.
4. Document this comparison explicitly in your technical brief's methodology section, including what you compared against and its limitations. This is honesty, not just covering yourselves — judges in a technical field respect a team that names its own limitation over one that hides it.

---

## 6. Preprocessing

```python
def preprocess_window(iq_complex, window_len=1024):
    """Convert complex IQ to (2, N) real array, normalized."""
    iq = iq_complex[:window_len]
    if len(iq) < window_len:
        iq = np.pad(iq, (0, window_len - len(iq)))
    arr = np.stack([iq.real, iq.imag])
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr.astype(np.float32)
```

Optional spectrogram path (recommended specifically for radar/FHSS, since their time-frequency signature is visually obvious):
```python
from scipy.signal import stft

def to_spectrogram(iq_complex, fs, nperseg=128):
    f, t, Zxx = stft(iq_complex, fs=fs, nperseg=nperseg, return_onesided=False)
    return np.abs(Zxx).astype(np.float32)
```

---

## 7. Model

Start with a straightforward 1D-CNN (fast to train, well-published baseline for AMC — don't reach for a Transformer under this time pressure, it buys you nothing but risk):

```python
import torch
import torch.nn as nn

class AMC_CNN(nn.Module):
    def __init__(self, num_classes, input_len=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        flat_len = 128 * (input_len // 4 + 1)  # adjust after checking actual shape
        self.fc1 = nn.Linear(flat_len, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
```

**Recommended head structure**: hierarchical — first predict coarse category (Civilian / Military / Jamming), then fine-grained class within that category. This concentrates model capacity on getting the judged categories (Military, Jamming) right rather than spreading effort evenly across all classes.

---

## 8. Training Strategy

```python
import torch.nn.functional as F

# Class-weighted loss — upweight Military + Jamming since these are minority
# classes in your dataset and the ones judged hardest
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0])  # example: last 3 = radar/FHSS/jamming
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
```

**Augmentation (apply during training, not just once during data generation)**:
```python
def augment_iq(arr):
    # random phase rotation
    theta = np.random.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    arr = rot @ arr
    # random time shift
    shift = np.random.randint(-20, 20)
    arr = np.roll(arr, shift, axis=1)
    return arr
```

**Train/val/test split**: stratify by both class AND SNR bin, so every split has representation across the full noise range — otherwise you won't know if your model actually handles low-SNR until the live stress test (Phase 2), which is too late.

---

## 9. Evaluation Plan

Track and report:
- **Per-class recall** (the exact metric named in the rules) — especially Military/CEMA and Jamming, must exceed 90%.
- **Confusion matrix** — reveals which classes get confused with which (e.g., is jamming being misclassified as noisy civilian signal?).
- **Accuracy-vs-SNR curve** — standard AMC evaluation plot, plot accuracy (y-axis) against SNR bins (x-axis), one line per class or overall. This is expected content for your technical brief and poster.

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
cm = confusion_matrix(y_true, y_pred)

# accuracy vs SNR
for snr in snr_bins:
    mask = (snr_true == snr)
    acc = (y_pred[mask] == y_true[mask]).mean()
    # collect and plot
```

---

## 10. Tools & Environment

| Purpose | Tool |
|---|---|
| Signal synthesis | NumPy, SciPy |
| Optional radar blocks | GNU Radio + `gr-radar` OOT module |
| Model training | PyTorch |
| Compute | Google Colab (1D-CNN is light enough to also run on CPU if needed) |
| Visualization/QA | Matplotlib (spectrograms, accuracy-vs-SNR plots) |
| Experiment tracking | TensorBoard or Weights & Biases |
| Version control | GitHub + Git LFS (for any large arrays that must be committed) |
| Brief writing | Google Docs → export PDF |
| Video demo | OBS Studio (recording) + CapCut/DaVinci Resolve (editing) |

---

## 11. GitHub Repository Structure & Workflow

```
/data          → gitignored; synthetic arrays hosted on Google Drive, not committed raw
/scripts
    gen_radar.py
    gen_fhss.py
    gen_jamming.py
    preprocess.py
    train.py
    evaluate.py
    infer.py
/notebooks     → Colab training notebook
/results       → confusion matrices, accuracy-vs-SNR plots, classification logs (small files)
/docs          → technical brief draft, this documentation, README
```

**Workflow rules:**
- `.gitignore`: all raw `.npy`/HDF5 signal arrays, model checkpoints (`.pt`), `wandb/` or `runs/` output folders.
- One script per person's task — avoids simultaneous edits to the same file.
- Branch per person (`radar-gen`, `fhss-jamming-gen`, `training-pipeline`, `docs-video`), merge to `main` at end of day only, announced in team chat first.
- Small, frequent commits with clear messages — don't let a day's work sit uncommitted.

---

## 12. Regulations & Compliance

- **Open-source requirement** (explicit in the rules): PyTorch, NumPy, SciPy, GNU Radio are all open-source — compliant.
- **No classified/restricted signal specs**: only use publicly published radar/FHSS parameter ranges from academic papers or textbooks. Never imply access to real/classified military signal specifications.
- **Dataset licensing**: cite RadioML/DeepSig's license and attribution requirements explicitly in your technical brief.
- **No real RF transmission**: keep everything in software/simulation. Do not transmit actual signals over the air via SDR hardware — that would require spectrum-authority authorization you don't have time to arrange, and isn't needed for this submission format (Phase 1 is a file-based classification log, not a live RF demo).

---

## 13. Team Roles & 4-Day Timeline

| Person | Day 1 | Day 2 | Day 3 | Day 4 |
|---|---|---|---|---|
| **A** | Download RadioML; build LFM radar generator + spectrogram QA plots | Fix radar generator per QA findings; help merge full dataset | Help evaluate model; iterate if recall <90% on Military class | Package submission (code, log, benchmark) |
| **B** | Build FHSS generator + spectrogram QA plots | Fix FHSS generator; help merge dataset | Monitor training; tune class weights/hyperparameters | Finish technical brief |
| **C** | Build jamming generator (barrage/tone/sweep) + spectrogram QA plots | Fix jamming generator; kick off real training run | Run inference on organizer's Qualifier IQ Data Stream → classification log | Generate accuracy-vs-SNR plots + confusion matrix for brief |
| **D** | Set up training pipeline; dry-run on RadioML-only data to confirm pipeline works end-to-end | Continue pipeline; prep evaluation scripts | Start recording video demo | Final video edit; submit with buffer time before deadline |

**Structural risk to note explicitly**: Days 1–2 have no independent check — the same 4 people generating the synthetic signals are also the ones QA-checking them. Unlike Track 2 (where a labeling gap just costs you some accuracy points), a flaw here that survives self-QA doesn't surface until the organizer's Qualifier IQ Data Stream is run against your model — which is also your Phase 1 submission, with no time left to fix it.

---

## 14. Submission Checklist

- [ ] Model source code (PyTorch), clean and runnable
- [ ] Classification log generated by running the model on the organizer's Qualifier IQ Data Stream
- [ ] Performance benchmark: recall >90% on Military/CEMA and Jamming classes, documented
- [ ] Technical brief PDF: dataset methodology (including synthetic generation + self-QA process), architecture, training details, confusion matrix, accuracy-vs-SNR curve, honest discussion of limitations
- [ ] Video demonstration (≤5 min, uploaded to YouTube): explain architecture, show classification log results
- [ ] All sources/licenses cited (RadioML/DeepSig, any reference literature used for parameter validation)

---

## 15. Appendix — Quick Reference: Parameter Ranges Used in Literature

| Signal | Parameter | Typical range (for realism) |
|---|---|---|
| LFM Radar | Pulse width | 10–100 µs |
| LFM Radar | Bandwidth | 50 kHz – 1 MHz |
| LFM Radar | PRI | 1–10 ms |
| FHSS | Hop rate | 100–1000 hops/sec |
| FHSS | Number of channels | 8–64 |
| Jamming | JSR | 0–20 dB (vary across examples) |

*(These are general literature-informed starting points, not guarantees of matching the organizer's actual Qualifier IQ Data Stream — validate against your own research before finalizing.)*
