# sedicAI_NEXA

**SEDIC 2026 — RF/Signal Track ("Project Overwatch")**

**Branch:** `eavan-multilabel-jamming-overlay`

AI model that detects and classifies radio signals from raw IQ data: civilian
modulations (BPSK/QPSK/16QAM/64QAM), military/tactical signals (LFM radar,
FHSS), and hostile jamming — across clean and noisy (low-SNR) conditions, and
when more than one of these overlaps in the same window (e.g. a jammer
overlaid on top of a real signal).

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/SEDIC2026_Track1_Documentation.md`](docs/SEDIC2026_Track1_Documentation.md) | Full technical plan, team roles, 4-day timeline |
| [`docs/TOOLS.md`](docs/TOOLS.md) | Every tool, library, and dataset — with licences |
| [`docs/pipeline/`](docs/pipeline/) | One doc per stage: data → generators → preprocessing → training → evaluation → submission |
| [`docs/pipeline/09-multilabel-composite-guide.md`](docs/pipeline/09-multilabel-composite-guide.md) | RadioML usage, composite generation, what to run |
| [`docs/WORKTREES.md`](docs/WORKTREES.md) | Parallel branch workflow |

---

## Setup

```bash
pip install -r requirements.txt
```

Verify everything works before touching real data:

```bash
pytest -q
```

104 tests should pass.

---

## Architecture

One model, two feature-extraction branches feeding one fused classifier —
`src/models/amc_cnn.py`, ~148,938 parameters, trained from scratch (no
pretrained backbone exists for raw RF IQ).

```
RAW IQ INPUT (2, 512) = (I, Q channels, 512 samples = 160 us)
        │
┌───────┴────────────────────┐
▼                              ▼
IQ BRANCH                  STFT BRANCH
(dilated 1D-CNN,           (spectrogram + 2D-CNN,
 128 ch, full resolution)   64 ch, coarser resolution)
        │                              │
        │                    upsample to IQ branch's resolution
        └──────────────┬───────────────┘
                        ▼
           CONCATENATE (192 channels)
                        │
           ENERGY-GATED ATTENTION POOLING
           (learned weighted pooling over time,
            also reads raw I²+Q² power directly)
                        │
           Linear(192→256) → ReLU → Dropout(0.5)
                        │
                  Linear(256→8)
                        │
                     SIGMOID
          each of the 8 classes judged independently
        → any subset can be true at once in one window
```

**Why two branches, not one representation**: raw IQ preserves phase, which
civilian modulations (BPSK/QPSK/QAM) are *defined* by — but a spectrogram
makes frequency movement (radar chirps, FHSS hops) far more explicit than
raw IQ does. Neither representation alone does both jobs well, so both
branches run over the *same* input window and get fused, rather than one
branch handling "type A" and the other "type B".

**Why the output can flag more than one class**: the branches and fusion
produce one shared feature vector before any per-class decision is made.
Multi-label capability lives entirely in the last step — an independent
sigmoid per class, not "one winner" — not in how many branches feed into it.
A window containing a real signal with a jammer overlaid on top correctly
reads as both classes present, e.g. `{QPSK, JAMMING}`.

---

## Data pipeline

Four sources combine into the training set (`src/data/build_dataset.py`):

| Source | Classes | Real or synthetic |
|---|---|---|
| RadioML 2018.01A | BPSK, QPSK, 16QAM, 64QAM | Real, external |
| RadChar | LFM_RADAR (partial, `radchar_fraction`) | Real, external |
| In-house generators (`src/generators/`) | LFM_RADAR (remainder), FHSS, JAMMING, NOISE_FLOOR | Synthetic |
| Composite overlay (`src/data/composite.py`) | Jammer overlaid on any of the above 6 (not NOISE_FLOOR, not JAMMING-on-JAMMING) | Combines the sources above |

No usable public dataset exists for FHSS or JAMMING — checked and confirmed,
see [`docs/pipeline/01-data-sources.md`](docs/pipeline/01-data-sources.md).

**Composite generation**: `overlay_jamming()` draws a random jammer
(barrage/tone/sweep) and mixes it onto a victim signal via the tested
`apply_jamming()`, at a randomized JSR (`jamming.jsr_db` range). Labeled with
**both** classes present. Additive to the dataset — standalone examples are
untouched. `overlay_fraction: 0.3` sizes how many composite examples get
generated per victim class per SNR bin.

At the default config: ~6,000 standalone examples/class, ~10,800 composite
examples, **~58,800 total**. Full detail on what happens to RadioML
specifically (it's loaded *twice* — once standalone, once as a separate
composite-victim pool) is in
[`docs/pipeline/09-multilabel-composite-guide.md`](docs/pipeline/09-multilabel-composite-guide.md).

Labels (`y.npy`) are multi-hot, shape `(N, 8)` — one column per class, `1` if
present. A standalone example has one `1`; a composite example has two.

---

## Project structure

```
configs/
  default.yaml        real run config (all tunables, with reasoning in comments)
  smoke.yaml           seconds-long dry run, same shape as default

src/
  config.py            loads YAML (SEDIC_CONFIG env var to override); multi_hot() label helper
  generators/           radar.py, fhss.py, jamming.py, noise.py — synthetic signal synthesis
  data/
    build_dataset.py     assembles X.npy/y.npy/snr_labels.npy from all 4 sources
    composite.py          jammer-overlaid-on-victim generation
    preprocess.py          windowing, AWGN, normalization, augmentation
    radchar.py              RadChar loader
    diagnose_jamming.py, diagnose_radar.py   diagnostic-only; see "Known gaps"
  models/
    amc_cnn.py           dual-branch fusion CNN, see Architecture above
  train.py              training loop, stratified by (label-combination, SNR)
  evaluate.py           per-class recall, tier metrics, comms-vs-jamming, scorecard
  infer.py              runs the model on the Qualifier IQ Stream -> classification log

scripts/
  train_ensemble.py      N-seed ensemble (averages sigmoid outputs) — the main robustness result
  measure_variance.py    quantifies seed-to-seed swing, so a real improvement can be told from noise
  inference_ui.py        local Gradio app — upload raw IQ, see every detected component
  inspect_attention.py   attention-weight-vs-amplitude inspection (Section 4 limitation writeup)
  validate_external_jamming.py, append_noise_floor.py

notebooks/
  colab_training_multilabel.ipynb   Colab training workflow for this branch

tests/                  DSP correctness + pipeline wiring + metrics correctness (104 tests)
data/                   raw/ interim/ processed/ — all gitignored
evals/                  scorecards, plots, classification logs — gitignored
results/                model checkpoints — gitignored
docs/                   technical documentation, pipeline stage docs
```

**Nothing heavy is committed.** Datasets, checkpoints, and plots are all
gitignored — share those via Google Drive, not Git.

---

## Running the pipeline

Dry run first (~15 seconds, proves the wiring, results are meaningless):

```bash
SEDIC_CONFIG=configs/smoke.yaml python -m src.data.build_dataset && SEDIC_CONFIG=configs/smoke.yaml python -m src.train && SEDIC_CONFIG=configs/smoke.yaml python -m src.evaluate
```

Then the real run:

```bash
python -m src.data.build_dataset
```
```bash
python -m src.train
```
```bash
python -m src.evaluate
```
```bash
python -m src.infer --input data/raw/qualifier_iq_stream.bin --output evals/classification_log.csv
```

For a more robust number, run the ensemble (5 seeds averaged) and check the
variance floor before trusting any single result:

```bash
python scripts/train_ensemble.py --models 5
python scripts/measure_variance.py --runs 5
```

Training on Google Colab: `notebooks/colab_training_multilabel.ipynb`. It
builds nothing itself — build locally (where RadioML/RadChar live), upload
the three small processed arrays, then train.

`src/evaluate.py` writes `evals/scorecard.json` stating plainly whether each
judged class (LFM_RADAR, FHSS, JAMMING) clears the **>80% recall** benchmark
the rules require.

---

## Output format

`src/infer.py` writes a CSV per classified window with:

- `detected_classes` — every class that crossed `multilabel_threshold`
  (default 0.5), semicolon-joined; `NONE` if nothing did
- per-class confidence
- `is_threat` — true if any detected class is in `judged_classes`
- `status` — `TRACKED` / `MONITOR` / `INVESTIGATE`
- one presence flag per tier (`civilian_present`, `military_present`,
  `hostile_present`, `empty_present`) — a tier is present if any of its
  member classes is detected, so a window can show multiple tiers present
  at once

`scripts/inference_ui.py` (local Gradio app) shows the same thing visually —
upload a raw IQ file and it renders one tier-colored badge per detected
class, so an overlapping signal shows every component it's made of.

`src/evaluate.py`'s tier metrics (`coarse_tier_metrics`, `comms_vs_jamming`)
work the same way: presence-based per tier/class, not "which single class
was predicted" — so a civilian window that's also jammed is evaluated on
whether *both* were caught, not forced into an either/or bucket.

---

## Known gaps / still to do

- `src/data/diagnose_jamming.py` and `diagnose_radar.py` still assume
  single-label `argmax` output. Diagnostic only (not part of the scored
  benchmark), safe to skip for now.
- `overlay_fraction` (0.3) is an untested default — validate against real
  composite-recall numbers before trusting it, the same way
  `examples_per_class_per_snr` was validated (see that key's comments in
  `configs/default.yaml` for the actual experiment that settled it).
- Composite generation currently covers only jammer-overlaid-on-victim
  (matches the rules' "comms vs hostile CEMA" scoring criterion). Other
  physically-realistic overlaps — two civilian signals, civilian+military,
  military+military, all without a jammer — are not yet generated.
- 3+-signal composites (not just pairs) are architecturally representable
  (sigmoid doesn't care how many bits are true) but have no training data
  yet — untested generalization from 2-signal training.
- `src/infer.py::load_iq_file()` assumes interleaved float32 — confirm the
  organizer's actual Qualifier Stream format before submitting.

## Tests: why they exist

The biggest risk on this track is shipping synthetic training data whose
physics is subtly wrong — the model then scores well on our own data and
fails the organizer's real Qualifier IQ Stream, with no time left to fix it.

`tests/` asserts the maths does what the class name claims: the LFM chirp
sweeps *linearly* at the requested rate, FHSS hops land on declared channels,
`add_awgn` achieves the SNR it was asked for, no generator exceeds Nyquist,
composite overlays hit their requested JSR, and the multi-label tier/comms-
vs-jamming metrics handle composite windows correctly (both components
caught, one caught, or neither).

These verify **internal consistency** — they cannot verify **realism**
(whether the equations match a real-world emitter). That still needs
reference literature or a signal-processing expert.

## Team workflow

Four parallel workstreams, each on its own branch and git worktree — see
[`docs/WORKTREES.md`](docs/WORKTREES.md). Day-by-day role split is in section
13 of the technical documentation.
