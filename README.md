# sedicAI_NEXA

**SEDIC 2026 — RF/Signal Track ("Project Overwatch")**

AI model that detects and classifies radio signals from raw IQ data: civilian
modulations (BPSK/QPSK/16QAM/64QAM), military/tactical signals (LFM radar,
FHSS), and hostile jamming — across clean and noisy (low-SNR) conditions.

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/SEDIC2026_Track1_Documentation.md`](docs/SEDIC2026_Track1_Documentation.md) | Full technical plan, team roles, 4-day timeline |
| [`docs/TOOLS.md`](docs/TOOLS.md) | **Every tool, library, and dataset — with licences** |
| [`docs/TEAM_ROLES.md`](docs/TEAM_ROLES.md) | **Who owns what** — one signal category per person, end to end |
| [`docs/pipeline/`](docs/pipeline/) | One doc per stage: data → generators → preprocessing → training → evaluation → submission |
| [`docs/WORKTREES.md`](docs/WORKTREES.md) | Parallel branch workflow (optional, not needed yet) |

---

## Setup

```bash
pip install -r requirements.txt
```

Verify everything works before touching real data:

```bash
pytest -q
```

## Project structure

```
configs/        default.yaml (real run) + smoke.yaml (seconds-long dry run)
src/
  config.py       loads YAML; override with SEDIC_CONFIG env var
  generators/     radar.py, fhss.py, jamming.py — synthetic signal synthesis
  data/           preprocess.py (windowing/SNR/augment), build_dataset.py
  models/         amc_cnn.py — 1D-CNN classifier
  train.py        training loop, stratified by class AND SNR
  evaluate.py     confusion matrix, accuracy-vs-SNR, pass/fail scorecard
  infer.py        runs the model on the Qualifier IQ Stream -> classification log
tests/          DSP correctness + pipeline wiring (46 tests)
data/           raw/ interim/ processed/ — all gitignored
evals/          scorecards, plots, classification logs — gitignored
results/        model checkpoints — gitignored
docs/           technical documentation
```

**Nothing heavy is committed.** Datasets, checkpoints, and plots are all
gitignored — share those via Google Drive, not Git.

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

`src/evaluate.py` writes `evals/scorecard.json` stating plainly whether each
judged class clears the **>90% recall** benchmark the rules require.

## Tests: why they exist

The biggest risk on this track is shipping synthetic training data whose
physics is subtly wrong — the model then scores 95% on our own data and fails
the organizer's real Qualifier IQ Stream, with no time left to fix it.

`tests/` asserts the maths does what the class name claims: the LFM chirp
sweeps *linearly* at the requested rate, FHSS hops land on declared channels,
`add_awgn` achieves the SNR it was asked for, and no generator exceeds Nyquist.

These already caught two real aliasing bugs in the initial config. They verify
**internal consistency** — they cannot verify **realism** (whether the equations
match a real-world emitter). That still needs reference literature or a
signal-processing expert.

## Still to do

- `src/data/build_dataset.py::load_radioml_civilian()` is a stub — needs
  RadioML2018.01a download + parsing for the four civilian classes.
- `src/infer.py::load_iq_file()` assumes interleaved float32 — confirm the
  organizer's actual format before submitting.

## Team workflow

Four parallel workstreams, each on its own branch and git worktree — see
[`docs/WORKTREES.md`](docs/WORKTREES.md). Day-by-day role split is in section 13
of the technical documentation.
