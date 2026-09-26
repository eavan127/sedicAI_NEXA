# The submission bundle — what is in it and which numbers to trust

The Drive folder `sedic/results_keep_c2/` holds the model we submit: the
**C2 (`stft_keep_rows`) 5-model ensemble**. This page says what each file is,
which ones are authoritative, and how to install the bundle.

Put a copy of this file in the Drive folder alongside the checkpoints.

## The headline numbers

Measured on the held-out test split of `eavan-retrain`, 5-model ensemble, at the
calibrated thresholds:

| judged class | recall | |
|---|---|---|
| LFM_RADAR | 0.8415 | PASS |
| FHSS | 0.8291 | PASS |
| JAMMING | 0.8399 | PASS |

All five unjudged classes also clear 80% recall. JAMMING precision is 0.9973 —
7 false positives in 19,260 windows — and the comms-vs-jamming false-alarm rate
is 7.6e-05.

Quote **recall** for the benchmark. If you quote precision, quote it next to the
threshold it was measured at, because these thresholds are deliberately low
(see below).

## What each file is

| File | Keep? | What it is |
|---|---|---|
| `ensemble_0..4.pt` | ✅ | the submission. 181,898 parameters each |
| `best_model.pt` | ✅ | single-model baseline; the app's Model dropdown and the ONNX export use it |
| `thresholds_rows_ens.json` | ✅ | the calibrated thresholds these checkpoints need |
| `eval_rows_ens.json` | ✅ | full evaluation at those thresholds — per-class, by-SNR, by-context |
| `jsr_rows_*.json` | ✅ | FHSS recall vs jammer strength (`scripts/probe_jsr.py`) |
| `high_snr_rows_*.json` | ✅ | recall split by standalone / mixture / overlay (`scripts/high_snr_probe.py`) |
| `evals/scorecard.json` | ⚠️ **replace** | see below |
| `evals/ensemble_scorecard.json` | ⚠️ **replace** | see below |
| `evals/accuracy_vs_snr.png` | ⚠️ **replace** | drawn at the old thresholds |
| `evals/confusion_matrix.png` | ⚠️ **replace** | drawn at the old thresholds |
| `evals/csv/` | ⚠️ **replace** | same numbers as the old scorecard, reshaped for Power BI |

## Why those five need replacing

**Both scorecards predate calibration.** Thresholds are chosen *after* training,
by `scripts/calibrate_thresholds.py`, so anything written during the training
run measures the model at thresholds it does not ship with.

The tell is in the shape of the numbers. At the calibrated thresholds QPSK reads
**0.847 recall / 0.393 precision**. The stale scorecard reads **0.477 / 0.792** —
high precision, low recall, which is what a too-high threshold looks like. Same
model, same weights; only the decision points differ.

Nothing in the old files said so, which is why this was easy to miss. Scorecards
written from now on carry a `provenance` block naming their thresholds,
checkpoints, architecture flags, parameter count, dataset fingerprint and test-set
size. **If a scorecard has no `provenance` block, it predates this fix — check it
before trusting it.**

## Regenerating them

Run `notebooks/colab_rescore_keep_c2.ipynb`. It is **evaluation only, no
training** — a forward pass over the test split, a few minutes, CPU is fine.

It must run against `sedic/eavan-retrain`. That matters: the work machine's
`data/processed` is the Sep-22 RadChar-fix build, which differs from
`eavan-retrain` on the radar windows specifically. Measured, the civilian recalls
agree to four decimals (BPSK 0.8291, QPSK 0.8471, 16QAM 0.8500, 64QAM 0.8584)
while **LFM_RADAR reads 0.8082 instead of 0.8415**. So 0.8082 is the signature of
the wrong dataset — if you see it, check `DRIVE_DATA`.

The two scorecards come from different scripts:

| File | Written by |
|---|---|
| `scorecard.json` | `python -m src.evaluate --ensemble --n-models 5` |
| `ensemble_scorecard.json` | `python scripts/train_ensemble.py --models 5 --eval-only` |

`--eval-only` re-scores the existing checkpoints at the current thresholds. It
does not retrain and does not touch `ensemble_*.pt`. It is also the only place
the per-member recalls appear, which is where seed spread is visible.

## Installing the bundle

```bash
cp ensemble_*.pt best_model.pt  results/
cp evals/*.json evals/*.png     evals/
```

Then set the thresholds from `thresholds_rows_ens.json` into
`configs/default.yaml` under `multilabel_thresholds_per_class`, and make sure the
architecture flags read:

```yaml
stft_freq_summary: false
stft_keep_rows:    true
cumulant_features: false
```

**The config and the checkpoints move together.** With `stft_keep_rows: true` the
model is 181,898 parameters, and nothing trained flag-off will load into it —
`src/evaluate.py`, `calibrate_thresholds.py`, the UI console and `web/build.py`
all build a model from the config and load `results/*.pt` into it. Get them out of
step and every one of those breaks. `tests/test_amc_cnn.py` guards this and names
both parameter counts when they disagree.

Confirm before going further:

```bash
python -m pytest tests/test_amc_cnn.py -q
python -m src.evaluate --ensemble --n-models 5
```

## Publishing to the web app

```bash
python scripts/export_onnx.py     # results/onnx/, verified against PyTorch
python web/build.py               # web/models/, web/data/
```

`export_onnx.py` exits non-zero if any export disagrees with its checkpoint, so a
green run is a real guarantee. `web/build.py` reads `evals/*.json` — replace those
**before** building or the site publishes the stale numbers. `evals/` is
gitignored, so the figures reach the site through `web/data/`, which is committed.

`web/vercel-build.mjs` runs at deploy time and only injects `SUPABASE_ANON_KEY`.
It does not rebuild models, so a model change always needs the two commands above.
